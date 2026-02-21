import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .config import RTDETRTrainConfig
from ..v9.trainer import LinearLRScheduler, CosineAnnealingScheduler
from ..v9.transforms import V9TrainTransform, V9MosaicMixupDataset
from ..training.ema import ModelEMA
from ..training.dataset import YOLODataset, COCODataset, create_dataloader
from ..data import load_data_config
from .loss import RTDETRLoss

logger = logging.getLogger(__name__)

def convert_targets_for_detr(targets, batch_size):
    # targets is [B, max_labels, 5] where columns are (cls, x1, y1, x2, y2)
    detr_targets = []
    for i in range(batch_size):
        batch_targets = targets[i]
        
        # Valid boxes have x2 > x1 and y2 > y1
        mask = (batch_targets[:, 3] > batch_targets[:, 1]) & (batch_targets[:, 4] > batch_targets[:, 2])
        valid_targets = batch_targets[mask]
        
        labels = valid_targets[:, 0].long()
        xyxy = valid_targets[:, 1:5]
        
        if len(labels) == 0:
            detr_targets.append({
                "labels": torch.zeros(0, dtype=torch.int64, device=targets.device),
                "boxes": torch.zeros(0, 4, dtype=torch.float32, device=targets.device)
            })
        else:
            # Convert xyxy to cxcywh
            w = xyxy[:, 2] - xyxy[:, 0]
            h = xyxy[:, 3] - xyxy[:, 1]
            cx = xyxy[:, 0] + w / 2
            cy = xyxy[:, 1] + h / 2
            boxes = torch.stack([cx, cy, w, h], dim=-1)
            detr_targets.append({"labels": labels, "boxes": boxes})
            
    return detr_targets


class RTDETRTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: RTDETRTrainConfig,
        wrapper_model: Optional[Any] = None,
    ):
        self.model = model
        self.config = config
        self.wrapper_model = wrapper_model

        # Setup device
        self.device = self._setup_device()

        # Training state
        self.start_epoch = 0
        self.current_epoch = 0
        self.current_iter = 0

        # Metric tracking
        self.best_mAP50_95 = 0.0
        self.best_mAP50 = 0.0
        self.best_epoch = 0
        self.final_loss = 0.0
        self.patience_counter = 0

        # Will be initialized in setup()
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        self.ema_model = None
        self.train_loader = None
        self.tensorboard_writer = None
        
        # Loss
        self.criterion = RTDETRLoss(num_classes=config.num_classes)

    def _setup_device(self) -> torch.device:
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)
        return device

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

        for k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, (nn.BatchNorm2d, nn.LayerNorm)):
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)

        lr = self.config.effective_lr

        if self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(pg0, lr=lr)
        elif self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(pg0, lr=lr)
        else:
            optimizer = torch.optim.SGD(pg0, lr=lr, momentum=self.config.momentum, nesterov=self.config.nesterov)

        optimizer.add_param_group({"params": pg1, "lr": lr, "weight_decay": self.config.weight_decay})
        optimizer.add_param_group({"params": pg2, "lr": lr})

        return optimizer

    def _setup_scheduler(self, iters_per_epoch: int):
        if self.config.scheduler == "linear":
            return LinearLRScheduler(
                lr=self.config.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=self.config.warmup_epochs,
                warmup_lr_start=self.config.warmup_lr_start,
                min_lr_ratio=self.config.min_lr_ratio,
            )
        else:
            return CosineAnnealingScheduler(
                lr=self.config.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=self.config.warmup_epochs,
                warmup_lr_start=self.config.warmup_lr_start,
                min_lr_ratio=self.config.min_lr_ratio,
            )

    def _get_save_dir(self) -> Path:
        project = Path(self.config.project)
        name = self.config.name

        if self.config.exist_ok:
            save_dir = project / name
        else:
            save_dir = project / name
            if save_dir.exists():
                i = 2
                while (project / f"{name}{i}").exists():
                    i += 1
                save_dir = project / f"{name}{i}"

        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def _setup_data(self):
        img_size = self.config.input_size

        # V9TrainTransform provides normalized cxcywh output compatible with DETR
        preproc = V9TrainTransform(max_labels=300, flip_prob=self.config.flip_prob, hsv_prob=self.config.hsv_prob)

        if self.config.data:
            data_cfg = load_data_config(self.config.data)
            data_dir = data_cfg['root']
            self.num_classes = data_cfg.get('nc', self.config.num_classes)

            if (Path(data_dir) / "annotations").exists():
                train_dataset = COCODataset(data_dir=data_dir, json_file="instances_train2017.json", name="train2017", img_size=img_size, preproc=preproc)
            else:
                # YOLO format
                train_path = data_cfg.get('train', 'images/train')
                if train_path.endswith('.txt'):
                    img_files_ = data_cfg['train_img_files']
                    label_files_ = data_cfg['train_label_files']
                    # check if image file and label file exists
                    img_files = []
                    label_files = []
                    for img_file, label_file in zip(img_files_, label_files_):
                        img_path = Path(data_dir) / img_file
                        lbl_path = Path(data_dir) / label_file
                        if img_path.exists() and lbl_path.exists():
                            img_files.append(img_path)
                            label_files.append(lbl_path)
                else:
                    # Handle both absolute and relative paths
                    train_path = Path(train_path)
                    if train_path.is_absolute():
                        train_img_dir = train_path
                    else:
                        train_img_dir = Path(data_dir) / train_path

                    # Collect image files
                    img_files = []
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                        img_files.extend(train_img_dir.glob(ext))
                        img_files.extend(train_img_dir.glob(ext.upper()))
                    img_files = sorted(img_files)

                    if len(img_files) == 0:
                        raise FileNotFoundError(f"No images found in {train_img_dir}")

                    # Infer label paths
                    label_files = []
                    for img_file in img_files:
                        label_file = Path(str(img_file).replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt')
                        label_files.append(label_file)
                        
                train_dataset = YOLODataset(
                    img_files=img_files,
                    label_files=label_files,
                    img_size=img_size,
                    preproc=preproc,
                )
        elif self.config.data_dir:
            data_dir = self.config.data_dir
            self.num_classes = self.config.num_classes

            if (Path(data_dir) / "annotations").exists():
                train_dataset = COCODataset(data_dir=data_dir, json_file="instances_train2017.json", name="train2017", img_size=img_size, preproc=preproc)
            else:
                train_dataset = YOLODataset(data_dir=data_dir, split="train", img_size=img_size, preproc=preproc)
        else:
            raise ValueError("Either 'data' or 'data_dir' must be specified in config")

        train_dataset = V9MosaicMixupDataset(
            dataset=train_dataset, img_size=img_size, mosaic=True, preproc=preproc,
            degrees=self.config.degrees, translate=self.config.translate,
            mosaic_scale=self.config.mosaic_scale, mixup_scale=self.config.mixup_scale,
            shear=self.config.shear, enable_mixup=self.config.mixup_prob > 0,
            mosaic_prob=self.config.mosaic_prob, mixup_prob=self.config.mixup_prob,
        )

        self.train_loader = create_dataloader(train_dataset, batch_size=self.config.batch, num_workers=self.config.workers, shuffle=True, pin_memory=True)
        return train_dataset

    def setup(self):
        self.model.to(self.device)
        train_dataset = self._setup_data()
        self.optimizer = self._setup_optimizer()
        self.lr_scheduler = self._setup_scheduler(len(self.train_loader))
        self.criterion.to(self.device)

        if self.config.amp and self.device.type == "cuda":
            self.scaler = GradScaler("cuda")
        else:
            self.scaler = None

        if self.config.ema:
            self.ema_model = ModelEMA(self.model, decay=self.config.ema_decay)

        self.save_dir = self._get_save_dir()
        self.config.to_yaml(self.save_dir / "train_config.yaml")

    def train(self) -> Dict:
        self.setup()
        start_time = time.time()

        for epoch in range(self.start_epoch, self.config.epochs):
            self.current_epoch = epoch
            if epoch == self.config.epochs - self.config.no_aug_epochs:
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic()

            epoch_loss, val_metrics = self._train_epoch(epoch)
            self.final_loss = epoch_loss

            if (epoch + 1) % self.config.save_period == 0 or epoch == self.config.epochs - 1:
                self._save_checkpoint(epoch, epoch_loss, val_metrics)

            if self.patience_counter >= self.config.patience:
                break

        return {
            'final_loss': self.final_loss,
            'best_mAP50': self.best_mAP50,
            'best_mAP50_95': self.best_mAP50_95,
            'best_epoch': self.best_epoch,
            'save_dir': str(self.save_dir),
            'best_checkpoint': str(self.save_dir / "weights" / 'best.pt'),
            'last_checkpoint': str(self.save_dir / "weights" / 'last.pt'),
        }

    def _validate_epoch(self, epoch: int) -> Optional[Dict[str, float]]:
        try:
            from libreyolo.validation import ValidationConfig
            from libreyolo.rtdetr.validator import RTDETRValidator

            val_config = ValidationConfig(
                data=self.config.data,
                batch_size=self.config.batch,
                imgsz=self.config.imgsz,
                conf_thres=0.001,
                iou_thres=0.65,
                device=str(self.device),
                half=self.config.amp and self.device.type == "cuda",
                verbose=False,
            )

            if self.wrapper_model is None:
                return None

            eval_pytorch_model = self.ema_model.ema if self.ema_model else self.model
            original_model = self.wrapper_model.model
            self.wrapper_model.model = eval_pytorch_model

            try:
                validator = RTDETRValidator(
                    model=self.wrapper_model,
                    config=val_config,
                )
                results = validator.run()
            finally:
                self.wrapper_model.model = original_model

            metrics = {
                'mAP50': results.get('metrics/mAP50', 0.0),
                'mAP50_95': results.get('metrics/mAP50-95', 0.0),
            }

            print(f"Validation - mAP50: {metrics['mAP50']:.4f}, mAP50-95: {metrics['mAP50_95']:.4f}")
            return metrics
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return None

    def _train_epoch(self, epoch: int) -> Tuple[float, Optional[Dict[str, float]]]:
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (imgs, targets, _, _) in enumerate(pbar):
            self.current_iter = epoch * len(self.train_loader) + batch_idx
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            detr_targets = convert_targets_for_detr(targets, imgs.shape[0])

            if self.scaler is not None:
                with autocast(device_type=self.device.type):
                    outputs = self.model(imgs, targets=detr_targets)
                    loss_dict = self.criterion(outputs, detr_targets)
                    loss = loss_dict["total_loss"]

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(imgs, targets=detr_targets)
                loss_dict = self.criterion(outputs, detr_targets)
                loss = loss_dict["total_loss"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.ema_model is not None:
                self.ema_model.update(self.model)

            loss_val = loss.item()
            cls_loss_val = loss_dict.get('loss_vfl', loss_dict.get('loss_focal', 0))
            if isinstance(cls_loss_val, torch.Tensor): cls_loss_val = cls_loss_val.item()
            box_loss_val = loss_dict.get('loss_bbox', 0)
            if isinstance(box_loss_val, torch.Tensor): box_loss_val = box_loss_val.item()
            giou_loss_val = loss_dict.get('loss_giou', 0)
            if isinstance(giou_loss_val, torch.Tensor): giou_loss_val = giou_loss_val.item()
            
            total_loss += loss_val

            lr = self.lr_scheduler.update_lr(self.current_iter + 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "lr": f"{lr:.6f}",
                "cls": f"{cls_loss_val:.4f}",
                "box": f"{box_loss_val:.4f}",
                "giou": f"{giou_loss_val:.4f}",
            })

        avg_loss = total_loss / num_batches

        val_metrics = None
        if self.config.eval_interval > 0 and (epoch + 1) % self.config.eval_interval == 0:
            val_metrics = self._validate_epoch(epoch)

        return avg_loss, val_metrics

    def _save_checkpoint(self, epoch: int, loss: float, val_metrics: Optional[Dict[str, float]] = None):
        model_to_save = self.ema_model.ema if self.ema_model is not None else self.model

        checkpoint = {
            "epoch": epoch,
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "loss": loss,
            "best_mAP50_95": self.best_mAP50_95,
            "best_mAP50": self.best_mAP50,
            "best_epoch": self.best_epoch,
            "nc": self.config.num_classes,
            "size": self.config.size,
            "model_family": "rtdetr",
        }
        if self.wrapper_model is not None:
            checkpoint["names"] = self.wrapper_model.names

        if self.ema_model is not None:
            checkpoint["ema_updates"] = self.ema_model.updates

        weights_dir = self.save_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        torch.save(checkpoint, weights_dir / "last.pt")

        if val_metrics and val_metrics['mAP50_95'] > self.best_mAP50_95:
            self.best_mAP50_95 = val_metrics['mAP50_95']
            self.best_mAP50 = val_metrics['mAP50']
            self.best_epoch = epoch + 1
            self.patience_counter = 0
            torch.save(checkpoint, weights_dir / "best.pt")
        elif val_metrics:
            self.patience_counter += 1

        if (epoch + 1) % self.config.save_period == 0:
            torch.save(checkpoint, weights_dir / f"epoch_{epoch + 1}.pt")
