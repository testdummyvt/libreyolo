"""
Trainer for DAB-DETR.

Subclasses RTDETRTrainer, swapping in:
  - DABDETRTrainConfig
  - DABDETRLoss
  - DABDETRValidator (for in-training validation)
  - Checkpoint model_family = "dabdetr"
"""

from typing import Any, Dict, Optional
import torch.nn as nn

from ..rtdetr.trainer import RTDETRTrainer
from .config import DABDETRTrainConfig
from .loss import DABDETRLoss


class DABDETRTrainer(RTDETRTrainer):
    """Trainer for DAB-DETR models.

    Inherits the full training loop from RTDETRTrainer and overrides
    the loss criterion, checkpoint family tag, and validator import.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DABDETRTrainConfig,
        wrapper_model: Optional[Any] = None,
    ):
        # Call grandparent __init__ indirectly by calling super().__init__
        # RTDETRTrainer.__init__ sets self.criterion = RTDETRLoss(...)
        # We'll call it then immediately replace the criterion.
        super().__init__(model=model, config=config, wrapper_model=wrapper_model)

        # Replace RT-DETR loss with DAB-DETR focal loss
        self.criterion = DABDETRLoss(num_classes=config.num_classes)

    def _validate_epoch(self, epoch: int) -> Optional[Dict]:
        """Run validation using DABDETRValidator."""
        try:
            from libreyolo.validation import ValidationConfig
            from .validator import DABDETRValidator

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

            eval_model = self.ema_model.ema if self.ema_model else self.model
            original_model = self.wrapper_model.model
            self.wrapper_model.model = eval_model

            try:
                validator = DABDETRValidator(
                    model=self.wrapper_model, config=val_config
                )
                results = validator.run()
            finally:
                self.wrapper_model.model = original_model

            metrics = {
                "mAP50": results.get("metrics/mAP50", 0.0),
                "mAP50_95": results.get("metrics/mAP50-95", 0.0),
            }
            print(
                f"Validation - mAP50: {metrics['mAP50']:.4f}, mAP50-95: {metrics['mAP50_95']:.4f}"
            )
            return metrics
        except Exception as e:
            import logging

            logging.getLogger(__name__).error(f"Validation failed: {e}")
            return None

    def _save_checkpoint(
        self, epoch: int, loss: float, val_metrics: Optional[Dict] = None
    ):
        """Save checkpoint with model_family='dabdetr'."""
        import torch
        from pathlib import Path

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
            "model_family": "dabdetr",
        }
        if self.wrapper_model is not None:
            checkpoint["names"] = self.wrapper_model.names
        if self.ema_model is not None:
            checkpoint["ema_updates"] = self.ema_model.updates

        weights_dir = self.save_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        torch.save(checkpoint, weights_dir / "last.pt")

        if val_metrics and val_metrics["mAP50_95"] > self.best_mAP50_95:
            self.best_mAP50_95 = val_metrics["mAP50_95"]
            self.best_mAP50 = val_metrics["mAP50"]
            self.best_epoch = epoch + 1
            self.patience_counter = 0
            torch.save(checkpoint, weights_dir / "best.pt")
        elif val_metrics:
            self.patience_counter += 1

        if (epoch + 1) % self.config.save_period == 0:
            torch.save(checkpoint, weights_dir / f"epoch_{epoch + 1}.pt")
