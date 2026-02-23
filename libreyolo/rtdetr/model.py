"""
LibreYOLORTDETR inference and training wrapper.

Provides a high-level API for RT-DETR object detection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image

from ..common.base_model import LibreYOLOBase
from ..common.image_loader import ImageInput
from ..common.utils import preprocess_image
from .nn import RTDETRModel
from .utils import postprocess

# Model configs — derived from official RT-DETR YAML configs:
# https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch/configs/rtdetr
RTDETR_MODELS = {
    "r18": {
        "backbone_depth": 18,
        "backbone_variant": "d",
        "backbone_pretrained": True,
        "backbone_freeze_norm": False,
        "hidden_dim": 256,
        "dim_feedforward": 1024,
        "expansion": 0.5,
        "num_decoder_layers": 3,
        "num_denoising": 100,
    },
    "r34": {
        "backbone_depth": 34,
        "backbone_variant": "d",
        "backbone_pretrained": True,
        "backbone_freeze_norm": False,
        "hidden_dim": 256,
        "dim_feedforward": 1024,
        "expansion": 0.5,
        "num_decoder_layers": 4,
        "num_denoising": 100,
    },
    "r50": {
        "backbone_depth": 50,
        "backbone_variant": "d",
        "backbone_pretrained": True,
        "backbone_freeze_norm": True,
        "hidden_dim": 256,
        "dim_feedforward": 1024,
        "expansion": 1.0,
        "num_decoder_layers": 6,
        "num_denoising": 100,
    },
    "r50m": {
        "backbone_depth": 50,
        "backbone_variant": "d",
        "backbone_pretrained": True,
        "backbone_freeze_norm": True,
        "hidden_dim": 256,
        "dim_feedforward": 1024,
        "expansion": 0.5,
        "num_decoder_layers": 6,
        "num_denoising": 100,
        "eval_idx": 2,
    },
    "r101": {
        "backbone_depth": 101,
        "backbone_variant": "d",
        "backbone_pretrained": True,
        "backbone_freeze_norm": True,
        "hidden_dim": 384,
        "dim_feedforward": 2048,
        "expansion": 1.0,
        "num_decoder_layers": 6,
        "num_denoising": 100,
    },
}

class LIBREYOLORTDETR(LibreYOLOBase):
    """
    LibreYOLORTDETR model for object detection.

    Args:
        model_path: Model weights source. Can be:
            - str: Path to a .pt/.pth weights file
            - dict: Pre-loaded state_dict (e.g., from torch.load())
        size: Model size variant (required). Must be one of: "r18", "r34", "r50", "r101", "x"
        nb_classes: Number of classes (default: 80 for COCO)
        device: Device for inference. "auto" uses CUDA if available, else MPS, else CPU.

    Example:
        >>> model = LIBREYOLORTDETR(model_path="path/to/weights.pt", size="r18")
        >>> detections = model(image=image_path, save=True)
    """

    def __init__(
        self,
        model_path,
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )

    def _get_valid_sizes(self) -> List[str]:
        # Commonly RT-DETR uses ResNet18/34/50/101 or HGNet as backbones.
        # We define a few standard codes here.
        return list(RTDETR_MODELS.keys())

    def _get_model_name(self) -> str:
        return "RTDETR"

    def _get_input_size(self) -> int:
        return 640

    def _get_val_preprocessor(self, img_size: int = 640) -> Any:
        from ..validation.preprocessors import RTDETRValPreprocessor
        return RTDETRValPreprocessor(img_size=img_size)

    def _init_model(self) -> nn.Module:
        cfg = RTDETR_MODELS[self.size]
        return RTDETRModel(
            num_classes=self.nb_classes,
            num_queries=300,
            nhead=8,
            num_decoder_points=4,
            aux_loss=True,
            **cfg,
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "encoder": self.model.encoder,
            "decoder": self.model.decoder,
        }

    def _preprocess(
        self, image: ImageInput, color_format: str = "auto", input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
        """
        RTDETR typically operates on resized and normalized (0-1) images, without letterboxing if using standard DETR paradigms,
        but to be consistent with LibreYOLO we can use the shared letterbox preprocess and handle it in postprocess.
        """
        effective_size = input_size if input_size is not None else self._get_input_size()
        return preprocess_image(image, input_size=effective_size, color_format=color_format)

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        # No targets during inference — denoising is training-only
        return self.model(input_tensor, targets=None)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        actual_input_size = kwargs.get('input_size', self._get_input_size())
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=actual_input_size,
            original_size=original_size,
            max_det=max_det,
            letterbox=kwargs.get('letterbox', False),  # preprocess_image uses simple resize, not letterbox
        )

    def _strict_loading(self) -> bool:
        return False

    def train(self, data: str, epochs: int = 100, batch_size: int = 4, imgsz: int = 640, device: str = "auto", **kwargs) -> Dict:
        """
        Train the RT-DETR model using the RTDETRTrainer.
        """
        from .trainer import RTDETRTrainer
        from .config import RTDETRTrainConfig

        # Handle 'batch' in kwargs
        batch_kw = kwargs.pop('batch', batch_size)

        # Handle optimizer casing
        if 'optimizer' in kwargs and isinstance(kwargs['optimizer'], str):
            kwargs['optimizer'] = kwargs['optimizer'].lower()

        config = RTDETRTrainConfig(
            data=data,
            epochs=epochs,
            batch=batch_kw,
            imgsz=imgsz,
            device=device,
            name=kwargs.pop('name', 'rtdetr_train'),
            project=kwargs.pop('project', 'runs/train'),
            num_classes=self.nb_classes,
            **kwargs
        )
        trainer = RTDETRTrainer(model=self.model, config=config, wrapper_model=self)
        results = trainer.train()
        return results

    def val(self, data: str, batch_size: int = 4, imgsz: int = 640, device: str = "auto", half: bool = True, **kwargs) -> Dict:
        """
        Validate the RT-DETR model.
        """
        from .validator import RTDETRValidator
        from ..validation import ValidationConfig

        batch_kw = kwargs.pop('batch', batch_size)

        config = ValidationConfig(
            data=data,
            batch_size=batch_kw,
            imgsz=imgsz,
            device=device,
            half=half,
            **kwargs
        )
        validator = RTDETRValidator(model=self, config=config)
        return validator.run()

