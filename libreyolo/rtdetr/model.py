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

# Model configs
RTDETR_MODELS = {
    "r18": {
        "backbone_depth": 18,
        "backbone_variant": "d",
        "backbone_pretrained": True,
        "backbone_freeze_norm": False,
    },
    "r34": {
        "backbone_depth": 34,
        "backbone_variant": "d",
        "backbone_pretrained": True,
        "backbone_freeze_norm": False,
    },
    "r50": {
        "backbone_depth": 50,
        "backbone_variant": "d",
        "backbone_pretrained": True,
        "backbone_freeze_norm": False,
    },
    "r101": {
        "backbone_depth": 101,
        "backbone_variant": "d",
        "backbone_pretrained": True,
        "backbone_freeze_norm": False,
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
        return ["r18"]

    def _get_model_name(self) -> str:
        return "RTDETR"

    def _get_input_size(self) -> int:
        return 640

    def _get_val_preprocessor(self, img_size: int = 640) -> Any:
        from ..validation.preprocessors import RFDETRValPreprocessor
        # RTDETR shares input characteristics with RFDETR
        return RFDETRValPreprocessor(img_size=img_size)

    def _init_model(self) -> nn.Module:
        return RTDETRModel(
            num_classes=self.nb_classes,
            hidden_dim=256,
            num_queries=300,
            num_decoder_layers=3,
            nhead=8,
            dim_feedforward=1024,
            num_denoising=100,
            num_decoder_points=4,
            aux_loss=True,
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

