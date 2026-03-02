"""
LibreYOLODABDETR inference and training wrapper.

Provides a high-level API for DAB-DETR object detection compatible with the
LibreYOLO base-model contract.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ..common.base_model import LibreYOLOBase
from ..common.image_loader import ImageInput
from ..common.utils import preprocess_image
from .nn import DABDETRModel
from .config import DAB_DETR_PRESETS
from ..rtdetr.utils import postprocess  # reuse RT-DETR postprocess (same format)


# Variant → model constructor kwargs
DABDETR_MODELS = {
    "r50": {
        "backbone_dilation": False,
        "num_patterns": 0,
    },
    "r50-dc5": {
        "backbone_dilation": True,
        "num_patterns": 0,
    },
    "r50-3pat": {
        "backbone_dilation": False,
        "num_patterns": 3,
    },
    "r50-dc5-3pat": {
        "backbone_dilation": True,
        "num_patterns": 3,
    },
}


class LIBREYOLODABDETR(LibreYOLOBase):
    """LibreYOLO wrapper for DAB-DETR (Dynamic Anchor Boxes DETR).

    Args:
        model_path: Path to a .pt/.pth weights file, or None to create from scratch.
        size: Variant code — one of "r50", "r50-dc5", "r50-3pat", "r50-dc5-3pat".
        nb_classes: Number of object classes (default: 80 for COCO).
        device: Device for inference. "auto" selects CUDA > MPS > CPU.

    Example:
        >>> model = LIBREYOLODABDETR(model_path=None, size="r50")
        >>> results = model(image="image.jpg", conf=0.5)
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
        return list(DABDETR_MODELS.keys())

    def _get_model_name(self) -> str:
        return "DABDETR"

    def _get_input_size(self) -> int:
        return 640

    def _get_val_preprocessor(self, img_size: int = 640) -> Any:
        from ..validation.preprocessors import RTDETRValPreprocessor

        return RTDETRValPreprocessor(img_size=img_size)

    def _init_model(self) -> nn.Module:
        cfg = DABDETR_MODELS[self.size]
        return DABDETRModel(
            num_classes=self.nb_classes,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            num_queries=300,
            modulate_hw_attn=True,
            aux_loss=True,
            backbone_pretrained=False,
            **cfg,
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "encoder": self.model.encoder,
            "decoder": self.model.decoder,
        }

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
        effective_size = (
            input_size if input_size is not None else self._get_input_size()
        )
        return preprocess_image(
            image, input_size=effective_size, color_format=color_format
        )

    def _forward(self, input_tensor: torch.Tensor) -> Any:
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
        actual_input_size = kwargs.get("input_size", self._get_input_size())
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=actual_input_size,
            original_size=original_size,
            max_det=max_det,
            letterbox=kwargs.get("letterbox", False),
        )

    def _strict_loading(self) -> bool:
        return False

    def train(
        self,
        data: str,
        epochs: int = 50,
        batch_size: int = 4,
        imgsz: int = 640,
        device: str = "auto",
        **kwargs,
    ) -> Dict:
        """Train the DAB-DETR model."""
        from .trainer import DABDETRTrainer
        from .config import DABDETRTrainConfig

        batch_kw = kwargs.pop("batch", batch_size)

        if "optimizer" in kwargs and isinstance(kwargs["optimizer"], str):
            kwargs["optimizer"] = kwargs["optimizer"].lower()

        config = DABDETRTrainConfig(
            data=data,
            epochs=epochs,
            batch=batch_kw,
            imgsz=imgsz,
            device=device,
            size=self.size,
            name=kwargs.pop("name", "dabdetr_train"),
            project=kwargs.pop("project", "runs/train"),
            num_classes=self.nb_classes,
            **kwargs,
        )
        trainer = DABDETRTrainer(model=self.model, config=config, wrapper_model=self)
        return trainer.train()

    def val(
        self,
        data: str,
        batch_size: int = 4,
        imgsz: int = 640,
        device: str = "auto",
        half: bool = True,
        **kwargs,
    ) -> Dict:
        """Validate the DAB-DETR model."""
        from .validator import DABDETRValidator
        from ..validation import ValidationConfig

        batch_kw = kwargs.pop("batch", batch_size)

        config = ValidationConfig(
            data=data,
            batch_size=batch_kw,
            imgsz=imgsz,
            device=device,
            half=half,
            **kwargs,
        )
        validator = DABDETRValidator(model=self, config=config)
        return validator.run()
