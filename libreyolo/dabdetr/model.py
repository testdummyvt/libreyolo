"""
LIBREYOLODABDETR inference wrapper.

Provides a high-level API for DAB-DETR object detection.
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
from .nn import DABDETRModel
from .utils import postprocess


# Model configs for DAB-DETR variants.
# Reference: https://github.com/IDEA-Research/DAB-DETR
DABDETR_MODELS = {
    "r50": {
        "backbone_name": "resnet50",
        "backbone_dilation": False,
        "backbone_pretrained": True,
        "d_model": 256,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "num_queries": 300,
        "num_patterns": 0,
        "random_refpoints_xy": False,
        "temperature_height": 20,
        "temperature_width": 20,
    },
    "r50-dc5": {
        "backbone_name": "resnet50",
        "backbone_dilation": True,
        "backbone_pretrained": True,
        "d_model": 256,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "num_queries": 300,
        "num_patterns": 0,
        "random_refpoints_xy": False,
        "temperature_height": 10,
        "temperature_width": 10,
    },
    "r50-3pat": {
        "backbone_name": "resnet50",
        "backbone_dilation": False,
        "backbone_pretrained": True,
        "d_model": 256,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "num_queries": 300,
        "num_patterns": 3,
        "random_refpoints_xy": False,
        "temperature_height": 20,
        "temperature_width": 20,
    },
    "r50-dc5-3pat": {
        "backbone_name": "resnet50",
        "backbone_dilation": True,
        "backbone_pretrained": True,
        "d_model": 256,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "num_queries": 300,
        "num_patterns": 3,
        "random_refpoints_xy": False,
        "temperature_height": 10,
        "temperature_width": 10,
    },
}


class LIBREYOLODABDETR(LibreYOLOBase):
    """LibreYOLO DAB-DETR model for object detection.

    DAB-DETR (Dynamic Anchor Boxes) is a DETR variant that uses learnable
    anchor boxes as queries, improving convergence and detection quality.

    Args:
        model_path: Model weights source. Can be:
            - str: Path to a .pt/.pth weights file
            - dict: Pre-loaded state_dict
            - None: Random initialization
        size: Model variant. One of: "r50", "r50-dc5", "r50-3pat", "r50-dc5-3pat"
        nb_classes: Number of classes (default: 80 for COCO)
        device: Device for inference. "auto" uses CUDA if available.

    Example:
        >>> model = LIBREYOLODABDETR(model_path="dab_detr_r50.pt", size="r50")
        >>> detections = model(image_path, conf=0.5)
    """

    def __init__(
        self,
        model_path,
        size: str = "r50",
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
        return 1333

    def _get_val_preprocessor(self, img_size: int = 1333) -> Any:
        from ..validation.preprocessors import DABDETRValPreprocessor
        return DABDETRValPreprocessor(img_size=(img_size, img_size))

    def _init_model(self) -> nn.Module:
        cfg = DABDETR_MODELS[self.size]
        return DABDETRModel(
            num_classes=self.nb_classes,
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
        """Preprocess image for DAB-DETR inference.

        Uses aspect-ratio preserving resize and ImageNet normalziation, padding to input_size.
        """
        from ..common.image_loader import ImageLoader
        import cv2
        import numpy as np

        effective_size = input_size if input_size is not None else self._get_input_size()
        
        img_pil = ImageLoader.load(image, color_format=color_format)
        original_size = img_pil.size
        
        img = np.array(img_pil)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
        orig_w, orig_h = original_size
        min_size = 800
        max_size = effective_size
        
        ratio = min_size / min(orig_h, orig_w)
        if ratio * max(orig_h, orig_w) > max_size:
            ratio = max_size / max(orig_h, orig_w)
            
        new_h, new_w = int(orig_h * ratio), int(orig_w * ratio)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Apply ImageNet normalization
        resized_img = resized_img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        resized_img = (resized_img - mean) / std
        
        # Pad to effective_size
        pad_h = max(0, effective_size - new_h)
        pad_w = max(0, effective_size - new_w)
        padded_img = np.pad(
            resized_img, 
            ((0, pad_h), (0, pad_w), (0, 0)), 
            mode='constant', 
            constant_values=0
        )
        
        img_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0)
        return img_tensor, img_pil, original_size

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
        logits = output['pred_logits']
        boxes_cxcywh = output['pred_boxes']
        if logits.dim() == 3:
            logits = logits[0]
            boxes_cxcywh = boxes_cxcywh[0]
            
        scores = logits.sigmoid()
        max_scores, class_ids = torch.max(scores, dim=-1)
        mask = max_scores > conf_thres
        
        if not mask.any():
            return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}
            
        filtered_boxes = boxes_cxcywh[mask]
        filtered_scores = max_scores[mask]
        filtered_class_ids = class_ids[mask]
        
        cx, cy, w, h = filtered_boxes.unbind(-1)
        
        pad_size = kwargs.get('input_size', self._get_input_size())
        
        x1 = (cx - w / 2) * pad_size
        y1 = (cy - h / 2) * pad_size
        x2 = (cx + w / 2) * pad_size
        y2 = (cy + h / 2) * pad_size
        
        absolute_xyxy = torch.stack((x1, y1, x2, y2), dim=-1)
        
        orig_w, orig_h = original_size
        min_size = 800
        max_size = pad_size
        
        ratio = min_size / min(orig_h, orig_w)
        if ratio * max(orig_h, orig_w) > max_size:
            ratio = max_size / max(orig_h, orig_w)
            
        absolute_xyxy /= ratio
        
        from ..common.utils import postprocess_detections
        return postprocess_detections(
            boxes=absolute_xyxy,
            scores=filtered_scores,
            class_ids=filtered_class_ids,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=pad_size,
            original_size=None,
            max_det=max_det,
            letterbox=False,
        )

    def _strict_loading(self) -> bool:
        return False
