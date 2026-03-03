"""
Utility functions for LibreYOLO DAB-DETR.

Provides postprocessing and core math helpers.
"""

import math
import torch
from typing import Tuple, Dict

from ..common.utils import postprocess_detections


# =============================================================================
# Core Utilities
# =============================================================================

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse of sigmoid function, clamped for numerical stability."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# =============================================================================
# Postprocessing
# =============================================================================

def postprocess(
    output: Dict,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: int = 800,
    original_size: Tuple[int, int] = None,
    max_det: int = 300,
    letterbox: bool = False,
) -> Dict:
    """
    Postprocess DAB-DETR model outputs to get final detections.

    Args:
        output: Model output dictionary with 'pred_logits' and 'pred_boxes'
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        input_size: Input image size
        original_size: Original image size (width, height) for scaling
        max_det: Maximum number of detections
        letterbox: Whether letterbox was used in preprocessing

    Returns:
        Dictionary with boxes, scores, classes, num_detections
    """
    logits = output['pred_logits']
    boxes_cxcywh = output['pred_boxes']

    if logits.dim() == 3:
        logits = logits[0]
        boxes_cxcywh = boxes_cxcywh[0]

    # Convert logits to probabilities via sigmoid (focal loss style)
    scores = logits.sigmoid()

    # Get max class score and class id
    max_scores, class_ids = torch.max(scores, dim=-1)

    # Apply confidence threshold
    mask = max_scores > conf_thres
    if not mask.any():
        return {
            "boxes": [],
            "scores": [],
            "classes": [],
            "num_detections": 0
        }

    filtered_boxes = boxes_cxcywh[mask]
    filtered_scores = max_scores[mask]
    filtered_class_ids = class_ids[mask]

    # Convert normalized cx, cy, w, h to absolute x1, y1, x2, y2
    cx, cy, w, h = filtered_boxes.unbind(-1)

    x1 = (cx - w / 2) * input_size
    y1 = (cy - h / 2) * input_size
    x2 = (cx + w / 2) * input_size
    y2 = (cy + h / 2) * input_size

    absolute_xyxy = torch.stack((x1, y1, x2, y2), dim=-1)

    # Use shared postprocess pipeline
    return postprocess_detections(
        boxes=absolute_xyxy,
        scores=filtered_scores,
        class_ids=filtered_class_ids,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        input_size=input_size,
        original_size=original_size,
        max_det=max_det,
        letterbox=letterbox,
    )
