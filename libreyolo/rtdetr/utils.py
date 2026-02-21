"""
Utility functions for LibreYOLO RTDETR.

Provides preprocessing, postprocessing, and core attention functions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

# Import shared utilities
from ..common.utils import postprocess_detections


# =============================================================================
# Core Utilities (used by nn.py, denoising.py, loss.py)
# =============================================================================

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse of sigmoid function, clamped for numerical stability."""
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    """Initialize conv/fc bias value according to a given probability."""
    return float(-math.log((1 - prior_prob) / prior_prob))


def get_activation(act: str, inplace: bool = True) -> nn.Module:
    """Get activation module by name."""
    if act is None:
        return nn.Identity()
    
    act = act.lower()
    if act == 'silu':
        m = nn.SiLU()
    elif act == 'relu':
        m = nn.ReLU()
    elif act == 'leaky_relu':
        m = nn.LeakyReLU()
    elif act == 'gelu':
        m = nn.GELU()
    else:
        raise RuntimeError(f'Unknown activation: {act}')

    if hasattr(m, 'inplace'):
        m.inplace = inplace

    return m


def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Pure PyTorch implementation of multi-scale deformable attention core.
    
    Uses F.grid_sample for bilinear interpolation at sampling locations.
    
    Args:
        value: [bs, value_length, n_head, c]
        value_spatial_shapes: List of [h, w] for each level
        sampling_locations: [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights: [bs, query_length, n_head, n_levels, n_points]
    
    Returns:
        output: [bs, query_length, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # [bs, H*W, n_head, c] -> [bs, n_head*c, H, W]
        value_l_ = value_list[level].flatten(2).permute(0, 2, 1).reshape(bs * n_head, c, h, w)
        # [bs, Lq, n_head, n_points, 2] -> [bs*n_head, Lq, n_points, 2]
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)
        # [bs*n_head, c, Lq, n_points]
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)

    # [bs, Lq, n_head, n_levels, n_points] -> [bs*n_head, 1, Lq, n_levels*n_points]
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points)
    
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


# =============================================================================
# Postprocessing (used by model.py)
# =============================================================================

def postprocess(
    output: Dict,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: int = 640,
    original_size: Tuple[int, int] = None,
    max_det: int = 300,
    letterbox: bool = False,
) -> Dict:
    """
    Postprocess RTDETR model outputs to get final detections.

    Args:
        output: Model output dictionary with 'pred_logits' and 'pred_boxes'
        conf_thres: Confidence threshold (default: 0.25)
        iou_thres: IoU threshold for NMS (default: 0.45)
        input_size: Input image size (default: 640)
        original_size: Original image size (width, height) for scaling
        max_det: Maximum number of detections to return (default: 300)

    Returns:
        Dictionary with boxes, scores, classes, num_detections
    """
    logits = output['pred_logits']
    boxes_cywh = output['pred_boxes']

    if logits.dim() == 3:
        logits = logits[0]
        boxes_cywh = boxes_cywh[0]

    # Convert logits to probabilities
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
        
    filtered_boxes = boxes_cywh[mask]
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
