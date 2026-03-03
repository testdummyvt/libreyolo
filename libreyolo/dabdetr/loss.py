"""
Loss functions for DAB-DETR.

DABDETRLoss creates a SetCriterion configured for DAB-DETR training
using focal loss (not VFL) with Hungarian matching.
"""

from ..rtdetr.loss import SetCriterion, HungarianMatcher


def DABDETRLoss(num_classes: int = 80):
    """Create a DABDETRLoss (SetCriterion) with standard DAB-DETR settings.

    DAB-DETR uses sigmoid focal loss for classification (not VFL like RT-DETR).

    Args:
        num_classes: Number of object classes.

    Returns:
        SetCriterion instance configured for DAB-DETR.
    """
    matcher_weight_dict = {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}
    matcher = HungarianMatcher(
        weight_dict=matcher_weight_dict,
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
    )

    weight_dict = {
        "loss_focal": 2.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
    }

    losses = ["focal", "boxes"]

    return SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        alpha=0.25,
        gamma=2.0,
        num_classes=num_classes,
    )
