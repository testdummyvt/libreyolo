"""
Loss functions for DAB-DETR.

Uses sigmoid focal loss (not VFL), L1 and GIoU box regression losses,
and Hungarian bipartite matching with weights:
  class cost = 2, L1 cost = 5, GIoU cost = 2.

Reference: https://github.com/IDEA-Research/DAB-DETR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou


# =============================================================================
# Hungarian Matcher
# =============================================================================


class HungarianMatcher(nn.Module):
    """Bipartite matching between predictions and ground-truth using focal cost."""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, (
            "All costs cannot be 0"
        )

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'pred_logits' [B, Q, C] and 'pred_boxes' [B, Q, 4]
            targets: list of dicts with 'labels' and 'boxes'

        Returns:
            List of (pred_indices, target_indices) per batch element.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))  # [B*Q, C]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*Q, 4]

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if len(tgt_ids) == 0:
            return [
                (
                    torch.as_tensor([], dtype=torch.int64),
                    torch.as_tensor([], dtype=torch.int64),
                )
                for _ in range(bs)
            ]

        # Focal classification cost
        out_prob_tgt = out_prob[:, tgt_ids]
        neg_cost = (
            (1 - self.alpha)
            * (out_prob_tgt**self.gamma)
            * (-(1 - out_prob_tgt + 1e-8).log())
        )
        pos_cost = (
            self.alpha
            * ((1 - out_prob_tgt) ** self.gamma)
            * (-(out_prob_tgt + 1e-8).log())
        )
        cost_class = pos_cost - neg_cost

        # L1 box cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # GIoU box cost
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox),
        )

        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


# =============================================================================
# Set Criterion
# =============================================================================


class SetCriterion(nn.Module):
    """DAB-DETR loss: sigmoid focal loss + L1 + GIoU with auxiliary outputs."""

    def __init__(
        self,
        matcher: HungarianMatcher,
        num_classes: int = 80,
        weight_dict: dict = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.alpha = alpha
        self.gamma = gamma

        if weight_dict is None:
            weight_dict = {
                "loss_ce": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
            }
        self.weight_dict = weight_dict

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Sigmoid focal classification loss."""
        src_logits = outputs["pred_logits"]  # [B, Q, C]
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_one_hot = F.one_hot(target_classes, num_classes=self.num_classes + 1)[
            ..., :-1
        ]
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target_one_hot.float(), self.alpha, self.gamma, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_ce": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """L1 + GIoU bounding-box regression losses."""
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        if len(target_boxes) == 0:
            return {
                "loss_bbox": src_boxes.sum() * 0,
                "loss_giou": src_boxes.sum() * 0,
            }

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses = {"loss_bbox": loss_bbox.sum() / num_boxes}

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _cast_to_float32(self, outputs):
        result = {}
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.float()
            elif isinstance(v, list):
                result[k] = [
                    {
                        kk: vv.float() if isinstance(vv, torch.Tensor) else vv
                        for kk, vv in item.items()
                    }
                    if isinstance(item, dict)
                    else item
                    for item in v
                ]
            else:
                result[k] = v
        return result

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'pred_logits', 'pred_boxes', optionally 'aux_outputs'.
            targets: list of dicts with 'labels' and 'boxes'.

        Returns:
            Dict of weighted losses including 'total_loss'.
        """
        outputs = self._cast_to_float32(outputs)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device
            if not isinstance(next(iter(outputs.values())), list)
            else targets[0]["labels"].device,
        )
        num_boxes = torch.clamp(num_boxes, min=1).item()

        losses = {}
        for loss_fn in [self.loss_labels, self.loss_boxes]:
            l_dict = loss_fn(outputs_without_aux, targets, indices, num_boxes)
            l_dict = {
                k: l_dict[k] * self.weight_dict.get(k, 1.0)
                for k in l_dict
                if k in self.weight_dict
            }
            losses.update(l_dict)

        # Auxiliary losses from intermediate decoder layers
        if "aux_outputs" in outputs:
            for i, aux_out in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_out, targets)
                for loss_fn in [self.loss_labels, self.loss_boxes]:
                    l_dict = loss_fn(aux_out, targets, aux_indices, num_boxes)
                    l_dict = {
                        k: l_dict[k] * self.weight_dict.get(k, 1.0)
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {f"{k}_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
        losses["total_loss"] = total_loss
        return losses


# =============================================================================
# Convenience factory
# =============================================================================


def DABDETRLoss(num_classes: int = 80) -> SetCriterion:
    """Create a DABDETRLoss (SetCriterion) with standard DAB-DETR settings.

    Args:
        num_classes: Number of object classes.

    Returns:
        SetCriterion instance.
    """
    matcher = HungarianMatcher(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        alpha=0.25,
        gamma=2.0,
    )
    weight_dict = {
        "loss_ce": 1.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
    }
    return SetCriterion(
        matcher=matcher,
        num_classes=num_classes,
        weight_dict=weight_dict,
        alpha=0.25,
        gamma=2.0,
    )
