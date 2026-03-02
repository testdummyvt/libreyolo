"""
ResNet-50 backbone for DAB-DETR.

Faithfully ported from the original DAB-DETR repository:
https://github.com/IDEA-Research/DAB-DETR/blob/main/models/backbone.py

Uses torchvision ResNet-50 with IntermediateLayerGetter to extract the C5
feature map only (layer4 → 2048 channels).  The dc5 variant replaces the
last stage's stride with dilation (replace_stride_with_dilation=[False,
False, True]) to double the spatial resolution of the C5 output.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with frozen statistics and affine parameters.

    Identical to the version used in the original DETR/DAB-DETR repos —
    keeps running stats fixed during training to match the pretrained
    ImageNet backbone BN statistics.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self) -> str:
        return f"num_features={self.weight.shape[0]}, eps={self.eps}"


def _replace_bn_with_frozen(module: nn.Module) -> nn.Module:
    """Recursively replace all BatchNorm2d layers with FrozenBatchNorm2d."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features)
            # Copy trained parameters if they exist
            if child.weight is not None:
                frozen.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                frozen.bias.data.copy_(child.bias.data)
            if child.running_mean is not None:
                frozen.running_mean.data.copy_(child.running_mean.data)
            if child.running_var is not None:
                frozen.running_var.data.copy_(child.running_var.data)
            setattr(module, name, frozen)
        else:
            _replace_bn_with_frozen(child)
    return module


class ResNet50Backbone(nn.Module):
    """Standard ResNet-50 backbone for DAB-DETR.

    Extracts only the C5 (layer4) feature map:
    - Standard:  stride-32, output channels = 2048
    - DC5 (dilation): stride-16, output channels = 2048

    Args:
        dilation (bool): If True, replace C5 stride with dilation (DC5 variant).
        pretrained (bool): Load ImageNet-pretrained weights from torchvision.
        freeze_bn (bool): Replace all BatchNorm2d with FrozenBatchNorm2d.
        train_backbone (bool): If False, freeze all backbone parameters.
    """

    NUM_CHANNELS = 2048  # C5 output channels for ResNet-50

    def __init__(
        self,
        dilation: bool = False,
        pretrained: bool = False,
        freeze_bn: bool = True,
        train_backbone: bool = True,
    ):
        super().__init__()
        self.dilation = dilation
        self.num_channels = self.NUM_CHANNELS

        # Build torchvision ResNet-50
        replace_stride = [False, False, True] if dilation else None
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = tv_models.resnet50(
            weights=weights,
            replace_stride_with_dilation=replace_stride,
        )

        if freeze_bn:
            resnet = _replace_bn_with_frozen(resnet)

        # Extract only C5 (layer4) — identical to original DAB-DETR
        self.body = IntermediateLayerGetter(resnet, return_layers={"layer4": "C5"})

        if not train_backbone:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor [B, 3, H, W], normalised to [0, 1].

        Returns:
            features: [B, 2048, H/32, W/32]  (H/16 for dc5 variant)
        """
        features: Dict[str, torch.Tensor] = self.body(x)
        return features["C5"]
