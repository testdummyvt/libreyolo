"""
Utility functions for DAB-DETR.
"""

import math
import torch
import torch.nn as nn


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse of sigmoid function, clamped for numerical stability."""
    x = x.clamp(min=0.0, max=1.0)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    """Initialize conv/fc bias according to a given probability."""
    return float(-math.log((1 - prior_prob) / prior_prob))


def get_activation(act: str, inplace: bool = True) -> nn.Module:
    """Get activation module by name."""
    if act is None:
        return nn.Identity()
    act = act.lower()
    if act == "relu":
        m = nn.ReLU()
    elif act == "gelu":
        m = nn.GELU()
    elif act == "silu":
        m = nn.SiLU()
    elif act == "leaky_relu":
        m = nn.LeakyReLU()
    else:
        raise RuntimeError(f"Unknown activation: {act}")
    if hasattr(m, "inplace"):
        m.inplace = inplace
    return m
