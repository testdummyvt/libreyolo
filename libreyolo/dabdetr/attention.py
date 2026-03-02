"""
Custom MultiheadAttention with vdim support for DAB-DETR cross-attention.

The original DAB-DETR cross-attention concatenates content embedding and
sinusoidal positional embedding to form Q and K (doubling their dimension
to 2*d_model), while V remains at d_model.  PyTorch's built-in
nn.MultiheadAttention requires vdim == embed_dim when batch_first is used
via the F.multi_head_attention_forward path, so we implement a thin custom
module that handles this mismatch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiheadAttentionWithVdim(nn.Module):
    """Multi-head attention where value dimension can differ from embed_dim.

    Used in DAB-DETR decoder cross-attention:
        embed_dim = 2 * d_model   (Q and K are content + sin-pos concatenated)
        vdim      = d_model       (V is the encoder memory)

    Args:
        embed_dim (int): Dimension of Q and K projections (2*d_model for cross-attn).
        num_heads (int): Number of attention heads.
        vdim (int): Dimension of V. Defaults to embed_dim if not specified.
        dropout (float): Dropout probability on attention weights.
        bias (bool): Add bias to input projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads  # head dim for Q/K
        # V head dim scales with vdim / num_heads
        assert self.vdim % num_heads == 0, "vdim must be divisible by num_heads"
        self.v_head_dim = self.vdim // num_heads

        self.scaling = float(self.head_dim) ** -0.5

        # Separate projections — keep them outside the attention module
        # as in the original DAB-DETR implementation.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, self.vdim, bias=bias)
        self.out_proj = nn.Linear(self.vdim, self.vdim, bias=bias)

        self.attn_drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: [B, Lq, embed_dim]
            key:   [B, Lk, embed_dim]
            value: [B, Lk, vdim]
            key_padding_mask: [B, Lk] boolean mask (True = ignore)
            attn_mask: [Lq, Lk] additive mask

        Returns:
            output: [B, Lq, vdim]
            attn_weights: None (not returned for efficiency)
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # Project
        q = self.q_proj(query)  # [B, Lq, embed_dim]
        k = self.k_proj(key)  # [B, Lk, embed_dim]
        v = self.v_proj(value)  # [B, Lk, vdim]

        # Reshape to [B, num_heads, L, head_dim]
        q = q.reshape(B, Lq, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, H, Lq, head_dim]
        k = k.reshape(B, Lk, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, H, Lk, head_dim]
        v = v.reshape(B, Lk, self.num_heads, self.v_head_dim).transpose(
            1, 2
        )  # [B, H, Lk, v_head_dim]

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # [B, H, Lq, Lk]

        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Attend to values
        out = torch.matmul(attn, v)  # [B, H, Lq, v_head_dim]
        out = out.transpose(1, 2).reshape(B, Lq, self.vdim)  # [B, Lq, vdim]
        out = self.out_proj(out)
        return out, None
