"""
Neural network architecture for LibreYOLO DAB-DETR.

Complete DAB-DETR implementation including:
- ResNet backbone (via timm) with FrozenBatchNorm2d
- Sine position embeddings (2D spatial + 4D query)
- Transformer encoder with query_scale MLP
- DAB decoder with content/position attention, ref_point_head, ref_anchor_head
- Detection head (class_embed + bbox_predictor)

Reference: https://github.com/huggingface/transformers/tree/main/src/transformers/models/dab_detr
Paper: DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None

from .utils import inverse_sigmoid


# =============================================================================
# Frozen BatchNorm2d
# =============================================================================

class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d where the batch statistics and affine params are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rsqrt.
    """

    def __init__(self, n: int):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        epsilon = 1e-5
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


def _replace_batch_norm(model: nn.Module):
    """Recursively replace all nn.BatchNorm2d with FrozenBatchNorm2d."""
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = FrozenBatchNorm2d(module.num_features)
            if module.weight.device != torch.device("meta"):
                new_module.weight.copy_(module.weight)
                new_module.bias.copy_(module.bias)
                new_module.running_mean.copy_(module.running_mean)
                new_module.running_var.copy_(module.running_var)
            model._modules[name] = new_module
        if len(list(module.children())) > 0:
            _replace_batch_norm(module)


# =============================================================================
# Backbone
# =============================================================================

class DabDetrBackbone(nn.Module):
    """ResNet backbone via timm with FrozenBatchNorm2d.

    Extracts the last feature map from the backbone.

    Args:
        backbone_name: timm model name (e.g. 'resnet50')
        dilation: Whether to use dilation in the last stage (DC5)
        pretrained: Whether to use pretrained weights
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        dilation: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for DAB-DETR backbone. Install with: pip install timm")

        kwargs = {
            "features_only": True,
            "out_indices": (1, 2, 3, 4),
            "pretrained": pretrained,
        }
        if dilation:
            kwargs["output_stride"] = 16

        self.body = timm.create_model(backbone_name, **kwargs)

        # Replace batch norms with frozen batch norms
        with torch.no_grad():
            _replace_batch_norm(self.body)

        # Get output channel size of last feature map
        self.num_channels = self.body.feature_info.channels()[-1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Tuple of (feature_map, mask) where:
                feature_map: Last feature map [B, C, H', W']
                mask: Spatial mask [B, H', W'] (all ones = valid)
        """
        features = self.body(x)
        feature_map = features[-1]  # Last stage feature

        # Create mask (all valid = ones)
        batch_size = x.shape[0]
        mask = torch.ones(
            (batch_size, feature_map.shape[2], feature_map.shape[3]),
            dtype=torch.bool,
            device=x.device,
        )
        return feature_map, mask


# =============================================================================
# Position Embeddings
# =============================================================================

class DabDetrSinePositionEmbedding(nn.Module):
    """Sine-based 2D position embedding for spatial features.

    Generates position embeddings concatenating sine/cosine for
    x and y coordinates into a hidden_size-dimensional embedding.

    Args:
        hidden_size: Total embedding dimension (split equally for x and y)
        temperature_height: Temperature for height coordinates
        temperature_width: Temperature for width coordinates
    """

    def __init__(
        self,
        hidden_size: int = 256,
        temperature_height: int = 20,
        temperature_width: int = 20,
    ):
        super().__init__()
        self.embedding_dim = hidden_size / 2
        self.temperature_height = temperature_height
        self.temperature_width = temperature_width
        self.scale = 2 * math.pi

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor) -> torch.Tensor:
        """Generate position embeddings.

        Args:
            pixel_values: Feature map [B, C, H, W] (used for device/dtype)
            pixel_mask: Boolean mask [B, H, W]

        Returns:
            Position embeddings [B, hidden_size, H, W]
        """
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_tx = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_tx = dim_tx // 2
        dim_tx = dim_tx * (2 / self.embedding_dim)
        dim_tx = self.temperature_width ** dim_tx
        pos_x = x_embed[:, :, :, None] / dim_tx

        dim_ty = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_ty = dim_ty // 2
        dim_ty = dim_ty * (2 / self.embedding_dim)
        dim_ty = self.temperature_height ** dim_ty
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def gen_sine_position_embeddings(pos_tensor: torch.Tensor, hidden_size: int = 256) -> torch.Tensor:
    """Generate sine position embeddings for 4D coordinates (x, y, w, h).

    Used for anchor-based position encoding in the decoder.

    Args:
        pos_tensor: [batch_size, num_queries, 4] — (x, y, w, h)
        hidden_size: Embedding dimension per coordinate pair

    Returns:
        Position embeddings [batch_size, num_queries, hidden_size * 2]
    """
    scale = 2 * math.pi
    dim = hidden_size // 2
    dim_t = torch.arange(dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / dim)

    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

    if pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")

    return pos.to(pos_tensor.dtype)


# =============================================================================
# MLP
# =============================================================================

class DabDetrMLP(nn.Module):
    """Multi-layer perceptron with ReLU activations (except last layer).

    Used for bbox prediction, ref_point_head, query_scale, ref_anchor_head.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of linear layers
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# =============================================================================
# Attention Modules
# =============================================================================

class DetrAttention(nn.Module):
    """Standard multi-head attention for the encoder.

    Position embeddings are added to queries and keys before projection.

    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads
        dropout: Attention dropout probability
    """

    def __init__(self, hidden_size: int = 256, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, \
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, D]
            attention_mask: [B, 1, L, L] additive mask
            object_queries: [B, L, D] position embeddings

        Returns:
            Output tensor [B, L, D]
        """
        batch_size, q_len, _ = hidden_states.size()

        # Add position embeddings
        if object_queries is not None:
            hidden_states_original = hidden_states
            hidden_states = hidden_states + object_queries

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states_original if object_queries is not None else hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, q_len, self.hidden_size)
        return self.out_proj(attn_output)


class DabDetrAttention(nn.Module):
    """DAB-DETR decoder attention.

    Different from standard attention: Q, K, V projections are external.
    Supports different dimensions for Q/K vs V (is_cross=True doubles Q/K dim).

    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads
        dropout: Attention dropout
        is_cross: If True, Q/K dimension is 2x hidden_size (for cross-attention)
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        is_cross: bool = False,
    ):
        super().__init__()
        self.embed_dim = hidden_size * 2 if is_cross else hidden_size
        self.output_dim = hidden_size
        self.num_heads = num_heads
        self.attention_head_dim = self.embed_dim // num_heads
        self.values_head_dim = self.output_dim // num_heads
        self.scaling = self.attention_head_dim ** -0.5
        self.output_proj = nn.Linear(self.output_dim, self.output_dim)
        self.dropout = dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Pre-projected queries [B, Q, D_qk]
            attention_mask: Additive attention mask
            key_states: Pre-projected keys [B, KV, D_qk]
            value_states: Pre-projected values [B, KV, D_v]

        Returns:
            Output tensor [B, Q, D_v]
        """
        batch_size, q_len, _ = hidden_states.size()

        query_states = hidden_states * self.scaling
        query_states = query_states.view(batch_size, -1, self.num_heads, self.attention_head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.num_heads, self.attention_head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_heads, self.values_head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, q_len, self.output_dim)
        return self.output_proj(attn_output)


# =============================================================================
# Decoder Sub-Layers
# =============================================================================

class DabDetrDecoderLayerSelfAttention(nn.Module):
    """Self-attention sub-layer for DAB-DETR decoder.

    Uses separate content/position projections for queries and keys.
    """

    def __init__(self, hidden_size: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.self_attn_query_content_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn_query_pos_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn_key_content_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn_key_pos_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn_value_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn = DabDetrAttention(hidden_size, num_heads, dropout=0.0)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        query_content = self.self_attn_query_content_proj(hidden_states)
        query_pos = self.self_attn_query_pos_proj(query_position_embeddings)
        key_content = self.self_attn_key_content_proj(hidden_states)
        key_pos = self.self_attn_key_pos_proj(query_position_embeddings)
        value = self.self_attn_value_proj(hidden_states)

        query = query_content + query_pos
        key = key_content + key_pos

        hidden_states = self.self_attn(
            hidden_states=query,
            attention_mask=attention_mask,
            key_states=key,
            value_states=value,
        )

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        return hidden_states


class DabDetrDecoderLayerCrossAttention(nn.Module):
    """Cross-attention sub-layer for DAB-DETR decoder.

    Concatenates content and position embeddings for Q and K, then
    uses DabDetrAttention with 2x hidden_size for the Q/K dimension.

    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        keep_query_pos: If True, always use query_pos_proj (not just first layer)
        is_first: Whether this is the first decoder layer
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        keep_query_pos: bool = False,
        is_first: bool = False,
    ):
        super().__init__()
        self.cross_attn_query_content_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_query_pos_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_key_content_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_key_pos_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_value_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_query_pos_sine_proj = nn.Linear(hidden_size, hidden_size)
        self.num_heads = num_heads
        self.cross_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = DabDetrAttention(hidden_size, num_heads, dropout=0.0, is_cross=True)

        self.keep_query_pos = keep_query_pos
        if not keep_query_pos and not is_first:
            self.cross_attn_query_pos_proj = None

        self.is_first = is_first
        self.dropout = dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        query_position_embeddings: torch.Tensor,
        object_queries: torch.Tensor,
        query_sine_embed: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_content = self.cross_attn_query_content_proj(hidden_states)
        key_content = self.cross_attn_key_content_proj(encoder_hidden_states)
        value = self.cross_attn_value_proj(encoder_hidden_states)

        batch_size, num_queries, n_model = query_content.shape
        _, hw, _ = key_content.shape

        key_pos = self.cross_attn_key_pos_proj(object_queries)

        # For the first decoder layer (or keep_query_pos), add position embedding
        if self.is_first or self.keep_query_pos:
            query_pos = self.cross_attn_query_pos_proj(query_position_embeddings)
            query = query_content + query_pos
            key = key_content + key_pos
        else:
            query = query_content
            key = key_content

        # Reshape and concatenate with sine embeddings
        query = query.view(batch_size, num_queries, self.num_heads, n_model // self.num_heads)
        query_sine_embed = self.cross_attn_query_pos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(
            batch_size, num_queries, self.num_heads, n_model // self.num_heads
        )
        query = torch.cat([query, query_sine_embed], dim=3).view(batch_size, num_queries, n_model * 2)

        key = key.view(batch_size, hw, self.num_heads, n_model // self.num_heads)
        key_pos = key_pos.view(batch_size, hw, self.num_heads, n_model // self.num_heads)
        key = torch.cat([key, key_pos], dim=3).view(batch_size, hw, n_model * 2)

        # Cross-attention
        residual = hidden_states
        hidden_states = self.cross_attn(
            hidden_states=query,
            attention_mask=encoder_attention_mask,
            key_states=key,
            value_states=value,
        )

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)
        return hidden_states


class DabDetrDecoderLayerFFN(nn.Module):
    """Feed-forward sub-layer for DAB-DETR decoder.

    Uses PReLU activation by default (DAB-DETR specific).

    Args:
        hidden_size: Model dimension
        ffn_dim: Feed-forward intermediate dimension
        dropout: Dropout probability
        activation: Activation function ('prelu' or 'relu')
    """

    def __init__(
        self,
        hidden_size: int = 256,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "prelu",
    ):
        super().__init__()
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_size)
        if activation == "prelu":
            self.activation_fn = nn.PReLU()
        elif activation == "relu":
            self.activation_fn = nn.ReLU()
        else:
            self.activation_fn = nn.PReLU()
        self.dropout = dropout

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=0.0, training=self.training)  # activation_dropout = 0
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


# =============================================================================
# Encoder
# =============================================================================

class DabDetrEncoderLayer(nn.Module):
    """Single encoder layer with self-attention + FFN."""

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "prelu",
    ):
        super().__init__()
        self.self_attn = DetrAttention(hidden_size, num_heads, dropout=0.0)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = dropout
        if activation == "prelu":
            self.activation_fn = nn.PReLU()
        elif activation == "relu":
            self.activation_fn = nn.ReLU()
        else:
            self.activation_fn = nn.PReLU()
        self.fc1 = nn.Linear(hidden_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_size)
        self.final_layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            object_queries=object_queries,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class DabDetrEncoder(nn.Module):
    """Transformer encoder with query_scale MLP.

    The encoder scales position embeddings by a learned query-dependent factor.

    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads
        ffn_dim: FFN intermediate dimension
        num_layers: Number of encoder layers
        dropout: Dropout probability
        activation: Activation function
        normalize_before: Whether to use pre-norm (unused, kept for config compat)
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.1,
        activation: str = "prelu",
        normalize_before: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.query_scale = DabDetrMLP(hidden_size, hidden_size, hidden_size, 2)
        self.layers = nn.ModuleList([
            DabDetrEncoderLayer(hidden_size, num_heads, ffn_dim, dropout, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size) if normalize_before else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Flattened features [B, HW, D]
            attention_mask: [B, 1, HW, HW] additive mask (or None)
            object_queries: Position embeddings [B, HW, D]

        Returns:
            Encoded features [B, HW, D]
        """
        for layer in self.layers:
            # Scale position embeddings
            pos_scales = self.query_scale(hidden_states)
            scaled_queries = object_queries * pos_scales if object_queries is not None else None

            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                object_queries=scaled_queries,
            )

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        return hidden_states


# =============================================================================
# Decoder
# =============================================================================

class DabDetrDecoderLayer(nn.Module):
    """Single decoder layer: self-attn → cross-attn → FFN."""

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "prelu",
        keep_query_pos: bool = False,
        is_first: bool = False,
    ):
        super().__init__()
        self.self_attn = DabDetrDecoderLayerSelfAttention(hidden_size, num_heads, dropout)
        self.cross_attn = DabDetrDecoderLayerCrossAttention(
            hidden_size, num_heads, dropout, keep_query_pos, is_first
        )
        self.mlp = DabDetrDecoderLayerFFN(hidden_size, ffn_dim, dropout, activation)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        object_queries: torch.Tensor,
        query_position_embeddings: torch.Tensor,
        query_sine_embed: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            query_position_embeddings=query_position_embeddings,
            attention_mask=attention_mask,
        )

        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_position_embeddings=query_position_embeddings,
            object_queries=object_queries,
            query_sine_embed=query_sine_embed,
            encoder_attention_mask=encoder_attention_mask,
        )

        hidden_states = self.mlp(hidden_states=hidden_states)
        return hidden_states


class DabDetrDecoder(nn.Module):
    """DAB-DETR decoder with dynamic anchor boxes.

    Features:
    - ref_point_head: Converts anchor sine embeddings to position queries
    - ref_anchor_head: Modulates H/W attention based on learned anchor size
    - query_scale: Learned scaling of position embeddings (skipped at layer 0)
    - bbox_embed: Optional iterative box refinement

    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads
        ffn_dim: FFN intermediate dimension
        num_layers: Number of decoder layers
        dropout: Dropout probability
        activation: Activation function
        query_dim: Anchor dimension (must be 4)
        keep_query_pos: Whether to use query position in all layers
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.1,
        activation: str = "prelu",
        query_dim: int = 4,
        keep_query_pos: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.query_dim = query_dim

        self.layers = nn.ModuleList([
            DabDetrDecoderLayer(
                hidden_size, num_heads, ffn_dim, dropout, activation,
                keep_query_pos, is_first=(i == 0),
            )
            for i in range(num_layers)
        ])

        self.layernorm = nn.LayerNorm(hidden_size)
        self.query_scale = DabDetrMLP(hidden_size, hidden_size, hidden_size, 2)
        self.ref_point_head = DabDetrMLP(query_dim // 2 * hidden_size, hidden_size, hidden_size, 2)
        self.bbox_embed = None  # Set externally for iterative refinement
        self.ref_anchor_head = DabDetrMLP(hidden_size, hidden_size, 2, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        object_queries: torch.Tensor,
        query_position_embeddings: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: Query embeddings [B, Q, D] (zeros or pattern embeddings)
            encoder_hidden_states: Encoder output [B, HW, D]
            object_queries: Spatial position embeddings [B, HW, D]
            query_position_embeddings: Anchor embeddings [B, Q, query_dim]
            memory_key_padding_mask: [B, HW] padding mask for encoder output

        Returns:
            Dict with 'hidden_states', 'intermediate', 'reference_points'
        """
        intermediate = []
        reference_points = query_position_embeddings.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.query_dim]
            query_sine_embed = gen_sine_position_embeddings(obj_center, self.hidden_size)
            query_pos = self.ref_point_head(query_sine_embed)

            # For the first decoder layer, skip query_scale transformation
            pos_transformation = 1 if layer_id == 0 else self.query_scale(hidden_states)

            # Apply transformation
            query_sine_embed = query_sine_embed[..., :self.hidden_size] * pos_transformation

            # Modulated Height/Width attentions
            reference_anchor_size = self.ref_anchor_head(hidden_states).sigmoid()
            query_sine_embed[..., self.hidden_size // 2:] *= (
                reference_anchor_size[..., 0] / obj_center[..., 2]
            ).unsqueeze(-1)
            query_sine_embed[..., :self.hidden_size // 2] *= (
                reference_anchor_size[..., 1] / obj_center[..., 3]
            ).unsqueeze(-1)

            hidden_states = layer(
                hidden_states,
                None,  # attention_mask (self-attention)
                object_queries,
                query_pos,
                query_sine_embed,
                encoder_hidden_states,
                encoder_attention_mask=None,
            )

            # Iterative box refinement
            if self.bbox_embed is not None:
                new_ref = self.bbox_embed(hidden_states)
                new_ref[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_ref = new_ref[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_ref)
                reference_points = new_ref.detach()

            intermediate.append(self.layernorm(hidden_states))

        hidden_states = self.layernorm(hidden_states)

        return {
            "hidden_states": hidden_states,
            "intermediate": torch.stack(intermediate),
            "reference_points": torch.stack(ref_points),
        }


# =============================================================================
# Full DAB-DETR Model
# =============================================================================

class DABDETRModel(nn.Module):
    """Complete DAB-DETR model for object detection.

    Architecture: ResNet backbone → 1x1 projection → Transformer encoder →
    DAB decoder → class_embed + bbox_predictor detection heads.

    Args:
        num_classes: Number of detection classes
        d_model: Hidden dimension (default 256)
        nhead: Number of attention heads (default 8)
        num_encoder_layers: Number of encoder layers (default 6)
        num_decoder_layers: Number of decoder layers (default 6)
        dim_feedforward: FFN intermediate dimension (default 2048)
        dropout: Dropout probability (default 0.1)
        activation: Activation function (default 'prelu')
        num_queries: Number of object queries (default 300)
        query_dim: Anchor box dimension (must be 4)
        num_patterns: Number of pattern embeddings (default 0)
        random_refpoints_xy: Whether to randomize anchor x/y coords
        keep_query_pos: Whether to keep query position in all decoder layers
        normalize_before: Whether to use pre-norm in encoder
        aux_loss: Whether to return auxiliary outputs for training
        backbone_name: Backbone model name (default 'resnet50')
        backbone_dilation: Whether to use dilation (DC5)
        backbone_pretrained: Whether to use pretrained backbone
        temperature_height: Position embedding temperature (height)
        temperature_width: Position embedding temperature (width)
    """

    def __init__(
        self,
        num_classes: int = 80,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "prelu",
        num_queries: int = 300,
        query_dim: int = 4,
        num_patterns: int = 0,
        random_refpoints_xy: bool = False,
        keep_query_pos: bool = False,
        normalize_before: bool = False,
        aux_loss: bool = False,
        backbone_name: str = "resnet50",
        backbone_dilation: bool = False,
        backbone_pretrained: bool = True,
        temperature_height: int = 20,
        temperature_width: int = 20,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_size = d_model
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        self.aux_loss = aux_loss
        self.query_dim = query_dim

        # Backbone
        self.backbone = DabDetrBackbone(backbone_name, backbone_dilation, backbone_pretrained)

        # Position embedding
        self.position_embedding = DabDetrSinePositionEmbedding(
            d_model, temperature_height, temperature_width,
        )

        # Input projection: reduce backbone channels to d_model
        self.input_projection = nn.Conv2d(self.backbone.num_channels, d_model, kernel_size=1)

        # Query reference point embeddings
        self.query_refpoint_embeddings = nn.Embedding(num_queries, query_dim)
        if random_refpoints_xy:
            self.query_refpoint_embeddings.weight.data[:, :2].uniform_(0, 1)
            self.query_refpoint_embeddings.weight.data[:, :2] = inverse_sigmoid(
                self.query_refpoint_embeddings.weight.data[:, :2]
            )
            self.query_refpoint_embeddings.weight.data[:, :2].requires_grad = False

        # Encoder
        self.encoder = DabDetrEncoder(
            d_model, nhead, dim_feedforward, num_encoder_layers, dropout, activation, normalize_before,
        )

        # Decoder
        self.decoder = DabDetrDecoder(
            d_model, nhead, dim_feedforward, num_decoder_layers, dropout, activation, query_dim, keep_query_pos,
        )

        # Pattern embeddings
        if num_patterns > 0:
            self.patterns = nn.Embedding(num_patterns, d_model)

        # Detection heads
        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_predictor = DabDetrMLP(d_model, d_model, 4, 3)

        # Wire bbox_embed for iterative refinement
        self.decoder.bbox_embed = self.bbox_predictor

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following DAB-DETR conventions."""
        # Initialize bbox predictor last layer to zero
        nn.init.constant_(self.bbox_predictor.layers[-1].weight, 0)
        nn.init.constant_(self.bbox_predictor.layers[-1].bias, 0)

        # Initialize class embed bias with prior prob
        prior_prob = 1 / (self.num_classes + 1)
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_embed.bias, bias_value)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[List[Dict]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input images [B, 3, H, W]
            targets: Optional list of target dicts (unused, kept for API compat)

        Returns:
            Dict with 'pred_logits' [B, Q, C] and 'pred_boxes' [B, Q, 4],
            and optionally 'aux_outputs' list.
        """
        batch_size = x.shape[0]
        device = x.device

        # 1. Backbone → feature map + mask
        feature_map, mask = self.backbone(x)

        # 2. Position embedding
        pos_embed = self.position_embedding(feature_map, mask)

        # 3. Input projection
        projected = self.input_projection(feature_map)

        # 4. Flatten: [B, C, H, W] → [B, HW, C]
        flattened_features = projected.flatten(2).permute(0, 2, 1)
        object_queries = pos_embed.flatten(2).permute(0, 2, 1)

        # 5. Reference point embeddings
        reference_position_embeddings = self.query_refpoint_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # 6. Encode
        encoder_output = self.encoder(
            hidden_states=flattened_features,
            attention_mask=None,
            object_queries=object_queries,
        )

        # 7. Prepare decoder queries
        num_queries = reference_position_embeddings.shape[1]
        if self.num_patterns == 0:
            queries = torch.zeros(batch_size, num_queries, self.hidden_size, device=device)
        else:
            queries = (
                self.patterns.weight[:, None, None, :]
                .repeat(1, self.num_queries, batch_size, 1)
                .flatten(0, 1)
                .permute(1, 0, 2)
            )  # [B, Q*P, D]
            reference_position_embeddings = reference_position_embeddings.repeat(1, self.num_patterns, 1)

        # 8. Decode
        decoder_output = self.decoder(
            hidden_states=queries,
            encoder_hidden_states=encoder_output,
            object_queries=object_queries,
            query_position_embeddings=reference_position_embeddings,
            memory_key_padding_mask=None,
        )

        intermediate_hidden_states = decoder_output["intermediate"]
        reference_points = decoder_output["reference_points"]

        # 9. Detection heads
        logits = self.class_embed(intermediate_hidden_states[-1])

        reference_before_sigmoid = inverse_sigmoid(reference_points)
        bbox_with_refinement = self.bbox_predictor(intermediate_hidden_states)
        bbox_with_refinement[..., :self.query_dim] += reference_before_sigmoid
        outputs_coord = bbox_with_refinement.sigmoid()

        pred_boxes = outputs_coord[-1]

        output = {
            "pred_logits": logits,
            "pred_boxes": pred_boxes,
        }

        if self.aux_loss:
            outputs_class = self.class_embed(intermediate_hidden_states)
            output["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]

        return output
