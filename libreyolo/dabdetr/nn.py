"""
Neural network architecture for DAB-DETR.

Implements:
- PositionEmbeddingSine: 2-D sinusoidal position encoding for the encoder.
- gen_sineembed_for_position: 4-D anchor → sinusoidal embedding for decoder.
- MLP: Multi-layer perceptron helper.
- TransformerEncoder / DABTransformerEncoderLayer
- DABTransformerDecoder / DABTransformerDecoderLayer
- DABDETRModel: top-level nn.Module (backbone + input_proj + encoder + decoder + heads).

Reference: https://github.com/IDEA-Research/DAB-DETR
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .backbone import ResNet50Backbone
from .attention import MultiheadAttentionWithVdim
from .utils import inverse_sigmoid, bias_init_with_prob


__all__ = ["DABDETRModel"]


# =============================================================================
# Position Encodings
# =============================================================================


class PositionEmbeddingSine(nn.Module):
    """2-D sinusoidal position encoding for encoder feature maps.

    Output shape: [B, H*W, d_model].
    """

    def __init__(
        self,
        d_model: int = 256,
        temperature: float = 10000.0,
        normalize: bool = True,
        scale: float = None,
    ):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        self.d_model = d_model
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature map (used only for shape).

        Returns:
            pos: [B, H*W, d_model]
        """
        B, _, H, W = x.shape
        device = x.device

        y_embed = (
            torch.arange(1, H + 1, dtype=torch.float32, device=device)
            .unsqueeze(1)
            .expand(H, W)
        )
        x_embed = (
            torch.arange(1, W + 1, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .expand(H, W)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.d_model // 2, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.d_model // 2))

        pos_x = x_embed.unsqueeze(-1) / dim_t  # [H, W, d_model/2]
        pos_y = y_embed.unsqueeze(-1) / dim_t  # [H, W, d_model/2]

        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1)
        pos_x = pos_x.flatten(-2)  # [H, W, d_model/2]
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1)
        pos_y = pos_y.flatten(-2)  # [H, W, d_model/2]

        pos = torch.cat([pos_y, pos_x], dim=-1)  # [H, W, d_model]
        pos = pos.reshape(H * W, self.d_model).unsqueeze(0).expand(B, -1, -1)
        return pos


def gen_sineembed_for_position(
    pos_tensor: torch.Tensor, d_model: int = 256
) -> torch.Tensor:
    """Generate sinusoidal embeddings from normalised anchor coordinates.

    Args:
        pos_tensor: [B, Q, 4] normalised (cx, cy, w, h) anchors in [0, 1].
        d_model: embedding dimension (must be divisible by 4).

    Returns:
        sineembed: [B, Q, d_model]
    """
    assert d_model % 4 == 0, "d_model must be divisible by 4"
    half = d_model // 2  # used per spatial dim (cx, cy) and (w, h)
    quarter = d_model // 4  # dim per coordinate

    temperature = 10000.0
    dim_t = torch.arange(quarter, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / quarter)

    # pos_tensor: [B, Q, 4] -> split into cx, cy, w, h
    cx = pos_tensor[..., 0:1] / dim_t  # [B, Q, quarter]
    cy = pos_tensor[..., 1:2] / dim_t
    w = pos_tensor[..., 2:3] / dim_t
    h = pos_tensor[..., 3:4] / dim_t

    def _sin_cos(t):
        """Interleave sin/cos: [B, Q, quarter] -> [B, Q, half]"""
        s = torch.stack([t[..., 0::2].sin(), t[..., 1::2].cos()], dim=-1)
        return s.flatten(-2)

    embed = torch.cat([_sin_cos(cx), _sin_cos(cy), _sin_cos(w), _sin_cos(h)], dim=-1)
    return embed  # [B, Q, d_model]


# =============================================================================
# MLP helper
# =============================================================================


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU activations (no activation on last layer)."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
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
# Transformer Encoder
# =============================================================================


class DABTransformerEncoderLayer(nn.Module):
    """Standard transformer encoder layer (self-attention only).

    DAB-DETR uses a ``query_scale`` MLP to modulate the position embedding
    fed into each encoder layer.  However that modulation lives inside the
    encoder stack rather than here — this layer just accepts a pre-computed
    positional bias.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = getattr(F, activation)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            src: [B, N, d_model]
            pos: [B, N, d_model] positional encoding (added to Q and K)
            src_key_padding_mask: [B, N]

        Returns:
            src: [B, N, d_model]
        """
        if self.normalize_before:
            src2 = self.norm1(src)
            q = k = self.with_pos_embed(src2, pos)
            src2, _ = self.self_attn(
                q, k, value=src2, key_padding_mask=src_key_padding_mask
            )
            src = src + self.dropout1(src2)
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        else:
            q = k = self.with_pos_embed(src, pos)
            src2, _ = self.self_attn(
                q, k, value=src, key_padding_mask=src_key_padding_mask
            )
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src


class DABTransformerEncoder(nn.Module):
    """Stack of DAB-DETR encoder layers with query_scale position modulation."""

    def __init__(
        self, encoder_layer: DABTransformerEncoderLayer, num_layers: int, d_model: int
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.d_model = d_model
        # query_scale: MLP(d_model -> d_model), applied per layer to modulate pos
        self.query_scale = MLP(d_model, d_model, d_model, 2)

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            src: [B, N, d_model] flattened feature map
            pos: [B, N, d_model] sinusoidal position encoding
            src_key_padding_mask: [B, N]

        Returns:
            output: [B, N, d_model]
        """
        output = src
        for i, layer in enumerate(self.layers):
            # Modulate pos by query_scale (keeps position scale information)
            pos_scales = self.query_scale(output)
            modulated_pos = pos * pos_scales
            output = layer(
                output, pos=modulated_pos, src_key_padding_mask=src_key_padding_mask
            )
        return output


# =============================================================================
# Transformer Decoder
# =============================================================================


class DABTransformerDecoderLayer(nn.Module):
    """Single DAB-DETR decoder layer.

    Self-attention → cross-attention (with concatenated content+pos Q/K) → FFN.

    The cross-attention uses a custom MultiheadAttentionWithVdim because Q and K
    have dimension 2*d_model (content ⊕ sinusoidal-pos) while V stays at d_model.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        modulate_hw_attn: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.modulate_hw_attn = modulate_hw_attn

        # Self-attention (queries attend to each other)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention: Q/K dim = 2*d_model, V dim = d_model
        self.cross_attn = MultiheadAttentionWithVdim(
            embed_dim=2 * d_model,
            num_heads=nhead,
            vdim=d_model,
            dropout=dropout,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation = getattr(F, activation)

        # query_scale and ref_point_head are set externally by the decoder stack
        # (they are shared/independent MLPs not duplicated per-layer)

        if modulate_hw_attn:
            # Predicts w and h scale factors for the sinusoidal pos embedding
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)

    def with_pos_embed(self, tensor: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: torch.Tensor,  # [B, Q, d_model]  query content
        memory: torch.Tensor,  # [B, N, d_model]  encoder output
        query_sine_embed: torch.Tensor,  # [B, Q, 2*d_model] sin-embed from anchor (full)
        query_pos: torch.Tensor,  # [B, Q, d_model]   content-side self-attn bias
        memory_pos: torch.Tensor,  # [B, N, d_model]   encoder pos for cross K
        is_first_layer: bool = False,
        self_attn_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Returns:
            tgt: [B, Q, d_model]
        """
        # ── Self-attention ────────────────────────────────────────────────────
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=self_attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ── Cross-attention ───────────────────────────────────────────────────
        # Q: concat(content, sin_pos_embed_xy) — first half of query_sine_embed
        # K: concat(memory_content, memory_pos)
        # V: memory_content

        # query_sine_embed is full d_model embed from (cx,cy,w,h).
        # For Q/K we use the (cx,cy) half only (first d_model/2*2 = d_model dims).
        # The original repo uses the full 4D embed for Q and only 2D for K.
        # We follow the reference implementation exactly:
        #   Q_pos = query_sine_embed (full 2*d_model) after optional HW modulation
        #   K_pos = memory_pos       (d_model)

        if self.modulate_hw_attn:
            # ref_anchor_head predicts w,h scale factors from current query content
            refhw = self.ref_anchor_head(tgt).sigmoid()  # [B, Q, 2]
            # query_sine_embed: [B, Q, 2*d_model]
            # first d_model dims correspond to (cx,cy), last d_model to (w,h)
            q_sine_xy = query_sine_embed[..., : self.d_model]  # [B, Q, d_model]
            q_sine_wh = query_sine_embed[..., self.d_model :]  # [B, Q, d_model]
            # Scale wh sine dims: view as [B, Q, d_model//2, 2], multiply by
            # refhw[..., None, :] which broadcasts as [B, Q, 1, 2], then reshape back.
            B_, Q_, _ = tgt.shape
            q_sine_wh = (
                q_sine_wh.view(B_, Q_, self.d_model // 2, 2)
                * refhw.unsqueeze(2)  # [B, Q, 1, 2]
            ).reshape(B_, Q_, self.d_model)
            q_pos = torch.cat([q_sine_xy, q_sine_wh], dim=-1)  # [B, Q, 2*d_model]
        else:
            q_pos = query_sine_embed  # [B, Q, 2*d_model]

        # Concatenate content and positional parts for Q and K
        q_cross = torch.cat(
            [tgt, q_pos[..., : self.d_model]], dim=-1
        )  # [B, Q, 2*d_model]
        k_cross = torch.cat([memory, memory_pos], dim=-1)  # [B, N, 2*d_model]

        tgt2, _ = self.cross_attn(
            query=q_cross,
            key=k_cross,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ── FFN ──────────────────────────────────────────────────────────────
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class DABTransformerDecoder(nn.Module):
    """DAB-DETR transformer decoder with iterative anchor box refinement.

    At each layer:
      1. Computes sinusoidal position embedding from current anchors.
      2. Runs decoder layer.
      3. Updates anchors via bbox_embed MLP delta on inverse_sigmoid coords.

    Returns all intermediate predictions for auxiliary loss computation.
    """

    def __init__(
        self,
        decoder_layer: DABTransformerDecoderLayer,
        num_layers: int,
        d_model: int,
        return_intermediate: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.d_model = d_model
        self.return_intermediate = return_intermediate

        # Shared MLPs across decoder layers
        # ref_point_head: (2*d_model -> d_model) transforms 4D sine-embed to query bias
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        # query_scale: (d_model -> d_model) modulates position per layer (identity at layer 0)
        self.query_scale = MLP(d_model, d_model, d_model, 2)

    def forward(
        self,
        tgt: torch.Tensor,  # [B, Q, d_model]  initial query content
        refpoints_sigmoid: torch.Tensor,  # [B, Q, 4]  initial normalised anchors
        memory: torch.Tensor,  # [B, N, d_model]  encoder output
        memory_pos: torch.Tensor,  # [B, N, d_model]  encoder pos embed
        bbox_embed: nn.ModuleList,  # per-layer or shared bbox MLP
        memory_key_padding_mask: torch.Tensor = None,
        self_attn_mask: torch.Tensor = None,
    ):
        """
        Returns:
            out_bboxes:  [num_layers, B, Q, 4]  sigmoid-activated box predictions
            out_logits_placeholder:  None (logits come from score_head in DABDETRModel)
            intermediate: list of [B, Q, d_model] hidden states (for score_head)
        """
        intermediate = []
        intermediate_ref_points = []

        ref_points = refpoints_sigmoid  # [B, Q, 4]

        for layer_idx, layer in enumerate(self.layers):
            # Generate sinusoidal embedding from current 4D anchor
            query_sine_embed = gen_sineembed_for_position(ref_points, self.d_model * 2)
            # [B, Q, 2*d_model] — full embed for (cx,cy,w,h)

            # ref_point_head: projects 4D sine embed to d_model content bias
            query_pos = self.ref_point_head(query_sine_embed)  # [B, Q, d_model]

            # query_scale modulates position per layer (skip at layer 0)
            if layer_idx == 0:
                pos_scale = torch.ones_like(query_pos)
            else:
                pos_scale = self.query_scale(tgt)  # [B, Q, d_model]
            query_pos = query_pos * pos_scale

            tgt = layer(
                tgt=tgt,
                memory=memory,
                query_sine_embed=query_sine_embed,
                query_pos=query_pos,
                memory_pos=memory_pos,
                is_first_layer=(layer_idx == 0),
                self_attn_mask=self_attn_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

            # Iterative anchor refinement via delta on inverse_sigmoid
            delta_unsig = bbox_embed[layer_idx](tgt)  # [B, Q, 4]
            outputs_unsig = delta_unsig + inverse_sigmoid(ref_points)
            new_ref_points = outputs_unsig.sigmoid()

            if self.return_intermediate:
                intermediate.append(tgt)
                intermediate_ref_points.append(new_ref_points)

            ref_points = new_ref_points.detach()

        if self.return_intermediate:
            return (
                torch.stack(intermediate_ref_points, dim=0),  # [L, B, Q, 4]
                intermediate,  # list[L] of [B, Q, d_model]
            )

        return (
            new_ref_points.unsqueeze(0),
            [tgt],
        )


# =============================================================================
# DAB-DETR Top-Level Model
# =============================================================================


class DABDETRModel(nn.Module):
    """Complete DAB-DETR model: backbone + input_proj + pos_enc + encoder + decoder + heads.

    Args:
        num_classes: Number of object classes.
        backbone_dilation: If True, use DC5 variant (replace C5 stride with dilation).
        d_model: Hidden dimension for the transformer.
        nhead: Number of attention heads.
        num_encoder_layers: Number of transformer encoder layers.
        num_decoder_layers: Number of transformer decoder layers.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout probability.
        num_queries: Number of base object queries.
        num_patterns: Number of pattern embeddings (0 = disabled, 3 for 3-pat variants).
        modulate_hw_attn: Whether to apply HW attention modulation in decoder.
        aux_loss: Return auxiliary outputs from all decoder layers.
        backbone_pretrained: Load ImageNet-pretrained backbone weights.
    """

    def __init__(
        self,
        num_classes: int = 80,
        backbone_dilation: bool = False,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_queries: int = 300,
        num_patterns: int = 0,
        modulate_hw_attn: bool = True,
        aux_loss: bool = True,
        backbone_pretrained: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        self.aux_loss = aux_loss
        self.num_classes = num_classes

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = ResNet50Backbone(
            dilation=backbone_dilation,
            pretrained=backbone_pretrained,
            freeze_bn=True,
            train_backbone=True,
        )

        # ── Input projection: 2048 → d_model ─────────────────────────────────
        self.input_proj = nn.Conv2d(self.backbone.num_channels, d_model, kernel_size=1)

        # ── Position encoding ─────────────────────────────────────────────────
        self.pos_enc = PositionEmbeddingSine(d_model=d_model)

        # ── Encoder ───────────────────────────────────────────────────────────
        enc_layer = DABTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = DABTransformerEncoder(enc_layer, num_encoder_layers, d_model)

        # ── Decoder ───────────────────────────────────────────────────────────
        dec_layer = DABTransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            modulate_hw_attn=modulate_hw_attn,
        )
        self.decoder = DABTransformerDecoder(
            dec_layer, num_decoder_layers, d_model, return_intermediate=True
        )

        # ── Learnable anchor queries ──────────────────────────────────────────
        # refpoint_embed: Q × 4, activated with sigmoid to get normalised boxes
        self.refpoint_embed = nn.Embedding(num_queries, 4)
        nn.init.uniform_(self.refpoint_embed.weight)

        # ── Pattern embeddings (for 3-pat variants) ───────────────────────────
        if num_patterns > 0:
            self.pattern_embed = nn.Embedding(num_patterns, d_model)
        else:
            self.pattern_embed = None

        # ── Per-layer bbox refinement heads ──────────────────────────────────
        self.bbox_embed = nn.ModuleList(
            [MLP(d_model, d_model, 4, 3) for _ in range(num_decoder_layers)]
        )

        # ── Classification head ───────────────────────────────────────────────
        self.class_embed = nn.Linear(d_model, num_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        # Input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)

        # Bbox heads: zero final layer weights/biases
        for bbox_head in self.bbox_embed:
            nn.init.constant_(bbox_head.layers[-1].weight, 0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0)

        # Classification head: prior-probability bias
        prior_prob = 0.01
        bias_value = bias_init_with_prob(prior_prob)
        nn.init.constant_(self.class_embed.bias, bias_value)

    def _build_query_tgt(self, batch_size: int, device: torch.device):
        """Build initial query content tensor [B, Q_eff, d_model]."""
        if self.num_patterns == 0:
            # Standard: zero initial content
            q = self.num_queries
            tgt = torch.zeros(batch_size, q, self.d_model, device=device)
        else:
            # Pattern tiling: tile num_patterns patterns over num_queries
            # effective_q = num_patterns * num_queries
            tgt = self.pattern_embed.weight.unsqueeze(1).expand(
                self.num_patterns, self.num_queries, self.d_model
            )  # [P, Q, d_model]
            tgt = tgt.reshape(1, self.num_patterns * self.num_queries, self.d_model)
            tgt = tgt.expand(batch_size, -1, -1)  # [B, P*Q, d_model]
        return tgt

    def _build_refpoints(self, batch_size: int, device: torch.device):
        """Build initial reference points [B, Q_eff, 4] in (0,1)."""
        refpoints = self.refpoint_embed.weight.sigmoid()  # [Q, 4]
        if self.num_patterns > 0:
            # Tile: each base query gets num_patterns copies
            refpoints = (
                refpoints.unsqueeze(0)
                .expand(self.num_patterns, -1, -1)
                .reshape(self.num_patterns * self.num_queries, 4)
            )
        refpoints = refpoints.unsqueeze(0).expand(batch_size, -1, -1)  # [B, Q_eff, 4]
        return refpoints

    def forward(self, x: torch.Tensor, targets=None):
        """
        Args:
            x: Input images [B, 3, H, W].
            targets: Unused during inference; present for trainer API compatibility.

        Returns:
            dict with:
                'pred_logits': [B, Q_eff, num_classes]
                'pred_boxes':  [B, Q_eff, 4]
                'aux_outputs': list of dicts (training only, when aux_loss=True)
        """
        B, _, H, W = x.shape
        device = x.device

        # ── Backbone ──────────────────────────────────────────────────────────
        feat = self.backbone(x)  # [B, 2048, H', W']

        # ── Input projection + positional encoding ────────────────────────────
        feat_proj = self.input_proj(feat)  # [B, d_model, H', W']
        pos = self.pos_enc(feat_proj)  # [B, H'*W', d_model]
        feat_flat = feat_proj.flatten(2).permute(0, 2, 1)  # [B, H'*W', d_model]

        # ── Encoder ───────────────────────────────────────────────────────────
        memory = self.encoder(feat_flat, pos)  # [B, H'*W', d_model]

        # ── Decoder ───────────────────────────────────────────────────────────
        tgt = self._build_query_tgt(B, device)  # [B, Q_eff, d_model]
        refpoints = self._build_refpoints(B, device)  # [B, Q_eff, 4]

        out_bboxes, hidden_states = self.decoder(
            tgt=tgt,
            refpoints_sigmoid=refpoints,
            memory=memory,
            memory_pos=pos,
            bbox_embed=self.bbox_embed,
        )
        # out_bboxes:   [L, B, Q_eff, 4]
        # hidden_states: list[L] of [B, Q_eff, d_model]

        # Final predictions (last decoder layer)
        pred_logits = self.class_embed(hidden_states[-1])  # [B, Q_eff, num_classes]
        pred_boxes = out_bboxes[-1]  # [B, Q_eff, 4]

        out = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(hidden_states, out_bboxes)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, hidden_states, out_bboxes):
        """Build auxiliary loss outputs from intermediate decoder layers."""
        return [
            {"pred_logits": self.class_embed(h), "pred_boxes": b}
            for h, b in zip(hidden_states[:-1], out_bboxes[:-1])
        ]
