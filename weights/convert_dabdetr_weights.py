"""
Convert HuggingFace DAB-DETR weights to LibreYOLO format.

Downloads weights from IDEA-Research/dab-detr-resnet-50 on HuggingFace Hub
and converts them to LibreYOLO's DABDETRModel state dict format.

Usage:
    python weights/convert_dabdetr_weights.py

Output:
    weights/dab_detr_r50.pt
"""

import re
import gc
from pathlib import Path

import torch


# Mapping from HuggingFace key patterns → LibreYOLO key patterns.
# The HF model wraps everything under "model." prefix (DabDetrForObjectDetection.model = DabDetrModel).
# We strip that prefix and map to our structure.
HF_TO_LIBREYOLO_KEY_MAPPING = {
    # Backbone: HF uses backbone.conv_encoder.model._backbone.* → we use backbone.body.*
    r"model\.backbone\.conv_encoder\.model\._backbone\.(.*)": r"backbone.body.\1",

    # Input projection
    r"model\.input_projection\.(bias|weight)": r"input_projection.\1",

    # Query reference point embeddings
    r"model\.query_refpoint_embeddings\.weight": r"query_refpoint_embeddings.weight",

    # Pattern embeddings (for 3pat variants)
    r"model\.patterns\.weight": r"patterns.weight",

    # =========================================================================
    # Encoder
    # =========================================================================
    # Encoder query_scale MLP
    r"model\.encoder\.query_scale\.layers\.(\d+)\.(bias|weight)": r"encoder.query_scale.layers.\1.\2",

    # Encoder layer self-attention Q, K, V projections
    r"model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.(bias|weight)": r"encoder.layers.\1.self_attn.q_proj.\2",
    r"model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.(bias|weight)": r"encoder.layers.\1.self_attn.k_proj.\2",
    r"model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.(bias|weight)": r"encoder.layers.\1.self_attn.v_proj.\2",
    r"model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.(bias|weight)": r"encoder.layers.\1.self_attn.out_proj.\2",

    # Encoder layer FFN
    r"model\.encoder\.layers\.(\d+)\.fc1\.(bias|weight)": r"encoder.layers.\1.fc1.\2",
    r"model\.encoder\.layers\.(\d+)\.fc2\.(bias|weight)": r"encoder.layers.\1.fc2.\2",

    # Encoder layer norms
    r"model\.encoder\.layers\.(\d+)\.self_attn_layer_norm\.(bias|weight)": r"encoder.layers.\1.self_attn_layer_norm.\2",
    r"model\.encoder\.layers\.(\d+)\.final_layer_norm\.(bias|weight)": r"encoder.layers.\1.final_layer_norm.\2",

    # Encoder activation (PReLU)
    r"model\.encoder\.layers\.(\d+)\.activation_fn\.weight": r"encoder.layers.\1.activation_fn.weight",

    # =========================================================================
    # Decoder
    # =========================================================================
    # Decoder layernorm
    r"model\.decoder\.layernorm\.(bias|weight)": r"decoder.layernorm.\1",

    # Decoder query_scale MLP
    r"model\.decoder\.query_scale\.layers\.(\d+)\.(bias|weight)": r"decoder.query_scale.layers.\1.\2",

    # Decoder ref_point_head MLP
    r"model\.decoder\.ref_point_head\.layers\.(\d+)\.(bias|weight)": r"decoder.ref_point_head.layers.\1.\2",

    # Decoder ref_anchor_head MLP
    r"model\.decoder\.ref_anchor_head\.layers\.(\d+)\.(bias|weight)": r"decoder.ref_anchor_head.layers.\1.\2",

    # Decoder bbox_embed (tied with bbox_predictor)
    r"model\.decoder\.bbox_embed\.layers\.(\d+)\.(bias|weight)": r"bbox_predictor.layers.\1.\2",

    # Decoder layer self-attention projections
    r"model\.decoder\.layers\.(\d+)\.self_attn\.self_attn_query_content_proj\.(bias|weight)": r"decoder.layers.\1.self_attn.self_attn_query_content_proj.\2",
    r"model\.decoder\.layers\.(\d+)\.self_attn\.self_attn_query_pos_proj\.(bias|weight)": r"decoder.layers.\1.self_attn.self_attn_query_pos_proj.\2",
    r"model\.decoder\.layers\.(\d+)\.self_attn\.self_attn_key_content_proj\.(bias|weight)": r"decoder.layers.\1.self_attn.self_attn_key_content_proj.\2",
    r"model\.decoder\.layers\.(\d+)\.self_attn\.self_attn_key_pos_proj\.(bias|weight)": r"decoder.layers.\1.self_attn.self_attn_key_pos_proj.\2",
    r"model\.decoder\.layers\.(\d+)\.self_attn\.self_attn_value_proj\.(bias|weight)": r"decoder.layers.\1.self_attn.self_attn_value_proj.\2",
    r"model\.decoder\.layers\.(\d+)\.self_attn\.self_attn\.output_proj\.(bias|weight)": r"decoder.layers.\1.self_attn.self_attn.output_proj.\2",
    r"model\.decoder\.layers\.(\d+)\.self_attn\.self_attn_layer_norm\.(bias|weight)": r"decoder.layers.\1.self_attn.self_attn_layer_norm.\2",

    # Decoder layer cross-attention projections
    r"model\.decoder\.layers\.(\d+)\.cross_attn\.cross_attn_query_content_proj\.(bias|weight)": r"decoder.layers.\1.cross_attn.cross_attn_query_content_proj.\2",
    r"model\.decoder\.layers\.(\d+)\.cross_attn\.cross_attn_query_pos_proj\.(bias|weight)": r"decoder.layers.\1.cross_attn.cross_attn_query_pos_proj.\2",
    r"model\.decoder\.layers\.(\d+)\.cross_attn\.cross_attn_key_content_proj\.(bias|weight)": r"decoder.layers.\1.cross_attn.cross_attn_key_content_proj.\2",
    r"model\.decoder\.layers\.(\d+)\.cross_attn\.cross_attn_key_pos_proj\.(bias|weight)": r"decoder.layers.\1.cross_attn.cross_attn_key_pos_proj.\2",
    r"model\.decoder\.layers\.(\d+)\.cross_attn\.cross_attn_value_proj\.(bias|weight)": r"decoder.layers.\1.cross_attn.cross_attn_value_proj.\2",
    r"model\.decoder\.layers\.(\d+)\.cross_attn\.cross_attn_query_pos_sine_proj\.(bias|weight)": r"decoder.layers.\1.cross_attn.cross_attn_query_pos_sine_proj.\2",
    r"model\.decoder\.layers\.(\d+)\.cross_attn\.cross_attn\.output_proj\.(bias|weight)": r"decoder.layers.\1.cross_attn.cross_attn.output_proj.\2",
    r"model\.decoder\.layers\.(\d+)\.cross_attn\.cross_attn_layer_norm\.(bias|weight)": r"decoder.layers.\1.cross_attn.cross_attn_layer_norm.\2",

    # Decoder layer FFN
    r"model\.decoder\.layers\.(\d+)\.mlp\.fc1\.(bias|weight)": r"decoder.layers.\1.mlp.fc1.\2",
    r"model\.decoder\.layers\.(\d+)\.mlp\.fc2\.(bias|weight)": r"decoder.layers.\1.mlp.fc2.\2",
    r"model\.decoder\.layers\.(\d+)\.mlp\.final_layer_norm\.(bias|weight)": r"decoder.layers.\1.mlp.final_layer_norm.\2",
    r"model\.decoder\.layers\.(\d+)\.mlp\.activation_fn\.weight": r"decoder.layers.\1.mlp.activation_fn.weight",

    # =========================================================================
    # Detection heads
    # In HF, class_embed is at top level (no "model." prefix)
    # bbox_predictor is accessed via decoder.bbox_embed (tied weights), handled above
    # =========================================================================
    r"class_embed\.(bias|weight)": r"class_embed.\1",
}


def convert_hf_key(key: str) -> str:
    """Convert a single HuggingFace key to LibreYOLO key."""
    for pattern, replacement in HF_TO_LIBREYOLO_KEY_MAPPING.items():
        if re.match(f"^{pattern}$", key):
            return re.sub(f"^{pattern}$", replacement, key)
    return None  # Key not matched


def convert_hf_to_libreyolo(hf_state_dict: dict) -> dict:
    """Convert HuggingFace DAB-DETR state dict to LibreYOLO format.

    Args:
        hf_state_dict: State dict from HuggingFace model

    Returns:
        Converted state dict for LibreYOLO DABDETRModel
    """
    new_state_dict = {}
    skipped_keys = []

    for key, value in hf_state_dict.items():
        new_key = convert_hf_key(key)
        if new_key is not None:
            new_state_dict[new_key] = value
        else:
            skipped_keys.append(key)

    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} keys:")
        for k in skipped_keys:
            print(f"  {k}")

    return new_state_dict


def main():
    """Download and convert DAB-DETR ResNet-50 weights from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface-hub")
        return

    output_dir = Path(__file__).parent
    output_path = output_dir / "dab_detr_r50.pt"

    print("Downloading DAB-DETR ResNet-50 weights from HuggingFace...")
    model_path = hf_hub_download(
        repo_id="IDEA-Research/dab-detr-resnet-50",
        filename="model.safetensors",
    )

    print(f"Loading weights from {model_path}...")
    try:
        from safetensors.torch import load_file
        hf_state_dict = load_file(model_path)
    except ImportError:
        print("safetensors not available, trying torch.load...")
        hf_state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    print(f"Converting {len(hf_state_dict)} keys...")
    new_state_dict = convert_hf_to_libreyolo(hf_state_dict)

    del hf_state_dict
    gc.collect()

    print(f"Saving converted weights to {output_path}...")
    print(f"  Total keys: {len(new_state_dict)}")
    torch.save(new_state_dict, output_path)

    # Verify loading
    print("\nVerifying weights load into DABDETRModel...")
    import sys
    sys.path.insert(0, str(output_dir.parent))
    from libreyolo.dabdetr.nn import DABDETRModel

    model = DABDETRModel(
        num_classes=91,  # COCO 91 classes (HF model)
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        num_queries=300,
        backbone_pretrained=False,
    )

    result = model.load_state_dict(new_state_dict, strict=False)
    print(f"  Missing keys: {len(result.missing_keys)}")
    print(f"  Unexpected keys: {len(result.unexpected_keys)}")

    if result.missing_keys:
        print("  Missing:")
        for k in result.missing_keys:
            print(f"    {k}")
    if result.unexpected_keys:
        print("  Unexpected:")
        for k in result.unexpected_keys:
            print(f"    {k}")

    print(f"\n✓ Weights saved to {output_path}")


if __name__ == "__main__":
    main()
