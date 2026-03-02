import pytest
import torch
from libreyolo import LIBREYOLODABDETR
from libreyolo.dabdetr.nn import DABDETRModel
from libreyolo.dabdetr.loss import DABDETRLoss
from PIL import Image
import numpy as np


@pytest.mark.unit
def test_dabdetr_initialization():
    model = LIBREYOLODABDETR(model_path=None, size="r50")
    assert model is not None
    assert model.size == "r50"
    assert model._get_model_name() == "DABDETR"


@pytest.mark.unit
def test_dabdetr_all_variants_initialize():
    for size in ("r50", "r50-dc5", "r50-3pat", "r50-dc5-3pat"):
        model = LIBREYOLODABDETR(model_path=None, size=size, nb_classes=80)
        assert model.size == size


@pytest.mark.unit
def test_dabdetr_forward_pass():
    model = LIBREYOLODABDETR(model_path=None, size="r50", nb_classes=80)

    dummy_img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))

    results = model(dummy_img, conf=0.1)

    assert hasattr(results, "boxes")
    assert hasattr(results.boxes, "xyxy")
    assert hasattr(results.boxes, "conf")
    assert hasattr(results.boxes, "cls")


@pytest.mark.unit
def test_dabdetr_training_forward():
    model = DABDETRModel(
        num_classes=80,
        d_model=256,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        num_queries=10,
        aux_loss=True,
        backbone_pretrained=False,
    )
    model.train()
    x = torch.randn(2, 3, 320, 320)
    out = model(x, targets=None)
    assert "pred_logits" in out
    assert "pred_boxes" in out
    assert "aux_outputs" in out
    assert out["pred_logits"].shape == (2, 10, 80)
    assert out["pred_boxes"].shape == (2, 10, 4)


@pytest.mark.unit
def test_dabdetr_training_forward_3pat():
    model = DABDETRModel(
        num_classes=80,
        d_model=256,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        num_queries=10,
        num_patterns=3,
        aux_loss=True,
        backbone_pretrained=False,
    )
    model.train()
    x = torch.randn(2, 3, 320, 320)
    out = model(x, targets=None)
    assert "pred_logits" in out
    assert "pred_boxes" in out
    # Effective queries = 3 * 10 = 30
    assert out["pred_logits"].shape == (2, 30, 80)
    assert out["pred_boxes"].shape == (2, 30, 4)


@pytest.mark.unit
def test_dabdetr_loss():
    model = DABDETRModel(
        num_classes=80,
        d_model=256,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        num_queries=10,
        aux_loss=True,
        backbone_pretrained=False,
    )
    criterion = DABDETRLoss(num_classes=80)
    model.train()
    x = torch.randn(2, 3, 320, 320)
    targets = [
        {
            "labels": torch.tensor([0, 1]),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
        },
        {
            "labels": torch.tensor([2]),
            "boxes": torch.tensor([[0.7, 0.7, 0.3, 0.3]]),
        },
    ]
    out = model(x, targets=targets)
    losses = criterion(out, targets)
    assert "total_loss" in losses
    assert torch.isfinite(losses["total_loss"])


@pytest.mark.unit
def test_dabdetr_valid_sizes():
    model = LIBREYOLODABDETR(model_path=None, size="r50")
    valid = model._get_valid_sizes()
    assert "r50" in valid
    assert "r50-dc5" in valid
    assert "r50-3pat" in valid
    assert "r50-dc5-3pat" in valid


if __name__ == "__main__":
    test_dabdetr_initialization()
    test_dabdetr_all_variants_initialize()
    test_dabdetr_forward_pass()
    test_dabdetr_training_forward()
    test_dabdetr_training_forward_3pat()
    test_dabdetr_loss()
    test_dabdetr_valid_sizes()
    print("All tests passed!")
