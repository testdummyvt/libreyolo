import pytest
import torch
from libreyolo import LIBREYOLORTDETR
from libreyolo.rtdetr.nn import RTDETRModel
from libreyolo.rtdetr.loss import RTDETRLoss
from PIL import Image
import numpy as np

@pytest.mark.unit
def test_rtdetr_initialization():
    model = LIBREYOLORTDETR(model_path=None, size="r18")
    assert model is not None
    assert model.size == "r18"
    assert model._get_model_name() == "RTDETR"
    
@pytest.mark.unit
def test_rtdetr_forward_pass():
    model = LIBREYOLORTDETR(model_path=None, size="r18", nb_classes=80)
    
    # Create a dummy image
    dummy_img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
    
    # Run inference
    results = model(dummy_img, conf=0.1)
    
    # Check results structure
    assert hasattr(results, "boxes")
    assert hasattr(results.boxes, "xyxy")
    assert hasattr(results.boxes, "conf")
    assert hasattr(results.boxes, "cls")

@pytest.mark.unit
def test_rtdetr_training_forward():
    model = RTDETRModel(num_classes=80, hidden_dim=256, num_queries=300,
                        num_decoder_layers=3, nhead=8, num_denoising=100)
    model.train()
    x = torch.randn(2, 3, 640, 640)
    targets = [
        {'labels': torch.tensor([0, 1]), 'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]])},
        {'labels': torch.tensor([2]),     'boxes': torch.tensor([[0.7, 0.7, 0.3, 0.3]])}
    ]
    out = model(x, targets=targets)
    assert 'pred_logits' in out
    assert 'pred_boxes' in out
    assert 'aux_outputs' in out
    assert 'dn_aux_outputs' in out
    assert 'dn_meta' in out

@pytest.mark.unit
def test_rtdetr_loss():
    model = RTDETRModel(num_classes=80, hidden_dim=256, num_queries=300,
                        num_decoder_layers=3, nhead=8, num_denoising=100)
    criterion = RTDETRLoss(num_classes=80)
    model.train()
    x = torch.randn(2, 3, 640, 640)
    targets = [
        {'labels': torch.tensor([0, 1]), 'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]])},
        {'labels': torch.tensor([2]),     'boxes': torch.tensor([[0.7, 0.7, 0.3, 0.3]])}
    ]
    out = model(x, targets=targets)
    losses = criterion(out, targets)
    assert 'total_loss' in losses
    assert torch.isfinite(losses['total_loss'])

if __name__ == "__main__":
    test_rtdetr_initialization()
    test_rtdetr_forward_pass()
    test_rtdetr_training_forward()
    test_rtdetr_loss()
    print("All tests passed!")
