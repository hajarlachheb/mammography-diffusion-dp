"""Reference head architecture, trainable-parameter count, Opacus compatibility and checkpoint loading.
Uses torchvision's Swin-T definition with random weights (no download, no medical data)."""
import inspect

import pytest
import torch
import torch.nn as nn
from torchvision import models

from mammo_diffusion_dp.classification import evaluate_classifier, train_classifier


def ref_head():
    return nn.Sequential(nn.Linear(768, 384), nn.LeakyReLU(0.2), nn.Linear(384, 2))


def test_trainer_source_defines_reference_head():
    src = inspect.getsource(train_classifier.main)
    assert "nn.Linear(in_features, 384),\n                nn.LeakyReLU(0.2),\n                nn.Linear(384, 2)" in src
    assert "inplace=True" not in src.split("head_type == 'ref_mlp'")[1].split("else:")[0]


def test_head_parameter_count_with_frozen_backbone():
    m = models.swin_t(weights=None)
    m.head = ref_head()
    for n, p in m.named_parameters():
        p.requires_grad = n.startswith("head.")
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 296066
    assert m.head[0].in_features == 768


def test_per_sample_gradients_with_non_inplace_leakyrelu():
    from opacus import GradSampleModule
    head = GradSampleModule(ref_head())
    x = torch.randn(4, 768)
    head(x).sum().backward()
    assert all(p.grad_sample.shape[0] == 4 for p in head.parameters())


def test_checkpoint_round_trip_through_evaluator(tmp_path):
    torch.manual_seed(0)
    m = models.swin_t(weights=None)
    m.head = ref_head()
    m.eval()
    sd = {f"_module.{k}": v for k, v in m.state_dict().items() if not k.endswith("relative_position_index")}
    ck = tmp_path / "best_model.pth"
    torch.save({"model_state_dict": sd, "head_type": "ref_mlp", "epoch": 0}, ck)  # Opacus-style key prefix
    loaded, meta = evaluate_classifier.load_model(str(ck), model_type="swin_t", device="cpu")
    loaded.eval()
    assert meta["head_type"] == "ref_mlp"
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(loaded(x), m(x), atol=1e-6)


def test_evaluator_rejects_unknown_head(tmp_path):
    ck = tmp_path / "bad.pth"
    torch.save({"model_state_dict": {}, "head_type": "unknown"}, ck)
    with pytest.raises(ValueError):
        evaluate_classifier.load_model(str(ck), model_type="swin_t", device="cpu")
