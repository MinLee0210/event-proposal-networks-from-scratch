import pytest
import torch

from src.config import SEDConfig
from src.modules.model import (
    BoundaryAwareCRNN,
    FrameCRNN,
    build_model,
    presence_probability,
)

B, C, T, F = 2, 4, 64, 16


def make_cfg(**overrides) -> SEDConfig:
    base = dict(
        num_classes=C,
        num_frames=T,
        n_mels=F,
        cnn_channels=(4, 8),
        rnn_hidden=8,
        rnn_layers=1,
        epn_hidden=4,
        epn_layers=1,
        dropout=0.0,
    )
    base.update(overrides)
    return SEDConfig(**base)


def test_boundary_model_shapes():
    cfg = make_cfg()
    model = BoundaryAwareCRNN(cfg).eval()
    x = torch.randn(B, 1, T, F)
    with torch.no_grad():
        out = model(x)
    assert out["presence"].shape == (B, C, T)
    assert out["onset"].shape == (B, C, T)
    assert out["offset"].shape == (B, C, T)
    assert out["don"].shape == (B, C, T)
    assert out["doff"].shape == (B, C, T)
    assert torch.isfinite(out["presence"]).all()
    assert (out["presence"] >= 0).all() and (out["presence"] <= 1).all()
    assert (out["don"] >= 0).all() and (out["doff"] >= 0).all()


def test_baseline_model_shapes():
    cfg = make_cfg(use_red=False, use_epn=False)
    model = build_model(cfg).eval()
    assert isinstance(model, FrameCRNN)
    x = torch.randn(B, 1, T, F)
    with torch.no_grad():
        out = model(x)
    assert out["pres_logits"].shape == (B, C, T)
    assert presence_probability(out).shape == (B, C, T)


def test_red_without_epn_has_no_durations():
    cfg = make_cfg(use_epn=False)
    model = BoundaryAwareCRNN(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(B, 1, T, F))
    assert "presence" in out
    assert "don" not in out


def test_epn_requires_red():
    with pytest.raises(ValueError):
        make_cfg(use_red=False, use_epn=True)


def test_single_gru_variant():
    cfg = make_cfg(epn_variant="single")
    model = BoundaryAwareCRNN(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(B, 1, T, F))
    assert out["don"].shape == (B, C, T)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        make_cfg(cnn_channels=())
    with pytest.raises(ValueError):
        make_cfg(dropout=1.0)
    with pytest.raises(ValueError):
        make_cfg(epn_variant="unknown")
