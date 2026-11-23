import torch

from src.config import SEDConfig
from src.modules.loss import SEDLoss, focal_loss, iou_loss
from src.modules.model import BoundaryAwareCRNN

B, C, T, F = 2, 3, 40, 12


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


def test_focal_loss_prefers_correct_predictions():
    targets = torch.zeros(1, 1, 10)
    targets[0, 0, 4] = 1.0
    good = torch.full_like(targets, 1e-3)
    good[0, 0, 4] = 0.999
    bad = 1.0 - good
    assert focal_loss(good, targets) < focal_loss(bad, targets)


def test_iou_loss_is_zero_for_perfect_intervals():
    don = torch.full((1, 2, 5), 0.3)
    doff = torch.full((1, 2, 5), 0.2)
    active = torch.ones_like(don)
    loss = iou_loss(don, doff, don, doff, active)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_iou_loss_penalises_wrong_duration():
    don = torch.full((1, 1, 4), 0.3)
    doff = torch.full((1, 1, 4), 0.2)
    active = torch.ones_like(don)
    wrong = iou_loss(don, doff, don + 0.5, doff + 0.5, active)
    assert wrong > 0.05


def test_sed_loss_components_and_gradients():
    cfg = make_cfg()
    model = BoundaryAwareCRNN(cfg)
    criterion = SEDLoss(cfg, lambda_ool=10.0, lambda_iou=1.0)
    batch = {
        "pres": (torch.rand(B, C, T) > 0.8).float(),
        "on": torch.zeros(B, C, T),
        "off": torch.zeros(B, C, T),
        "don": torch.zeros(B, C, T),
        "doff": torch.zeros(B, C, T),
    }
    batch["on"][:, :, 5] = 1.0
    batch["off"][:, :, 20] = 1.0
    batch["don"][:, :, 5:20] = 0.1
    batch["doff"][:, :, 5:20] = 0.2
    outputs = model(torch.randn(B, 1, T, F))
    losses = criterion(outputs, batch)
    assert set(losses) == {"loss", "pres", "ool", "iou"}
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_ablation_switches_disable_terms():
    cfg = make_cfg()
    model = BoundaryAwareCRNN(cfg)
    batch = {
        "pres": torch.zeros(B, C, T),
        "on": torch.zeros(B, C, T),
        "off": torch.zeros(B, C, T),
        "don": torch.zeros(B, C, T),
        "doff": torch.zeros(B, C, T),
    }
    outputs = model(torch.randn(B, 1, T, F))
    criterion = SEDLoss(cfg, use_ool=False, use_epn=False)
    losses = criterion(outputs, batch)
    assert losses["ool"].item() == 0.0
    assert losses["iou"].item() == 0.0
