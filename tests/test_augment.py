import torch

from src.modules.augment import mixup_batch, spec_augment


def make_batch(b=4, c=3, t=20, f=8):
    return {
        "mel": torch.randn(b, 1, t, f),
        "pres": torch.zeros(b, c, t),
        "on": torch.zeros(b, c, t),
        "off": torch.zeros(b, c, t),
        "don": torch.zeros(b, c, t),
        "doff": torch.zeros(b, c, t),
        "index": torch.arange(b),
    }


def test_mixup_union_targets():
    batch = make_batch()
    batch["pres"][0, 0, 2:6] = 1.0
    batch["pres"][1, 1, 10:14] = 1.0
    batch["don"][0, 0, 2:6] = 0.5
    out = mixup_batch(batch, prob=1.0, generator=torch.Generator().manual_seed(0))
    assert out["pres"].shape == batch["pres"].shape
    assert (out["pres"] >= batch["pres"]).all()
    assert (out["don"] >= 0).all()


def test_mixup_prob_zero_is_noop():
    batch = make_batch()
    out = mixup_batch(batch, prob=0.0)
    assert out is batch


def test_spec_augment_shapes_and_masks():
    mel = torch.ones(2, 1, 20, 8)
    out = spec_augment(mel, time_masks=1, time_width=5, freq_masks=1, freq_width=3)
    assert out.shape == mel.shape
    assert out.min().item() == 0.0
    assert out.max().item() == 1.0
