import torch

from src.modules.red import RED, parallel_scan, sequential_red


def _logits_from_probs(p: torch.Tensor) -> torch.Tensor:
    return torch.logit(p.clamp(1e-4, 1 - 1e-4))


def test_parallel_scan_matches_sequential():
    torch.manual_seed(0)
    b, c, t = 2, 3, 40
    p_start = torch.rand(b, c, t) * 0.3
    p_end = torch.rand(b, c, t) * 0.3
    pres, onset, offset = sequential_red(p_start, p_end)

    red = RED()
    out = red(_logits_from_probs(p_start), _logits_from_probs(p_end))
    assert torch.allclose(out["presence"], pres, atol=1e-4)
    assert torch.allclose(out["onset"], onset, atol=1e-4)
    assert torch.allclose(out["offset"], offset, atol=1e-4)


def test_single_event_has_localised_onset_and_offset():
    t = 20
    p_start = torch.zeros(1, 1, t)
    p_end = torch.zeros(1, 1, t)
    p_start[0, 0, 3] = 0.9
    p_end[0, 0, 9] = 0.9
    out = RED()(_logits_from_probs(p_start), _logits_from_probs(p_end))

    pres = out["presence"][0, 0]
    assert pres[2] < 0.01
    assert (pres[4:8] > 0.7).all()
    assert out["onset"][0, 0, 3] > 0.8
    assert out["offset"][0, 0, 9] > 0.8
    assert out["onset"].sum() < 1.1
    assert out["offset"].sum() < 1.1


def test_mask_zeroes_padded_tail():
    t = 12
    p_start = torch.zeros(1, 1, t)
    p_end = torch.zeros(1, 1, t)
    p_start[0, 0, 2] = 0.9
    p_start[0, 0, 9] = 0.9  # inside the padded region
    mask = torch.tensor([[True] * 6 + [False] * 6])
    out = RED()(_logits_from_probs(p_start), _logits_from_probs(p_end), mask=mask)
    assert (out["presence"][0, 0, 6:] == 0).all()
    assert (out["onset"][0, 0, 6:] == 0).all()


def test_red_is_differentiable():
    p_start = torch.rand(2, 2, 16, requires_grad=True)
    p_end = torch.rand(2, 2, 16, requires_grad=True)
    out = RED()(_logits_from_probs(p_start), _logits_from_probs(p_end))
    loss = out["presence"].sum() + out["onset"].sum() + out["offset"].sum()
    loss.backward()
    assert p_start.grad is not None and torch.isfinite(p_start.grad).all()
    assert p_end.grad is not None and torch.isfinite(p_end.grad).all()


def test_scan_identity():
    a = torch.ones(2, 5)
    b = torch.zeros(2, 5)
    _, out = parallel_scan(a, b)
    assert torch.equal(out, torch.zeros(2, 5))
