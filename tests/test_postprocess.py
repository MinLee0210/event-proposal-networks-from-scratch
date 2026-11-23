import torch

from src.modules.postprocess import (
    epn_inference,
    median_filter_events,
    median_filter_probs,
)


def test_median_filter_removes_isolated_spike():
    probs = torch.full((1, 1, 9), 0.1)
    probs[0, 0, 4] = 0.9
    filtered = median_filter_probs(probs, kernel_size=5)
    assert torch.allclose(filtered, torch.full_like(probs, 0.1))


def test_median_filter_events_groups_consecutive_frames():
    probs = torch.zeros(1, 1, 50)
    probs[0, 0, 10:20] = 0.9
    events = median_filter_events(probs, frame_rate=10.0, threshold=0.5, filter_len=3)
    assert len(events[0]) == 1
    ev = events[0][0]
    assert ev.class_id == 0
    assert abs(ev.onset - 1.0) < 1e-6
    assert abs(ev.offset - 2.0) < 1e-6


def test_median_filter_min_duration_drops_short_events():
    probs = torch.zeros(1, 1, 50)
    probs[0, 0, 10:12] = 0.9
    events = median_filter_events(
        probs, frame_rate=10.0, threshold=0.5, filter_len=1, min_duration=0.5
    )
    assert events[0] == []


def test_epn_inference_recovers_one_proposal():
    presence = torch.full((1, 1, 50), 1e-4)
    presence[0, 0, 5] = 0.9
    don = torch.full((1, 1, 50), 0.2)
    doff = torch.full((1, 1, 50), 0.3)
    events = epn_inference(presence, don, doff, frame_rate=10.0, clip_duration=5.0, k=1)
    assert len(events[0]) == 1
    ev = events[0][0]
    assert abs(ev.onset - 0.3) < 1e-5
    assert abs(ev.offset - 0.8) < 1e-5


def test_epn_inference_respects_k_and_m():
    presence = torch.zeros(1, 3, 50)
    presence[0, 0, 5] = 0.9
    presence[0, 1, 25] = 0.8
    presence[0, 2, 45] = 0.7
    don = torch.full_like(presence, 0.1)
    doff = torch.full_like(presence, 0.1)

    all_events = epn_inference(
        presence, don, doff, frame_rate=10.0, clip_duration=5.0, k=15
    )
    assert len(all_events[0]) == 3

    top_one = epn_inference(
        presence, don, doff, frame_rate=10.0, clip_duration=5.0, k=15, m=1
    )
    assert len(top_one[0]) == 1 and top_one[0][0].class_id == 0

    limited = epn_inference(
        presence, don, doff, frame_rate=10.0, clip_duration=5.0, k=1
    )
    assert sum(1 for e in limited[0] if e.class_id == 0) == 1
