import numpy as np

from src.modules.events import Event
from src.modules.metrics import (
    best_collar_f1,
    collar_f1,
    frame_metrics,
    match_counts,
    psds1,
    psds2,
)


def _clips():
    gt = [[Event(0, 1.0, 3.0)], [Event(1, 4.0, 5.0)]]
    pred = [[Event(0, 1.0, 3.0, 0.9)], [Event(1, 4.0, 5.0, 0.8)]]
    return pred, gt


def test_perfect_predictions_score_one():
    pred, gt = _clips()
    psds = psds1(pred, gt, dataset_duration=20.0)
    psds_two = psds2(pred, gt, dataset_duration=20.0)
    collar = collar_f1(pred, gt, threshold=0.5)
    assert abs(psds["psds1"] - 1.0) < 1e-6
    assert abs(psds_two["psds2"] - 1.0) < 1e-6
    assert abs(collar["f1"] - 1.0) < 1e-6
    assert collar["tp"] == 2 and collar["fp"] == 0 and collar["fn"] == 0


def test_psds2_penalises_cross_triggers():
    gt = [[Event(0, 1.0, 3.0)]]
    correct = [[Event(0, 1.0, 3.0, 0.9)]]
    wrong_class = [[Event(1, 1.0, 3.0, 0.9)]]
    good = psds2(correct, gt, dataset_duration=10.0)["psds2"]
    bad = psds2(wrong_class, gt, dataset_duration=10.0)["psds2"]
    assert good > bad


def test_boundary_shift_breaks_collar_match():
    pred = [[Event(0, 1.5, 3.5, 0.9)]]
    gt = [[Event(0, 1.0, 3.0)]]
    collar = collar_f1(pred, gt, threshold=0.5, collar=0.2)
    assert collar["f1"] == 0.0


def test_matching_counts_are_per_class():
    pred = [Event(0, 1.0, 2.0, 0.9), Event(1, 3.0, 4.0, 0.9)]
    gt = [Event(0, 1.0, 2.0), Event(0, 5.0, 6.0)]
    counts = match_counts(pred, gt, iou_threshold=0.5)
    assert counts[0]["tp"] == 1
    assert counts[0]["fn"] == 1
    assert counts[1]["fp"] == 1


def test_false_positive_lowers_psds():
    pred, gt = _clips()
    pred = [p + [Event(2, 0.5, 1.5, 0.95)] for p in pred]
    psds = psds1(pred, gt, dataset_duration=20.0)
    assert 0.0 < psds["psds1"] < 1.0


def test_no_predictions_scores_zero():
    _, gt = _clips()
    pred = [[], []]
    psds = psds1(pred, gt, dataset_duration=20.0)
    collar = collar_f1(pred, gt, threshold=0.5)
    assert psds["psds1"] == 0.0
    assert collar["f1"] == 0.0


def test_no_ground_truth_is_nan():
    pred = [[Event(0, 1.0, 2.0, 0.9)]]
    gt = [[]]
    assert np.isnan(psds1(pred, gt, dataset_duration=10.0)["psds1"])


def test_best_collar_f1_sweeps_thresholds():
    pred = [[Event(0, 1.0, 3.0, 0.3)]]
    gt = [[Event(0, 1.0, 3.0)]]
    best = best_collar_f1(pred, gt, thresholds=[0.1, 0.5])
    assert best["threshold"] == 0.1
    assert best["f1"] == 1.0


def test_frame_metrics():
    target = np.zeros((1, 1, 10))
    target[0, 0, 2:6] = 1.0
    perfect = frame_metrics(target, target)
    assert perfect["f1"] == 1.0
    empty = frame_metrics(np.zeros_like(target), target)
    assert empty["f1"] == 0.0
    assert empty["recall"] == 0.0
