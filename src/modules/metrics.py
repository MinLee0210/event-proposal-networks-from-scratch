"""Event-based SED metrics: PSDS1 and collar-based F1.

Both metrics operate on `list[list[Event]]` (one list per clip), so the same
functions evaluate EPN inference and the median-filter baseline.

PSDS1 (arXiv:2601.04178, Sec. III-A) follows the DCASE definition: for a grid
of operating points (one global score threshold per point), count per-class
true positives with an IoU-0.5 matching criterion, build the TPR vs effective
false-positive-rate curve, average TPR over classes, and integrate. We omit the
cross-trigger term and the inter-class variance penalty (`alpha_ct=0`,
`alpha_st=0`), which is exactly the paper's "PSDS1 without variance penalty".

PSDS2 is the DCASE scenario-2 metric: DTC/GTC 0.1, CTTC 0.3, cross-trigger
weight 0.5 and inter-class variance penalty 1.0. It is reported alongside PSDS1
because it emphasises class recognition rather than boundary precision.

Collar F1 matches events with a 200 ms onset collar and an offset collar of
`max(200 ms, 20% of the ground-truth duration)`, then pools TP/FP/FN.
"""

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.modules.events import Event, intersection, iou

Counts = Dict[int, Dict[str, int]]


def _empty_counts() -> Counts:
    """Return an empty per-class count table.

    Returns:
        An empty dict.
    """
    return {}


def _bump(counts: Counts, class_id: int, key: str, amount: int = 1) -> None:
    """Increment one TP/FP/FN counter for a class.

    Args:
        counts: Mutable per-class count table.
        class_id: Event class index.
        key: One of `tp`, `fp`, `fn`.
        amount: Increment.
    """
    counts.setdefault(class_id, {"tp": 0, "fp": 0, "fn": 0})[key] += amount


def match_counts(
    predictions: Sequence[Event],
    ground_truth: Sequence[Event],
    iou_threshold: Optional[float] = 0.5,
    collar: Optional[float] = None,
    offset_collar_ratio: float = 0.2,
) -> Counts:
    """Greedy per-class event matching, returned as `{class: {tp, fp, fn}}`.

    Predictions are consumed in descending score order. With `iou_threshold`,
    a match requires IoU >= threshold; with `collar`, it requires both boundary
    deviations to fit their collars. Each ground-truth event matches once.
    """
    if iou_threshold is None and collar is None:
        raise ValueError("provide either iou_threshold or collar")

    gts_by_class: Dict[int, List[int]] = {}
    for idx, gt in enumerate(ground_truth):
        gts_by_class.setdefault(gt.class_id, []).append(idx)
    matched = [False] * len(ground_truth)
    counts: Counts = {}

    for pred in sorted(predictions, key=lambda e: e.score, reverse=True):
        best_idx: Optional[int] = None
        best_score = -1.0
        for idx in gts_by_class.get(pred.class_id, []):
            if matched[idx]:
                continue
            gt = ground_truth[idx]
            if iou_threshold is not None:
                value = iou(pred, gt)
                if value >= iou_threshold and value > best_score:
                    best_idx, best_score = idx, value
            else:
                onset_err = abs(pred.onset - gt.onset)
                offset_err = abs(pred.offset - gt.offset)
                offset_collar = max(float(collar), offset_collar_ratio * gt.duration)
                if onset_err <= float(collar) and offset_err <= offset_collar:
                    # Prefer the closest boundary pair.
                    value = 1.0 / (1.0 + onset_err + offset_err)
                    if value > best_score:
                        best_idx, best_score = idx, value
        if best_idx is None:
            _bump(counts, pred.class_id, "fp")
        else:
            matched[best_idx] = True
            _bump(counts, pred.class_id, "tp")

    for idx, gt in enumerate(ground_truth):
        if not matched[idx]:
            _bump(counts, gt.class_id, "fn")
    return counts


def _totals(counts: Counts) -> tuple[int, int, int]:
    """Sum per-class TP/FP/FN counts.

    Args:
        counts: Per-class count table.

    Returns:
        `(tp, fp, fn)` pooled over classes.
    """
    tp = sum(c["tp"] for c in counts.values())
    fp = sum(c["fp"] for c in counts.values())
    fn = sum(c["fn"] for c in counts.values())
    return tp, fp, fn


def collar_f1(
    predictions: Sequence[Sequence[Event]],
    ground_truth: Sequence[Sequence[Event]],
    threshold: float = 0.5,
    collar: float = 0.2,
    offset_collar_ratio: float = 0.2,
) -> Dict[str, object]:
    """Collar-based event F1 at one score threshold.

    Returns micro precision/recall/F1 plus per-class F1.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have the same length")

    counts: Counts = {}
    for preds, gts in zip(predictions, ground_truth):
        kept = [e for e in preds if e.score >= threshold]
        batch = match_counts(
            kept,
            gts,
            iou_threshold=None,
            collar=collar,
            offset_collar_ratio=offset_collar_ratio,
        )
        for c, row in batch.items():
            for key, value in row.items():
                _bump(counts, c, key, value)

    tp, fp, fn = _totals(counts)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    per_class = {}
    for c, row in counts.items():
        p = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0.0
        r = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0.0
        per_class[c] = {
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            **row,
        }
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_class": per_class,
        "threshold": threshold,
    }


def best_collar_f1(
    predictions: Sequence[Sequence[Event]],
    ground_truth: Sequence[Sequence[Event]],
    thresholds: Iterable[float] = tuple(np.arange(0.1, 0.95, 0.05)),
    collar: float = 0.2,
) -> Dict[str, object]:
    """Sweep score thresholds and return the best collar-F1 operating point.

    Reported for completeness; model selection during training monitors PSDS1,
    which is threshold-independent.
    """
    best: Optional[Dict[str, object]] = None
    for t in thresholds:
        result = collar_f1(predictions, ground_truth, threshold=float(t), collar=collar)
        if best is None or result["f1"] > best["f1"]:
            best = result
    assert best is not None
    return best


def _step_curve(x: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """Step interpolation of a ROC curve, as in psds_eval.

    Duplicate x values keep the highest y; y is made monotonically
    non-decreasing; missing points carry the previous value forward.
    """
    order = np.argsort(xp)
    xp, yp = xp[order], yp[order]
    uniq_x, idx = np.unique(xp, return_index=True)
    uniq_y = np.maximum.reduceat(yp, idx)
    uniq_y = np.maximum.accumulate(uniq_y)

    out = np.zeros_like(x)
    pos = np.searchsorted(uniq_x, x, side="right") - 1
    valid = pos >= 0
    out[valid] = uniq_y[pos[valid]]
    return out


def _dtc_gtc_counts(
    predictions: Sequence[Event],
    ground_truth: Sequence[Event],
    dtc: float,
    gtc: float,
    cttc: float,
) -> tuple[Counts, Counts, dict[tuple[int, int], int]]:
    """Count TP/FP and cross-triggers with the DTC/GTC/CTTC criteria.

    Implements the intersection-based matching used by DCASE PSDS2: a detection
    is relevant when its summed same-class intersection covers at least `dtc`
    of its duration; a ground-truth event is a TP when relevant detections cover
    at least `gtc` of it; a false positive that covers at least `cttc` of a
    different-class ground truth is a cross-trigger.

    Args:
        predictions: Detections for one clip.
        ground_truth: Ground-truth events for the same clip.
        dtc: Detection tolerance criterion in `[0, 1]`.
        gtc: Ground-truth intersection criterion in `[0, 1]`.
        cttc: Cross-trigger tolerance criterion in `[0, 1]`.

    Returns:
        `(tp, fp, ct)` where `tp`/`fp` are per-class count tables and `ct` maps
        `(predicted_class, ground_truth_class)` to a cross-trigger count.
    """
    tp: Counts = {}
    fp: Counts = {}
    ct: dict[tuple[int, int], int] = {}
    if not predictions or not ground_truth:
        for pred in predictions:
            _bump(fp, pred.class_id, "fp")
        return tp, fp, ct

    inter = [[intersection(pred, gt) for gt in ground_truth] for pred in predictions]
    relevant = []
    for i, pred in enumerate(predictions):
        dur = max(pred.duration, 1e-9)
        same = sum(
            inter[i][j]
            for j, gt in enumerate(ground_truth)
            if gt.class_id == pred.class_id
        )
        relevant.append(same / dur >= dtc)

    tp_gt: list[int] = []
    for j, gt in enumerate(ground_truth):
        dur = max(gt.duration, 1e-9)
        coverage = sum(inter[i][j] for i, pred in enumerate(predictions) if relevant[i])
        if coverage / dur >= gtc:
            tp_gt.append(j)
            _bump(tp, gt.class_id, "tp")

    for i, pred in enumerate(predictions):
        # Relevant detections that contribute to a TP are not false positives;
        # this mirrors psds_eval dropping duplicate TPs.
        if relevant[i] and any(inter[i][j] > 0.0 for j in tp_gt):
            continue
        _bump(fp, pred.class_id, "fp")
        dur = max(pred.duration, 1e-9)
        for j, gt in enumerate(ground_truth):
            if gt.class_id == pred.class_id:
                continue
            if inter[i][j] / dur >= cttc:
                key = (pred.class_id, gt.class_id)
                ct[key] = ct.get(key, 0) + 1
    return tp, fp, ct


def _psd_score(
    rows: list[dict],
    classes: list[int],
    max_efpr: float,
    alpha_st: float,
) -> tuple[float, list[float], list[float]]:
    """Integrate per-class PSD-ROC curves into a PSDS value.

    Args:
        rows: Operating points with `efpr` and `tpr` dicts keyed by class.
        classes: Class ids present in the evaluation.
        max_efpr: Maximum effective FPR considered, in errors per duration unit.
        alpha_st: Inter-class variance penalty weight (0 disables it).

    Returns:
        `(score, x_points, mean_tpr)` for the integrated curve.
    """
    x_points = {c: [0.0] for c in classes}
    y_points = {c: [0.0] for c in classes}
    for row in rows:
        for c in classes:
            x_points[c].append(float(row["efpr"][c]))
            y_points[c].append(float(row["tpr"][c]))

    all_x = np.unique(
        np.concatenate([np.asarray(x_points[c]) for c in classes] + [[max_efpr]])
    )
    all_x = all_x[all_x <= max_efpr]
    curves = np.stack(
        [
            _step_curve(all_x, np.asarray(x_points[c]), np.asarray(y_points[c]))
            for c in classes
        ],
        axis=0,
    )
    mean = np.nan_to_num(curves.mean(axis=0), nan=0.0)
    if alpha_st > 0:
        std = np.nan_to_num(curves.std(axis=0), nan=0.0)
        score_curve = np.clip(mean - alpha_st * std, 0.0, None)
    else:
        score_curve = mean

    if max_efpr <= 0:
        score = 1.0 if np.allclose(score_curve, 1.0) else 0.0
    else:
        score = float(np.sum(np.diff(all_x) * score_curve[:-1]) / max_efpr)
    return score, all_x.tolist(), score_curve.tolist()


def psds1(
    predictions: Sequence[Sequence[Event]],
    ground_truth: Sequence[Sequence[Event]],
    dataset_duration: float,
    thresholds: Optional[Iterable[float]] = None,
    iou_threshold: float = 0.5,
    max_efpr: float = 100.0,
    duration_unit_seconds: float = 3600.0,
) -> Dict[str, object]:
    """Threshold-independent PSDS1 (localisation-focused, no variance penalty).

    The paper defines PSDS1 with an IoU-0.5 matching criterion and explicitly
    omits the inter-class variance penalty. The false-positive rate is expressed
    per hour and integrated up to `max_efpr` (DCASE uses 100 FP/hour).

    Args:
        predictions: One event list per clip.
        ground_truth: One event list per clip.
        dataset_duration: Total audio duration of the evaluation set (seconds).
        thresholds: Operating points; defaults to 0.01 ... 0.99.
        iou_threshold: 0.5 for PSDS1.
        max_efpr: Integration limit in false positives per hour.
        duration_unit_seconds: FPR unit (3600 = per hour).

    Returns:
        A dict with `psds1`, `curve`, `operating_points`, `max_efpr` and
        `num_gt`.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have the same length")
    if dataset_duration <= 0:
        raise ValueError("dataset_duration must be > 0")
    thresholds = (
        list(thresholds) if thresholds is not None else list(np.arange(0.01, 1.0, 0.01))
    )

    classes = sorted(
        {e.class_id for clip in ground_truth for e in clip}
        | {e.class_id for clip in predictions for e in clip}
    )
    has_gt = any(clip for clip in ground_truth)
    if not classes or not has_gt:
        return {
            "psds1": float("nan"),
            "curve": {"fpr": [], "tpr": []},
            "operating_points": [],
            "max_efpr": max_efpr,
            "num_gt": {},
        }
    n_gt = {c: 0 for c in classes}
    for clip in ground_truth:
        for e in clip:
            if e.class_id in n_gt:
                n_gt[e.class_id] += 1

    rows = []
    for t in thresholds:
        counts: Counts = {}
        for preds, gts in zip(predictions, ground_truth):
            kept = [e for e in preds if e.score >= float(t)]
            batch = match_counts(kept, gts, iou_threshold=iou_threshold)
            for c, row in batch.items():
                for key, value in row.items():
                    _bump(counts, c, key, value)

        tpr, efpr = {}, {}
        for c in classes:
            row = counts.get(c, {"tp": 0, "fp": 0, "fn": 0})
            tpr[c] = row["tp"] / n_gt[c] if n_gt[c] else 0.0
            efpr[c] = row["fp"] * duration_unit_seconds / dataset_duration
        rows.append({"threshold": float(t), "tpr": tpr, "efpr": efpr})

    score, x_points, y_points = _psd_score(rows, classes, max_efpr, alpha_st=0.0)
    return {
        "psds1": score,
        "curve": {"fpr": x_points, "tpr": y_points},
        "operating_points": rows,
        "max_efpr": max_efpr,
        "num_gt": n_gt,
    }


def psds2(
    predictions: Sequence[Sequence[Event]],
    ground_truth: Sequence[Sequence[Event]],
    dataset_duration: float,
    thresholds: Optional[Iterable[float]] = None,
    dtc: float = 0.1,
    gtc: float = 0.1,
    cttc: float = 0.3,
    alpha_ct: float = 0.5,
    alpha_st: float = 1.0,
    max_efpr: float = 100.0,
    duration_unit_seconds: float = 3600.0,
) -> Dict[str, object]:
    """Threshold-independent PSDS2 (classification-focused DCASE scenario 2).

    Uses DCASE's scenario-2 parameters: DTC/GTC 0.1, CTTC 0.3, cross-trigger
    weight 0.5, inter-class variance penalty 1.0, integrated up to 100 FP/hour.

    Args:
        predictions: One event list per clip.
        ground_truth: One event list per clip.
        dataset_duration: Total audio duration of the evaluation set (seconds).
        thresholds: Operating points; defaults to 0.01 ... 0.99.
        dtc: Detection tolerance criterion.
        gtc: Ground-truth intersection criterion.
        cttc: Cross-trigger tolerance criterion.
        alpha_ct: Cross-trigger penalty weight.
        alpha_st: Inter-class variance penalty weight.
        max_efpr: Integration limit in false positives per hour.
        duration_unit_seconds: FPR unit (3600 = per hour).

    Returns:
        A dict with `psds2`, `curve`, `operating_points`, `max_efpr` and
        `num_gt`.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have the same length")
    if dataset_duration <= 0:
        raise ValueError("dataset_duration must be > 0")
    thresholds = (
        list(thresholds) if thresholds is not None else list(np.arange(0.01, 1.0, 0.01))
    )

    classes = sorted(
        {e.class_id for clip in ground_truth for e in clip}
        | {e.class_id for clip in predictions for e in clip}
    )
    if not classes or not any(ground_truth):
        return {
            "psds2": float("nan"),
            "curve": {"fpr": [], "tpr": []},
            "operating_points": [],
            "max_efpr": max_efpr,
            "num_gt": {},
        }

    n_gt = {c: 0 for c in classes}
    gt_duration = {c: 0.0 for c in classes}
    for clip in ground_truth:
        for e in clip:
            if e.class_id in n_gt:
                n_gt[e.class_id] += 1
                gt_duration[e.class_id] += e.duration

    rows = []
    for t in thresholds:
        tp_counts: Counts = {}
        fp_counts: Counts = {}
        ct_counts: dict[tuple[int, int], int] = {}
        for preds, gts in zip(predictions, ground_truth):
            kept = [e for e in preds if e.score >= float(t)]
            tp, fp, ct = _dtc_gtc_counts(kept, gts, dtc=dtc, gtc=gtc, cttc=cttc)
            for c, row in tp.items():
                _bump(tp_counts, c, "tp", row["tp"])
            for c, row in fp.items():
                _bump(fp_counts, c, "fp", row["fp"])
            for key, value in ct.items():
                ct_counts[key] = ct_counts.get(key, 0) + value

        tpr, efpr = {}, {}
        other_classes = [c for c in classes if n_gt[c] > 0]
        for c in classes:
            tp_c = tp_counts.get(c, {"tp": 0})["tp"]
            fp_c = fp_counts.get(c, {"fp": 0})["fp"]
            tpr[c] = tp_c / n_gt[c] if n_gt[c] else 0.0
            fpr = fp_c * duration_unit_seconds / dataset_duration
            ctr = 0.0
            if len(other_classes) > 1:
                ctr = sum(
                    ct_counts.get((c, k), 0)
                    * duration_unit_seconds
                    / max(gt_duration[k], 1e-9)
                    for k in other_classes
                    if k != c
                ) / (len(other_classes) - 1)
            efpr[c] = fpr + alpha_ct * ctr
        rows.append({"threshold": float(t), "tpr": tpr, "efpr": efpr})

    score, x_points, y_points = _psd_score(rows, classes, max_efpr, alpha_st)
    return {
        "psds2": score,
        "curve": {"fpr": x_points, "tpr": y_points},
        "operating_points": rows,
        "max_efpr": max_efpr,
        "num_gt": n_gt,
    }


def frame_metrics(
    presence: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Frame-wise precision/recall/F1/accuracy for monitoring (not PSDS)."""
    pred = np.asarray(presence) >= threshold
    target = np.asarray(targets) >= 0.5
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    tn = float(np.logical_and(~pred, ~target).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
    }
