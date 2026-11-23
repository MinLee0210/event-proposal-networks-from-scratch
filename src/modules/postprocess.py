"""Post-processing and event inference.

Two ways to turn frame-wise probabilities into events:

1. `median_filter_events` — the standard frame-wise baseline: median filter
   presence probabilities, threshold, group consecutive active frames.
2. `epn_inference` — Algorithm 1 from arXiv:2601.04178: sort proposals by RED
   presence, emit `[t - d_on, t + d_off]`, and suppress overlapping proposals
   (non-maximum-suppression-like), keeping at most `k` events per class and
   considering only the `m` most active classes.

Both return `list[list[Event]]` (one list per clip) so the evaluator can consume
them uniformly.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.config import SEDConfig
from src.modules.events import Event


def median_filter_probs(probs: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Median filter `(B, C, T)` probabilities along time (replicate padding)."""
    if kernel_size <= 1:
        return probs
    if kernel_size % 2 == 0:
        raise ValueError(f"median filter length must be odd, got {kernel_size}")
    pad = kernel_size // 2
    x = F.pad(probs, (pad, pad), mode="replicate")
    windows = x.unfold(-1, kernel_size, 1)
    return windows.median(dim=-1).values


def _runs(mask: np.ndarray) -> List[tuple[int, int]]:
    """Find half-open runs of True in a 1-D boolean array.

    Args:
        mask: Boolean activity mask.

    Returns:
        A list of `(start, end)` pairs with `end` exclusive.
    """
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(changes[i]), int(changes[i + 1])) for i in range(0, len(changes), 2)]


def median_filter_events(
    presence: torch.Tensor,
    frame_rate: float,
    threshold: float = 0.5,
    filter_len: int = 7,
    min_duration: float = 0.0,
) -> List[List[Event]]:
    """Extract events from presence probabilities via median filtering.

    Args:
        presence: Presence probabilities, `(B, C, T)`.
        frame_rate: Frames per second.
        threshold: Activity threshold applied after filtering.
        filter_len: Odd median-filter length in frames.
        min_duration: Drop shorter events (seconds).

    Returns:
        One event list per clip.

    Raises:
        ValueError: If `presence` is not 3-D.
    """
    if presence.dim() != 3:
        raise ValueError(f"expected (B, C, T), got {tuple(presence.shape)}")
    filtered = median_filter_probs(presence.detach().float().cpu(), filter_len)
    active = (filtered >= threshold).numpy()
    out: List[List[Event]] = []
    for b in range(active.shape[0]):
        events: List[Event] = []
        for c in range(active.shape[1]):
            for start, end in _runs(active[b, c]):
                onset, offset = start / frame_rate, end / frame_rate
                if offset - onset < min_duration or end <= start:
                    continue
                score = float(filtered[b, c, start:end].mean().item())
                events.append(Event(c, onset, offset, score))
        out.append(events)
    return out


def epn_inference(
    presence: torch.Tensor,
    don: torch.Tensor,
    doff: torch.Tensor,
    frame_rate: float,
    clip_duration: float,
    k: int = 15,
    m: Optional[int] = None,
    min_score: float = 1e-3,
    min_duration: float = 0.0,
) -> List[List[Event]]:
    """Algorithm 1 (event inference) for `(B, C, T)` RED/EPN outputs.

    Args:
        presence: RED presence probabilities, `(B, C, T)`.
        don, doff: EPN durations in seconds, `(B, C, T)`.
        k: maximum events per class.
        m: number of most-active classes to consider; `None` means all.
        min_score: ignore proposal frames below this presence probability.
        min_duration: drop proposals shorter than this (seconds).

    Returns:
        One event list per clip, sorted by onset then class.

    Raises:
        ValueError: If the three tensors do not share the same shape.
    """
    if presence.shape != don.shape or presence.shape != doff.shape:
        raise ValueError("presence/don/doff must have the same (B, C, T) shape")
    pres = presence.detach().float().cpu().numpy()
    don_np = don.detach().float().cpu().numpy()
    doff_np = doff.detach().float().cpu().numpy()

    batch, classes, frames = pres.shape
    m = classes if m is None else max(1, min(int(m), classes))
    out: List[List[Event]] = []

    for b in range(batch):
        class_scores = pres[b].mean(axis=1)
        selected_classes = np.argsort(-class_scores)[:m]
        events: List[Event] = []

        for c in selected_classes:
            cand = np.flatnonzero(pres[b, c] > min_score)
            if cand.size == 0:
                continue
            cand = cand[np.argsort(-pres[b, c, cand])]
            removed = np.zeros(frames, dtype=bool)
            n_class_events = 0

            for t in cand:
                if n_class_events >= k:
                    break
                if removed[t]:
                    continue
                left = t / frame_rate - don_np[b, c, t]
                right = t / frame_rate + doff_np[b, c, t]
                left = float(np.clip(left, 0.0, clip_duration))
                right = float(np.clip(right, 0.0, clip_duration))
                if right - left < min_duration or right <= left:
                    removed[t] = True
                    continue

                # Suppress every proposal that overlaps the chosen interval.
                cand_left = cand / frame_rate - don_np[b, c, cand]
                cand_right = cand / frame_rate + doff_np[b, c, cand]
                overlapping = (cand_left < right) & (cand_right > left)
                removed[cand[overlapping]] = True

                f0 = max(int(np.floor(left * frame_rate)), 0)
                f1 = min(int(np.ceil(right * frame_rate)), frames)
                score = float(pres[b, c, f0:f1].mean()) if f1 > f0 else 0.0
                events.append(Event(int(c), left, right, score))
                n_class_events += 1

        events.sort(key=lambda e: (e.onset, e.class_id))
        out.append(events)
    return out


def events_from_model_output(
    outputs: dict[str, torch.Tensor],
    cfg: SEDConfig,
    method: str = "epn",
    threshold: float = 0.5,
    filter_len: int = 7,
    k: int = 15,
    m: Optional[int] = None,
    min_duration: float = 0.0,
) -> List[List[Event]]:
    """Dispatch RED/EPN outputs to the requested event extractor."""
    from src.modules.model import presence_probability

    presence = presence_probability(outputs)
    if method == "epn":
        if "don" not in outputs:
            raise KeyError("method='epn' requires EPN outputs (don/doff)")
        return epn_inference(
            presence,
            outputs["don"],
            outputs["doff"],
            frame_rate=cfg.frame_rate,
            clip_duration=cfg.num_frames / cfg.frame_rate,
            k=k,
            m=m,
            min_duration=min_duration,
        )
    if method == "median":
        return median_filter_events(
            presence,
            frame_rate=cfg.frame_rate,
            threshold=threshold,
            filter_len=filter_len,
            min_duration=min_duration,
        )
    raise ValueError(f"unknown event extraction method: {method!r}")
