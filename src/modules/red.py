"""Recurrent Event Detection (RED) layer.

RED (arXiv:2601.04178, Sec. II-A) turns two conditional boundary probabilities
into frame-wise event presence, onset and offset probabilities with a
parameter-free probabilistic recurrence. Omitting the class index:

    p_t       = p_start_t * (1 - p_{t-1}) + (1 - p_end_t) * p_{t-1}
    onset_t   = p_start_t * (1 - p_{t-1})
    offset_t  = p_end_t   * p_{t-1}
    p_{-1}    = 0

where `p_start_t = P(e_t | not e_{t-1})`, `p_end_t = P(not e_t | e_{t-1})`,
and `p_t = P(e_t)` is event presence.

The recurrence is a first-order linear scan `p_t = a_t p_{t-1} + b_t` with
`a_t = 1 - p_start_t - p_end_t` and `b_t = p_start_t`. It is evaluated with a
Hillis-Steele parallel prefix scan (the same associative composition as
Heinsen's scan [arXiv:2311.06281]), so the Python loop is over `log2(T)` steps
instead of `T`. A sequential reference implementation is kept for testing.
"""

from typing import Optional

import torch
import torch.nn as nn


def _scan_compose(
    a_left: torch.Tensor,
    b_left: torch.Tensor,
    a_right: torch.Tensor,
    b_right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose two affine recurrences applied left then right.

    Args:
        a_left: Multipliers of the left recurrence.
        b_left: Offsets of the left recurrence.
        a_right: Multipliers of the right recurrence.
        b_right: Offsets of the right recurrence.

    Returns:
        The composed `(a, b)` with `a = a_right * a_left` and
        `b = a_right * b_left + b_right`.
    """
    return a_right * a_left, a_right * b_left + b_right


def parallel_scan(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive prefix scan of `x_t = a_t * x_{t-1} + b_t` along the last dim.

    `a` and `b` are `(..., T)` and the initial state is zero. Returns
    `(A, B)` such that `x_t = A_t * x_{-1} + B_t`; with `x_{-1}=0`, `x_t = B_t`.
    The scan is exact in exact arithmetic and uses `log2(T)` vectorised steps.

    Args:
        a: Recurrence multipliers, `(..., T)`.
        b: Recurrence offsets, `(..., T)`.

    Returns:
        `(A, B)`, both `(..., T)`, the inclusive scan coefficients.

    Raises:
        ValueError: If `a` and `b` do not have the same shape.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"a and b must have the same shape, got {a.shape} vs {b.shape}"
        )
    t = a.shape[-1]
    aa, bb = a, b
    step = 1
    while step < t:
        a_prev = torch.ones_like(aa)
        b_prev = torch.zeros_like(bb)
        a_prev[..., step:] = aa[..., :-step]
        b_prev[..., step:] = bb[..., :-step]
        aa, bb = _scan_compose(a_prev, b_prev, aa, bb)
        step *= 2
    return aa, bb


def sequential_red(
    p_start: torch.Tensor, p_end: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate RED one frame at a time as a reference implementation.

    Args:
        p_start: Conditional start probabilities, `(B, C, T)`.
        p_end: Conditional end probabilities, `(B, C, T)`.

    Returns:
        `(presence, onset, offset)` probabilities, each `(B, C, T)`.
    """
    batch, classes, frames = p_start.shape
    pres = []
    prev = torch.zeros(batch, classes, device=p_start.device, dtype=p_start.dtype)
    for t in range(frames):
        s, e = p_start[..., t], p_end[..., t]
        cur = s * (1.0 - prev) + (1.0 - e) * prev
        pres.append(cur)
        prev = cur
    pres_t = torch.stack(pres, dim=-1)
    prev = torch.cat(
        [
            torch.zeros(batch, classes, 1, device=p_start.device, dtype=p_start.dtype),
            pres_t[..., :-1],
        ],
        dim=-1,
    )
    onset = p_start * (1.0 - prev)
    offset = p_end * prev
    return pres_t, onset, offset


class RED(nn.Module):
    """Parameter-free, differentiable RED layer.

    `forward` accepts start/end logits (the acoustic model's raw outputs) and an
    optional boolean `mask` of shape `(B, T)` where `False` marks padded frames.
    Padded frames carry the previous state but their outputs are zeroed, so a
    padded tail cannot create spurious events.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        """Initialise the RED layer.

        Args:
            eps: Clamp bound applied to presence probabilities before they are
                used as the previous-frame state.
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Convert start/end logits into boundary-aware probabilities.

        Padded frames carry the previous state but their outputs are zeroed, so
        a padded tail cannot create spurious events.

        Args:
            start_logits: Acoustic model start logits, `(B, C, T)`.
            end_logits: Acoustic model end logits, `(B, C, T)`.
            mask: Optional boolean `(B, T)`; `False` marks padded frames.

        Returns:
            A dict with `presence`, `onset`, `offset`, `p_start` and `p_end`,
            each `(B, C, T)`.

        Raises:
            ValueError: If the start and end tensors have different shapes.
        """
        if start_logits.shape != end_logits.shape:
            raise ValueError(
                "start/end logits must match: "
                f"{tuple(start_logits.shape)} vs {tuple(end_logits.shape)}"
            )
        p_start = torch.sigmoid(start_logits)
        p_end = torch.sigmoid(end_logits)

        a = 1.0 - p_start - p_end
        b = p_start
        if mask is not None:
            # Carry the state through padded frames; their outputs are masked
            # below. `a=1, b=0` is the identity of the affine scan.
            valid = mask.to(dtype=a.dtype).unsqueeze(1)
            a = torch.where(valid > 0, a, torch.ones_like(a))
            b = torch.where(valid > 0, b, torch.zeros_like(b))

        _, pres = parallel_scan(a, b)
        pres = pres.clamp(self.eps, 1.0 - self.eps)

        prev = torch.cat(
            [
                torch.zeros_like(pres[..., :1]),
                pres[..., :-1],
            ],
            dim=-1,
        )
        onset = (p_start * (1.0 - prev)).clamp(0.0, 1.0)
        offset = (p_end * prev).clamp(0.0, 1.0)

        if mask is not None:
            valid = mask.unsqueeze(1)
            pres = pres * valid
            onset = onset * valid
            offset = offset * valid

        return {
            "presence": pres,
            "onset": onset,
            "offset": offset,
            "p_start": p_start,
            "p_end": p_end,
        }
