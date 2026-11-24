"""Batch augmentations for SED training.

Two augmentations are provided:

* `mixup_batch` — mixes two clips in the batch by taking the element-wise
  maximum of their log-mel features and the union of their frame targets. The
  result contains events from both clips, i.e. a synthetic soundscape with
  randomly added events.
* `spec_augment` — random time/frequency masking on the log-mel input.

Both are applied in `src.pipelines.train` only; validation and test data are
never augmented.
"""

from typing import Optional

import torch


def mixup_batch(
    batch: dict,
    prob: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> dict:
    """Randomly union two clips in a batch.

    Args:
        batch: Dict with `mel` `(B, 1, T, F)` and `pres`/`on`/`off`/`don`/
            `doff` `(B, C, T)`.
        prob: Probability of applying the mix to the whole batch.
        generator: Optional torch RNG for reproducibility.

    Returns:
        A new batch dict. Durations are taken from the clip that is active at
        each frame; presence/onset/offset labels are the union of both clips.
    """
    if prob <= 0.0 or batch["mel"].shape[0] < 2:
        return batch
    if torch.rand(1, generator=generator).item() >= prob:
        return batch

    b = batch["mel"].shape[0]
    perm = torch.randperm(b, generator=generator)
    out = dict(batch)
    out["mel"] = torch.maximum(batch["mel"], batch["mel"][perm])
    out["pres"] = torch.maximum(batch["pres"], batch["pres"][perm])
    out["on"] = torch.maximum(batch["on"], batch["on"][perm])
    out["off"] = torch.maximum(batch["off"], batch["off"][perm])

    # Use the original clip's durations where it has an event; otherwise take
    # the donor clip's durations.
    use_main = batch["pres"] >= batch["pres"][perm]
    out["don"] = torch.where(use_main, batch["don"], batch["don"][perm])
    out["doff"] = torch.where(use_main, batch["doff"], batch["doff"][perm])
    return out


def spec_augment(
    mel: torch.Tensor,
    time_masks: int = 2,
    time_width: int = 50,
    freq_masks: int = 2,
    freq_width: int = 8,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply SpecAugment-style time and frequency masking.

    Args:
        mel: Log-mel batch `(B, 1, T, F)`.
        time_masks: Number of time masks per clip.
        time_width: Maximum mask width in frames.
        freq_masks: Number of frequency masks per clip.
        freq_width: Maximum mask width in mel bins.
        generator: Optional torch RNG.

    Returns:
        The masked tensor (a new tensor).
    """
    if time_masks <= 0 and freq_masks <= 0:
        return mel
    out = mel.clone()
    b, _, t, f = out.shape
    for i in range(b):
        for _ in range(time_masks):
            width = int(torch.randint(0, time_width + 1, (1,), generator=generator))
            if width == 0:
                continue
            start = int(torch.randint(0, max(t - width, 1), (1,), generator=generator))
            out[i, :, start : start + width, :] = 0.0
        for _ in range(freq_masks):
            width = int(torch.randint(0, freq_width + 1, (1,), generator=generator))
            if width == 0:
                continue
            start = int(torch.randint(0, max(f - width, 1), (1,), generator=generator))
            out[i, :, :, start : start + width] = 0.0
    return out
