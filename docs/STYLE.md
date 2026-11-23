# Python docstring convention

Every module, class, public function and method has a docstring. We follow
[PEP 257](https://peps.python.org/pep-0257/) for placement and add
[Google-style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
sections when the callable has arguments, a return value, or raises.

Rules:

- First line is a one-sentence imperative summary ending with a period.
- Leave one blank line after the summary before prose or sections.
- Document every non-obvious argument in `Args`, the return in `Returns`, and
  every deliberate exception in `Raises`.
- Types live in the signature, so the docstring describes meaning, units and
  valid ranges, not the type again.
- Include `Example:` for non-trivial public APIs. Examples must be executable
  and match the current behaviour.
- Private helpers get a one-line docstring explaining *why* they exist.

## Type hints

Every function, method and module-level callable is fully annotated, including
`-> None` on constructors. Use built-in generics (`list[str]`, `dict[str, int]`)
and `Optional[...]` rather than implicit `Any`. Torch types are imported
normally; optional/third-party typing-only imports go under `TYPE_CHECKING`.

```python
from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torchaudio


def waveform_to_logmel(
    waveform: torch.Tensor,
    transform: "torchaudio.transforms.MelSpectrogram",
    num_frames: int,
) -> "np.ndarray":
    """Convert a mono waveform into a fixed-length log-mel spectrogram."""
```


## Example

```python
def compute_iou(
    pred: Event,
    target: Event,
    eps: float = 1e-7,
) -> float:
    """Compute temporal intersection-over-union for two events.

    Intervals are half-open `[onset, offset)`. IoU is sensitive to boundary
    errors, which is why it is used both by the PSDS1 matcher and by the EPN
    duration loss.

    Args:
        pred: Predicted event.
        target: Ground-truth event.
        eps: Guard against division by zero for zero-length intervals.

    Returns:
        IoU in `[0, 1]`. Returns 0.0 when the events do not overlap.

    Raises:
        ValueError: If either event has a negative duration.

    Example:
        >>> compute_iou(Event(0, 1.0, 3.0), Event(0, 2.0, 4.0))
        0.3333333333333333
    """
    if pred.duration < 0 or target.duration < 0:
        raise ValueError("event durations must be non-negative")
    inter = intersection(pred, target)
    denom = pred.duration + target.duration - inter
    return inter / max(denom, eps)
```
