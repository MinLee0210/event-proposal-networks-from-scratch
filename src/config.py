"""Architecture config for the boundary-aware SED model.

This is the ARCHITECTURE-ONLY config (frontend, CRNN widths, RED/EPN switches).
Training-loop hyperparameters (epochs, lr, data paths, loss weights, callback
wiring) live in `src.pipelines.config.TrainingConfig`, so a checkpoint can
rebuild the network without the training YAML.

The model follows "Sound Event Detection with Boundary-Aware Optimization and
Inference" (arXiv:2601.04178):

    log-mel -> CRNN -> 2 logits/class (start, end)
                       -> RED layer -> presence / onset / offset probabilities
                       -> EPN       -> time-since-onset & time-to-offset

Set `use_red=False` for the plain frame-wise BCE baseline, and `use_epn=False`
for the +RED / +OOL ablations.
"""

from dataclasses import dataclass
from typing import Tuple

EPN_VARIANTS = ("per_class", "single")


@dataclass
class SEDConfig:
    """CRNN acoustic model + RED + EPN.

    `num_classes` is filled in at runtime from the dataset. The frontend fields
    must match the values used by `scripts/data/prepare_urbansed.py`; they are
    stored in the processed dataset's `meta.json`.
    """

    # --- task ---
    num_classes: int = 0

    # --- log-mel frontend (must match the precomputed features) ---
    sample_rate: int = 16000
    n_fft: int = 1024
    win_length: int = 400  # 25 ms at 16 kHz
    hop_length: int = 320  # 20 ms -> 50 frames/s
    n_mels: int = 64
    f_min: float = 0.0
    f_max: float = 8000.0
    num_frames: int = 500  # 10 s clips

    # --- CRNN backbone ---
    cnn_channels: Tuple[int, ...] = (16, 32, 64, 128)
    rnn_hidden: int = 128
    rnn_layers: int = 2
    dropout: float = 0.3

    # --- RED / EPN ---
    use_red: bool = True
    use_epn: bool = True
    epn_variant: str = "per_class"
    epn_hidden: int = 32
    epn_layers: int = 2

    def __post_init__(self) -> None:
        if self.num_classes < 0:
            raise ValueError(f"num_classes must be >= 0, got {self.num_classes}")
        if self.sample_rate <= 0 or self.hop_length <= 0:
            raise ValueError("sample_rate and hop_length must be > 0")
        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be > 0, got {self.num_frames}")
        if self.n_mels <= 0:
            raise ValueError(f"n_mels must be > 0, got {self.n_mels}")
        if not self.cnn_channels or any(c <= 0 for c in self.cnn_channels):
            raise ValueError("cnn_channels must be a non-empty tuple of positive ints")
        if self.rnn_hidden <= 0 or self.rnn_layers <= 0:
            raise ValueError("rnn_hidden and rnn_layers must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.epn_variant not in EPN_VARIANTS:
            raise ValueError(
                f"unknown epn_variant {self.epn_variant!r}; choose from {EPN_VARIANTS}"
            )
        if self.epn_hidden <= 0 or self.epn_layers <= 0:
            raise ValueError("epn_hidden and epn_layers must be > 0")
        if self.use_epn and not self.use_red:
            raise ValueError("EPN consumes RED outputs; use_epn requires use_red=True")

    @property
    def frame_rate(self) -> float:
        """Frames per second of the model output."""
        return self.sample_rate / self.hop_length
