"""Training-loop hyperparameters for boundary-aware SED.

Model architecture lives in `src.config.SEDConfig`; this dataclass holds data
paths, optimisation, loss weights, post-processing settings and callback wiring.
`arch` and `loss` are dicts so a YAML can override architecture / objective
fields without duplicating the whole config.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from src.config import SEDConfig
from src.utils.io_utils import read_yaml


@dataclass
class TrainingConfig:
    """Run configuration for `src.pipelines.train`.

    Attributes:
        data_root: Processed dataset directory (contains `<split>_mel.npy`).
        train_split/val_split/test_split: Split names inside `data_root`.
        batch_size: Clips per optimisation step.
        epochs: Maximum number of epochs.
        lr: Peak AdamW learning rate.
        weight_decay: AdamW decoupled weight decay.
        warmup_steps: Linear warmup steps for the cosine schedule.
        grad_clip: Gradient-norm clip, 0 disables.
        arch: Overrides forwarded to `SEDConfig` (num_classes is injected).
        loss: `lambda_ool`, `lambda_iou`, `focal_alpha`.
        eval: Post-processing and monitoring settings.
    """

    # --- data ---
    data_root: str = "data/processed/urbansed"
    train_split: str = "train"
    val_split: str = "validate"
    test_split: str = "test"
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 42

    # --- output ---
    ckpt_dir: str = "checkpoints"
    result_dir: str = "results"
    run_name: Optional[str] = None

    # --- optimization ---
    epochs: int = 40
    batch_size: int = 128
    accum_steps: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-3
    warmup_steps: int = 1000
    grad_clip: float = 5.0
    amp: bool = False
    log_every: int = 20
    eval_every: int = 2
    ckpt_every: int = 10

    # --- architecture overrides for SEDConfig ---
    arch: Optional[dict] = None

    # --- augmentation (train only) ---
    augment: dict = field(
        default_factory=lambda: {
            "mixup_prob": 0.0,
            "time_masks": 0,
            "time_width": 50,
            "freq_masks": 0,
            "freq_width": 8,
        }
    )

    # --- objective (Eq. 5) ---
    loss: dict = field(
        default_factory=lambda: {
            "lambda_ool": 100.0,
            "lambda_iou": 2.0,
            "focal_alpha": 2.0,
        }
    )

    # --- evaluation / post-processing ---
    eval: dict = field(
        default_factory=lambda: {
            "method": "epn",  # "epn" | "median"
            "threshold": 0.5,
            "filter_len": 7,
            "k": 15,
            "m": None,
            "min_duration": 0.0,
        }
    )

    # --- callbacks ---
    lr_scheduler: Optional[dict] = None
    early_stopping: Optional[dict] = None
    save_best: bool = True
    best_metric: str = "psds1"
    best_mode: str = "max"
    wandb: Optional[dict] = None
    resume_from: Optional[str] = None

    def model_config(self, num_classes: int) -> SEDConfig:
        """Build the architecture config with the dataset's class count.

        Args:
            num_classes: Number of event classes in the dataset.

        Returns:
            `SEDConfig` with YAML `arch` overrides applied.

        Raises:
            ValueError: If `arch` tries to set `num_classes` itself.
        """
        arch = dict(self.arch or {})
        if "num_classes" in arch:
            raise ValueError(
                "set num_classes through the dataset, not the `arch` override"
            )
        return SEDConfig(num_classes=num_classes, **arch)

    def eval_settings(self) -> dict:
        """Return post-processing settings with defaults filled in.

        Returns:
            A dict with `method`, `threshold`, `filter_len`, `k`, `m` and
            `min_duration`.
        """
        defaults = {
            "method": "epn",
            "threshold": 0.5,
            "filter_len": 7,
            "k": 15,
            "m": None,
            "min_duration": 0.0,
        }
        defaults.update(self.eval or {})
        return defaults


def load_training_config(
    config_path: Optional[Union[str, Path]] = None, **cli_overrides: Any
) -> TrainingConfig:
    """Build a `TrainingConfig` from defaults, YAML, then CLI overrides.

    CLI overrides have the highest precedence, but only keys whose value is not
    `None` are applied (argparse defaults are `None`).

    Args:
        config_path: Optional YAML file; unknown keys raise.
        **cli_overrides: Field overrides from the command line.

    Returns:
        The resolved configuration.

    Raises:
        ValueError: If the YAML contains an unknown field.
    """
    cfg = TrainingConfig()

    if config_path is not None:
        yaml_data = read_yaml(config_path)
        known_fields = set(asdict(cfg).keys())
        for key, value in yaml_data.items():
            if key not in known_fields:
                raise ValueError(
                    f"unknown training config key {key!r} in {config_path}"
                )
            setattr(cfg, key, value)

    for key, value in cli_overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    return cfg
