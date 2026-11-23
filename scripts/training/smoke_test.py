"""End-to-end smoke test on a tiny synthetic SED dataset.

Generates 48 synthetic clips, trains for one epoch with a narrow CRNN, runs the
best checkpoint on the test split, and asserts that PSDS1 / collar-F1 / frame-F1
are finite. No external dataset required.

Usage:
    uv run python scripts/training/smoke_test.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.config import TrainingConfig  # noqa: E402
from src.pipelines.train import train  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402


def main() -> None:
    """Run the synthetic end-to-end check."""
    tmp = Path(tempfile.mkdtemp(prefix="epn_smoke_"))
    try:
        data_root = tmp / "synthetic"
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "data" / "make_synthetic_sed.py"),
                "--out",
                str(data_root),
                "--n_train",
                "32",
                "--n_val",
                "8",
                "--n_test",
                "8",
                "--num_classes",
                "4",
                "--num_frames",
                "100",
            ],
            check=True,
        )

        cfg = TrainingConfig(
            data_root=str(data_root),
            train_split="train",
            val_split="validate",
            test_split="test",
            ckpt_dir=str(tmp / "checkpoints"),
            result_dir=str(tmp / "results"),
            run_name="smoke",
            epochs=1,
            batch_size=8,
            lr=3e-3,
            weight_decay=1e-4,
            warmup_steps=2,
            log_every=2,
            eval_every=1,
            ckpt_every=1,
            arch={
                "num_frames": 100,
                "hop_length": 400,
                "cnn_channels": [8, 16, 32],
                "rnn_hidden": 16,
                "rnn_layers": 1,
                "epn_hidden": 4,
                "epn_layers": 1,
                "dropout": 0.1,
            },
            loss={"lambda_ool": 10.0, "lambda_iou": 1.0},
            eval={"method": "epn", "threshold": 0.5, "k": 5, "m": 4},
            lr_scheduler={"type": "warmup_cosine", "warmup_steps": 2},
            early_stopping={"enabled": False},
            best_metric="psds1",
            best_mode="max",
            wandb={"enabled": False},
        )
        train(cfg)

        metrics_path = tmp / "results" / "smoke" / "test_metrics.json"
        metrics = json.loads(metrics_path.read_text())
        print(json.dumps(metrics, indent=2))
        for key in ("psds1", "collar_f1", "frame_f1", "frame_accuracy"):
            value = float(metrics[key])
            if value != value:  # NaN check
                raise AssertionError(f"{key} is NaN")
        if not (tmp / "checkpoints" / "smoke" / "best.pt").exists():
            raise AssertionError("best.pt was not produced")
        print(f"device: {detect_device()}")
        print("\nsmoke test: PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
