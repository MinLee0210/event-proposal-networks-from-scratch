"""Generate a synthetic URBAN-SED-shaped dataset for smoke runs and tests.

The features are random log-mel-like arrays with injected energy bands on the
event intervals, so a model can learn above chance. The output schema is
identical to `prepare_urbansed.py`, letting the full training/eval pipeline run
without the 6.6 GB archive.

Usage:
    uv run python scripts/data/make_synthetic_sed.py --out data/processed/synthetic
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import SEDConfig  # noqa: E402
from src.data import build_targets, write_meta  # noqa: E402
from src.modules.events import Event  # noqa: E402

CLASS_NAMES = [f"event_{i}" for i in range(10)]


def _make_split(
    n: int,
    cfg: SEDConfig,
    rng: np.random.Generator,
    signal: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """Generate one synthetic split.

    Args:
        n: Number of clips.
        cfg: Shape/frontend config.
        rng: NumPy random generator.
        signal: Energy added on active time-frequency cells.

    Returns:
        `(mel (N, T, F) float16, targets dict (N, C, T) float16)`.
    """
    t, f, c = cfg.num_frames, cfg.n_mels, cfg.num_classes
    mel = rng.normal(0.0, 1.0, size=(n, t, f)).astype(np.float32)
    keys = ("pres", "on", "off", "don", "doff")
    targets = {key: np.zeros((n, c, t), dtype=np.float16) for key in keys}

    for i in range(n):
        events: list[Event] = []
        n_events = int(rng.integers(1, 5))
        for _ in range(n_events):
            class_id = int(rng.integers(0, c))
            duration = float(rng.uniform(0.3, 2.5))
            onset = float(rng.uniform(0.1, 9.5 - duration))
            events.append(Event(class_id, onset, onset + duration))
            start = int(round(onset * cfg.frame_rate))
            end = min(int(round((onset + duration) * cfg.frame_rate)), t)
            band = slice(
                (class_id * 4) % max(f - 8, 1), (class_id * 4) % max(f - 8, 1) + 8
            )
            mel[i, start:end, band] += signal
        built = build_targets(events, cfg)
        for key in keys:
            targets[key][i] = built[key].astype(np.float16)
    return mel.astype(np.float16), targets


def main() -> None:
    """Entry point for the synthetic dataset CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data" / "processed" / "synthetic"
    )
    parser.add_argument("--n_train", type=int, default=128)
    parser.add_argument("--n_val", type=int, default=32)
    parser.add_argument("--n_test", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = SEDConfig(
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        n_mels=args.n_mels,
        hop_length=400,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    split_sizes = {"train": args.n_train, "validate": args.n_val, "test": args.n_test}
    for split, n in split_sizes.items():
        print(f"[{split}] generating {n} clips")
        mel, targets = _make_split(n, cfg, rng)
        np.save(args.out / f"{split}_mel.npy", mel)
        np.savez(args.out / f"{split}_targets.npz", **targets)

    train = np.load(args.out / "train_mel.npy", mmap_mode="r")
    sample = np.asarray(train[: min(len(train), 64)], dtype=np.float32)
    write_meta(
        args.out,
        cfg,
        CLASS_NAMES[: args.num_classes],
        split_sizes,
        float(sample.mean()),
        float(sample.std()) or 1.0,
        "synthetic",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
