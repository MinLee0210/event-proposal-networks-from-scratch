"""Precompute URBAN-SED log-mel features and frame-level targets.

Run after extracting `URBAN-SED_v2.0.0.tar.gz` under `data/`:

    uv run python scripts/data/prepare_urbansed.py
    uv run python scripts/data/prepare_urbansed.py --limit 200 --out data/processed/urbansed_smoke

The output schema is consumed by `src.data.SEDDataset`:

    <split>_mel.npy       (N, 500, 64) float16 log-mel (dB)
    <split>_targets.npz   pres/on/off/don/doff (N, 10, 500) float16
    meta.json             classes, frontend, normalisation stats, source files
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import SEDConfig  # noqa: E402
from src.data import (  # noqa: E402
    URBAN_SED_CLASSES,
    build_targets,
    discover_split,
    load_audio,
    make_mel_transform,
    waveform_to_logmel,
    write_events,
    write_meta,
)

_TRANSFORM = None
_CFG: SEDConfig | None = None


def _init_worker(cfg: SEDConfig) -> None:
    """Create one mel transform per worker process.

    Args:
        cfg: Frontend config copied into the worker.
    """
    global _TRANSFORM, _CFG
    _CFG = cfg
    _TRANSFORM = make_mel_transform(cfg)


def _process_one(audio_path: str) -> np.ndarray:
    """Load, resample and log-mel one wav file.

    Args:
        audio_path: Path to a URBAN-SED wav.

    Returns:
        `(num_frames, n_mels)` float16 log-mel.
    """
    assert _CFG is not None and _TRANSFORM is not None
    waveform = load_audio(audio_path, _CFG.sample_rate)
    mel = waveform_to_logmel(waveform, _TRANSFORM, _CFG.num_frames)
    return mel.astype(np.float16)


def _split_features(
    rows: list[dict], cfg: SEDConfig, workers: int, split: str
) -> tuple[np.ndarray, dict]:
    """Compute features and targets for one split.

    Args:
        rows: Output of `discover_split`.
        cfg: Frontend/model config.
        workers: Number of worker processes.
        split: Split name used for progress messages.

    Returns:
        `(mel (N, T, F) float16, targets dict of (N, C, T) float32)`.
    """
    n = len(rows)
    mel = np.empty((n, cfg.num_frames, cfg.n_mels), dtype=np.float16)
    audio_paths = [row["audio"] for row in rows]

    if workers <= 1:
        _init_worker(cfg)
        for i, path in enumerate(audio_paths):
            mel[i] = _process_one(path)
            if (i + 1) % 500 == 0 or i + 1 == n:
                print(f"  {split}: mel {i + 1}/{n}")
    else:
        with mp.Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for i, feat in enumerate(
                pool.imap(_process_one, audio_paths, chunksize=8), start=1
            ):
                mel[i - 1] = feat
                if i % 500 == 0 or i == n:
                    print(f"  {split}: mel {i}/{n}")

    keys = ("pres", "on", "off", "don", "doff")
    targets = {
        key: np.empty((n, cfg.num_classes, cfg.num_frames), dtype=np.float16)
        for key in keys
    }
    for i, row in enumerate(rows):
        built = build_targets(row["events"], cfg)
        for key in keys:
            targets[key][i] = built[key].astype(np.float16)
    return mel, targets


def main() -> None:
    """Entry point for the feature-preparation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=REPO_ROOT / "data" / "URBAN-SED_v2.0.0"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data" / "processed" / "urbansed"
    )
    parser.add_argument(
        "--splits", type=str, default="train,validate,test", help="comma-separated"
    )
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    parser.add_argument(
        "--limit", type=int, default=0, help="cap clips per split (0 = all)"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = SEDConfig(num_classes=len(URBAN_SED_CLASSES))
    if cfg.num_frames != 500:
        raise ValueError("this frontend assumes 10 s clips at the configured hop")

    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"output directory is not empty: {args.out}\npass --overwrite to replace it"
        )
    args.out.mkdir(parents=True, exist_ok=True)

    class_to_idx = {name: i for i, name in enumerate(URBAN_SED_CLASSES)}
    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    split_sizes: dict[str, int] = {}
    source_files: dict[str, list[str]] = {}
    train_mel_path: Path | None = None

    for split in split_names:
        print(f"\n[{split}] discovering annotations")
        rows = discover_split(args.source, split, class_to_idx)
        if args.limit:
            rows = rows[: args.limit]
        print(f"[{split}] {len(rows)} clips")

        mel, targets = _split_features(rows, cfg, args.workers, split)
        mel_path = args.out / f"{split}_mel.npy"
        target_path = args.out / f"{split}_targets.npz"
        np.save(mel_path, mel)
        np.savez(target_path, **targets)
        write_events(args.out, split, rows)
        split_sizes[split] = len(rows)
        source_files[split] = [Path(row["audio"]).name for row in rows]
        print(f"  saved {mel_path} {mel.shape} {mel.dtype}")
        print(f"  saved {target_path} {targets['pres'].shape}")
        if split == "train":
            train_mel_path = mel_path

    stats_path = train_mel_path or (args.out / f"{split_names[0]}_mel.npy")
    print(f"\ncomputing normalisation stats from {stats_path.name}")
    mel_mm = np.load(stats_path, mmap_mode="r")
    stride = max(len(mel_mm) // 500, 1)
    sample = np.asarray(mel_mm[::stride], dtype=np.float32)
    mel_mean = float(sample.mean())
    mel_std = float(sample.std()) or 1.0
    print(f"  mel_mean={mel_mean:.4f}  mel_std={mel_std:.4f}")

    write_meta(
        args.out,
        cfg,
        URBAN_SED_CLASSES,
        split_sizes,
        mel_mean,
        mel_std,
        str(args.source),
        source_files,
    )
    print(f"\nwrote {args.out / 'meta.json'}  splits={split_sizes}")


if __name__ == "__main__":
    main()
