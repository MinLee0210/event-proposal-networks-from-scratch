"""Single-clip inference for a trained boundary-aware SED model.

Loads a checkpoint, rebuilds the model from its saved architecture config, and
converts frame probabilities into events either with EPN Algorithm 1 or median
filtering. Input can be a raw `.wav` or a row of a processed split.

Usage:
    uv run python -m src.pipelines.infer \
        --ckpt checkpoints/<run>/best.pt --wav data/URBAN-SED_v2.0.0/audio/test/x.wav
    uv run python -m src.pipelines.infer \
        --ckpt checkpoints/<run>/best.pt --split test --index 0
"""

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, Optional

import torch

from src.config import SEDConfig
from src.data import (
    SEDDataset,
    load_audio,
    load_meta,
    make_mel_transform,
    waveform_to_logmel,
)
from src.modules.events import Event
from src.modules.postprocess import events_from_model_output
from src.modules.model import build_model
from src.utils.model_utils import detect_device


def config_from_checkpoint(ckpt: dict) -> SEDConfig:
    """Rebuild `SEDConfig` from checkpoint metadata.

    Supports both the periodic-checkpoint schema (`extra.model_cfg`) and the
    best-checkpoint schema (`extra.cfg.model`).

    Args:
        ckpt: Loaded checkpoint dict.

    Returns:
        The architecture config used to train the checkpoint.

    Raises:
        ValueError: If the checkpoint has no recognisable model config.
    """
    extra = ckpt.get("extra", {})
    saved = extra.get("model_cfg")
    if saved is None:
        cfg = extra.get("cfg", {})
        saved = cfg.get("model", cfg) if isinstance(cfg, dict) else {}
    known = {f.name for f in fields(SEDConfig)}
    filtered = {k: v for k, v in saved.items() if k in known}
    if not filtered.get("num_classes"):
        raise ValueError("checkpoint does not contain a usable model config")
    return SEDConfig(**filtered)


def load_model(
    ckpt_path: Path, device: torch.device
) -> tuple[torch.nn.Module, SEDConfig]:
    """Load a checkpoint and its architecture.

    Args:
        ckpt_path: Path to a `.pt` checkpoint.
        device: Torch device.

    Returns:
        `(model, cfg)` with the model in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location=str(device))
    cfg = config_from_checkpoint(raw)
    model = build_model(cfg).to(device)
    model.load_state_dict(raw["model"])
    model.eval()
    return model, cfg


def _load_clip(
    wav: Optional[Path],
    split: Optional[str],
    index: int,
    data_root: str,
    cfg: SEDConfig,
) -> tuple[torch.Tensor, Optional[list[Event]]]:
    """Load one clip as a normalised `(1, 1, T, F)` tensor.

    Args:
        wav: Optional raw audio path.
        split: Processed split name used when `wav` is omitted.
        index: Row index in `split`.
        data_root: Processed dataset directory.
        cfg: Architecture config for the frontend.

    Returns:
        `(mel_batch, ground_truth_events_or_None)`.

    Raises:
        ValueError: If neither `wav` nor `split` is provided.
    """
    if wav is not None:
        meta = {}
        try:
            meta = load_meta(data_root)
        except FileNotFoundError:
            pass
        waveform = load_audio(wav, cfg.sample_rate)
        mel = waveform_to_logmel(waveform, make_mel_transform(cfg), cfg.num_frames)
        mean = float(meta.get("mel_mean", 0.0))
        std = float(meta.get("mel_std", 1.0)) or 1.0
        x = torch.from_numpy((mel - mean) / std).unsqueeze(0).unsqueeze(0)
        return x.float(), None

    if split is None:
        raise ValueError("provide either --wav or --split")
    dataset = SEDDataset(data_root, split)
    row = dataset[index]
    gt = dataset.ground_truth_events(index, threshold=0.5)
    return row["mel"].unsqueeze(0), gt


@torch.no_grad()
def infer(
    ckpt_path: Path,
    wav: Optional[Path] = None,
    split: Optional[str] = None,
    index: int = 0,
    data_root: str = "data/processed/urbansed",
    method: str = "epn",
    threshold: float = 0.5,
    filter_len: int = 7,
    k: int = 15,
    m: Optional[int] = None,
    out_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Run inference on one clip and print/save the predicted events.

    Args:
        ckpt_path: Checkpoint path.
        wav: Raw audio path; mutually exclusive with `split`.
        split: Processed split name.
        index: Row index in `split`.
        data_root: Processed dataset directory.
        method: `epn` or `median`.
        threshold: Median-filter threshold (also used for scoring output).
        filter_len: Median-filter length in frames.
        k: Maximum events per class for EPN inference.
        m: Number of most-active classes for EPN inference.
        out_path: Optional JSON output path.

    Returns:
        A dict with `events`, `presence` and `ground_truth`.
    """
    device = detect_device()
    model, cfg = load_model(ckpt_path, device)
    x, gt = _load_clip(wav, split, index, data_root, cfg)
    x = x.to(device)
    outputs = model(x)
    if method == "epn" and "don" not in outputs:
        method = "median"
    events = events_from_model_output(
        outputs,
        cfg,
        method=method,
        threshold=threshold,
        filter_len=filter_len,
        k=k,
        m=m,
    )[0]

    print(f"checkpoint: {ckpt_path}")
    print(f"events: {len(events)}")
    for ev in sorted(events, key=lambda e: (e.onset, e.class_id)):
        print(
            f"  class={ev.class_id:>2d}  onset={ev.onset:6.2f}s  "
            f"offset={ev.offset:6.2f}s  score={ev.score:.3f}"
        )
    if gt is not None:
        print(f"ground truth: {len(gt)} events")
        for ev in sorted(gt, key=lambda e: (e.onset, e.class_id)):
            print(f"  class={ev.class_id:>2d}  [{ev.onset:6.2f}, {ev.offset:6.2f}]")

    result = {
        "events": [asdict(ev) for ev in events],
        "ground_truth": [asdict(ev) for ev in gt] if gt is not None else None,
        "source": str(wav) if wav is not None else f"{split}[{index}]",
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"saved: {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--wav", type=Path, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--data_root", type=str, default="data/processed/urbansed")
    parser.add_argument("--method", choices=["epn", "median"], default="epn")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--filter_len", type=int, default=7)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    infer(
        args.ckpt,
        wav=args.wav,
        split=args.split,
        index=args.index,
        data_root=args.data_root,
        method=args.method,
        threshold=args.threshold,
        filter_len=args.filter_len,
        k=args.k,
        m=args.m,
        out_path=args.out,
    )
