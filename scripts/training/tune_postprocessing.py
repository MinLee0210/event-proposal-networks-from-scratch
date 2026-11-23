"""Grid-search post-processing on a validation split.

For the frame-wise baseline, sweep median-filter length and threshold. For EPN,
sweep the inference `k` / `m` and score threshold. Model outputs are collected
once, so every grid point is cheap.

Usage:
    uv run python scripts/training/tune_postprocessing.py \
        --ckpt checkpoints/<run>/best.pt --method median --split validate
"""

import argparse
import itertools
import json
import sys
from typing import Optional
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import SEDConfig  # noqa: E402
from src.data import SEDDataset, collate, load_meta  # noqa: E402
from src.modules.events import frames_to_events  # noqa: E402
from src.modules.metrics import collar_f1, psds1  # noqa: E402
from src.modules.postprocess import epn_inference, median_filter_events  # noqa: E402
from src.modules.model import presence_probability  # noqa: E402
from src.pipelines.infer import load_model  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402


@torch.no_grad()
def collect_outputs(
    ckpt: Path, data_root: str, split: str, batch_size: int = 64
) -> tuple[
    torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], list, SEDConfig
]:
    """Collect frame probabilities and ground-truth events for one split.

    Args:
        ckpt: Checkpoint path.
        data_root: Processed dataset directory.
        split: Split name.
        batch_size: Inference batch size.

    Returns:
        `(presence, don, doff, gt_events, cfg)` with CPU tensors/arrays.
    """
    device = detect_device()
    model, cfg = load_model(ckpt, device)
    meta = load_meta(data_root)
    dataset = SEDDataset(data_root, split, meta=meta)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    presence, don, doff, gt_events = [], [], [], []
    for batch in loader:
        outputs = model(batch["mel"].float().to(device))
        presence.append(presence_probability(outputs).cpu())
        if "don" in outputs:
            don.append(outputs["don"].cpu())
            doff.append(outputs["doff"].cpu())
        for i in range(batch["pres"].shape[0]):
            gt_events.append(
                frames_to_events(
                    batch["pres"][i].numpy().tolist(), cfg.frame_rate, threshold=0.5
                )
            )
    return (
        torch.cat(presence),
        torch.cat(don) if don else None,
        torch.cat(doff) if doff else None,
        gt_events,
        cfg,
    )


def main() -> None:
    """Run the post-processing grid search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data_root", type=str, default="data/processed/urbansed")
    parser.add_argument("--split", type=str, default="validate")
    parser.add_argument("--method", choices=["median", "epn"], default="median")
    parser.add_argument(
        "--thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
    )
    parser.add_argument("--filter_lens", type=str, default="1,3,5,7,11,15,21,31,41,51")
    parser.add_argument("--ks", type=str, default="5,10,15,20")
    parser.add_argument("--ms", type=str, default="5,10")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    presence, don, doff, gt_events, cfg = collect_outputs(
        args.ckpt, args.data_root, args.split
    )
    duration = len(gt_events) * (cfg.num_frames / cfg.frame_rate)
    thresholds = [float(x) for x in args.thresholds.split(",")]

    rows = []
    if args.method == "median":
        grid = itertools.product(
            thresholds, [int(x) for x in args.filter_lens.split(",")]
        )
        for threshold, filter_len in grid:
            preds = median_filter_events(
                presence, cfg.frame_rate, threshold=threshold, filter_len=filter_len
            )
            psds = psds1(preds, gt_events, dataset_duration=duration)["psds1"]
            collar = collar_f1(preds, gt_events, threshold=threshold)["f1"]
            rows.append(
                {
                    "threshold": threshold,
                    "filter_len": filter_len,
                    "psds1": psds,
                    "collar_f1": collar,
                }
            )
    else:
        if don is None or doff is None:
            raise SystemExit("checkpoint has no EPN outputs; use --method median")
        grid = itertools.product(
            thresholds,
            [int(x) for x in args.ks.split(",")],
            [None if x == "all" else int(x) for x in args.ms.split(",")],
        )
        for threshold, k, m in grid:
            preds = epn_inference(
                presence,
                don,
                doff,
                frame_rate=cfg.frame_rate,
                clip_duration=cfg.num_frames / cfg.frame_rate,
                k=k,
                m=m,
            )
            psds = psds1(preds, gt_events, dataset_duration=duration)["psds1"]
            collar = collar_f1(preds, gt_events, threshold=threshold)["f1"]
            rows.append(
                {
                    "threshold": threshold,
                    "k": k,
                    "m": m,
                    "psds1": psds,
                    "collar_f1": collar,
                }
            )

    rows.sort(key=lambda r: (-r["psds1"], -r["collar_f1"]))
    print(f"{'threshold':>9} {'filter/k':>9} {'m':>4} {'PSDS1':>8} {'collarF1':>9}")
    for row in rows[:15]:
        which = row.get("filter_len", row.get("k", 0))
        print(
            f"{row['threshold']:9.2f} {which:9d} {str(row.get('m', '-')):>4} "
            f"{row['psds1'] * 100:8.2f} {row['collar_f1'] * 100:9.2f}"
        )

    out = args.out or (
        REPO_ROOT / "results" / Path(args.ckpt).parent.name / f"tune_{args.method}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {"method": args.method, "split": args.split, "rows": rows}, f, indent=2
        )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
