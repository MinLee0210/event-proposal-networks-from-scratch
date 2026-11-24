"""Probability-ensemble several SED checkpoints.

Averaging frame probabilities from different checkpoints/seeds is a standard,
low-risk PSDS1 improvement in DCASE systems. Models must share the processed
feature shape; architectures may differ as long as they all produce presence
probabilities. EPN durations are averaged only when every model provides them.

Usage:
    uv run python scripts/training/ensemble.py \
        --ckpts checkpoints/crnn_large_urbansed/best.pt \
                checkpoints/crnn_urbansed_iou05/best.pt \
                checkpoints/crnn_red_urbansed/best.pt \
        --out results/ensemble_large_iou05_red.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import SEDDataset, collate  # noqa: E402
from src.modules.events import frames_to_events  # noqa: E402
from src.modules.metrics import collar_f1, psds1, psds2  # noqa: E402
from src.modules.model import presence_probability  # noqa: E402
from src.modules.postprocess import epn_inference, median_filter_events  # noqa: E402
from src.pipelines.infer import load_model  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402


@torch.no_grad()
def ensemble(
    ckpts: list[Path], data_root: str, split: str, batch_size: int = 32
):
    """Collect averaged presence/duration outputs for a split."""
    device = detect_device()
    models, cfgs = [], []
    for path in ckpts:
        model, cfg = load_model(path, device)
        models.append(model)
        cfgs.append(cfg)
    dataset = SEDDataset(data_root, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    presence, don, doff, gt = [], [], [], []
    for batch in loader:
        mel = batch["mel"].float().to(device)
        pres_sum = None
        don_sum = None
        doff_sum = None
        all_epn = True
        for model in models:
            out = model(mel)
            p = presence_probability(out)
            pres_sum = p if pres_sum is None else pres_sum + p
            if "don" in out and "doff" in out:
                don_sum = out["don"] if don_sum is None else don_sum + out["don"]
                doff_sum = out["doff"] if doff_sum is None else doff_sum + out["doff"]
            else:
                all_epn = False
        presence.append((pres_sum / len(models)).cpu())
        if all_epn and don_sum is not None:
            don.append((don_sum / len(models)).cpu())
            doff.append((doff_sum / len(models)).cpu())
        if dataset.events:
            gt.extend([dataset.events[int(i)] for i in batch["index"].tolist()])
        else:
            for i in range(batch["pres"].shape[0]):
                gt.append(
                    frames_to_events(
                        batch["pres"][i].numpy().tolist(),
                        cfgs[0].frame_rate,
                        threshold=0.5,
                    )
                )
    return (
        torch.cat(presence),
        torch.cat(don) if don else None,
        torch.cat(doff) if doff else None,
        gt,
        cfgs[0],
    )


def main() -> None:
    """Run ensemble evaluation on validation and optionally test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpts", type=Path, nargs="+", required=True)
    parser.add_argument("--data_root", type=str, default="data/processed/urbansed")
    parser.add_argument("--val_split", type=str, default="validate")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--filter_len", type=int, default=7)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = {"checkpoints": [str(p) for p in args.ckpts]}
    for split in (args.val_split, args.test_split):
        if not split:
            continue
        presence, don, doff, gt, cfg = ensemble(args.ckpts, args.data_root, split)
        duration = len(gt) * (cfg.num_frames / cfg.frame_rate)
        if don is not None:
            preds = epn_inference(
                presence,
                don,
                doff,
                frame_rate=cfg.frame_rate,
                clip_duration=cfg.num_frames / cfg.frame_rate,
                k=args.k,
                m=None,
            )
        else:
            preds = median_filter_events(
                presence,
                frame_rate=cfg.frame_rate,
                threshold=args.threshold,
                filter_len=args.filter_len,
            )
        result[split] = {
            "psds1": psds1(preds, gt, dataset_duration=duration)["psds1"],
            "psds2": psds2(preds, gt, dataset_duration=duration)["psds2"],
            "collar_f1": collar_f1(preds, gt, threshold=args.threshold)["f1"],
            "events": len(preds),
        }
        print(
            f"{split}: PSDS1={result[split]['psds1'] * 100:.2f}  "
            f"PSDS2={result[split]['psds2'] * 100:.2f}  "
            f"collar={result[split]['collar_f1'] * 100:.2f}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
