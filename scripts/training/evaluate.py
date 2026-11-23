"""Evaluate a trained SED checkpoint on a processed split.

Usage:
    uv run python scripts/training/evaluate.py \
        --ckpt checkpoints/<run>/best.pt --split test --method epn
"""

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import SEDDataset, collate, load_meta  # noqa: E402
from src.modules.loss import SEDLoss  # noqa: E402
from src.pipelines.eval import evaluate, format_report  # noqa: E402
from src.pipelines.infer import load_model  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402


def main() -> None:
    """Entry point for checkpoint evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data_root", type=str, default="data/processed/urbansed")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--method", choices=["epn", "median"], default="epn")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--filter_len", type=int, default=7)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    device = detect_device()
    model, cfg = load_model(args.ckpt, device)
    meta = load_meta(args.data_root)
    dataset = SEDDataset(args.data_root, args.split, meta=meta)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    settings = {
        "method": args.method,
        "threshold": args.threshold,
        "filter_len": args.filter_len,
        "k": args.k,
        "m": args.m,
    }
    criterion = SEDLoss(cfg, use_ool=cfg.use_red, use_epn=cfg.use_epn)
    result = evaluate(
        model,
        loader,
        device,
        cfg,
        settings=settings,
        criterion=criterion,
        save_json_path=args.json,
        gt_events_by_index=dataset.events or None,
    )
    print(format_report(result))
    if args.json:
        print(f"saved: {args.json}")


if __name__ == "__main__":
    main()
