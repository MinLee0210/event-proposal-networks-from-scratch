"""Class-wise median-filter tuning for PSDS1.

DCASE systems tune median-filter length per class, not one global value. This
script performs coordinate descent on the validation split, then optionally
applies the learned settings to the test split.

Usage:
    uv run python scripts/training/tune_classwise.py \
        --ckpt checkpoints/crnn_large_urbansed/best.pt \
        --out results/crnn_large_urbansed/tune_classwise.json
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
from src.modules.events import Event, frames_to_events  # noqa: E402
from src.modules.metrics import collar_f1, psds1, psds2  # noqa: E402
from src.modules.model import presence_probability  # noqa: E402
from src.modules.postprocess import median_filter_events  # noqa: E402
from src.pipelines.infer import load_model  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402


@torch.no_grad()
def collect(ckpt: Path, data_root: str, split: str, batch_size: int = 64):
    """Collect presence probabilities and ground-truth events."""
    device = detect_device()
    model, cfg = load_model(ckpt, device)
    dataset = SEDDataset(data_root, split)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    presence, gt = [], []
    for batch in loader:
        out = model(batch["mel"].float().to(device))
        presence.append(presence_probability(out).cpu())
        if dataset.events:
            gt.extend([dataset.events[int(i)] for i in batch["index"].tolist()])
        else:
            for i in range(batch["pres"].shape[0]):
                gt.append(
                    frames_to_events(
                        batch["pres"][i].numpy().tolist(), cfg.frame_rate, threshold=0.5
                    )
                )
    return torch.cat(presence), gt, cfg


def class_events(
    presence: torch.Tensor, class_id: int, threshold: float, filter_len: int
):
    """Median-filter events for one class channel."""
    events = median_filter_events(
        presence[:, class_id : class_id + 1],
        frame_rate=50.0,
        threshold=threshold,
        filter_len=filter_len,
    )
    return [
        [Event(class_id, e.onset, e.offset, e.score) for e in clip] for clip in events
    ]


def combine(base, replacement, class_id: int):
    """Replace all events of one class in `base` with `replacement`."""
    return [
        [e for e in clip if e.class_id != class_id] + replacement[i]
        for i, clip in enumerate(base)
    ]


def main() -> None:
    """Run coordinate descent and optionally evaluate on test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data_root", type=str, default="data/processed/urbansed")
    parser.add_argument("--split", type=str, default="validate")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--thresholds", type=str, default="0.2,0.3,0.4,0.5")
    parser.add_argument("--filter_lens", type=str, default="1,3,7,15,31")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",")]
    filter_lens = [int(x) for x in args.filter_lens.split(",")]
    presence, gt, cfg = collect(args.ckpt, args.data_root, args.split)
    duration = len(gt) * (cfg.num_frames / cfg.frame_rate)
    num_classes = presence.shape[1]
    print(f"val clips {len(gt)}  classes {num_classes}")

    settings = {c: (thresholds[0], filter_lens[0]) for c in range(num_classes)}
    preds = median_filter_events(
        presence,
        frame_rate=cfg.frame_rate,
        threshold=settings[0][0],
        filter_len=settings[0][1],
    )
    best = psds1(preds, gt, dataset_duration=duration)["psds1"]
    print(f"start: {best * 100:.2f}  {settings}")

    for p in range(args.passes):
        for c in range(num_classes):
            best_local = best
            best_setting = settings[c]
            for th in thresholds:
                for fl in filter_lens:
                    candidate = class_events(presence, c, th, fl)
                    combined = combine(preds, candidate, c)
                    score = psds1(combined, gt, dataset_duration=duration)["psds1"]
                    if score > best_local:
                        best_local, best_setting = score, (th, fl)
            if best_setting != settings[c]:
                settings[c] = best_setting
                preds = combine(preds, class_events(presence, c, *best_setting), c)
                best = best_local
            print(f"pass {p} class {c}: {settings[c]}  psds1={best * 100:.2f}")

    result = {
        "settings": {
            str(c): {"threshold": s[0], "filter_len": s[1]} for c, s in settings.items()
        },
        "val_psds1": best,
    }

    if args.test_split:
        test_presence, test_gt, test_cfg = collect(
            args.ckpt, args.data_root, args.test_split
        )
        # Build the full test prediction list, then replace each class setting.
        base = median_filter_events(
            test_presence,
            frame_rate=test_cfg.frame_rate,
            threshold=settings[0][0],
            filter_len=settings[0][1],
        )
        for c in range(num_classes):
            base = combine(base, class_events(test_presence, c, *settings[c]), c)
        duration_test = len(test_gt) * (test_cfg.num_frames / test_cfg.frame_rate)
        result["test"] = {
            "psds1": psds1(base, test_gt, dataset_duration=duration_test)["psds1"],
            "psds2": psds2(base, test_gt, dataset_duration=duration_test)["psds2"],
            "collar_f1": collar_f1(base, test_gt, threshold=0.3)["f1"],
        }
        print(
            f"test: PSDS1={result['test']['psds1'] * 100:.2f}  "
            f"PSDS2={result['test']['psds2'] * 100:.2f}  "
            f"collar={result['test']['collar_f1'] * 100:.2f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
