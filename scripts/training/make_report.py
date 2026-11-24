"""Build a results table and Markdown report from `test_metrics.json` files.

Usage:
    uv run python scripts/training/make_report.py \
        --results results --out docs/reports/2026-11-24/urbansed-crnn

The script only reads existing metrics; it never invents numbers.
"""

import argparse
import csv
import json
from pathlib import Path

LABELS = {
    "crnn_baseline_urbansed": "BL (frame BCE + median)",
    "crnn_red_urbansed": "+RED",
    "crnn_red_ool_urbansed": "+OOL",
    "crnn_red_epn_urbansed": "+EPN (lambda_iou=2)",
    "crnn_urbansed_iou05": "+EPN (lambda_iou=0.5)",
    "crnn_urbansed_iou1": "+EPN (lambda_iou=1.0)",
    "crnn_large_urbansed": "Extension: wide model",
    "crnn_mixup_urbansed": "Extension: synthetic mixup",
    "crnn_sample_wandb": "Sample run (not a result)",
    "debug_single": "Debug run (not a result)",
    "notebook_smoke": "Notebook smoke run (not a result)",
}

FIELDS = ("psds1", "psds2", "collar_f1", "frame_f1", "loss")


def main() -> None:
    """Read metrics and write a CSV plus a Markdown summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="https://forge.coreweave.com/wandb/octoopt/epn-sed",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(args.results.glob("*/test_metrics.json")):
        run = path.parent.name
        metrics = json.loads(path.read_text())
        rows.append(
            {
                "run": run,
                "label": LABELS.get(run, run),
                **{field: metrics.get(field) for field in FIELDS},
            }
        )
    if not rows:
        raise SystemExit(f"no test_metrics.json under {args.results}")

    order = {name: i for i, name in enumerate(LABELS)}
    rows.sort(key=lambda row: order.get(row["run"], 999))

    with open(args.out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("run", "label", *FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# URBAN-SED boundary-aware SED results",
        "",
        "Generated from `results/*/test_metrics.json` by",
        "`scripts/training/make_report.py`.",
        "",
        f"W&B project: {args.wandb_project}",
        "",
        "| Run | PSDS1 | PSDS2 | collar-F1 | frame-F1 | loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {psds1:.2f} | {psds2:.2f} | {collar_f1:.2f} "
            "| {frame_f1:.2f} | {loss:.4f} |".format(
                label=row["label"],
                psds1=100 * row["psds1"],
                psds2=100 * row["psds2"],
                collar_f1=100 * row["collar_f1"],
                frame_f1=100 * row["frame_f1"],
                loss=row["loss"],
            )
        )
    lines += [
        "",
        "PSDS1/PSDS2 are threshold-independent; collar-F1 uses the configured",
        "score threshold (0.5 for the baseline, 0.3 for EPN runs).",
        "",
    ]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out / 'summary.csv'}")
    print(f"wrote {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
