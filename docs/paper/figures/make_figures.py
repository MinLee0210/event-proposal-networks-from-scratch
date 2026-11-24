"""Generate the data figures for the technical report.

Reads `results/*/test_metrics.json` and `train_log.json`; writes PDF (vector)
and PNG (preview) versions into `docs/paper/figures/`.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parent

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "-",
    }
)

COLORS = {
    "baseline": "#B0BEC5",
    "method": "#E76F51",
    "extension": "#2A9D8F",
    "wide": "#264653",
    "ablation": "#E9C46A",
    "muted": "#8C8C8C",
}

RUNS = {
    "crnn_baseline_urbansed": "BL: frame BCE + median",
    "crnn_red_urbansed": "+RED",
    "crnn_red_ool_urbansed": "+OOL",
    "crnn_red_epn_urbansed": r"+EPN, $\lambda_{iou}=2$",
    "crnn_urbansed_iou1": r"+EPN, $\lambda_{iou}=1$",
    "crnn_urbansed_iou05": r"+EPN, $\lambda_{iou}=0.5$",
    "crnn_mixup_urbansed": "Mixup (p=0.5)",
    "crnn_large_urbansed": "Wide, 40 ep",
    "crnn_large_ool10_urbansed": "Wide, OOL=10",
    "crnn_large100_urbansed": "Wide, 100 ep",
    "crnn_xl_urbansed": "XL",
    "crnn_large_mixup_mild": "Wide + mild mixup",
    "ensemble_xl_wide_ool10": "Ensemble: 3 models",
    "ensemble_final": "Ensemble: 4 models",
    "xl_classwise": "XL + class-wise MF",
}

ORDER = [
    "crnn_baseline_urbansed",
    "crnn_red_urbansed",
    "crnn_red_ool_urbansed",
    "crnn_red_epn_urbansed",
    "crnn_urbansed_iou1",
    "crnn_urbansed_iou05",
    "crnn_mixup_urbansed",
    "crnn_large_urbansed",
    "crnn_large_ool10_urbansed",
    "crnn_large100_urbansed",
    "crnn_xl_urbansed",
    "crnn_large_mixup_mild",
    "ensemble_xl_wide_ool10",
    "ensemble_final",
    "xl_classwise",
]


def metrics(run: str) -> dict:
    special = {
        "ensemble_xl_wide_ool10": ROOT / "results" / "ensemble_xl_wide_ool10.json",
        "ensemble_final": ROOT / "results" / "ensemble_final.json",
        "xl_classwise": ROOT / "results" / "crnn_xl_urbansed" / "tune_classwise.json",
    }
    if run in special:
        if not special[run].exists():
            return {}
        payload = json.loads(special[run].read_text())
        return payload.get("test", {})
    path = ROOT / "results" / run / "test_metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def color_for(run: str) -> str:
    if "baseline" in run:
        return COLORS["baseline"]
    if "ensemble" in run:
        return "#0072B2"
    if "xl" in run or "large" in run or "wide" in run:
        return COLORS["wide"]
    if "iou" in run or "epn" in run:
        return COLORS["method"]
    if "mixup" in run:
        return COLORS["extension"]
    return COLORS["muted"]


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


def main() -> None:
    available = [(r, metrics(r)) for r in ORDER if metrics(r)]
    labels = [RUNS[r] for r, _ in available]
    psds1 = [100 * m["psds1"] for _, m in available]
    collar = [100 * m["collar_f1"] for _, m in available]
    colors = [color_for(r) for r, _ in available]

    # --- Figure: main comparison (PSDS1 + collar-F1) ---
    order = np.argsort(psds1)
    labels_s = [labels[i] for i in order]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    y = np.arange(len(labels_s))
    for ax, values, title in (
        (axes[0], np.asarray(psds1)[order], "PSDS1"),
        (axes[1], np.asarray(collar)[order], "Collar-F1"),
    ):
        ax.barh(y, values, color=[colors[i] for i in order], height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(labels_s)
        ax.set_xlabel(f"{title} (%)")
        ax.set_title(title)
        ax.set_xlim(0, max(values) * 1.18)
        for yi, value in zip(y, values):
            ax.text(value + 0.4, yi, f"{value:.1f}", va="center", fontsize=7)
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, "fig_main_comparison")

    # --- Figure: lambda_iou sweep ---
    sweep = ["crnn_urbansed_iou05", "crnn_urbansed_iou1", "crnn_red_epn_urbansed"]
    sweep_labels = [r"$\lambda_{iou}=0.5$", r"$\lambda_{iou}=1$", r"$\lambda_{iou}=2$"]
    vals = [100 * metrics(r)["psds1"] for r in sweep]
    collar_vals = [100 * metrics(r)["collar_f1"] for r in sweep]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    x = np.arange(len(sweep))
    width = 0.36
    ax.bar(x - width / 2, vals, width, label="PSDS1", color=COLORS["method"])
    ax.bar(x + width / 2, collar_vals, width, label="Collar-F1", color=COLORS["wide"])
    ax.set_xticks(x)
    ax.set_xticklabels(sweep_labels)
    ax.set_ylabel("Score (%)")
    ax.set_title("IoU loss weight sensitivity")
    ax.legend()
    for xi, v in zip(x - width / 2, vals):
        ax.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=7)
    for xi, v in zip(x + width / 2, collar_vals):
        ax.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_iou_sweep")

    # --- Figure: validation training curves ---
    curves = [
        ("crnn_baseline_urbansed", "BL", COLORS["baseline"]),
        ("crnn_red_epn_urbansed", r"+EPN $\lambda=2$", COLORS["method"]),
        ("crnn_large_urbansed", "Wide, 40 ep", COLORS["wide"]),
        ("crnn_large100_urbansed", "Wide, 100 ep", COLORS["extension"]),
        ("crnn_large_ool10_urbansed", "Wide, OOL=10", COLORS["ablation"]),
        ("crnn_xl_urbansed", "XL", "#0072B2"),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for run, label, color in curves:
        path = ROOT / "results" / run / "train_log.json"
        if not path.exists():
            continue
        log = json.loads(path.read_text())
        epochs = [row["epoch"] for row in log if row.get("psds1") is not None]
        vals = [100 * row["psds1"] for row in log if row.get("psds1") is not None]
        ax.plot(epochs, vals, label=label, color=color, lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation PSDS1 (%)")
    ax.set_title("Validation PSDS1 during training")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    save(fig, "fig_training_curves")

    # --- Figure: post-processing comparison ---
    pp_labels = ["Default\nEPN", "Default\nmedian", "Wide\nEPN", "Wide\nmedian"]
    pp_psds1 = [
        100 * metrics("crnn_red_epn_urbansed")["psds1"],
        25.62,
        100 * metrics("crnn_large_urbansed")["psds1"],
        43.00,
    ]
    pp_collar = [
        100 * metrics("crnn_red_epn_urbansed")["collar_f1"],
        11.56,
        100 * metrics("crnn_large_urbansed")["collar_f1"],
        30.24,
    ]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    x = np.arange(len(pp_labels))
    width = 0.36
    ax.bar(x - width / 2, pp_psds1, width, label="PSDS1", color=COLORS["method"])
    ax.bar(x + width / 2, pp_collar, width, label="Collar-F1", color=COLORS["wide"])
    ax.set_xticks(x)
    ax.set_xticklabels(pp_labels)
    ax.set_ylabel("Score (%)")
    ax.set_title("EPN inference vs median filtering")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_postprocessing")


if __name__ == "__main__":
    main()
