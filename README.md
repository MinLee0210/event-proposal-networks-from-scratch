# Reproducing Event-Proposal Networks (EPN)

Implementation of **“Sound Event Detection with Boundary-Aware Optimization and
Inference”** ([arXiv:2601.04178](https://arxiv.org/abs/2601.04178)) on
**URBAN-SED v2.0.0**.

The paper replaces frame-wise thresholding + median filtering with three
components on top of any frame-wise acoustic model:

1. **RED** — a parameter-free recurrent layer that turns per-class
   start/end conditionals into presence, onset and offset probabilities.
2. **OOL** — a focal onset/offset loss on the impulse-like boundary labels.
3. **EPN** — per-class (or single) bidirectional GRUs that predict
   time-since-onset and time-to-offset, trained with a duration-weighted IoU
   loss, plus a non-maximum-suppression-like inference algorithm.

This repo implements the full stack (data preparation → training → event
inference → PSDS1/PSDS2 / collar-F1 evaluation), the frame-wise + median-filter
baseline, and the paper’s component ablation (`BL → +RED → +OOL → +EPN`).

Technical report: [`docs/paper/technical-report.pdf`](docs/paper/technical-report.pdf)
([LaTeX source](docs/paper/technical-report.tex)).

## Status

| Item | State |
|---|---|
| RED layer + parallel scan | implemented, unit-tested |
| OOL focal loss + duration-weighted IoU loss | implemented, unit-tested |
| EPN (per-class and single GRU) | implemented, unit-tested |
| Algorithm 1 event inference | implemented, unit-tested |
| PSDS1 + PSDS2 + 200 ms collar-F1 | implemented in-repo, unit-tested |
| Median-filter baseline | implemented |
| URBAN-SED download + preprocessing | working; see `data/processed/urbansed/` |
| End-to-end synthetic smoke run | passing (`scripts/training/smoke_test.py`) |
| Full URBAN-SED suite (40 epochs, VM GPU) | completed; see the [results report](docs/reports/2026-11-24/urbansed-crnn/report.md) |
| Baseline / +RED / +OOL / +EPN | 36.67 / 37.16 / 35.93 / 25.87 PSDS1 (test) |
| λ_iou sweep (0.5 / 1.0) | 35.42 / 33.08 PSDS1 (test) |
| Best single model (wide + mild mixup) | 45.77 PSDS1 / 76.11 PSDS2 (test) |
| **Best overall (4-model ensemble)** | **47.73 PSDS1 / 76.74 PSDS2** (test) |

No paper-scale numbers are claimed yet. The tables below are filled only from
actual runs; see `results/` and `docs/EXPERIMENTS.md`.

## Dataset

[URBAN-SED v2.0.0](https://zenodo.org/records/1324404) — 10,000 synthesized
10-second soundscapes, 10 classes, strong annotations, pre-split 6,000 / 2,000 /
2,000 (train / validate / test). It is redistributed on the Hub as
[`LakoreAI/URBAN-SED`](https://huggingface.co/datasets/LakoreAI/URBAN-SED).

```bash
# 6.6 GB download, verified md5, extracted to data/URBAN-SED_v2.0.0/
uv run python scripts/data/download_urbansed.py

# 16 kHz / 64-band log-mel + frame targets -> data/processed/urbansed/
uv run python scripts/data/prepare_urbansed.py --workers 8
```

The processed schema is `(N, 500, 64)` float16 log-mel plus
`pres/on/off/don/doff` targets of shape `(N, 10, 500)`.

## Quickstart

```bash
# `wandb` is an optional extra; the run configs log to W&B by default.
uv sync --extra wandb
cp .env_example .env   # then set WANDB_API_KEY (and HF_API_KEY if needed)

# 1. sanity check without the real dataset
uv run python scripts/data/make_synthetic_sed.py
uv run python scripts/training/smoke_test.py

# 2. full method
uv run python -m src.pipelines.train --config configs/crnn_urbansed.yaml

# 3. evaluate the best checkpoint
uv run python scripts/training/evaluate.py \
    --ckpt checkpoints/crnn_red_epn_urbansed/best.pt --split test --method epn

# 4. tune post-processing on validation (baseline or EPN)
uv run python scripts/training/tune_postprocessing.py \
    --ckpt checkpoints/crnn_baseline_urbansed/best.pt --method median
```

Training prints PSDS1, PSDS2, collar-F1 and frame-F1 every `eval_every` epochs,
saves `best.pt` by validation PSDS1, and writes `test_metrics.json` after
training. With W&B enabled it logs `train/loss`, `train/lr`,
`eval/psds1`, `eval/psds2`, `eval/collar_f1`, `eval/frame_f1`, `eval/val_loss`,
and the final `test/*` scalars, and uploads the best checkpoint as an artifact.

A 300-clip, 6-epoch sample run is logged at
[`epn-sed` run `crnn_sample_wandb`](https://forge.coreweave.com/wandb/octoopt/epn-sed/runs/s2ltfazu).

## Configs

| Config | What it trains | Event extraction |
|---|---|---|
| `configs/crnn_urbansed.yaml` | CRNN + RED + OOL + EPN (single-GRU EPN, default) | EPN Algorithm 1 |
| `configs/crnn_baseline.yaml` | CRNN + frame BCE (paper’s BL) | median filter |
| `configs/crnn_red.yaml` | + RED, BCE only | median filter |
| `configs/crnn_red_ool.yaml` | + RED + OOL | median filter |
| `configs/crnn_urbansed_per_class.yaml` | full method with per-class GRUs (paper config, slow) | EPN Algorithm 1 |
| `configs/crnn_smoke.yaml` | tiny synthetic run | EPN |
| `configs/crnn_sample.yaml` | 300-clip real URBAN-SED sample, W&B on | EPN |
| `configs/crnn_urbansed_large.yaml` | wider CRNN/EPN (~4M params) | EPN |
| `configs/crnn_urbansed_mixup.yaml` | max-mix synthetic event mixing + SpecAugment | EPN |
| `configs/crnn_urbansed_iou05.yaml` | full method, `λ_iou = 0.5` | EPN |
| `configs/crnn_urbansed_iou1.yaml` | full method, `λ_iou = 1.0` | EPN |
| `configs/crnn_large100.yaml` | wide model, 100 epochs | EPN |
| `configs/crnn_large_ool10.yaml` | wide model, `λ_ool = 10` | EPN |
| `configs/crnn_xl.yaml` | XL CRNN/EPN | EPN |
| `configs/crnn_large_mixup_mild.yaml` | wide model, mild mixup + SpecAugment | EPN |

Architecture is in `src/config.py` (`SEDConfig`); training, loss weights and
post-processing are in `configs/*.yaml`.

## Repository layout

```
src/
├── config.py                 SEDConfig (frontend + CRNN + RED/EPN switches)
├── data.py                   URBAN-SED parsing, log-mel, targets, Dataset
├── modules/
│   ├── model.py              CRNN backbone, FrameCRNN, BoundaryAwareCRNN
│   ├── red.py                RED layer + Hillis-Steele parallel scan
│   ├── epn.py                per-class / single-GRU duration heads
│   ├── loss.py               BCE + focal OOL + duration-weighted IoU
│   ├── postprocess.py        median filter + Algorithm 1 inference
│   ├── metrics.py            PSDS1, collar-F1, event matching
│   └── events.py             Event type + frame rasterisation
├── pipelines/                train / eval / infer entrypoints
└── callbacks/                checkpointing, early stopping, LR schedule, W&B
scripts/data/                 download, preprocess, synthetic generator
scripts/training/             train wrapper, evaluate, smoke test, tuning
notebooks/                    01 EDA, 02 full pipeline, 03 experiments
```

## Method notes

**RED recurrence.** With `s_t = P(e_t | ¬e_{t-1})` and
`z_t = P(¬e_t | e_{t-1})`:

```
p_t      = s_t (1 - p_{t-1}) + (1 - z_t) p_{t-1}
onset_t  = s_t (1 - p_{t-1})
offset_t = z_t p_{t-1}
```

`p_t = a_t p_{t-1} + b_t` is a linear scan, evaluated in `log2(T)` steps with an
associative prefix scan (`src/modules/red.py`).

**Loss.** `L = L_pres + λ_ool (L_on + L_off) + λ_iou L_iou`, with `λ_ool = 100`,
`λ_iou = 2`, focal `α = 2`, and the IoU computed in closed form for intervals
that both contain `t`:

```
IoU = (min(don, don_hat) + min(doff, doff_hat))
      / (max(don, don_hat) + max(doff, doff_hat))
```

**Inference.** Sort frames by RED presence, emit `[t - don, t + doff]`, suppress
overlapping proposals, keep at most `k = 15` events per class and consider the
`m` most active classes.

## Metrics

- **PSDS1**: localisation-focused; per-class TPR vs false-positive rate per
  hour, IoU-0.5 matching, averaged over classes, integrated up to 100 FP/hour;
  no cross-trigger term and no variance penalty (the paper's setting).
- **PSDS2**: classification-focused DCASE scenario 2; DTC/GTC 0.1, CTTC 0.3,
  cross-trigger weight 0.5, inter-class variance penalty 1.0, same 100 FP/hour
  integration.
- **Collar-F1**: 200 ms onset collar, offset collar
  `max(200 ms, 20% of ground-truth duration)`, greedy score-ordered matching.
- **Frame-F1**: monitoring only, threshold 0.5.

Both PSDS variants are reported everywhere (`eval_per_epoch`, training logs,
`test_metrics.json`, the tuning script). PSDS1 remains the model-selection
metric because it is the paper's primary metric.

## Reproduction caveats

The paper uses AudioSet Strong (447 classes; AS-Strong-10 for ablations). This
repo uses URBAN-SED because it is public, strongly annotated and runnable on a
single machine, so absolute numbers are **not comparable** to Tables I–III of
the paper. Deliberate scope choices:

- CRNN backbone only; no ATST-F / BEATs / MN-GRU.
- Frame-BCE + median filter baseline; SEBB and HSM3 are not reimplemented.
- **Capacity, mild augmentation and ensembling drive the gains.** The wide
  model reaches 43.58 PSDS1, the XL model 45.02, the wide model with mild mixup
  45.77, and a four-model probability ensemble **47.73** on test, above the
  36.67 frame-BCE baseline. The default ~1M model with `λ_iou=2` underperforms;
  see the report for the λ_iou sweep and analysis.
- **Default EPN is the single-GRU variant.** The paper uses per-class GRUs for
  AS-Strong-10 and the single GRU for the full 447 classes. On Apple-Silicon
  MPS the per-class variant launches 10 GRUs per forward and is ~12× slower; the
  faithful per-class config is kept in `configs/crnn_urbansed_per_class.yaml`.
- Freq-MixStyle / FilterAugment are not enabled by default.
- PSDS1 uses IoU-0.5 matching, following the paper’s wording; the official
  `psds_eval` DTC/GTC criterion is slightly looser.

## Citation

```bibtex
@article{schmid2026sedboundary,
  title   = {Sound Event Detection with Boundary-Aware Optimization and Inference},
  author  = {Schmid, Florian and Tang, Chi Ian and Parekh, Sanjeel and Ithapu, Vamsi Krishna and others},
  journal = {arXiv preprint arXiv:2601.04178},
  year    = {2026}
}

@dataset{salamon_urbansed_2017,
  title     = {URBAN-SED},
  author    = {Salamon, Justin and MacConnell, Duncan and Cartwright, Mark and Li, Peter and Bello, Juan Pablo},
  year      = {2017},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.1324404}
}
```
