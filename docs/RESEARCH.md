# Research

**Question.** Can the boundary-aware objective and Event Proposal Networks from
Schmid et al. (arXiv:2601.04178) improve event-level sound event detection over
a frame-wise BCE model with median-filter post-processing, on URBAN-SED?

**Why it matters.** Frame-wise SED models are the dominant paradigm, but their
final event boundaries depend on non-differentiable post-processing (median
filter, SEBB) whose hyperparameters are tuned outside training. RED + OOL + EPN
move boundary modelling into the model and remove that tuning step. A small,
fully public reproduction is useful for checking which component carries the
gain and how sensitive the method is to inference settings.

## Scope

- **Dataset:** URBAN-SED v2.0.0 (10,000 clips, 10 classes, strong annotations,
  official 6,000 / 2,000 / 2,000 split).
- **Model:** CRNN from scratch (4 conv blocks + 2-layer BiGRU), the paper’s
  smallest reported backbone and the only one runnable on one Apple-Silicon GPU.
- **Method:** RED, focal onset/offset loss (OOL), per-class-GRU EPN, Algorithm 1
  inference.
- **Baselines:** frame-BCE CRNN + median filtering. Ablation: `BL`, `+RED`,
  `+OOL`, `+EPN`.
- **Metrics:** PSDS1 (no variance penalty), 200 ms collar-F1, frame-F1
  (monitoring).
- **Out of scope:** ATST-F, BEATs, MN-GRU, SEBB, HSM3, AudioSet Strong,
  Freq-MixStyle/FilterAugment, knowledge distillation.

## Method

1. Precompute 16 kHz, 64-band log-mel features (25 ms window, 20 ms hop,
   500 frames per 10 s clip) and rasterise annotations into presence, onset,
   offset and duration targets.
2. Train a CRNN that emits two logits per class (event start / event end).
3. Apply RED to obtain presence/onset/offset probabilities.
4. Add the focal boundary loss and the duration-weighted IoU loss from EPN
   proposals.
5. At inference, compare EPN Algorithm 1 against median filtering.
6. Report PSDS1 and collar-F1 on the held-out test split.

## Acceptance criteria

- A run counts only if it uses the official test split and the metric code in
  `src/modules/metrics.py`.
- PSDS1 is the primary criterion; collar-F1 is secondary.
- Full-method vs baseline comparison must use identical backbones, splits and
  epoch budgets.
- Post-processing hyperparameters are tuned on validation, never on test.

## Results

- EDA summary: [`docs/analysis/2026-11-23/eda/summary.md`](analysis/2026-11-23/eda/summary.md)
- One-epoch real-data sanity: PSDS1 0.99, PSDS2 12.98, collar-F1 0.75 (test).
  Not converged; see `docs/EXPERIMENTS.md`.
- Full 40-epoch run: _pending; results will be written to
  `results/crnn_red_epn_urbansed/test_metrics.json` and linked here._
- Component study (`BL → +RED → +OOL → +EPN`): _pending._
