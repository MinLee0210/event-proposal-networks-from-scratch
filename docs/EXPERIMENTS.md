# Experiments

Runbook for gated experiments. Each experiment records: hypothesis, config,
command, result, and decision.

## Template

### <YYYY-MM-DD> — <short name>

- **Hypothesis:**
- **Config:** `configs/<file>.yaml`
- **Command:**

  ```bash
  uv run python -m src.pipelines.train --config configs/<file>.yaml
  ```

- **Result:** _PSDS1 / PSDS2 / collar-F1 / frame-F1, plus where the log lives._
- **Decision:** _adopt / discard / follow up with ..._

---

## 2026-11-23 — W&B-logged sample run (not a result)

- **Hypothesis:** the training pipeline logs train/eval/test metrics and the
  best-checkpoint artifact to W&B.
- **Config:** `configs/crnn_sample.yaml` (300 clips/split, 6 epochs).
- **Command:** `uv run python -m src.pipelines.train --config configs/crnn_sample.yaml`
- **Result:** test PSDS1 0.72, PSDS2 0.00, collar-F1 0.00, frame-F1 0.00;
  W&B run `crnn_sample_wandb`
  (https://forge.coreweave.com/wandb/octoopt/epn-sed/runs/s2ltfazu).
  Checkpoint `checkpoints/crnn_sample_wandb/best.pt`.
- **Decision:** logging path validated; sample is too small/undertrained for
  conclusions. Full runs go to the next VM.

## 2026-11-23 — one-epoch real-data sanity (not a result)

- **Hypothesis:** the full data → RED → EPN → PSDS path works on real features.
- **Config:** `configs/crnn_urbansed.yaml` (single-GRU EPN), 1 epoch.
- **Command:** `uv run python -m src.pipelines.train --config configs/crnn_urbansed.yaml --epochs 1 --run_name debug_single`
- **Result:** test PSDS1 0.99, PSDS2 12.98, collar-F1 0.75, frame-F1 17.41;
  checkpoint `checkpoints/debug_single/best.pt`, log `results/debug_single/`.
  These are one-epoch numbers and are not comparable to the paper.
- **Decision:** pipeline validated; run the full 40-epoch budget.

## 2026-11-23 — component study on URBAN-SED

- **Hypothesis:** each paper component contributes monotonically:
  `BL < +RED < +OOL < +EPN` on PSDS1 and collar-F1.
- **Configs:** `crnn_baseline.yaml`, `crnn_red.yaml`, `crnn_red_ool.yaml`,
  `crnn_urbansed.yaml`.
- **Command:**

  ```bash
  for c in crnn_baseline crnn_red crnn_red_ool crnn_urbansed; do
    uv run python -m src.pipelines.train --config configs/$c.yaml
  done
  ```

- **Result:** _pending — full runs not yet executed. Smoke test passes (PSDS1 and PSDS2 are finite)._
- **Decision:** _pending._

## 2026-11-23 — median-filter vs EPN inference

- **Hypothesis:** EPN removes the need for tuning a median-filter grid and
  improves PSDS1, especially for short events.
- **Config:** full method checkpoint (`crnn_urbansed.yaml`), evaluated twice:
  once with EPN Algorithm 1, once with median filtering on the same presence
  probabilities.
- **Command:**

  ```bash
  uv run python scripts/training/tune_postprocessing.py \
      --ckpt checkpoints/crnn_urbansed_urbansed/best.pt --method median

  uv run python scripts/training/tune_postprocessing.py \
      --ckpt checkpoints/crnn_urbansed_urbansed/best.pt --method epn
  ```

- **Result:** _pending._
- **Decision:** _pending._

## 2026-11-23 — inference hyperparameter sensitivity

- **Hypothesis:** PSDS1 is stable for `k` in 10–20 and for `m` near the number
  of active classes; collar-F1 degrades if the score threshold is too high.
- **Grid:** `k ∈ {5, 10, 15, 20}`, `m ∈ {5, 10}`, threshold `0.1 … 0.9`.
- **Command:** `scripts/training/tune_postprocessing.py --method epn`.
- **Result:** _pending._
- **Decision:** _pending._
