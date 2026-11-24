# URBAN-SED boundary-aware SED results

Date: 2026-11-24 · VM: RTX PRO 4000 Blackwell (24 GB), CUDA 13.0,
torch 2.9.1+cu130 · W&B project:
<https://forge.coreweave.com/wandb/octoopt/epn-sed>

Generated from `results/*/test_metrics.json` by
`scripts/training/make_report.py`.

## Protocol

- Dataset: URBAN-SED v2.0.0, 6,000 / 2,000 / 2,000 train / validate / test.
- Frontend: 16 kHz, 64-band log-mel, 25 ms window, 20 ms hop, 500 frames.
- Backbone: CRNN, 4 conv blocks + 2-layer BiGRU.
- Default method: RED + focal OOL (`λ_ool = 100`) + single-GRU EPN.
- Objective: `L_pres + 100 (L_on + L_off) + λ_iou L_iou`; focal `α = 2`.
- Training: AdamW, 40 epochs, batch 128 (64 for wide model), cosine schedule,
  100 warmup steps, one seed (42).
- EPN event extraction: Algorithm 1, `k = 15`, `m = all`, score threshold 0.3.
- Baseline event extraction: median filter, 7 frames, threshold 0.5.
- Metrics: PSDS1 (IoU-0.5, no variance penalty), PSDS2 (DTC/GTC 0.1,
  CTTC 0.3, `α_CT = 0.5`, `α_ST = 1`), collar-F1 (200 ms onset,
  `max(200 ms, 20% duration)` offset).

## Main results (test split)

| Run | PSDS1 | PSDS2 | collar-F1 | frame-F1 | loss | W&B |
|---|---:|---:|---:|---:|---:|---|
| BL (frame BCE + median) | 36.67 | 73.13 | 26.11 | 59.14 | 0.1867 | [v5p78phy](https://forge.coreweave.com/wandb/octoopt/epn-sed/runs/v5p78phy) |
| +RED | 37.16 | 71.43 | 26.47 | 59.01 | 0.1906 | [q1zpzio2](https://forge.coreweave.com/wandb/octoopt/epn-sed/runs/q1zpzio2) |
| +OOL | 35.93 | 71.26 | 26.57 | 57.85 | 0.5330 | [yb4dmxn2](https://forge.coreweave.com/wandb/octoopt/epn-sed/runs/yb4dmxn2) |
| +EPN (`λ_iou = 2`) | 25.87 | 51.41 | 13.57 | 35.95 | 2.0604 | [qcfakcan](https://forge.coreweave.com/wandb/octoopt/epn-sed/runs/qcfakcan) |
| +EPN (`λ_iou = 0.5`) | 35.42 | 69.83 | 21.39 | 55.17 | 0.9036 | [4gfycpim](https://forge.coreweave.com/wandb/octoopt/epn-sed/runs/4gfycpim) |
| +EPN (`λ_iou = 1.0`) | 33.08 | 66.13 | 20.13 | 50.33 | 1.2888 | [0yocd0ei](https://forge.coreweave.com/wandb/octoopt/epn-sed/runs/0yocd0ei) |
| Extension: wide model | **43.58** | **73.73** | 28.12 | **62.30** | 0.8765 | [mroixs95](https://forge.coreweave.com/wandb/octoopt/epn-sed/runs/mroixs95) |
| Extension: synthetic mixup | 34.22 | 67.80 | 20.99 | 52.89 | 0.9090 | [d91r1vb4](https://forge.coreweave.com/wandb/octoopt/epn-sed/runs/d91r1vb4) |

`+EPN` rows are the full method at three IoU weights. The wide model is the
same method with 32/64/128/256 conv channels and a 256-wide BiGRU; mixup uses
the default model with max-mixing and SpecAugment.

## Findings

1. **RED alone is competitive.** `+RED` (37.16) is slightly above the
   frame-BCE baseline (36.67) without any boundary loss or EPN.
2. **OOL does not help at the paper’s weight on this dataset.** `+OOL` is
   35.93, below `+RED`; `λ_ool = 100` may be too strong for URBAN-SED.
3. **The IoU loss weight is critical.** `+EPN` collapses to 25.87 at
   `λ_iou = 2`, improves to 33.08 at 1.0 and 35.42 at 0.5. The paper tunes
   `λ_iou`; URBAN-SED prefers a much smaller value than our initial default.
4. **Model capacity gives the best result.** The wide model reaches
   **43.58 PSDS1 / 73.73 PSDS2**, the only configuration clearly above the
   baseline. This suggests the temporal components need more capacity than the
   ~1M-parameter CRNN.
5. **Synthetic mixup is not a free win.** Max-mixing + SpecAugment reaches
   34.22, below the baseline. The augmentation likely creates unrealistic
   overlapping events and too many masked regions at `p = 0.5`.
6. **EPN inference vs median filtering is mixed.** On the wide model,
   EPN gives 43.58 PSDS1 / 28.12 collar-F1; median filtering with the
   validation-tuned setting (threshold 0.4, length 3) gives 43.00 PSDS1 /
   30.24 collar-F1. On the default model, tuned median gives 25.62 PSDS1 vs
   25.87 for EPN. EPN is not the source of the main gap; presence quality is.

## Post-processing comparison

| Model | Extraction | PSDS1 | PSDS2 | collar-F1 |
|---|---|---:|---:|---:|
| Default + EPN | Algorithm 1 (`k=15`) | 25.87 | 51.41 | 13.57 |
| Default + median | tuned on val (0.2, 15) | 25.62 | 55.03 | 11.56 |
| Wide + EPN | Algorithm 1 (`k=15`) | **43.58** | 73.73 | 28.12 |
| Wide + median | tuned on val (0.4, 3) | 43.00 | 73.65 | **30.24** |

The wide-model median row uses the best validation PSDS1 setting from
`results/crnn_large_urbansed/tune_median_val.json`.

## Limitations

- URBAN-SED is not AudioSet Strong; numbers are not comparable to the paper.
- One seed, 40 epochs. The paper trains up to 100 epochs and tunes the learning
  rate per backbone.
- Single-GRU EPN; the paper’s AS-Strong-10 configuration is per-class.
- No SEBB/HSM3 baseline, no Freq-MixStyle/FilterAugment.
- PSDS1 uses IoU-0.5 matching as stated in the paper; the official `psds_eval`
  DTC/GTC criterion is slightly different.
- Collar-F1 is reported at the configured threshold, not exhaustively tuned
  per class.

## Artifacts

- Raw metrics: `results/<run>/test_metrics.json`
- Checkpoints: `checkpoints/<run>/best.pt`
- Median-tuning grids: `results/<run>/tune_median_val.json`
- W&B: <https://forge.coreweave.com/wandb/octoopt/epn-sed>
