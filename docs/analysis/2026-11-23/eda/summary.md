# URBAN-SED EDA summary (2026-11-23)

Source: `notebooks/01_eda.ipynb`, raw `data/URBAN-SED_v2.0.0/`, processed
`data/processed/urbansed/`.

## Splits and annotations

| Split | Clips | Raw events | Events/clip |
|---|---:|---:|---:|
| train | 6,000 | 29,919 | 4.99 |
| validate | 2,000 | 9,956 | 4.98 |
| test | 2,000 | 9,956 | 4.98 |

- No annotation outside `[0, 10] s`, no non-positive durations.
- No filename overlap between train/validate/test.
- Class counts in train are balanced: 2,941–3,055 events per class.
- Event durations: mean 2.01 s, median 1.87 s, p95 3.89 s, max 4.79 s.
- Mean summed active fraction per clip: 1.005 (classes overlap; this is not a
  probability).

## Processed features

- Log-mel: `(6000, 500, 64)`, `(2000, 500, 64)`, `(2000, 500, 64)` float16.
- 16 kHz, 25 ms window, 20 ms hop, 64 mel bands, 10 s clips.
- Global normalisation: mean `-18.0155`, std `13.4189` dB.
- Frame presence active fraction: 9.39% (train), 9.25% (validate), 9.30%
  (test), averaged over class channels.
- Raw events vs merged frame regions: 29,919 raw train events become 26,448
  merged same-class regions (26,448 onsets/offsets). Training uses merged
  regions; evaluation uses the raw annotation sidecar `events_train.json`, so
  overlapping same-class events still count as separate ground truth.

## Smoke run (synthetic, not a result)

`notebooks/02_pipeline.ipynb` trains one epoch on the 32-clip synthetic set:

| PSDS1 | PSDS2 | collar-F1 | frame-F1 |
|---:|---:|---:|---:|
| 0.22 | 0.00 | 2.20 | 5.52 |

These numbers only confirm that the full path (data → RED → EPN → events →
metrics) executes; they say nothing about URBAN-SED performance.
