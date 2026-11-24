# scripts/

Utility entrypoints, grouped by category. Run them from the repo root, e.g.
`uv run python scripts/<category>/<name>.py ...`.

```
scripts/
├── data/       download / preprocess / synthetic generators
└── training/   train / evaluate / smoke-test / post-processing tuning
```

## data/

| Script | Purpose |
|---|---|
| `data/download_urbansed.py` | Download the 6.6 GB URBAN-SED archive, verify md5, extract to `data/` |
| `data/prepare_urbansed.py` | Resample to 16 kHz, compute 64-band log-mel, rasterise annotations into `pres/on/off/don/doff` targets |
| `data/make_synthetic_sed.py` | Generate a tiny URBAN-SED-shaped dataset for smoke runs/tests |

## training/

| Script | Purpose |
|---|---|
| `training/train.py` | Thin wrapper over `src.pipelines.train` (the real CLI) |
| `training/evaluate.py` | Evaluate a checkpoint on a processed split with PSDS1/PSDS2/collar-F1 |
| `training/smoke_test.py` | Synthetic end-to-end run: generate → train → test, assert finite metrics |
| `training/tune_postprocessing.py` | Grid-search median-filter length/threshold or EPN `k`/`m` on validation |
| `training/run_ablations.sh` | Run BL / +RED / +OOL / +EPN sequentially |
| `training/run_extensions.sh` | Run the wider-model and synthetic-mixup extensions |

Typical order:

```bash
uv run python scripts/data/download_urbansed.py
uv run python scripts/data/prepare_urbansed.py --workers 8
uv run python -m src.pipelines.train --config configs/crnn_urbansed.yaml
uv run python scripts/training/evaluate.py --ckpt checkpoints/crnn_red_epn_urbansed/best.pt --split test
```
