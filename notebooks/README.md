# Notebooks

| Notebook | Contents |
|---|---|
| `01_eda.ipynb` | Split/class/event-duration/polyphony statistics, log-mel + annotation overlays, co-occurrence, sanity checks, processed-target audit |
| `02_pipeline.ipynb` | Dataset batch → CRNN → RED → losses → training → PSDS1/PSDS2/collar-F1 → EPN vs median-filter inference |
| `03_experiments.ipynb` | Component study, median-filter grid, EPN `k`/`m` sensitivity, boundary-error analysis |

All three notebooks are **self-contained / Colab-ready**: they do not import
`src`. `02_pipeline.ipynb` and `03_experiments.ipynb` start with seven
documented `%%writefile epn_core.py` / `%%writefile -a epn_core.py` cells
(config/events, RED, EPN/CRNN, losses, post-processing, metrics, data), each
preceded by its own markdown note, then import that local module. `01_eda.ipynb`
inlines its small parsing/plotting helpers. Notes stay in markdown cells; one
logical block per code cell.

Rules:

- The repo `src/` remains the source of truth; the notebook core is a compact
  copy kept in sync for Colab. When changing a core module, regenerate the
  seven `%%writefile` core cells.
- `02_pipeline.ipynb` prefers `data/processed/urbansed_sample`, then the full
  processed set, then a synthetic fallback, so it always executes.
- `03_experiments.ipynb` prefers the sample checkpoint, then the full-method
  checkpoint; missing runs are reported rather than invented.
- Store generated tables/figures under `docs/analysis/<yyyy-mm-dd>/<topic>/`
  and link them from `docs/EXPERIMENTS.md`.
