# Paper Spec — reproduction report

## Venue & format
- **Author block:** _fill in before submission._

## Paper type
Empirical reproduction / component study.

## Thesis
- **Claim:** On URBAN-SED, adding RED + OOL + EPN to a frame-wise CRNN improves
  PSDS1 and collar-F1 over the same CRNN trained with frame BCE and evaluated
  with median filtering, without tuning post-processing hyperparameters.
- **Strength:** Single dataset, single backbone (CRNN), 10 classes. Do **not**
  claim AudioSet-Strong-scale results or state-of-the-art.
- **Novelty:** None in method; the contribution is an independent, fully public
  reproduction and a component/ablation analysis on a different dataset.

## Section plan
1. Introduction: frame-wise SED, post-processing brittleness, reproduction goal.
2. Method: RED recurrence, OOL focal loss, EPN duration heads, IoU loss,
   Algorithm 1. Equations copied from the paper, not re-derived.
3. Experimental setup: URBAN-SED, CRNN, features, metrics (PSDS1/PSDS2),
   training budget, baseline and ablation protocol.
4. Results: component study table, median-filter vs EPN table, k/m sensitivity,
   boundary-error analysis.
5. Discussion: which component contributes most, where PSDS1 and PSDS2 diverge,
   limitations (dataset shift, no transformers, no SEBB/HSM3).
6. Conclusion: no new claims.

## Content rules
- **Formal results:** No theorems. Define RED and IoU loss precisely.
- **Examples:** One worked RED figure (two events) and one event-extraction
  figure is enough.
- **Figures/tables:** Component study table (primary), PSDS curves, boundary
  error histograms. Tables over figures where possible.
- **Experiments:** Only numbers produced by `results/*/test_metrics.json`.
  Never invent or copy paper numbers into the reproduction table.

## Citations
- Cite the original paper, RED/Heinsen scan, focal loss, PSDS (Bilen et al. and
  Ebbers et al.), URBAN-SED and Scaper.
- Verify every arXiv ID and venue before submission.

## Style
- Formal, direct, no hype words.
- Report PSDS1 and PSDS2 as percentages (`×100`), state the integration limit
  (100 FP/hour) and matching criterion.

## Output contract
- One self-contained `.tex` file compiling with pdflatex twice.
- Hand-off message: page count, list of `% TODO: verify` items.

## Inputs
- `docs/RESEARCH.md`, `docs/EXPERIMENTS.md`, `docs/analysis/`, `results/`.
