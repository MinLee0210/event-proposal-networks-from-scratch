# Notes

Decisions log. Append newest at the top. Record *why*, not just *what*, so the
reasoning survives after the code changes.

## 2026-11-23

- **Use URBAN-SED instead of AS-Strong.** The paper evaluates on AudioSet
  Strong, which is not redistributable as audio and needs a large pretrained
  backbone. URBAN-SED is public, strongly annotated, 10 classes and 10 s clips,
  so the full method and its ablations fit on one machine. Absolute numbers are
  not comparable to the paper; the comparison is internal (baseline vs RED/EPN).
- **CRNN only.** ATST-F / BEATs / MN-GRU add pretrained checkpoints and a second
  training stack. The paper’s CRNN is the cleanest test of the temporal
  components and is small enough to train from scratch.
- **Median-filter baseline, no SEBB/HSM3.** MF is the standard frame-wise
  post-processing baseline and is easy to tune per class. SEBB would require a
  faithful reimplementation of its change-detection and tuning grid; HSM3 is a
  separate inference system. Both are documented as out of scope.
- **PSDS1 in-repo.** The official `psds_eval` package is pandas-heavy and its
  DTC/GTC matching is not exactly IoU-0.5. We implement the paper’s stated
  criterion directly (IoU-0.5, macro TPR vs FPR/hour, no variance penalty) and
  keep the implementation unit-tested. `psds_eval` was used only as a reference
  for the curve construction.
- **RED uses a Hillis-Steele scan.** A Python loop over 500 frames creates 500
  tiny kernels per forward. The affine scan `p_t = a_t p_{t-1} + b_t` composes
  associatively, so a log-depth vectorised scan is exact and much faster. A
  sequential implementation remains in `sequential_red` for tests.
- **Frame targets use a 20 ms hop (500 frames/clip).** This is a compromise
  between boundary resolution and memory/training time. The paper does not
  specify its exact frontend for CRNN; the processed `meta.json` records the
  choice so checkpoints and features cannot silently disagree.
- **Default EPN is single-GRU; per-class is opt-in.** The paper uses per-class
  GRUs for AS-Strong-10 and the single GRU for 447 classes. URBAN-SED has 10
  classes, so per-class is the faithful choice, but on Apple-Silicon MPS it is
  ~12× slower (10 GRUs × 2 layers × 2 directions = 40 RNN calls per forward;
  12.7 clips/s vs 152 clips/s). The default config uses the single GRU so a
  full 40-epoch run fits in a session; `crnn_urbansed_per_class.yaml` preserves
  the paper configuration for an overnight/CUDA run.
- **Both PSDS1 and PSDS2 are reported.** PSDS1 follows the paper: IoU-0.5
  matching, macro TPR vs FP/hour, no variance penalty. PSDS2 follows DCASE
  scenario 2: DTC/GTC 0.1, CTTC 0.3, cross-trigger weight 0.5, variance penalty
  1.0. They answer different questions (localisation vs class recognition), so
  reporting only one hides failures.
- **Evaluation ground truth is the raw annotation sidecar, not the merged
  frame targets.** URBAN-SED contains same-class overlaps (29,919 raw train
  events vs 26,448 merged regions). Training needs one presence region per
  class, but evaluation must count the original events; otherwise PSDS and
  collar-F1 silently drop ~12% of ground truth. `events_<split>.json` preserves
  the raw annotations.
- **W&B is on by default for real runs.** The training configs enable it, and
  `train()` logs train loss/LR, validation PSDS1/PSDS2/collar-F1/frame-F1,
  final test scalars and the best checkpoint artifact. `wandb` stays an optional
  extra (`uv sync --extra wandb`) so the smoke test and tests do not require it;
  the callback no-ops if the package is absent.
- **Warmup is 100 steps, not the paper's 1,000.** The paper's 1,000-step
  warmup assumes ~790 optimizer steps per epoch (100k AS-Strong clips at batch
  128). URBAN-SED has 6k clips, i.e. 46 steps/epoch, so 1,000 steps would cover
  22 of 40 epochs and leave the model undertrained. 100 steps is ~2 epochs.
- **Score threshold still exists for collar-F1.** PSDS1 is threshold-independent,
  but collar-F1 needs an operating point. We tune it on validation through
  `scripts/training/tune_postprocessing.py`; EPN event scores are mean RED
  presence over the proposal interval.
