#!/usr/bin/env bash
# Run the paper's component study on URBAN-SED, sequentially.
#
#   bash scripts/training/run_ablations.sh
#
# Each run writes checkpoints/<run_name>/best.pt and
# results/<run_name>/test_metrics.json. Expect roughly 40-60 min per run on an
# Apple-Silicon GPU with the single-GRU EPN; the per-class config is slower.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

for config in crnn_baseline crnn_red crnn_red_ool crnn_urbansed; do
  echo "=== ${config} ==="
  uv run python -m src.pipelines.train --config "configs/${config}.yaml"
done
