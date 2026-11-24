#!/usr/bin/env bash
# Run the extension experiments: larger model and synthetic event mixing.
#
#   bash scripts/training/run_extensions.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

for config in crnn_urbansed_large crnn_urbansed_mixup; do
  echo "=== ${config} ==="
  uv run python -m src.pipelines.train --config "configs/${config}.yaml"
done
