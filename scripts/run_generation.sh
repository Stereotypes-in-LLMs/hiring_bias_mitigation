#!/usr/bin/env bash
# Runs (or resumes) the semi-synthetic data generation, then assembles the datasets.
#
# Every pass checks for its own raw dump first, so an interrupted run continues from where it
# stopped rather than repeating hours of generation. Safe to re-run at any point.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f ./.env ] && . ./.env; set +a
export PATH="$PWD/.venv/bin:$PATH"

mkdir -p logs
LOG="logs/generate-$(date +%Y%m%d-%H%M%S).log"
echo "$LOG" > logs/generate.logpath
echo "log: $LOG"

{
  python scripts/generate_training_data.py --config configs/generation/teacher.yaml
  status=$?
  if [ $status -eq 0 ]; then
    echo "=== generation finished; assembling datasets ==="
    python scripts/assemble_datasets.py --config configs/generation/teacher.yaml
  else
    echo "!! generation exited $status — assembling whatever completed" >&2
    python scripts/assemble_datasets.py --config configs/generation/teacher.yaml || true
  fi
} > "$LOG" 2>&1
