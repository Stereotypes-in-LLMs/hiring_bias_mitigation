#!/usr/bin/env bash
# Waits for a running sweep to finish, then runs the configs given as arguments.
#
# Why this exists rather than "just add them to the CONFIGS array": bash reads a script
# incrementally as it executes it, so editing scripts/run_all_audit.sh while a multi-day
# sweep is mid-flight can corrupt the remainder of the queue. This appends work without
# touching the running file.
#
#   ./scripts/queue_after.sh configs/audit/qwen3.5-9b_en_baseline.yaml \
#                            configs/audit/qwen3.5-9b_uk_baseline.yaml
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f ./.env ] && . ./.env; set +a
export PATH="$PWD/.venv/bin:$PATH"

if [ $# -eq 0 ]; then
  echo "usage: $0 <config.yaml> [config.yaml ...]" >&2
  exit 1
fi

mkdir -p logs
LOG="logs/queued-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "queued $# config(s); log: $LOG"

# Wait for any in-flight sweep. Matching on the runner script only -- never a pattern that
# could also match this script's own command line.
while pgrep -f "run_all_[a-z]*\.sh" > /dev/null 2>&1; do
  echo "$(date +%H:%M:%S) waiting for the in-flight sweep to finish..."
  sleep 300
done
echo "$(date +%H:%M:%S) sweep finished; starting queued configs"

FAILED=()
for cfg in "$@"; do
  echo "=== QUEUED: $cfg ==="
  python scripts/run_audit.py --config "$cfg" || { echo "!! FAILED: $cfg" >&2; FAILED+=("$cfg"); }
done

python scripts/make_report.py
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "${#FAILED[@]} config(s) FAILED:" >&2
  for f in "${FAILED[@]}"; do echo "  $f" >&2; done
  exit 1
fi
echo "QUEUE COMPLETE"
