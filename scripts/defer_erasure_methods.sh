#!/usr/bin/env bash
# Once the embedding stage has stopped, marks INLP and mean-difference as deferred and
# regenerates the experiment plan so it reflects what will actually run.
#
# Sequenced rather than immediate: scripts/run_all_embedding.sh is read incrementally by the
# bash executing it, so it can only be rewritten after that process is gone.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f ./.env ] && . ./.env; set +a
export PATH="$PWD/.venv/bin:$PATH"

mkdir -p logs
LOG="logs/defer-erasure-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG"

while pgrep -f "run_all_embedding.sh" >/dev/null 2>&1; do
  echo "$(date +%H:%M:%S) embedding runner still up; waiting..."
  sleep 60
done
echo "$(date +%H:%M:%S) embedding runner gone — deferring INLP and mean-difference"

# LEACE stays enabled so the runner array still names the family; the deferred methods are
# commented out, which is what the plan generator reads to mark them "future work".
python scripts/select_configs.py --runner scripts/run_all_embedding.sh \
    configs/mitigation/embedding/qwen3.5-4b_en_leace.yaml \
    configs/mitigation/embedding/qwen3.5-4b_uk_leace.yaml \
    configs/mitigation/embedding/qwen3.5-9b_en_leace.yaml \
    configs/mitigation/embedding/qwen3.5-9b_uk_leace.yaml

python scripts/make_experiment_matrix.py
echo "plan regenerated — INLP and mean-difference now marked future work"
