#!/usr/bin/env bash
# Runs scripts/diagnose_adapter.py on the SFT and decision-only DPO adapters once the GPU is
# free. The student pass over the synthetic pool is left to finish first: its generations are
# cached and reused, and two models on this GPU at once would not fit.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f ./.env ] && . ./.env; set +a
export PATH="$PWD/.venv/bin:$PATH"
LOG="logs/adapter-diagnostics-$(date +%Y%m%d-%H%M%S).log"
echo "$LOG" > logs/diagnostics.logpath
exec > >(tee -a "$LOG") 2>&1

until [ "$(pgrep -x python | wc -l)" -eq 0 ]; do sleep 30; done
echo "$(date +%H:%M:%S) GPU free"

python scripts/diagnose_adapter.py --adapter outputs/sft/qwen3.5-9b_en_only \
  --audit-run Qwen3.5-9B--en--sft--adapter--en_only --train-split sft_train
python scripts/diagnose_adapter.py --adapter outputs/dpo/qwen3.5-9b_en_only_dpo_decision \
  --audit-run Qwen3.5-9B--en--dpo--adapter--en_only_dpo_decision --train-split dpo_decision_train
echo ">>> adapter diagnostics done <<<"
