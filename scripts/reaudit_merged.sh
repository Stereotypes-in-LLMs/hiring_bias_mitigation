#!/usr/bin/env bash
# Re-audits every trained adapter from merged weights.
#
# The first audits served adapters through vLLM's LoRA path, which does not reproduce
# Qwen3.5 adapters (see src/hiring_bias_mitigation/mitigation/merge.py). Their generations
# and scores are moved to *_vllm_lora_invalid/ rather than deleted, so the discrepancy
# stays documented. The runner now merges each adapter before serving it.
#
# Order: the four main SFT cells first (they are the paper's training rows), then the probes.
#
#     nohup bash scripts/reaudit_merged.sh > logs/reaudit_merged.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$PWD/.venv/bin:$PATH"  # vLLM compiles kernels with ninja from the venv
PY=.venv/bin/python

RAW="$HBM_OUTPUT_ROOT/outputs/raw"
OLD_RAW="$HBM_OUTPUT_ROOT/outputs/raw_vllm_lora_invalid"
OLD_RES=eval/results_vllm_lora_invalid
mkdir -p "$OLD_RAW" "$OLD_RES"

CONFIGS=(
  qwen3.5-9b_en_sft_en_only
  qwen3.5-9b_uk_sft_uk_only
  qwen3.5-4b_en_sft_en_only
  qwen3.5-4b_uk_sft_uk_only
  qwen3.5-9b_en_dpo_en_only_dpo_decision
  qwen3.5-9b_en_sft_en_only_v2_ckpt250
  qwen3.5-9b_en_dpo_en_only_dpo_v2_ckpt50
)

for name in "${CONFIGS[@]}"; do
  cfg="configs/audit/$name.yaml"
  run=$($PY -c "import sys; sys.path.insert(0, 'src')
from hiring_bias_mitigation.eval.runner import build_run_name
from hiring_bias_mitigation.utils.config import load_config
print(build_run_name(load_config('$cfg')))")
  if grep -q '"served_model": ".*merged' "$RAW/$run.meta.json" 2>/dev/null; then
    echo "$(date '+%F %T') skip $run (already audited from merged weights)"
    continue
  fi
  for f in "$RAW/$run.parquet" "$RAW/$run.meta.json"; do
    [ -f "$f" ] && mv "$f" "$OLD_RAW/"
  done
  [ -f "eval/results/$run.json" ] && mv "eval/results/$run.json" "$OLD_RES/"
  echo "$(date '+%F %T') start $run"
  if $PY scripts/run_audit.py --config "$cfg"; then
    echo "$(date '+%F %T') done $run"
  else
    echo "$(date '+%F %T') FAILED $run"
  fi
done
echo "$(date '+%F %T') queue finished"
