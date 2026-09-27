#!/usr/bin/env bash
# The remaining GPU work for the paper, in priority order. Replaces the tail of
# scripts/reaudit_merged.sh (which it waits for) so LAPA runs before the internal DPO probes.
#
#   1. merged-weight re-audits of the 4B SFT adapters
#   2. operating-point control for the four Qwen SFT cells (scripts/operating_point_control.py)
#   3. LAPA SFT, EN then UK: smoke step -> train -> merge + validate against HF+PEFT -> audit
#      -> operating-point control
#   4. internal probes, not for the paper: DPO decision-only, SFT v2 ckpt250, DPO v2 ckpt50
#   5. set stability + report
#
#     nohup bash scripts/run_final_queue.sh > logs/final_queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PATH="$PWD/.venv/bin:$PATH"  # vLLM compiles kernels with ninja from the venv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
RAW="$HBM_OUTPUT_ROOT/outputs/raw"
OLD_RAW="$HBM_OUTPUT_ROOT/outputs/raw_vllm_lora_invalid"
OLD_RES=eval/results_vllm_lora_invalid
mkdir -p "$OLD_RAW" "$OLD_RES"

say() { echo "$(date '+%F %T') $*"; }
run_name() { $PY -c "import sys; sys.path.insert(0, 'src')
from hiring_bias_mitigation.eval.runner import build_run_name
from hiring_bias_mitigation.utils.config import load_config
print(build_run_name(load_config('$1')))"; }

audit() {  # audit a trained adapter from merged weights; old vLLM-LoRA outputs moved aside
  local cfg="configs/audit/$1.yaml" run; run=$(run_name "$cfg")
  if grep -q '"served_model": ".*merged' "$RAW/$run.meta.json" 2>/dev/null; then
    say "skip audit $run (already from merged weights)"; return 0; fi
  for f in "$RAW/$run.parquet" "$RAW/$run.meta.json"; do [ -f "$f" ] && mv "$f" "$OLD_RAW/"; done
  [ -f "eval/results/$run.json" ] && mv "eval/results/$run.json" "$OLD_RES/"
  say "start audit $run"
  if $PY scripts/run_audit.py --config "$cfg"; then say "done audit $run"; else say "FAILED audit $run"; return 1; fi
}

op_control() {  # operating-point control for one adapter's audit config
  local cfg="configs/audit/$1.yaml"
  say "start op-control $1"
  $PY scripts/operating_point_control.py margins --config "$cfg" --which base &&
  $PY scripts/operating_point_control.py margins --config "$cfg" --which adapter &&
  $PY scripts/operating_point_control.py compare --config "$cfg" &&
  say "done op-control $1" || say "FAILED op-control $1"
}

train_lapa() {  # $1 = en|uk
  local lang=$1 train="configs/mitigation/sft/lapa-12b_${lang}_only.yaml"
  local adapter="outputs/sft/lapa-12b_${lang}_only"
  if [ -f "$HBM_OUTPUT_ROOT/$adapter/adapter_config.json" ]; then
    say "skip train lapa $lang (adapter exists)"
  else
    say "start smoke lapa $lang"
    $PY scripts/preflight_training.py --smoke --config "$train" || { say "FAILED smoke lapa $lang"; return 1; }
    say "start train lapa $lang"
    $PY -m hiring_bias_mitigation.mitigation.sft --config "$train" || { say "FAILED train lapa $lang"; return 1; }
    say "done train lapa $lang"
  fi
  say "start validate lapa $lang"
  local merged; merged=$($PY scripts/merge_adapter.py --adapter "$adapter" | tail -1)
  $PY scripts/diagnose_adapter.py --base-model lapa-llm/lapa-v0.1.2-instruct --adapter "$adapter" \
      --lang "$lang" --audit-run "lapa-v0.1.2-instruct--${lang}--baseline" --vllm-merged "$merged" \
    && say "done validate lapa $lang" || say "FAILED validate lapa $lang"
}

# 0. let the running 9B uk re-audit finish (its parent queue was stopped)
while pgrep -f "run_audit.py --config configs/audit/qwen3.5-9b_uk_sft_uk_only.yaml" >/dev/null; do sleep 60; done
say "9B uk re-audit finished"

# 1.
audit qwen3.5-4b_en_sft_en_only
audit qwen3.5-4b_uk_sft_uk_only
# 2.
for c in qwen3.5-9b_en_sft_en_only qwen3.5-9b_uk_sft_uk_only qwen3.5-4b_en_sft_en_only qwen3.5-4b_uk_sft_uk_only; do
  op_control "$c"
done
$PY scripts/set_stability.py >/dev/null 2>&1; $PY scripts/make_report.py >/dev/null 2>&1
say "paper core done (4 SFT cells + op-control)"
# 3.
for lang in en uk; do
  train_lapa "$lang" && audit "lapa-12b_${lang}_sft_${lang}_only" && op_control "lapa-12b_${lang}_sft_${lang}_only"
done
$PY scripts/set_stability.py >/dev/null 2>&1; $PY scripts/make_report.py >/dev/null 2>&1
say "lapa done"
# 4. internal probes
audit qwen3.5-9b_en_dpo_en_only_dpo_decision
audit qwen3.5-9b_en_sft_en_only_v2_ckpt250
audit qwen3.5-9b_en_dpo_en_only_dpo_v2_ckpt50
# 5.
$PY scripts/set_stability.py >/dev/null 2>&1; $PY scripts/make_report.py >/dev/null 2>&1
say "queue finished"
