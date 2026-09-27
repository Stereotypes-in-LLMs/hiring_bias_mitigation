#!/usr/bin/env bash
# Retrains the two Ukrainian SFT cells on corrected targets, then continues with LAPA.
#
# Why the retrain. The Ukrainian SFT targets carried the *canonical* decision word
# ("reject"/"hire") instead of the words the Ukrainian prompt asks for
# ("відхилити"/"найняти"): generation normalises decisions for language-agnostic analysis,
# and that normalised value was written into the training target. Both Ukrainian adapters
# learned to answer in English, which changes the output contract and confounds their
# results (Qwen3.5-9B UK also collapsed to a 10% hire rate). Fixed in
# `generation/dataset.py::_completion`, pinned by a test, and the datasets were rebuilt.
# The old adapters, generations and scores are kept under *_english_targets/.
#
#   1. Qwen3.5-9B UK: train -> audit from merged weights -> operating-point control
#   2. Qwen3.5-4B UK: train -> audit
#   3. LAPA EN, then LAPA UK: smoke -> train -> merge + validate -> audit -> op-control
#   4. internal probes (not for the paper), then the reports
#
#     nohup bash scripts/run_uk_retrain_then_lapa.sh > logs/uk_retrain.log 2>&1 &
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

audit() {
  local cfg="configs/audit/$1.yaml" run; run=$(run_name "$cfg")
  if grep -q '"served_model": ".*merged' "$RAW/$run.meta.json" 2>/dev/null; then
    say "skip audit $run (already from merged weights)"; return 0; fi
  for f in "$RAW/$run.parquet" "$RAW/$run.meta.json"; do [ -f "$f" ] && mv "$f" "$OLD_RAW/"; done
  [ -f "eval/results/$run.json" ] && mv "eval/results/$run.json" "$OLD_RES/"
  say "start audit $run"
  if $PY scripts/run_audit.py --config "$cfg"; then say "done audit $run"; else say "FAILED audit $run"; return 1; fi
}

op_control() {
  local cfg="configs/audit/$1.yaml"
  say "start op-control $1"
  $PY scripts/operating_point_control.py margins --config "$cfg" --which base &&
  $PY scripts/operating_point_control.py margins --config "$cfg" --which adapter &&
  $PY scripts/operating_point_control.py compare --config "$cfg" &&
  say "done op-control $1" || say "FAILED op-control $1"
}

train() {  # $1 = training config stem under configs/mitigation/sft
  local cfg="configs/mitigation/sft/$1.yaml" out
  out=$($PY -c "import yaml;print(yaml.safe_load(open('$cfg'))['output_dir'])")
  if [ -f "$HBM_OUTPUT_ROOT/$out/adapter_config.json" ]; then
    say "skip train $1 (adapter exists)"; return 0; fi
  say "start train $1"
  if $PY -m hiring_bias_mitigation.mitigation.sft --config "$cfg"; then
    say "done train $1"
  else
    say "FAILED train $1"; return 1
  fi
}

reports() { $PY scripts/set_stability.py >/dev/null 2>&1; $PY scripts/make_report.py >/dev/null 2>&1; }

# 1-2. the Ukrainian cells, on corrected targets
train qwen3.5-9b_uk_only && audit qwen3.5-9b_uk_sft_uk_only && op_control qwen3.5-9b_uk_sft_uk_only
reports; say "9B uk retrained and audited"
train qwen3.5-4b_uk_only && audit qwen3.5-4b_uk_sft_uk_only
reports; say "uk retrain done"

# 3. LAPA, the extension
for lang in en uk; do
  adapter="outputs/sft/lapa-12b_${lang}_only"
  if [ ! -f "$HBM_OUTPUT_ROOT/$adapter/adapter_config.json" ]; then
    say "start smoke lapa $lang"
    $PY scripts/preflight_training.py --smoke --config "configs/mitigation/sft/lapa-12b_${lang}_only.yaml" \
      || { say "FAILED smoke lapa $lang"; continue; }
  fi
  train "lapa-12b_${lang}_only" || continue
  say "start validate lapa $lang"
  merged=$($PY scripts/merge_adapter.py --adapter "$adapter" | tail -1)
  $PY scripts/diagnose_adapter.py --base-model lapa-llm/lapa-v0.1.2-instruct --adapter "$adapter" \
      --lang "$lang" --audit-run "lapa-v0.1.2-instruct--${lang}--baseline" --vllm-merged "$merged" \
    && say "done validate lapa $lang" || say "FAILED validate lapa $lang"
  audit "lapa-12b_${lang}_sft_${lang}_only" && op_control "lapa-12b_${lang}_sft_${lang}_only"
  reports
done
say "lapa done"

# 4. internal probes, not reported in the paper
audit qwen3.5-9b_en_dpo_en_only_dpo_decision
audit qwen3.5-9b_en_sft_en_only_v2_ckpt250
audit qwen3.5-9b_en_dpo_en_only_dpo_v2_ckpt50
reports
say "queue finished"
