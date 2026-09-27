#!/usr/bin/env bash
# Repeats the three prompt strategies the paper leans on, on Qwen3.5-9B English, over the full
# grid (all three groups, explicit + implicit) instead of the scoped military-status-implicit
# cell the original runs used. Makes the prompt-versus-SFT comparison in that cell like for
# like with every other cell; the scoped runs are kept under their own names.
#
#   nohup bash scripts/run_9b_en_fullscope.sh > logs/9b_en_fullscope.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PATH="$PWD/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { echo "$(date '+%F %T') $*"; }
for s in structured_rubric ignore_personal_info second_pass_verification; do
  # Resume after a crash (the machine lost power mid-run once): a finished audit leaves its
  # generations behind, and re-running it would cost two hours to reproduce the same file.
  run="Qwen3.5-9B--en--prompt--${s}--fullscope"
  if [ -f "$HBM_OUTPUT_ROOT/outputs/raw/$run.parquet" ]; then
    say "skip $s (already audited)"; continue
  fi
  say "start $s"
  if .venv/bin/python scripts/run_audit.py --config "configs/mitigation/prompt/qwen3.5-9b_en_${s}_fullscope.yaml"; then
    say "done $s"
  else
    say "FAILED $s"
  fi
done
.venv/bin/python scripts/set_stability.py >/dev/null 2>&1
.venv/bin/python scripts/make_report.py >/dev/null 2>&1
say "fullscope finished"
