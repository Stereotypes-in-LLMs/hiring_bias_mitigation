#!/usr/bin/env bash
# Reorders the tail of the sweep: audit the four Qwen adapters first, then train and audit
# LAPA. Puts readable results on the board a day earlier than the default order, which trains
# everything before measuring anything.
#
# Runs configs directly rather than through run_all_*.sh: the stage runners take a fixed
# CONFIGS array, and splitting one stage in two is exactly what they cannot express.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f ./.env ] && . ./.env; set +a
export PATH="$PWD/.venv/bin:$PATH"

mkdir -p logs
LOG="logs/reorder-$(date +%Y%m%d-%H%M%S).log"
echo "$LOG" > logs/reorder.logpath
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG"

QWEN_AUDITS=(
  configs/audit/qwen3.5-4b_en_sft_en_only.yaml
  configs/audit/qwen3.5-4b_uk_sft_uk_only.yaml
  configs/audit/qwen3.5-9b_en_sft_en_only.yaml
  configs/audit/qwen3.5-9b_uk_sft_uk_only.yaml
)
LAPA_SFT=(
  configs/mitigation/sft/lapa-12b_en_only.yaml
  configs/mitigation/sft/lapa-12b_uk_only.yaml
)
LAPA_AUDITS=(
  configs/audit/lapa-12b_en_sft_en_only.yaml
  configs/audit/lapa-12b_uk_sft_uk_only.yaml
)

running() {  # $1 substring; true if a python process carries it
  local pid cmd
  for pid in $(pgrep -x python 2>/dev/null); do
    cmd=$( { tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null ) || continue
    case "$cmd" in *"$1"*) return 0 ;; esac
  done
  return 1
}

# 1. Let the 9B Ukrainian run finish, then take the queue over before it reaches LAPA.
echo "$(date +%H:%M:%S) waiting for qwen3.5-9b_uk_only to finish..."
while running "sft/qwen3.5-9b_uk_only"; do sleep 20; done
echo "$(date +%H:%M:%S) 9B uk done — stopping the default stage order"

for p in $(pgrep -x bash 2>/dev/null); do
  mapfile -d '' -t argv < "/proc/$p/cmdline" 2>/dev/null || continue
  case "${argv[1]:-}" in
    */rerun_training_stages.sh|*/run_all_sft.sh) echo "  stopping ${argv[1]##*/}"; kill "$p" ;;
  esac
done
sleep 5
# The runner may already have launched LAPA in the seconds after 9B finished; it is re-run
# below in its proper place, so killing it here costs minutes, not work.
for pid in $(pgrep -x python 2>/dev/null); do
  cmd=$( { tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null ) || continue
  case "$cmd" in *mitigation.sft*) echo "  stopping premature LAPA start"; kill "$pid" ;; esac
done
sleep 8

run_all() {
  local label="$1"; shift
  local -a cfgs=("$@") ; local i=0
  echo
  echo "############ $label — ${#cfgs[@]} run(s) — $(date +%H:%M:%S) ############"
  for cfg in "${cfgs[@]}"; do
    i=$((i+1))
    echo "=== $label [$i/${#cfgs[@]}]: $cfg ==="
    case "$cfg" in
      configs/audit/*) python scripts/run_audit.py --config "$cfg" ;;
      *)               python -m hiring_bias_mitigation.mitigation.sft --config "$cfg" ;;
    esac || echo "!! FAILED: $cfg — continuing" >&2
    python scripts/make_report.py >/dev/null 2>&1 \
      && echo "--- reports/RESULTS.md updated ---"
  done
}

run_all "QWEN AUDIT" "${QWEN_AUDITS[@]}"
python scripts/analyze_results.py --json reports/analysis.json \
  --eval-scope targeted-no-intersections || true
echo
echo ">>> Qwen SFT results are in reports/RESULTS.md — LAPA starts now <<<"

python scripts/preflight_training.py || echo "!! preflight failed" >&2
run_all "LAPA SFT"   "${LAPA_SFT[@]}"
run_all "LAPA AUDIT" "${LAPA_AUDITS[@]}"

python scripts/make_report.py
python scripts/analyze_results.py --json reports/analysis.json \
  --eval-scope targeted-no-intersections || true
echo "ALL SFT RESULTS COMPLETE — see reports/RESULTS.md"
