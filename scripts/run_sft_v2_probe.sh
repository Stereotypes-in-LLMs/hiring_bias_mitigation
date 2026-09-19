#!/usr/bin/env bash
# Trains and audits the v2 SFT variant on Qwen3.5-9B en, then hands the GPU back.
#
# Slots in after the four Qwen audits and before LAPA: the v1 result (writing changed,
# decisions did not) is a reason to check whether SFT can move decisions at all before
# spending another ~30 GPU-hours training the 12B the same way.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f ./.env ] && . ./.env; set +a
export PATH="$PWD/.venv/bin:$PATH"

mkdir -p logs
LOG="logs/sftv2-$(date +%Y%m%d-%H%M%S).log"
echo "$LOG" > logs/sftv2.logpath
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG"

SELF_PGID=$(ps -o pgid= -p $$ | tr -d ' ')
gpu_busy() {
  local pid cmd pgid
  for pid in $(pgrep -x python 2>/dev/null); do
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$pgid" ] && [ "$pgid" = "$SELF_PGID" ] && continue
    cmd=$( { tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null ) || continue
    case "$cmd" in *run_audit.py*|*mitigation.sft*|*fit_eraser.py*) return 0 ;; esac
  done
  return 1
}

# Wait for the four Qwen audits, then take the slot before the sequencer starts LAPA.
echo "$(date +%H:%M:%S) waiting for the Qwen audits to finish..."
while ! grep -qa "Qwen SFT results are in" "$(cat logs/reorder.logpath)" 2>/dev/null; do
  sleep 120
done
echo "$(date +%H:%M:%S) audits done — pausing the LAPA sequencer"

for p in $(pgrep -x bash 2>/dev/null); do
  mapfile -d '' -t argv < "/proc/$p/cmdline" 2>/dev/null || continue
  case "${argv[1]:-}" in
    */audit_qwen_then_lapa.sh) echo "  stopping ${argv[1]##*/}"; kill "$p" ;;
  esac
done
sleep 5
for pid in $(pgrep -x python 2>/dev/null); do
  cmd=$( { tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null ) || continue
  case "$cmd" in *mitigation.sft*) echo "  stopping premature LAPA start"; kill "$pid" ;; esac
done
while gpu_busy; do sleep 30; done

echo
echo "############ SFT v2 probe — $(date +%H:%M:%S) ############"
python -m hiring_bias_mitigation.mitigation.sft \
  --config configs/mitigation/sft/qwen3.5-9b_en_only_v2.yaml \
  && python scripts/run_audit.py --config configs/audit/qwen3.5-9b_en_sft_en_only_v2.yaml \
  && python scripts/make_report.py

echo
echo ">>> v2 probe done — compare the two sft rows for Qwen3.5-9B en in reports/RESULTS.md <<<"
echo ">>> LAPA has NOT been started; run scripts/audit_qwen_then_lapa.sh again to continue <<<"
