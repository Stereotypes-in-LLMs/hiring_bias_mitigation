#!/usr/bin/env bash
# Re-runs SFT -> DPO -> adapter audits after the warmup_ratio fix.
#
# The first attempt died on `TrainingArguments.__init__() got an unexpected keyword argument
# 'warmup_ratio'` (transformers 5.x dropped it): all 16 training runs failed in 90 seconds,
# and the 16 audits then spent six hours loading base models to evaluate adapters that were
# never written. Both are fixed; this drains the stages again.
#
# Waits for the pending-rerun watcher so the LEACE runs keep the GPU to themselves.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f ./.env ] && . ./.env; set +a
export PATH="$PWD/.venv/bin:$PATH" HBM_KEEP_GOING=1

mkdir -p logs
LOG="logs/retrain-$(date +%Y%m%d-%H%M%S).log"
echo "$LOG" > logs/retrain.logpath
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG"

SELF_PGID=$(ps -o pgid= -p $$ | tr -d ' ')
busy() {
  local pid cmd pgid
  for pid in $(pgrep -x bash 2>/dev/null; pgrep -x python 2>/dev/null); do
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$pgid" ] && [ "$pgid" = "$SELF_PGID" ] && continue
    cmd=$( { tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null ) || continue
    case "$cmd" in
      *run_pending_after_queue.sh*|*rerun_pending.sh*|*fit_eraser.py*) return 0 ;;
      *run_audit.py*embedding*) return 0 ;;
    esac
  done
  return 1
}

while busy; do
  echo "$(date +%H:%M:%S) LEACE re-runs still going; waiting..."
  sleep 600
done
echo "$(date +%H:%M:%S) GPU free — restarting the training stages"

# Preflight: build the real TrainingArguments for every enabled config before spending a GPU
# hour on the first one. This is the check that was missing when the stage collapsed.
# --smoke runs one real training step on eight rows. Constructing the config objects is not
# enough: TRL validates trainer arguments against each other inside the *trainer*
# constructor, which is how a formatting_func incompatible with completion_only_loss passed a
# green preflight and then failed all six runs.
python scripts/preflight_training.py --smoke || {
  echo "!! preflight failed — not starting" >&2; exit 1; }

for stage in sft dpo audit; do
  n=$(grep -cE '^  configs/' "scripts/run_all_$stage.sh")
  echo
  echo "############ $stage — $n run(s) — $(date +%H:%M:%S) ############"
  bash "scripts/run_all_$stage.sh" || echo "!! $stage had failures — continuing" >&2
done

python scripts/make_report.py
python scripts/analyze_results.py --json reports/analysis.json \
  --eval-scope targeted-no-intersections || true
echo "TRAINING STAGES COMPLETE — see reports/RESULTS.md"
