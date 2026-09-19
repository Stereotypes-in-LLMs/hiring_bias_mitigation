#!/usr/bin/env bash
# Waits for the phase-1 orchestrator to exit, then runs whatever is listed in
# reports/pending_reruns.txt and refreshes the report.
#
# Why a separate process rather than an extra stage in the orchestrator: bash reads a script
# incrementally as it executes, so editing run_mitigation_phase1.sh while it is running can
# corrupt the rest of the queue. The same applies to the run_all_*.sh runners -- their CONFIGS
# array was fixed the moment bash parsed it, which is exactly why a config pulled out mid-stage
# can no longer be reached from inside that stage.
#
# Waiting for the whole queue (not just the embedding stage) keeps this off the GPU while the
# training stages are using it.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f ./.env ] && . ./.env; set +a
export PATH="$PWD/.venv/bin:$PATH"

mkdir -p logs
LOG="logs/pending-$(date +%Y%m%d-%H%M%S).log"
echo "$LOG" > logs/pending.logpath
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG"

SELF_PGID=$(ps -o pgid= -p $$ | tr -d ' ')
queue_running() {
  local pid cmd pgid
  for pid in $(pgrep -x bash 2>/dev/null); do
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$pgid" ] && [ "$pgid" = "$SELF_PGID" ] && continue
    cmd=$( { tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null ) || continue
    case "$cmd" in *run_mitigation_phase1.sh*) return 0 ;; esac
  done
  return 1
}

while queue_running; do
  echo "$(date +%H:%M:%S) phase-1 queue still running; waiting..."
  sleep 900
done
echo "$(date +%H:%M:%S) queue finished — running pending reruns"

bash reports/rerun_pending.sh
python scripts/make_report.py
python scripts/analyze_results.py --json reports/analysis.json \
  --eval-scope targeted-no-intersections || true
echo "PENDING RERUNS COMPLETE — reports/RESULTS.md refreshed"
