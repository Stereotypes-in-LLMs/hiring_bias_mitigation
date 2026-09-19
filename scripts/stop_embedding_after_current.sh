#!/usr/bin/env bash
# Ends the embedding stage after the run in flight, keeping LEACE only.
#
# Editing scripts/run_all_embedding.sh is not an option while it executes: bash reads a script
# incrementally, and its CONFIGS array was fixed the moment it was parsed. So the stage is
# stopped from outside instead -- the orchestrator sees the runner exit, logs it, and moves on
# to the next stage. The LEACE configs that never got their turn go to pending_reruns.txt,
# which the post-queue watcher already knows how to drain.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"

KEEP="qwen3.5-4b_en_leace"          # the run currently in flight
mkdir -p logs
LOG="logs/stop-embedding-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG"

pid_for() {  # pid of a python process whose argv contains $1
  local pid cmd
  for pid in $(pgrep -x python 2>/dev/null); do
    cmd=$( { tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null ) || continue
    case "$cmd" in *"$1"*) echo "$pid"; return 0 ;; esac
  done
  return 1
}

echo "$(date +%H:%M:%S) waiting for $KEEP to finish..."
while pid_for "$KEEP" >/dev/null; do sleep 30; done
echo "$(date +%H:%M:%S) $KEEP done"

# Kill the stage runner first so it cannot launch another config, then anything it already
# started. Order matters: the reverse leaves the runner free to start the next one.
for pid in $(pgrep -f "run_all_embedding.sh" 2>/dev/null); do
  cmd=$( { tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null )
  case "$cmd" in *run_all_embedding.sh*) echo "killing runner $pid"; kill "$pid" ;; esac
done
sleep 3
for name in fit_eraser.py run_audit.py; do
  while p=$(pid_for "$name"); do
    cmd=$( { tr '\0' ' ' < "/proc/$p/cmdline"; } 2>/dev/null )
    case "$cmd" in
      *embedding*) echo "killing leftover $name ($p)"; kill "$p"; sleep 3 ;;
      *) break ;;
    esac
  done
done
echo "$(date +%H:%M:%S) embedding stage ended; orchestrator continues with SFT"
