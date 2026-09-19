#!/usr/bin/env bash
# Waits for the semi-synthetic data generation to finish, then runs mitigation phase 1 in
# cost order, refreshing reports/RESULTS.md after every single run.
#
#   ./scripts/run_mitigation_phase1.sh              # wait, then run everything enabled
#   ./scripts/run_mitigation_phase1.sh --no-wait    # start immediately
#   ./scripts/run_mitigation_phase1.sh --stages prompt,scrub
#
# Order is deliberate. The zero-training arms come first: they cost nothing to run and they
# set the bar the training arms have to beat, so a training result is only interpretable once
# they are on the board. SFT precedes DPO because each DPO config continues from its SFT
# checkpoint. The trained adapters are audited last -- training produces weights, and the
# fairness numbers come from measuring them.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f ./.env ] && . ./.env; set +a
export PATH="$PWD/.venv/bin:$PATH" HBM_KEEP_GOING=1

STAGES="prompt,scrub,embedding,sft,dpo,audit"
WAIT=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stages) STAGES="$2"; shift 2 ;;
    --no-wait) WAIT=0; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

mkdir -p logs
LOG="logs/mitigation-$(date +%Y%m%d-%H%M%S).log"
echo "$LOG" > logs/mitigation.logpath
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG"
echo "stages: $STAGES"

# Match on the script path in a process's own /proc argv rather than `pgrep -f <name>`: a
# substring match over full command lines also hits the shell that invoked this check, whose
# own command line contains the name.
#
# Excluding `$$` alone is not enough. Every subshell this script forks inherits its argv under a
# fresh pid (the process *group* catches those), and every ancestor shell that was launched with
# this script's name on its command line carries the watched strings too (the ancestor walk
# catches those). Both classes have to go, or the script waits forever on itself.
#
# All three names matter, and the last two are why this is not merely a wait on the generation
# process: run_generation.sh calls assemble_datasets.py once generation exits, and it is that
# step which writes sft_train/dpo_train. Releasing at generation exit would start the training
# arms against datasets that are still being written.
SELF_PGID=$(ps -o pgid= -p $$ | tr -d ' ')
SELF_LINE=" $$ "
_p=$$
while [ -n "$_p" ] && [ "$_p" != "0" ] && [ "$_p" != "1" ]; do
  _p=$(awk '/^PPid:/{print $2}' "/proc/$_p/status" 2>/dev/null)
  [ -n "$_p" ] && SELF_LINE="$SELF_LINE$_p "
done
unset _p

data_generation_running() {
  local pid cmd pgid
  for pid in $(pgrep -x python 2>/dev/null; pgrep -x bash 2>/dev/null); do
    case "$SELF_LINE" in *" $pid "*) continue ;; esac
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$pgid" ] && [ "$pgid" = "$SELF_PGID" ] && continue
    # A pid can exit between pgrep and the read; suppress the shell's own redirect error.
    cmd=$( { tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null ) || continue
    case "$cmd" in
      *generate_training_data.py*|*assemble_datasets.py*|*run_generation.sh*) return 0 ;;
    esac
  done
  return 1
}

if [ "$WAIT" = "1" ]; then
  while data_generation_running; do
    echo "$(date +%H:%M:%S) waiting for data generation to finish..."
    sleep 600
  done
  echo "$(date +%H:%M:%S) data generation finished"
fi

# The training arms cannot start without the datasets; the inference arms can.
need_data=0
case ",$STAGES," in *,sft,*|*,dpo,*) need_data=1 ;; esac
if [ "$need_data" = "1" ]; then
  for f in sft_train dpo_train; do
    if [ ! -f "$HBM_OUTPUT_ROOT/artifacts/semisynthetic-v1/$f.parquet" ]; then
      echo "!! $f.parquet missing — run scripts/assemble_datasets.py first" >&2
      exit 1
    fi
  done
fi

run_stage() {
  local stage="$1" script="$2"
  case ",$STAGES," in *",$stage,"*) ;; *) echo "skipping $stage"; return 0 ;; esac
  local enabled
  enabled=$(grep -c '^  configs/' "$script" || true)
  if [ "$enabled" -eq 0 ]; then
    echo "=== $stage: nothing enabled in $script — skipping ==="
    return 0
  fi
  echo
  echo "############ $stage — $enabled run(s) — $(date +%H:%M:%S) ############"
  bash "$script" || echo "!! $stage had failures — continuing" >&2
}

run_stage prompt    scripts/run_all_prompt.sh
run_stage scrub     scripts/run_all_scrub.sh
run_stage embedding scripts/run_all_embedding.sh
run_stage sft       scripts/run_all_sft.sh
run_stage dpo       scripts/run_all_dpo.sh
run_stage audit     scripts/run_all_audit.sh

python scripts/make_report.py
python scripts/analyze_results.py --json reports/analysis.json \
  --eval-scope targeted-no-intersections || true
echo
echo "PHASE 1 COMPLETE — see reports/RESULTS.md"
