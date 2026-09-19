#!/usr/bin/env bash
# End-to-end run: benchmark -> audit -> report, then optionally the mitigation stages.
#
#   ./scripts/quickstart.sh --tier smoke     # 20 pairs, one model, minutes -- proves plumbing
#   ./scripts/quickstart.sh --tier stage1    # the baseline audit. START HERE.
#   ./scripts/quickstart.sh --tier core      # stage1 + the zero-training mitigations
#   ./scripts/quickstart.sh --tier full      # everything, including generation and training
#   ./scripts/quickstart.sh                  # whatever is currently uncommented
#
# --tier applies a preset, which OVERWRITES the manual selection in every
# scripts/run_all_*.sh. Without it, your current selection is used unchanged.
#
# Deliberately, `stage1` stops after the audit. Which models, groups and languages deserve a
# mitigation run is a decision to make by reading reports/RESULTS.md and the manual-review
# queue -- not one to guess in advance, and not one this script should make for you.
set -euo pipefail
cd "$(dirname "$0")/.."

TIER=""
KEEP_GOING=0
LOCAL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --keep-going) KEEP_GOING=1; shift ;;
    --local) LOCAL=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

mkdir -p logs
LOG="logs/quickstart-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "logging to $LOG"

if [[ -n "$TIER" ]]; then
  echo "=== applying preset: $TIER (this overwrites the CONFIGS selection) ==="
  python scripts/generate_experiment_configs.py --enable "$TIER"
fi

run() {
  local label="$1"; shift
  echo; echo "=================== $label ==================="
  if [[ $KEEP_GOING -eq 1 ]]; then
    "$@" || echo "!! $label failed, continuing (--keep-going)" >&2
  else
    "$@"
  fi
}

if [[ $LOCAL -eq 0 ]]; then
  docker compose build
  RUNNER=(docker compose run --rm)
  run "benchmark" "${RUNNER[@]}" benchmark
  run "audit (stage 1)" "${RUNNER[@]}" audit
  run "report" "${RUNNER[@]}" report
  if [[ "$TIER" == "core" || "$TIER" == "full" ]]; then
    run "prompt mitigations" "${RUNNER[@]}" prompt
    run "scrub mitigations" "${RUNNER[@]}" scrub
  fi
  if [[ "$TIER" == "full" ]]; then
    run "training-data generation" "${RUNNER[@]}" generate
    run "embedding mitigations" "${RUNNER[@]}" embedding
    run "SFT" "${RUNNER[@]}" sft
    run "preference optimisation" "${RUNNER[@]}" dpo
  fi
  run "report" "${RUNNER[@]}" report
else
  run "benchmark" python scripts/build_benchmark.py
  run "audit (stage 1)" bash scripts/run_all_audit.sh
  run "report" python scripts/make_report.py
  if [[ "$TIER" == "core" || "$TIER" == "full" ]]; then
    run "prompt mitigations" bash scripts/run_all_prompt.sh
    run "scrub mitigations" bash scripts/run_all_scrub.sh
  fi
  if [[ "$TIER" == "full" ]]; then
    run "training-data generation" python scripts/generate_training_data.py --config configs/generation/teacher.yaml
    run "embedding mitigations" bash scripts/run_all_embedding.sh
    run "SFT" bash scripts/run_all_sft.sh
    run "preference optimisation" bash scripts/run_all_dpo.sh
  fi
  run "report" python scripts/make_report.py
fi

echo
echo "================================================================"
cat reports/RESULTS.md
echo "================================================================"
echo "Full report: reports/RESULTS.md"
echo "Rows a human still needs to read: reports/manual_review/review_queue.csv"
