#!/usr/bin/env bash
# Re-runs configs listed in reports/pending_reruns.txt, then refreshes the report.
# These were killed mid-stage (a bad config, an interrupted run), so the stage's own
# CONFIGS array -- fixed when bash parsed the runner -- can no longer reach them.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f ./.env ] && . ./.env; set +a
export PATH="$PWD/.venv/bin:$PATH"

mapfile -t CFGS < <(grep -vE '^\s*(#|$)' reports/pending_reruns.txt)
[ ${#CFGS[@]} -eq 0 ] && { echo "nothing pending"; exit 0; }

for cfg in "${CFGS[@]}"; do
  echo "=== RERUN: $cfg ==="
  case "$cfg" in
    */embedding/*) python scripts/fit_eraser.py --config "$cfg" || continue ;;
  esac
  python scripts/run_audit.py --config "$cfg" && python scripts/make_report.py >/dev/null
done
