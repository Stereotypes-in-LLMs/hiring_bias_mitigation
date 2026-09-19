#!/usr/bin/env bash
# Decision-only DPO probe on Qwen3.5-9B en, queued behind the running DPO checkpoint-50 audit.
#   smoke on the real config -> train -> audit -> report
# Every GPU stage waits for the previous one; nothing here overlaps a running job.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f ./.env ] && . ./.env; set +a
export PATH="$PWD/.venv/bin:$PATH"

CFG=configs/mitigation/dpo/qwen3.5-9b_en_only_dpo_decision.yaml
AUDIT=configs/audit/qwen3.5-9b_en_dpo_en_only_dpo_decision.yaml
mkdir -p logs
LOG="logs/dpo-decision-$(date +%Y%m%d-%H%M%S).log"
echo "$LOG" > logs/dpodecision.logpath
exec > >(tee -a "$LOG") 2>&1

echo "$(date +%H:%M:%S) waiting for the DPO checkpoint-50 audit..."
until grep -qaE "DPO ckpt50 audit done" "$(cat logs/auditdpo.logpath)" 2>/dev/null; do sleep 60; done
until [ "$(pgrep -x python | wc -l)" -eq 0 ]; do sleep 20; done
echo "$(date +%H:%M:%S) GPU free"

# Smoke. The preflight process lingers after it succeeds, so watch its log and end it by
# exact process name once the verdict is written -- never by a pattern that could match this
# script's own command line.
rm -rf "$HBM_OUTPUT_ROOT/outputs/_preflight_smoke" "$HBM_OUTPUT_ROOT/outputs/dpo/qwen3.5-9b_en_only_dpo_decision"
SMOKE="logs/dpo-decision-smoke.log"
python scripts/preflight_training.py --smoke --config "$CFG" > "$SMOKE" 2>&1 &
until grep -qaE "smoke ok|Traceback|Killed" "$SMOKE"; do sleep 10; done
grep -qa "smoke ok" "$SMOKE" || { echo "!! smoke failed:"; tail -30 "$SMOKE"; exit 1; }
echo "smoke ok; mismatch warnings: $(grep -ac 'Mismatch between tokenized prompt' "$SMOKE")"
for p in $(pgrep -x python); do kill "$p"; done
until [ "$(pgrep -x python | wc -l)" -eq 0 ]; do sleep 5; done
rm -rf "$HBM_OUTPUT_ROOT/outputs/_preflight_smoke"

echo "############ train $(date +%H:%M:%S) ############"
python -m hiring_bias_mitigation.mitigation.dpo --config "$CFG" || { echo "!! training failed"; exit 1; }
echo "############ audit $(date +%H:%M:%S) ############"
python scripts/run_audit.py --config "$AUDIT" && python scripts/make_report.py
echo ">>> decision-only DPO probe done <<<"
