#!/usr/bin/env bash
# Consistency-target DPO probe on Qwen3.5-9B en:
#   student pass over the synthetic pool -> consistency pairs -> smoke -> train -> audit
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f ./.env ] && . ./.env; set +a
export PATH="$PWD/.venv/bin:$PATH"

CFG=configs/mitigation/dpo/qwen3.5-9b_en_only_dpo_consistency.yaml
AUDIT=configs/audit/qwen3.5-9b_en_dpo_en_only_dpo_consistency.yaml
mkdir -p logs
LOG="logs/dpo-consistency-$(date +%Y%m%d-%H%M%S).log"
echo "$LOG" > logs/dpoconsistency.logpath
exec > >(tee -a "$LOG") 2>&1

until [ "$(pgrep -x python | wc -l)" -eq 0 ]; do sleep 20; done

echo "############ build pairs $(date +%H:%M:%S) ############"
python scripts/build_consistency_pairs.py --model Qwen/Qwen3.5-9B --lang en \
  || { echo "!! pair build failed"; exit 1; }
until [ "$(pgrep -x python | wc -l)" -eq 0 ]; do sleep 10; done

rm -rf "$HBM_OUTPUT_ROOT/outputs/_preflight_smoke" "$HBM_OUTPUT_ROOT/outputs/dpo/qwen3.5-9b_en_only_dpo_consistency"
SMOKE="logs/dpo-consistency-smoke.log"
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
echo ">>> consistency DPO probe done <<<"
