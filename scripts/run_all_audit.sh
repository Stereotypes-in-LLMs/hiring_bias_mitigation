#!/usr/bin/env bash
# Runs the configs enabled below. This CONFIGS array is the SINGLE SOURCE OF TRUTH for what
# runs -- `make audit`, the docker-compose service and quickstart.sh all execute this
# script, so what you uncomment here is exactly what runs, however you launch it.
#
# Set a preset instead of editing by hand (rewrites every runner's array):
#     python scripts/generate_experiment_configs.py --enable stage1
#     python scripts/generate_experiment_configs.py --enable core
#     python scripts/generate_experiment_configs.py --enable full
#     python scripts/generate_experiment_configs.py --enable none
#
# Gated models (google/gemma-4-*) are never auto-enabled -- accept the licence, set HF_TOKEN,
# then uncomment them here yourself.
# Currently enabled: 6/41
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
#   configs/audit/smoke.yaml
#   configs/audit/gemma-4-e4b_en_baseline.yaml
#   configs/audit/gemma-4-e4b_en_dpo_en_only_dpo.yaml
#   configs/audit/gemma-4-e4b_en_dpo_en_only_kto.yaml
#   configs/audit/gemma-4-e4b_en_sft_en_only.yaml
#   configs/audit/gemma-4-e4b_uk_baseline.yaml
#   configs/audit/gemma-4-e4b_uk_dpo_uk_only_dpo.yaml
#   configs/audit/gemma-4-e4b_uk_dpo_uk_only_kto.yaml
#   configs/audit/gemma-4-e4b_uk_sft_uk_only.yaml
#   configs/audit/qwen3.5-4b_en_baseline.yaml
#   configs/audit/qwen3.5-4b_en_dpo_en_only_dpo.yaml
#   configs/audit/qwen3.5-4b_en_dpo_en_only_kto.yaml
  configs/audit/qwen3.5-4b_en_sft_en_only.yaml
#   configs/audit/qwen3.5-4b_uk_baseline.yaml
#   configs/audit/qwen3.5-4b_uk_dpo_uk_only_dpo.yaml
#   configs/audit/qwen3.5-4b_uk_dpo_uk_only_kto.yaml
  configs/audit/qwen3.5-4b_uk_sft_uk_only.yaml
#   configs/audit/qwen3.5-9b_en_baseline.yaml
#   configs/audit/qwen3.5-9b_en_dpo_en_only_dpo.yaml
#   configs/audit/qwen3.5-9b_en_dpo_en_only_kto.yaml
  configs/audit/qwen3.5-9b_en_sft_en_only.yaml
#   configs/audit/qwen3.5-9b_uk_baseline.yaml
#   configs/audit/qwen3.5-9b_uk_dpo_uk_only_dpo.yaml
#   configs/audit/qwen3.5-9b_uk_dpo_uk_only_kto.yaml
  configs/audit/qwen3.5-9b_uk_sft_uk_only.yaml
#   configs/audit/gemma-4-12b_en_baseline.yaml
#   configs/audit/gemma-4-12b_en_dpo_en_only_dpo.yaml
#   configs/audit/gemma-4-12b_en_dpo_en_only_kto.yaml
#   configs/audit/gemma-4-12b_en_sft_en_only.yaml
#   configs/audit/gemma-4-12b_uk_baseline.yaml
#   configs/audit/gemma-4-12b_uk_dpo_uk_only_dpo.yaml
#   configs/audit/gemma-4-12b_uk_dpo_uk_only_kto.yaml
#   configs/audit/gemma-4-12b_uk_sft_uk_only.yaml
#   configs/audit/lapa-12b_en_baseline.yaml
#   configs/audit/lapa-12b_en_dpo_en_only_dpo.yaml
#   configs/audit/lapa-12b_en_dpo_en_only_kto.yaml
  configs/audit/lapa-12b_en_sft_en_only.yaml
#   configs/audit/lapa-12b_uk_baseline.yaml
#   configs/audit/lapa-12b_uk_dpo_uk_only_dpo.yaml
#   configs/audit/lapa-12b_uk_dpo_uk_only_kto.yaml
  configs/audit/lapa-12b_uk_sft_uk_only.yaml
)

if [ ${#CONFIGS[@]} -eq 0 ]; then
  echo "No configs enabled in $0." >&2
  echo "Uncomment entries in the CONFIGS array, or run:" >&2
  echo "    python scripts/generate_experiment_configs.py --enable core" >&2
  exit 1
fi

# A full sweep is hours to days. By default it stops at the first failure, so a
# misconfiguration does not burn the rest of the queue producing the same broken output.
# Pass --keep-going (or set HBM_KEEP_GOING=1) for an unattended run: failures are recorded
# and reported at the end, and the remaining configs still run.
KEEP_GOING="${HBM_KEEP_GOING:-0}"
[ "${1:-}" = "--keep-going" ] && KEEP_GOING=1

echo "AUDIT: ${#CONFIGS[@]} config(s) enabled (keep-going=$KEEP_GOING)"
FAILED=()
DONE=0
for cfg in "${CONFIGS[@]}"; do
  DONE=$((DONE + 1))
  echo "=== AUDIT [$DONE/${#CONFIGS[@]}]: $cfg ==="
  if python scripts/run_audit.py --config "$cfg"; then
    # Refresh the report after every run, not once at the end. A sweep is days long, and a
    # result that only lands when the whole queue finishes is a result nobody can act on --
    # including the decision to stop the queue because an arm is clearly not working.
    python scripts/make_report.py >/dev/null 2>&1       && echo "--- reports/RESULTS.md updated ($DONE/${#CONFIGS[@]} done) ---"
  else
    if [ "$KEEP_GOING" = "1" ]; then
      echo "!! FAILED: $cfg -- continuing" >&2
      FAILED+=("$cfg")
    else
      echo "!! FAILED: $cfg -- stopping. Re-run with --keep-going to skip failures." >&2
      exit 1
    fi
  fi
done

python scripts/make_report.py >/dev/null 2>&1 || true

if [ ${#FAILED[@]} -gt 0 ]; then
  echo >&2
  echo "AUDIT: ${#FAILED[@]} of ${#CONFIGS[@]} config(s) FAILED:" >&2
  printf '  %s
' "${FAILED[@]}" >&2
  exit 1
fi
