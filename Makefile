# =========================================================================================
# Hiring Bias Mitigation -- Makefile
# =========================================================================================
# The `local-*` targets run against a uv-managed venv (fast iteration, no image rebuild);
# the plain targets run the same thing inside Docker so it behaves identically anywhere.
#
# The pipeline has an order, and it is not arbitrary:
#   benchmark -> audit (STAGE 1) -> [you decide what to mitigate] -> generate -> mitigate
# Stage 1 exists to be read. Which models, groups and languages deserve a mitigation run is
# a decision made from its output, not one guessed in advance -- see README "Workflow".

COMPOSE = docker compose

.PHONY: help benchmark audit generate prompt scrub embedding sft dpo report analyze smoke \
        build shell down status watch progress \
        enable-none enable-smoke enable-stage1 enable-core enable-full \
        local-setup local-benchmark local-audit local-generate local-prompt local-scrub \
        local-embedding local-sft local-dpo local-report local-analyze local-plan local-test lint

help:
	@echo "Pipeline (run in this order):"
	@echo "  make benchmark   fetch + validate the golden benchmark, write the holdout"
	@echo "  make audit       STAGE 1 -- baseline disparity for every enabled model x language"
	@echo "  make report      regenerate reports/RESULTS.md and the manual-review sheets"
	@echo "  make analyze     grade the evidence and produce the mitigation shortlist"
	@echo "  make plan        enumerate every planned run into reports/EXPERIMENT_PLAN.md"
	@echo "  ---- then choose what to mitigate, from the report ----"
	@echo "  make generate    build the semi-synthetic training data with the teacher model"
	@echo "  make prompt      prompt-based mitigations"
	@echo "  make scrub       attribute-scrubbing mitigations"
	@echo "  make embedding   fit erasers, then evaluate (transformers backend)"
	@echo "  make sft         SFT on counterfactually-consistent data"
	@echo "  make dpo         preference optimisation (needs sft first for the DPO configs)"
	@echo ""
	@echo "Selecting experiments (rewrites every scripts/run_all_*.sh CONFIGS array):"
	@echo "  make enable-smoke | enable-stage1 | enable-core | enable-full | enable-none"
	@echo ""
	@echo "Prefix any pipeline target with local- to run it outside Docker."

# ---- Experiment selection ---------------------------------------------------------------
enable-%:
	uv run python scripts/generate_experiment_configs.py --enable $*

# ---- Docker -----------------------------------------------------------------------------
build:
	$(COMPOSE) build

benchmark: build
	$(COMPOSE) run --rm benchmark

audit: build
	$(COMPOSE) run --rm audit

generate: build
	$(COMPOSE) run --rm generate

prompt: build
	$(COMPOSE) run --rm prompt

scrub: build
	$(COMPOSE) run --rm scrub

embedding: build
	$(COMPOSE) run --rm embedding

sft: build
	$(COMPOSE) run --rm sft

dpo: build
	$(COMPOSE) run --rm dpo

report: build
	$(COMPOSE) run --rm report
	@cat reports/RESULTS.md

analyze: build
	$(COMPOSE) run --rm analyze

smoke:
	./scripts/quickstart.sh --tier smoke

shell: build
	$(COMPOSE) run --rm shell

down:
	$(COMPOSE) down -v

# Follow the newest run log. vLLM draws its progress bar with carriage returns, so the last
# "line" of the file is megabytes long -- `-n 0` starts from now instead of replaying it.
watch:
	@tail -f -n 0 "$$(ls -t logs/*.log 2>/dev/null | head -1)" 2>/dev/null \
	  || echo "No logs/*.log yet -- start a run first."

# One-shot progress: which config is running, how far in, and how many runs are scored.
progress:
	@log="$$(ls -t logs/*.log 2>/dev/null | head -1)"; \
	if [ -z "$$log" ]; then echo "No logs/*.log yet."; exit 0; fi; \
	echo "log:      $$log"; \
	echo -n "running:  "; pgrep -f "run_all_.*\.sh" >/dev/null && echo "yes (pid $$(pgrep -f 'run_all_.*\.sh' | head -1))" || echo "no"; \
	echo "config:   $$(grep -E '^=== ' "$$log" | tail -1)"; \
	echo -n "progress: "; tr '\r' '\n' < "$$log" | grep -E 'Processed prompts' | tail -1 | cut -c1-100; \
	echo "scored:   $$(ls eval/results/*.json 2>/dev/null | grep -cv index.json) run(s) in eval/results/"; \
	echo -n "gpu:      "; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "n/a"

status:
	@echo "=== containers ==="; $(COMPOSE) ps
	@echo; echo "=== gpu ==="; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || echo "nvidia-smi unavailable"
	@echo; echo "=== enabled experiments ==="; grep -H "Currently enabled" scripts/run_all_*.sh | sed 's/:# / /'
	@echo; echo "=== scored runs ==="; ls eval/results/*.json 2>/dev/null | wc -l

# ---- Local (uv venv) --------------------------------------------------------------------
# `uv run` does not read .env and nothing calls load_dotenv, so without this the local
# targets run with HBM_OUTPUT_ROOT and HF_HOME unset and every download and checkpoint
# lands on the system disk. Keep this OFF the Docker targets: compose sets container paths,
# and sourcing host paths inside the container would override the bind mount.
DOTENV = set -a; [ -f ./.env ] && . ./.env; set +a;

local-setup:
	uv venv --python 3.12
	uv pip install -e ".[dev,ukr]"
	@echo "Now install torch for this machine, e.g. on GB10/arm64:"
	@echo "  uv pip install torch --index-url https://download.pytorch.org/whl/cu130"
	@echo "and, for the vLLM backend:  uv pip install -e '.[vllm]'"

local-benchmark:
	$(DOTENV) uv run python scripts/build_benchmark.py

local-audit:
	$(DOTENV) uv run bash scripts/run_all_audit.sh

local-generate:
	$(DOTENV) uv run python scripts/generate_training_data.py --config configs/generation/teacher.yaml

local-prompt:
	$(DOTENV) uv run bash scripts/run_all_prompt.sh

local-scrub:
	$(DOTENV) uv run bash scripts/run_all_scrub.sh

local-embedding:
	$(DOTENV) uv run bash scripts/run_all_embedding.sh

local-sft:
	$(DOTENV) uv run bash scripts/run_all_sft.sh

local-dpo:
	$(DOTENV) uv run bash scripts/run_all_dpo.sh

local-report:
	$(DOTENV) uv run python scripts/make_report.py

# Reads only; safe to run while a sweep is in flight.
local-analyze:
	$(DOTENV) uv run python scripts/analyze_results.py --json reports/analysis.json $(ARGS)

# Enumerated experiment plan. Reads reports/analysis.json, so run local-analyze first.
local-plan:
	$(DOTENV) uv run python scripts/make_experiment_matrix.py --csv reports/experiment_runs.csv

local-test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ scripts/ tests/
