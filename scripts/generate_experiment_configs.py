"""Generates every experiment config, and sets which ones are enabled.

The experiment matrix is large and highly regular -- 3 models x 2 languages x (1 baseline +
9 prompt strategies + 2 scrub modes + 3 erasure methods + SFT + 2 preference objectives) --
so the configs are generated rather than hand-written. Regenerating **preserves your current
selection**; only `--enable` changes it.

    python scripts/generate_experiment_configs.py                 # (re)write all configs
    python scripts/generate_experiment_configs.py --enable stage1 # audit only
    python scripts/generate_experiment_configs.py --enable core
    python scripts/generate_experiment_configs.py --enable full
    python scripts/generate_experiment_configs.py --enable none

Presets:
    smoke     one model, one language, 20 pairs -- proves the plumbing end to end in minutes
    stage1    the baseline audit, ungated models only (Qwen3.5-4B, LAPA) x 2 languages
    stage1-all  the same, plus google/gemma-4-12B-it. The ONE preset that enables a gated
              model, because "audit every model" is the whole point of the baseline stage --
              accept the licence and set HF_TOKEN first or the runs fail on download.
    core    stage1 plus the cheap mitigations (prompt, scrub) everywhere
    full    everything, including the training runs

Gated models (`google/gemma-4-*`) are never auto-enabled by a preset: accept the licence,
set HF_TOKEN, and uncomment them yourself.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.data.prompts import STRATEGIES  # noqa: E402
from hiring_bias_mitigation.utils.config import REPO_ROOT  # noqa: E402

CONFIG_ROOT = REPO_ROOT / "configs"
SCRIPT_ROOT = REPO_ROOT / "scripts"

LANGUAGES = ("en", "uk")
PROTECTED_GROUPS = ["military_status", "gender", "religion"]
INTERSECTIONS = ["military_status_x_gender", "military_status_x_religion"]

# --------------------------------------------------------------------------------------
# The three target models. Chosen for family spread, size spread, and Ukrainian coverage;
# every one is trainable on a single GB10 with the settings below.
# --------------------------------------------------------------------------------------
MODELS: dict[str, dict] = {
    "qwen3.5-4b": {
        "repo": "Qwen/Qwen3.5-4B",
        "params_b": 4,
        "layers": 32,
        "gated": False,
        # LoRA, like every other model here. A full fine-tune fits this one comfortably
        # (~32 GB at 8 bytes/param), but then the "SFT" row would mean a full fine-tune for
        # the 4B and an adapter for the 9B and 12Bs, and a difference between those rows
        # could not be attributed to model size rather than to how much of the model each
        # run was allowed to move.
        "use_lora": True,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1.0e-5,
        "note": (
            "Qwen3.5 dense 4B. The Qwen3.5 line ships as multimodal checkpoints "
            "(image-text-to-text); text-only use is unaffected, but the unused vision tower "
            "still occupies memory, which is why the batch size is conservative for a 4B."
        ),
    },
    "qwen3.5-9b": {
        "repo": "Qwen/Qwen3.5-9B",
        "params_b": 9,
        "layers": 32,
        "gated": False,
        # LoRA, not a full fine-tune. 9B x 8 bytes/param is ~72 GiB of weights, gradients and
        # optimizer state, which fits the Spark's 121.6 GiB on paper but leaves little for
        # activations once the desktop's ~14 GiB is subtracted.
        #
        # NOTE FOR THE WRITE-UP: this makes a 4B-vs-9B comparison confounded, because
        # qwen3.5-4b is the only full fine-tune in the matrix. A difference between them mixes
        # scale with training regime. If that comparison matters, add a 4B LoRA control -- it
        # is the cheapest run in the study.
        "use_lora": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1.0e-4,
        "note": (
            "Qwen3.5 dense 9B (32 layers). Sits between the 4B and the 12B targets, so the "
            "Qwen line alone spans a scale range within one family and one tokeniser."
        ),
    },
    "gemma-4-e4b": {
        "repo": "google/gemma-4-E4B-it",
        "params_b": 4,
        "layers": 42,
        "gated": False,
        # Sparse (expert) architecture, so total parameters exceed the "effective 4B" the
        # name refers to and a full fine-tune would cost far more than 4B x 8 bytes suggests.
        "use_lora": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1.0e-4,
        "note": (
            "Gemma 4 E4B instruct (Gemma4ForConditionalGeneration, 42 layers, sparse). The "
            "small Gemma. Without it the matrix has Gemma only at 12B and Qwen only at 4-9B, "
            "so any family comparison would be confounded by scale."
        ),
    },
    "gemma-4-12b": {
        "repo": "google/gemma-4-12B-it",
        "params_b": 12,
        "layers": 48,
        # Verified against the Hub: Gemma 4 is NOT gated, unlike Gemma 2 and 3. No licence
        # click and no HF_TOKEN needed, so the presets may enable it like any other model.
        "gated": False,
        "use_lora": True,  # ~26 GB weights + adapter; a full FT would leave no activation room
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1.0e-4,
        "note": (
            "Gemma 4 12B instruct (Gemma4UnifiedForConditionalGeneration, 48 layers). "
            "Continuity with the audit study, which evaluated the Gemma 2 line."
        ),
    },
    "lapa-12b": {
        "repo": "lapa-llm/lapa-v0.1.2-instruct",
        "params_b": 12,
        "layers": 48,
        "gated": False,
        "use_lora": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1.0e-4,
        "note": (
            "Ukrainian-native instruct model. The audit study could only evaluate Ukrainian "
            "on models whose Ukrainian was incidentally good enough, and reported that lower "
            "generation quality confounds the inconsistency rate. A Ukrainian-native model "
            "separates 'more biased in Ukrainian' from 'worse at Ukrainian'."
        ),
    },
}

TEACHER = "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4"

AUDIT_BASE = {
    "backend": "vllm",
    "max_model_len": 8192,
    # 0.80, not the 0.90 that is conventional on a discrete GPU. The Spark's 121.6 GiB is
    # UNIFIED memory shared with the desktop: Xorg and gnome-shell hold ~14 GiB, so a request
    # for 0.90 (109.5 GiB) exceeds what is actually free (~107.7 GiB) and vLLM refuses to
    # start. 0.80 leaves ~24 GiB of headroom, which also survives someone opening a browser
    # mid-run. Raise it only on a headless machine.
    "gpu_memory_utilization": 0.80,
    "seed": 42,
    "raw_dir": "outputs/raw",
    "conditions": ["explicit", "implicit", "attr_free"],
    # Thinking mode off -- see backends.DEFAULT_CHAT_TEMPLATE_KWARGS. A reasoning model
    # otherwise spends the whole token budget on a visible trace and is truncated before the
    # JSON, and non-thinking mode is what keeps a baseline comparable to the audit study's.
    "chat_template_kwargs": {"enable_thinking": False},
    # `null` = the full cross product. Capping it subsamples, and the subsample keeps every
    # cell that touches a reference level FIRST -- which for military x gender (5 x 20 = 100
    # cells, 24 of them reference-touching) means a cap of 24 keeps the reference "star" and
    # zero off-reference cells. Non-additivity lives entirely in the off-reference cells:
    # it is the claim that the effect of (veteran, female) differs from effect(veteran) +
    # effect(female), and without that cell there is nothing to measure. Cap this only when
    # you have accepted that the intersection becomes descriptive rather than a test.
    "max_intersection_cells": None,
    "generation": {"max_new_tokens": 512, "temperature": 0.0, "top_p": 1.0, "seed": 42},
}

HEADER = "# Generated by scripts/generate_experiment_configs.py -- edits are overwritten.\n"


def _yaml_dump(data: dict) -> str:
    import yaml

    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _write(path: Path, comment: str, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{HEADER}# {comment}\n{_yaml_dump(payload)}", encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# Config writers
# --------------------------------------------------------------------------------------

def write_audit_configs() -> list[Path]:
    """Baseline audit: one config per model x language. Stage 1 of the study."""
    written = []
    for key, model in MODELS.items():
        for lang in LANGUAGES:
            payload = {
                **AUDIT_BASE,
                "model": model["repo"],
                "lang": lang,
                "protected_groups": PROTECTED_GROUPS + INTERSECTIONS,
                "mitigation": {"family": "none"},
            }
            written.append(
                _write(
                    CONFIG_ROOT / "audit" / f"{key}_{lang}_baseline.yaml",
                    f"Baseline audit | {model['repo']} | {lang} | {model['note']}",
                    payload,
                )
            )
    # A smoke config that exercises every stage on 20 pairs and one group.
    written.append(
        _write(
            CONFIG_ROOT / "audit" / "smoke.yaml",
            "Smoke test: 20 pairs, one group, one language. Proves the plumbing, not a result.",
            {
                **AUDIT_BASE,
                "model": MODELS["qwen3.5-4b"]["repo"],
                "lang": "en",
                "protected_groups": ["military_status"],
                "limit_pairs": 20,
                # Without a suffix this run name collides with the real
                # qwen3.5-4b_en_baseline, and a smoke run would silently overwrite that
                # run's cached generations and its scored result.
                "run_suffix": "smoke",
                "mitigation": {"family": "none"},
            },
        )
    )
    return written


def write_prompt_configs() -> list[Path]:
    written = []
    strategies = [s for s in STRATEGIES if s != "baseline"]
    for key, model in MODELS.items():
        for lang in LANGUAGES:
            for strategy in strategies:
                payload = {
                    **AUDIT_BASE,
                    "model": model["repo"],
                    "lang": lang,
                    "protected_groups": _scoped_groups(
                        key, lang, PROTECTED_GROUPS + INTERSECTIONS
                    ),
                    "conditions": _scoped_conditions(
                        key, lang, AUDIT_BASE["conditions"]
                    ),
                    "mitigation": {"family": "prompt", "strategy": strategy},
                }
                written.append(
                    _write(
                        CONFIG_ROOT / "mitigation" / "prompt" / f"{key}_{lang}_{strategy}.yaml",
                        f"Prompt mitigation '{strategy}' | {model['repo']} | {lang}",
                        payload,
                    )
                )
    return written


def write_scrub_configs() -> list[Path]:
    written = []
    for key, model in MODELS.items():
        for lang in LANGUAGES:
            for mode in ("lexical", "llm"):
                payload = {
                    **AUDIT_BASE,
                    "model": model["repo"],
                    "lang": lang,
                    "protected_groups": _scoped_groups(
                        key, lang, PROTECTED_GROUPS + INTERSECTIONS
                    ),
                    "conditions": _scoped_conditions(
                        key, lang, AUDIT_BASE["conditions"]
                    ),
                    "mitigation": {
                        "family": "scrub",
                        "mode": mode,
                        "groups": [*PROTECTED_GROUPS, "marital_status"],
                    },
                }
                comment = (
                    f"Attribute scrubbing ({mode}) | {model['repo']} | {lang}"
                    + (
                        " | the rewrite is done by the model under evaluation, so its cost is "
                        "one extra call per item"
                        if mode == "llm"
                        else " | deterministic, zero extra calls"
                    )
                )
                written.append(
                    _write(
                        CONFIG_ROOT / "mitigation" / "scrub" / f"{key}_{lang}_{mode}.yaml",
                        comment,
                        payload,
                    )
                )
    return written


def write_embedding_configs() -> list[Path]:
    written = []
    for key, model in MODELS.items():
        # Mid-depth layers: early layers still carry surface form, late layers are already
        # committed to the output distribution. The exact index is a hyperparameter -- refit
        # at other depths and compare the probe accuracies in the diagnostics JSON.
        layers = [_eraser_layer(model)]
        for lang in LANGUAGES:
            for method in ("leace", "inlp", "mean_diff"):
                name = f"{key}_{lang}_{method}"
                payload = {
                    **AUDIT_BASE,
                    "backend": "transformers",  # forward hooks; vLLM cannot host them
                    "batch_size": 8,
                    "dtype": "bfloat16",
                    "model": model["repo"],
                    "lang": lang,
                    # Same targeted scope as every other family. Erasure was the one branch
                    # that evaluated the full group set regardless of where bias was actually
                    # confirmed, which both inflated its cost and made its cells incomparable
                    # with the prompt and scrub arms for the same model and language.
                    "protected_groups": _scoped_groups(key, lang, PROTECTED_GROUPS),
                    "conditions": _scoped_conditions(key, lang, AUDIT_BASE["conditions"]),
                    # The eraser is fitted on the military-status slice of the generated
                    # training data -- never on the benchmark. See scripts/fit_eraser.py.
                    "data_config": "configs/data/military_only.yaml",
                    "mitigation": {
                        "family": "embedding",
                        "method": method,
                        "protected_group": "military_status",
                        "layers": layers,
                        "strength": 1.0,
                        "n_fitting_rows": 2000,
                        "batch_size": 8,
                        "eraser_path": f"outputs/erasers/{name}.npz",
                    },
                }
                written.append(
                    _write(
                        CONFIG_ROOT / "mitigation" / "embedding" / f"{name}.yaml",
                        f"Concept erasure ({method}) at layer {layers[0]} | {model['repo']} | "
                        f"{lang} | fit with scripts/fit_eraser.py before running the audit",
                        payload,
                    )
                )
    return written


def _eraser_layer(model: dict) -> int:
    """Default layer for the concept eraser: mid-depth, from the checkpoint's own config.

    Early layers still carry surface form and late ones are already committed to the output
    distribution, so the midpoint is a reasonable starting point -- not a tuned value. Refit
    at other depths and compare `probe_accuracy_before` in the eraser's diagnostics JSON.

    Depths are recorded per model rather than guessed from parameter count: Qwen3.5-4B and
    Qwen3.5-9B both have 32 layers despite differing more than twofold in size, so a
    size-based estimate would have put the 9B eraser in the wrong place.
    """
    return max(2, model["layers"] // 2)


def write_data_configs(artifacts_dir: str) -> list[Path]:
    """Dataset views over one generation run: all groups, one group, one language."""
    written = []
    views = {
        "all_groups": {"protected_groups": PROTECTED_GROUPS, "languages": list(LANGUAGES)},
        "military_only": {"protected_groups": ["military_status"], "languages": list(LANGUAGES)},
        "uk_only": {"protected_groups": PROTECTED_GROUPS, "languages": ["uk"]},
        "en_only": {"protected_groups": PROTECTED_GROUPS, "languages": ["en"]},
    }
    for name, view in views.items():
        written.append(
            _write(
                CONFIG_ROOT / "data" / f"{name}.yaml",
                f"Training-data view '{name}' over the generated artifacts. Subsetting here "
                "is how a per-group or per-language mitigation runs without regenerating data.",
                {"name": name, "artifacts_dir": artifacts_dir, "seed": 42, **view},
            )
        )
    written.append(
        _write(
            CONFIG_ROOT / "data" / "smoke.yaml",
            "Tiny view for the smoke run.",
            {
                "name": "smoke",
                "artifacts_dir": artifacts_dir,
                "seed": 42,
                "protected_groups": ["military_status"],
                "languages": ["en"],
                "limit_rows": 256,
            },
        )
    )
    return written


def write_generation_config(artifacts_dir: str) -> Path:
    return _write(
        CONFIG_ROOT / "generation" / "teacher.yaml",
        f"Semi-synthetic training-data generation | teacher {TEACHER} | run on the DGX Spark",
        {
            "teacher_model": TEACHER,
            "backend": "vllm",
            "dtype": "auto",
            "max_model_len": 8192,
            # Two constraints pull in opposite directions here.
            #
            # Down: vLLM checks *free* memory, not *available*. After reading the Djinni
            # parquet files the page cache holds tens of GiB that would be reclaimed under
            # pressure, but vLLM refuses to start rather than rely on that -- so a request
            # sized to total memory fails.
            #
            # Up: this teacher is ~71 GiB of weights. At 0.62 (~75 GiB) only ~3 GiB was left
            # for the KV cache, and the Qwen3.5 Mamba/GDN layers could allocate just 66 of
            # the 256 per-sequence cache blocks vLLM wants -- startup failed outright with
            # "max_num_seqs exceeds available Mamba cache blocks".
            #
            # 0.68 (~83 GiB) is the compromise: it fits under the ~87 GiB that is genuinely
            # free once the page cache is warm, and still leaves ~12 GiB of KV cache -- around
            # 230 Mamba blocks, comfortably above the capped max_num_seqs below. Raising it
            # further only helps on a freshly-booted machine and risks the same startup
            # failure on a warm one.
            "gpu_memory_utilization": 0.68,
            "max_num_seqs": 64,
            "artifacts_dir": artifacts_dir,
            "dataset_name": "hiring-bias-mitigation-semisynthetic",
            "languages": list(LANGUAGES),
            "protected_groups": PROTECTED_GROUPS,
            "conditions": ["explicit", "implicit"],
            # 3,000 pairs at 2 jobs per candidate = 1,500 distinct CVs, against 1,000 if
            # each CV were paired with 3 jobs. CV diversity is the binding constraint, not
            # pair count: the counterfactual construction already multiplies one pair into
            # ~24 training rows (3 groups x 2 attributes x 2 conditions x chosen and
            # rejected), so a CV reused across 3 jobs contributes ~72 rows carrying the same
            # passage of text. Two jobs per candidate keeps the "same CV judged against
            # different requirements" signal -- which is realistic and worth having -- while
            # widening the pool of distinct people the model trains on.
            "n_pairs_per_language": 3000,
            "jobs_per_candidate": 2,
            "attributes_per_pair": 2,
            "rejected_source": "teacher_biased",
            "balance_decisions": True,
            "val_fraction": 0.05,
            "seed": 42,
            # A rationale is <=30 words; 320 tokens is ample and keeps the pass fast. The
            # teacher runs slightly warm so its rationales vary across near-identical
            # profiles -- an SFT set of 60,000 identical sentences teaches one sentence.
            "generation": {
                "max_new_tokens": 320,
                "temperature": 0.7,
                "top_p": 0.9,
                "seed": 42,
            },
        },
    )


#: Sequence budget per language, measured on the generated data rather than guessed.
#: English tops out at ~1,540 tokens; Ukrainian reaches ~2,370, and 8.7% of it exceeds 1,536.
#: Truncation here cuts the *completion* -- the label the model is trained on -- so the budget
#: has to clear the longest example, not the average one.
MAX_SEQ_LEN = {"en": 2048, "uk": 2560, "all": 2560}


def _seq_len_for(view: str) -> int:
    if view.startswith("en"):
        return MAX_SEQ_LEN["en"]
    if view.startswith("uk"):
        return MAX_SEQ_LEN["uk"]
    return MAX_SEQ_LEN["all"]


def _gradient_checkpointing(model: dict) -> bool:
    """Trades memory for speed, and changes nothing about the result.

    Recomputing activations in the backward pass costs roughly a third of the step time. The
    4B and 9B targets have the memory to skip it; the 12Bs do not reliably, and an OOM halfway
    through a run is more expensive than the time it saves.
    """
    return model.get("params_b", 0) >= 12


def write_sft_configs() -> list[Path]:
    written = []
    for key, model in MODELS.items():
        # Per-language views first: the military-status effect reverses direction between
        # English and Ukrainian (see reports/FINDINGS.md), so training on both at once lets
        # the halves cancel and makes a null result uninterpretable.
        for view in ("en_only", "uk_only", "all_groups", "military_only"):
            name = f"{key}_{view}"
            written.append(
                _write(
                    CONFIG_ROOT / "mitigation" / "sft" / f"{name}.yaml",
                    f"SFT on counterfactually-consistent data | {model['repo']} | view {view} "
                    f"| {'LoRA' if model['use_lora'] else 'full fine-tune'}",
                    {
                        "stage": "sft",
                        "base_model": model["repo"],
                        "data_config": f"configs/data/{view}.yaml",
                        "output_dir": f"outputs/sft/{name}",
                        "use_lora": model["use_lora"],
                        "lora_r": 32,
                        "lora_alpha": 64,
                        "max_seq_len": _seq_len_for(view),
                        "num_train_epochs": 2,
                        "learning_rate": model["learning_rate"],
                        "per_device_train_batch_size": model["per_device_train_batch_size"],
                        "gradient_accumulation_steps": model["gradient_accumulation_steps"],
                        "gradient_checkpointing": _gradient_checkpointing(model),
                        "dtype": "bfloat16",
                        "seed": 42,
                        "wandb_project": "hiring-bias-mitigation",
                        "wandb_run_name": f"sft-{name}",
                    },
                )
            )
    return written


def write_dpo_configs() -> list[Path]:
    written = []
    for key, model in MODELS.items():
        for view in ("en_only", "uk_only", "all_groups"):
          # ORPO was the original second objective; TRL 1.x removed it outright, and pinning
          # an older TRL would mean downgrading transformers below what vLLM needs. KTO takes
          # its place: same responses, scored unpaired instead of against a partner.
          for objective in ("dpo", "kto"):
            name = f"{key}_{view}_{objective}"
            payload = {
                "stage": "dpo",
                "objective": objective,
                "base_model": model["repo"],
                "data_config": f"configs/data/{view}.yaml",
                "output_dir": f"outputs/dpo/{name}",
                "use_lora": True,  # preference training is LoRA-only here; see dpo.py
                "lora_r": 32,
                "lora_alpha": 64,
                "beta": 0.1,
                "max_seq_len": _seq_len_for(view),
                "max_prompt_len": 1792,
                "num_train_epochs": 1,
                "learning_rate": 5.0e-6 if objective == "dpo" else 8.0e-6,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "gradient_checkpointing": _gradient_checkpointing(model),
                "dtype": "bfloat16",
                "seed": 42,
                "wandb_project": "hiring-bias-mitigation",
                "wandb_run_name": f"{objective}-{name}",
            }
            if objective == "dpo":
                # DPO continues from the SFT checkpoint, which is what makes SFT-only vs
                # SFT+preference a clean ablation on identical data. KTO starts from the base
                # model instead, so the pair of arms also separates "preference after SFT"
                # from "preference instead of SFT".
                payload["sft_checkpoint"] = f"outputs/sft/{key}_{view}"
            written.append(
                _write(
                    CONFIG_ROOT / "mitigation" / "dpo" / f"{name}.yaml",
                    f"Preference optimisation ({objective.upper()}) | {model['repo']}"
                    + (" | continues from the SFT checkpoint" if objective == "dpo" else
                       " | single-stage, from the base model, on the unpaired KTO split"),
                    payload,
                )
            )
    return written


def write_trained_audit_configs() -> list[Path]:
    """Audit configs that evaluate a trained adapter — one per training run, per language.

    Training produces weights; the fairness numbers come from auditing them. These configs
    must name the checkpoint the training stage will actually write, and must evaluate the
    same groups and conditions the mitigation was aimed at, or the mitigated numbers cannot
    be placed beside the baseline ones.

    Training is per language (`outputs/sft/<slug>_<lang>_only`), so each checkpoint is
    audited in **its own language only**. Auditing a Ukrainian-trained adapter in English
    would be a transfer experiment, which is a different question and is not on this plan.
    """
    written = []
    for key, model in MODELS.items():
        for lang in LANGUAGES:
            runs = [
                ("sft", f"outputs/sft/{key}_{lang}_only", f"{lang}_only"),
                ("dpo", f"outputs/dpo/{key}_{lang}_only_dpo", f"{lang}_only_dpo"),
                ("dpo", f"outputs/dpo/{key}_{lang}_only_kto", f"{lang}_only_kto"),
            ]
            for family, checkpoint, suffix in runs:
                mitigation = {"family": family}
                # LoRA runs load an adapter beside the base model; a full fine-tune replaces
                # it, so the checkpoint becomes the model itself.
                payload_model = model["repo"]
                if model["use_lora"]:
                    mitigation["lora_path"] = checkpoint
                else:
                    payload_model = checkpoint
                    mitigation["checkpoint"] = checkpoint

                written.append(
                    _write(
                        CONFIG_ROOT / "audit" / f"{key}_{lang}_{family}_{suffix}.yaml",
                        f"Audit of the trained {family.upper()} checkpoint `{checkpoint}` | "
                        f"{model['repo']} | {lang} | run the matching training config first. "
                        f"Evaluates all three protected groups -- training touched all of "
                        f"them, so the untargeted ones are where damage would show.",
                        {
                            **AUDIT_BASE,
                            "model": payload_model,
                            "lang": lang,
                            # Deliberately NOT scoped, unlike prompt and scrub. Those are
                            # aimed at named cells, so measuring those cells is fair.
                            # Training moves weights against all three protected groups at
                            # once, which makes the untargeted groups exactly where
                            # collateral damage would appear -- and scoping them out would
                            # make that damage unobservable rather than absent.
                            "protected_groups": PROTECTED_GROUPS,
                            "conditions": AUDIT_BASE["conditions"],
                            "run_suffix": suffix,
                            "mitigation": mitigation,
                        },
                    )
                )
    return written


# --------------------------------------------------------------------------------------
# Runner scripts: the CONFIGS arrays that decide what actually runs
# --------------------------------------------------------------------------------------

RUNNERS = {
    "run_all_audit.sh": ("configs/audit", "python scripts/run_audit.py --config"),
    "run_all_prompt.sh": ("configs/mitigation/prompt", "python scripts/run_audit.py --config"),
    "run_all_scrub.sh": ("configs/mitigation/scrub", "python scripts/run_audit.py --config"),
    "run_all_embedding.sh": (
        "configs/mitigation/embedding",
        "python scripts/fit_eraser.py --config CFG && python scripts/run_audit.py --config",
    ),
    "run_all_sft.sh": (
        "configs/mitigation/sft",
        "python -m hiring_bias_mitigation.mitigation.sft --config",
    ),
    "run_all_dpo.sh": (
        "configs/mitigation/dpo",
        "python -m hiring_bias_mitigation.mitigation.dpo --config",
    ),
}

RUNNER_TEMPLATE = """#!/usr/bin/env bash
# Runs the configs enabled below. This CONFIGS array is the SINGLE SOURCE OF TRUTH for what
# runs -- `make {make_target}`, the docker-compose service and quickstart.sh all execute this
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
# Currently enabled: {n_enabled}/{n_total}
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
{entries}
)

if [ ${{#CONFIGS[@]}} -eq 0 ]; then
  echo "No configs enabled in $0." >&2
  echo "Uncomment entries in the CONFIGS array, or run:" >&2
  echo "    python scripts/generate_experiment_configs.py --enable core" >&2
  exit 1
fi

# A full sweep is hours to days. By default it stops at the first failure, so a
# misconfiguration does not burn the rest of the queue producing the same broken output.
# Pass --keep-going (or set HBM_KEEP_GOING=1) for an unattended run: failures are recorded
# and reported at the end, and the remaining configs still run.
KEEP_GOING="${{HBM_KEEP_GOING:-0}}"
[ "${{1:-}}" = "--keep-going" ] && KEEP_GOING=1

echo "{label}: ${{#CONFIGS[@]}} config(s) enabled (keep-going=$KEEP_GOING)"
FAILED=()
DONE=0
for cfg in "${{CONFIGS[@]}}"; do
  DONE=$((DONE + 1))
  echo "=== {label} [$DONE/${{#CONFIGS[@]}}]: $cfg ==="
  if {command}; then
    # Refresh the report after every run, not once at the end. A sweep is days long, and a
    # result that only lands when the whole queue finishes is a result nobody can act on --
    # including the decision to stop the queue because an arm is clearly not working.
    python scripts/make_report.py >/dev/null 2>&1 \
      && echo "--- reports/RESULTS.md updated ($DONE/${{#CONFIGS[@]}} done) ---"
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

if [ ${{#FAILED[@]}} -gt 0 ]; then
  echo >&2
  echo "{label}: ${{#FAILED[@]}} of ${{#CONFIGS[@]}} config(s) FAILED:" >&2
  printf '  %s\n' "${{FAILED[@]}}" >&2
  exit 1
fi
"""


def _run_order(config_path: str) -> tuple:
    """Orders a runner's CONFIGS array cheapest-model-first.

    A full audit is days of GPU. Running the 4B before the 12Bs means the first complete,
    readable report arrives in hours instead of after a day and a half -- and if something is
    misconfigured, it surfaces at the cost of the short run rather than the long one. Ties
    fall back to the filename so the array stays stable across regenerations.
    """
    for key, model in MODELS.items():
        if config_path.rsplit("/", 1)[-1].startswith(key):
            return (model["params_b"], config_path)
    return (0, config_path)  # smoke and anything unmatched first


def _existing_selection(path: Path) -> set[str]:
    """Which configs are currently uncommented, so regeneration preserves the selection."""
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8")
    match = re.search(r"CONFIGS=\((.*?)\n\)", text, flags=re.S)
    if not match:
        return set()
    selected = set()
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            selected.add(stripped.split()[0])
    return selected


def _preset_selection(preset: str, configs: list[str], runner: str) -> set[str]:
    gated = tuple(k for k, m in MODELS.items() if m["gated"])

    def ungated(paths: list[str]) -> list[str]:
        return [p for p in paths if not any(g in p for g in gated)]

    if preset == "none":
        return set()
    if preset == "smoke":
        return {c for c in configs if "smoke" in c}
    if preset == "stage1":
        return set(ungated([c for c in configs if runner == "run_all_audit.sh"
                            and "baseline" in c]))
    if preset == "stage1-all":
        # Deliberately NOT filtered through `ungated`. Every other preset leaves gated models
        # commented out; this one is the explicit opt-in, and the user typed its name.
        return {c for c in configs if runner == "run_all_audit.sh" and "baseline" in c}
    if preset == "core":
        if runner in ("run_all_audit.sh",):
            return set(ungated([c for c in configs if "baseline" in c]))
        if runner in ("run_all_prompt.sh", "run_all_scrub.sh"):
            return set(ungated(configs))
        return set()
    if preset == "full":
        return set(ungated(configs))
    raise ValueError(f"unknown preset {preset!r}")


def write_runners(preset: str | None) -> list[Path]:
    written = []
    for runner, (config_dir, command) in RUNNERS.items():
        path = SCRIPT_ROOT / runner
        configs = sorted(
            (str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / config_dir).glob("*.yaml")),
            key=_run_order,
        )
        selected = (
            _preset_selection(preset, configs, runner)
            if preset is not None
            else _existing_selection(path)
        )
        entries = "\n".join(
            f"  {c}" if c in selected else f"#   {c}" for c in configs
        )
        label = runner.replace("run_all_", "").replace(".sh", "").upper()
        path.write_text(
            RUNNER_TEMPLATE.format(
                make_target=runner.replace("run_all_", "").replace(".sh", ""),
                n_enabled=len(selected),
                n_total=len(configs),
                entries=entries,
                label=label,
                command=command.replace("CFG", '"$cfg"') + ' "$cfg"'
                if "CFG" in command
                else f'{command} "$cfg"',
            ),
            encoding="utf-8",
        )
        path.chmod(0o755)
        written.append(path)
    return written


DEFAULT_EVAL_SCOPE = REPO_ROOT / "reports" / "eval_scope.yaml"

EVAL_SCOPE: dict[str, dict] = {}


def _scoped_groups(slug: str, lang: str, default: list[str]) -> list[str]:
    """The protected groups one mitigation config should evaluate."""
    groups = (EVAL_SCOPE.get("groups") or {}).get(slug, {}).get(lang)
    return list(groups) if groups else default


def _scoped_conditions(slug: str, lang: str, default: list[str]) -> list[str]:
    """The injection conditions one mitigation config should evaluate.

    A target is a group *under a condition*: "military status, explicit". Evaluating the
    other framing as well doubles the cost to answer a question that cell did not raise.
    `attr_free` is always retained -- it is one extra pass over the pairs and the only thing
    separating "the disparity fell" from "the operating point moved".
    """
    conditions = (EVAL_SCOPE.get("conditions") or {}).get(slug, {}).get(lang)
    return list(conditions) if conditions else default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enable", choices=("none", "smoke", "stage1", "stage1-all", "core", "full"),
        help="set which configs are enabled in every runner (omit to preserve the selection)",
    )
    parser.add_argument("--artifacts-dir", default="artifacts/semisynthetic-v1")
    parser.add_argument(
        "--eval-scope",
        default=str(DEFAULT_EVAL_SCOPE) if DEFAULT_EVAL_SCOPE.exists() else None,
        help="path to reports/eval_scope.yaml (written by scripts/analyze_results.py). "
             "Restricts each mitigation config to the protected groups that showed a "
             "confirmed disparity for that model and language, plus its control group. "
             "Defaults to reports/eval_scope.yaml when it exists -- a plain regeneration "
             "must not silently widen every config back to the full grid, which is days of "
             "extra GPU nobody asked for. Pass --full-grid to opt out.",
    )
    parser.add_argument(
        "--full-grid", action="store_true",
        help="ignore reports/eval_scope.yaml and evaluate every protected group.",
    )
    parser.add_argument(
        "--no-runners", action="store_true",
        help="write the config YAMLs but leave scripts/run_all_*.sh untouched. Use this "
             "while a sweep is running: bash reads a script incrementally as it executes, "
             "so rewriting one mid-run can corrupt the rest of the queue.",
    )
    args = parser.parse_args()

    if args.full_grid:
        args.eval_scope = None

    if args.eval_scope:
        import yaml as _yaml

        loaded = _yaml.safe_load(Path(args.eval_scope).read_text()) or {}
        # Accept the older flat {slug: {lang: [groups]}} shape as well as the current
        # {groups: ..., conditions: ...} one, so an existing scope file keeps working.
        EVAL_SCOPE.update(
            loaded if "groups" in loaded else {"groups": loaded, "conditions": {}}
        )
        n = len(EVAL_SCOPE.get("groups") or {})
        print(f"eval scope: {n} model(s) restricted from {args.eval_scope}")

    written = []
    written += write_audit_configs()
    written += write_prompt_configs()
    written += write_scrub_configs()
    written += write_data_configs(args.artifacts_dir)
    written += write_embedding_configs()
    written.append(write_generation_config(args.artifacts_dir))
    written += write_sft_configs()
    written += write_dpo_configs()
    written += write_trained_audit_configs()
    print(f"wrote {len(written)} config file(s)")

    if args.no_runners:
        print("--no-runners: scripts/run_all_*.sh left untouched")
        return

    runners = write_runners(args.enable)
    for path in runners:
        first = path.read_text(encoding="utf-8")
        n = re.search(r"Currently enabled: (\d+)/(\d+)", first)
        print(f"  {path.name}: {n.group(1)}/{n.group(2)} enabled")


if __name__ == "__main__":
    main()
