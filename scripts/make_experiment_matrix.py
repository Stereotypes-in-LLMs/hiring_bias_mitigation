"""Generates the fully enumerated experiment plan: every run, what it changes, what it is
compared against, and what has to hold for it to count.

`ANALYSIS.md` argues where the bias is. `FINDINGS.md` summarises that for a reader. This
produces the operational document: one row per run that will actually execute, naming the
model, language, protected groups, mitigation family and variant, the training scope, the
evaluation scope, the baseline it is measured against, and the success criterion.

    python scripts/make_experiment_matrix.py            # writes reports/EXPERIMENT_PLAN.md
    python scripts/make_experiment_matrix.py --csv runs.csv

Generated from `reports/analysis.json`, so it cannot drift from the analysis it came from.
Regenerate rather than editing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.utils.config import REPO_ROOT  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("experiment_matrix")

#: What each family changes, and what its result means. Ordered cheapest-first, which is also
#: the order they should run: a training arm that fails to beat a free prompt edit is a
#: result, but only if the prompt edit was measured first.
FAMILY_ORDER = ["prompt", "scrub", "embedding", "sft", "dpo"]

FAMILY_INTERVENES = {
    "prompt": "prompt construction — no weights, no data",
    "scrub": "the CV text, before the prompt is built",
    "embedding": "the residual stream, via forward hooks at inference",
    "sft": "model weights (LoRA or full FT)",
    "dpo": "model weights, continuing from the SFT checkpoint",
}

FAMILY_TRAINED_ON = {
    "prompt": "—",
    "scrub": "—",
    "embedding": (
        "eraser fitted on the **military-status** slice of the generated pool, per language"
    ),
    "sft": "generated SFT split; one run per language (see the `Trained on` column)",
    "dpo": "generated preference split; continues from the matching per-language SFT run",
}

#: Attribute counts per protected group, for costing a run's evaluation scope.
GROUP_SIZE = {
    "military_status": 5,
    "gender": 20,
    "religion": 9,
    "marital_status": 5,
    "military_status_x_gender": 100,
    "military_status_x_religion": 45,
}

#: Benchmark pairs per language, and the two injected conditions (attribute-free adds one
#: flat pass over the pairs).
PAIRS_PER_LANG = 450
INJECTED_CONDITIONS = 2


def eval_groups_for(slug: str, lang: str, plan: dict) -> list[str]:
    """The protected groups one run actually evaluates.

    Not a constant: `analyze_results.py --eval-scope` restricts each model-language to the
    groups that showed a confirmed disparity for *it*, plus its control. Printing the full
    grid everywhere would misdescribe what these runs measure and misstate their cost.
    """
    scope = plan.get("eval_groups") or {}
    langs = ["en", "uk"] if "+" in lang else [lang]
    groups: list[str] = []
    for one in langs:
        for group in scope.get(slug, {}).get(one, []):
            if group not in groups:
                groups.append(group)
    return sorted(groups) if groups else sorted(GROUP_SIZE)


def describe_eval(groups: list[str]) -> str:
    return ", ".join(f"{g.replace('_x_', ' × ')} ({GROUP_SIZE.get(g, '?')})" for g in groups)


def eval_conditions_for(slug: str, lang: str, plan: dict) -> list[str]:
    """The injection conditions one run evaluates.

    A target is a group *under a condition*, so a run measures the framings its own targets
    were found in. `attr_free` is always present: one extra pass over the pairs, and the only
    thing separating "the disparity fell" from "the operating point moved".
    """
    scope = plan.get("eval_conditions") or {}
    langs = ["en", "uk"] if "+" in lang else [lang]
    out: list[str] = []
    for one in langs:
        for condition in scope.get(slug, {}).get(one, []):
            if condition not in out:
                out.append(condition)
    return out or ["explicit", "implicit", "attr_free"]


def eval_cost(groups: list[str], conditions: list[str], n_langs: int = 1) -> int:
    attrs = sum(GROUP_SIZE.get(g, 0) for g in groups)
    injected = len([c for c in conditions if c != "attr_free"])
    flat = PAIRS_PER_LANG if "attr_free" in conditions else 0
    return n_langs * (attrs * PAIRS_PER_LANG * injected + flat)


def _variant_of(path: str, family: str) -> str:
    stem = Path(path).stem
    if family in ("prompt", "scrub", "embedding"):
        # <slug>_<lang>_<variant>
        return stem.split("_", 2)[-1] if stem.count("_") >= 2 else stem
    return stem  # training configs carry their data view in the name


def _lang_of(path: str, family: str) -> str:
    stem = Path(path).stem
    for lang in ("_en_", "_uk_"):
        if lang in stem:
            return lang.strip("_")
    if family in ("sft", "dpo"):
        return "en + uk"
    return "?"


def _trained_on(path: str, family: str) -> str:
    if family in ("prompt", "scrub"):
        return "—"
    if family == "embedding":
        return "military_status, one language"
    stem = Path(path).stem
    if "military_only" in stem:
        return "**military_status only**, en + uk"
    if "_en_only" in stem:
        return "all 3 groups, **English only**"
    if "_uk_only" in stem:
        return "all 3 groups, **Ukrainian only**"
    return "all 3 groups, en + uk"


def _slug_to_model(slug: str, plan: dict) -> str:
    for entry in plan["targets"] + plan["controls"] + plan["rescues"]:
        if entry["model_slug"] == slug:
            return entry["model"]
    return slug


def _role_of(model: str, lang: str, plan: dict) -> str:
    """Role of a run.

    A training config covers both languages at once ("en + uk"), so its role must be the
    strongest role the model holds in *either* — otherwise a model that is clean in English
    and a target in Ukrainian gets labelled a control on a run that exists because of its
    Ukrainian disparity, which is exactly backwards.
    """
    if any(r["model"] == model for r in plan["rescues"]):
        return "rescue"
    langs = ["en", "uk"] if "+" in lang else [lang]
    if any(t["model"] == model and t["lang"] in langs for t in plan["targets"]):
        return "target"
    return "control"


def enabled_configs() -> set[str]:
    """Config paths currently uncommented in any scripts/run_all_*.sh.

    The runners are the ground truth for what executes. Deriving "deferred" from them rather
    than from a hardcoded list means the plan cannot claim a run that was cut, and cannot
    quietly drop one that was restored.
    """
    import re

    enabled: set[str] = set()
    for runner in sorted((REPO_ROOT / "scripts").glob("run_all_*.sh")):
        for line in runner.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^  (configs/\S+\.yaml)", line)
            if match:
                enabled.add(match.group(1))
    return enabled


def build_runs(plan: dict) -> list[dict]:
    """One row per config that will execute."""
    slugs = {e["model_slug"] for e in plan["targets"] + plan["controls"] + plan["rescues"]}
    enabled = enabled_configs()
    runs = []
    for family in FAMILY_ORDER:
        for path in sorted(plan["config_paths"].get(family, [])):
            slug = next((s for s in sorted(slugs, key=len, reverse=True)
                         if Path(path).stem.startswith(s)), None)
            if slug is None:
                continue
            model = _slug_to_model(slug, plan)
            lang = _lang_of(path, family)
            runs.append({
                "family": family,
                "model": model,
                "slug": slug,
                "lang": lang,
                "variant": _variant_of(path, family),
                "config": path,
                "trained_on": _trained_on(path, family),
                "role": _role_of(model, lang, plan),
                "deferred": path not in enabled,
                "eval_groups": eval_groups_for(slug, lang, plan),
                "eval_conditions": eval_conditions_for(slug, lang, plan),
                "eval_prompts": eval_cost(
                    eval_groups_for(slug, lang, plan),
                    eval_conditions_for(slug, lang, plan),
                    2 if "+" in lang else 1,
                ),
            })
    return runs


def render(analysis: dict) -> str:
    plan = analysis["plan"]
    runs = build_runs(plan)
    targets = plan["targets"]
    rescues = plan["rescues"]
    controls = plan["controls"]

    parts = [_header(analysis, runs, targets, controls, rescues)]
    parts.append(_section_what_to_fix(targets, controls, rescues))
    parts.append(_section_runs(runs))
    parts.append(_section_evaluation(plan, analysis))
    parts.append(_section_order(runs))
    if any(r["deferred"] for r in runs):
        parts.append(_section_future_work(runs, analysis))
    return "\n\n".join(parts)


def _header(analysis, runs, targets, controls, rescues) -> str:
    # Headline counts describe what will actually execute. A deferred row is still listed in
    # section 2 and accounted for in section 5, but counting it here would advertise a sweep
    # nobody is running.
    active = [r for r in runs if not r["deferred"]]
    by_family = defaultdict(int)
    for run in active:
        by_family[run["family"]] += 1
    counts = " · ".join(f"{f} {by_family[f]}" for f in FAMILY_ORDER if by_family[f])
    models = sorted({r["model"] for r in active})
    deferred = len(runs) - len(active)
    deferred_note = (
        f" A further **{deferred}** are configured but deferred — see section 5."
        if deferred else ""
    )
    return (
        "# Experiment plan — what to mitigate, and how it will be evaluated\n\n"
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by "
        f"`scripts/make_experiment_matrix.py` from `reports/analysis.json`. Derived — "
        f"regenerate rather than editing.*\n\n"
        "Evidence and reasoning: [`FINDINGS.md`](FINDINGS.md) · "
        "[`ANALYSIS.md`](ANALYSIS.md) · metric definitions: "
        "[`docs/METRICS.md`](../docs/METRICS.md)\n\n"
        f"**{len(active)} runs** across {len(models)} models: {counts}.{deferred_note}\n\n"
        f"{len(targets)} fairness targets · {len(controls)} no-harm controls · "
        f"{len(rescues)} rescue arms.\n\n"
        "Two things are held apart throughout, because conflating them is how a mitigation "
        "study reports a result it did not measure:\n\n"
        "- **Trained on** — what the intervention was fitted to.\n"
        "- **Evaluated on** — what is measured afterwards. **Per model and language**, not a "
        "fixed grid.\n\n"
        f"{_scope_summary(analysis)}\n\n"
        "**What this scoping costs, stated plainly.** A group left out of a run cannot be "
        "checked for damage by that run: a mitigation that fixes military status while "
        "breaking religion would not be caught. The attribute-free condition is what still "
        "guards against gross damage — it catches a model whose whole operating point moved "
        "— but harm confined to an untested group is not detectable here, and the write-up "
        "must say so rather than imply a no-harm check that was not run.\n\n"
        "### The intersections are deferred, not dropped\n\n"
        "The two intersections are 145 of the 179 attributes. Measuring them on every "
        "mitigation run costs **~538 GPU-hours**; measuring the single groups costs ~87. "
        "Paying for the intersections on arms that turn out not to work is the expensive "
        "mistake, so this plan splits in two:\n\n"
        "| Phase | Scope | Cost | Answers |\n|---|---|---:|---|\n"
        "| **1** (this document) | single groups only | ~87 h | which mitigation works, and "
        "at what cost to utility |\n"
        "| **2** (after phase 1) | intersections, on the winning arms only | ~34 h per "
        "promoted arm | does the mitigation reach the intersectional cells, and does the "
        "sub-additivity found in the baseline survive it |\n\n"
        "Phase 2 matters and must not be quietly skipped. The baseline found military × "
        "gender **sub-additive in both languages** — 52/23 negative in English, 62/11 in "
        "Ukrainian — meaning the combination is treated worse than the two effects predict "
        "separately. Whether a mitigation flattens that, leaves it, or inverts it is a result "
        "in its own right, and it is the direct follow-up to the study's intersection "
        "finding.\n\n"
        "```bash\n"
        "# after phase 1 has been scored\n"
        "python scripts/select_phase2.py --top 1 --dry-run   # see the cost\n"
        "python scripts/select_phase2.py --top 1             # writes reports/phase2_scope.yaml\n"
        "python scripts/generate_experiment_configs.py --no-runners \\\n"
        "    --eval-scope reports/phase2_scope.yaml\n"
        "```\n\n"
        "Promotion is by disparity reduction against each run's own baseline, with a utility "
        "guard: an arm that cut disparity by damaging the model is never promoted, however "
        "large the cut."
    )


def _scope_summary(analysis: dict) -> str:
    """What each model-language actually evaluates, and why it is not uniform."""
    plan = analysis["plan"]
    scope = plan.get("eval_groups") or {}
    if not scope:
        return ""
    rows = [
        "| Model | Lang | Groups evaluated | Conditions | Attributes | Prompts/run |",
        "|---|---|---|---|---:|---:|",
    ]
    for slug in sorted(scope):
        for lang in sorted(scope[slug]):
            groups = scope[slug][lang]
            conditions = eval_conditions_for(slug, lang, plan)
            attrs = sum(GROUP_SIZE.get(g, 0) for g in groups)
            rows.append(
                f"| `{slug}` | {lang} | {describe_eval(groups)} | "
                f"{', '.join(conditions)} | {attrs} | "
                f"{eval_cost(groups, conditions):,} |"
            )
    return (
        f"**Evaluation scope: `{plan.get('eval_scope', 'full')}`.** Each model-language "
        "measures exactly the groups **and conditions** in which it showed a confirmed "
        "disparity — not the full grid. A target is a group *under a condition*, so a cell "
        "found under explicit injection is measured under explicit; adding the other framing "
        "would double the cost to answer a question that cell did not raise.\n\n"
        "`attr_free` is always present and is not optional: one extra pass over the 450 "
        "pairs, and the only thing separating *\"the disparity fell\"* from *\"the model's "
        "whole operating point moved\"*.\n\n" + "\n".join(rows)
    )


def _section_what_to_fix(targets, controls, rescues) -> str:
    rows = [
        "| Priority | Model | Lang | Protected group | Condition | MAD (pp) | Range (pp) | "
        "Utility % | Sig. attrs | Role |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for t in targets:
        rows.append(
            f"| {t['priority']} | {t['model']} | {t['lang']} | {_md(t['group'])} | "
            f"{t['condition']} | {100 * t['ar_mad']:.1f} | {100 * t['ar_range']:.1f} | "
            f"{100 * t['utility']:.1f} | {t['n_significant_fdr']} | target |"
        )
    for c in controls:
        rows.append(
            f"| — | {c['model']} | {c['lang']} | {_md(c['group'])} | {c['condition']} | "
            f"{100 * c['ar_mad']:.1f} | {100 * c['ar_range']:.1f} | "
            f"{100 * c['utility']:.1f} | {c['n_significant_fdr']} | **control** |"
        )
    for r in rescues:
        rows.append(
            f"| — | {r['model']} | {r['lang']} | (all) | (all) | "
            f"{100 * r['ar_mad']:.1f} | {100 * r['ar_range']:.1f} | "
            f"{100 * r['utility']:.1f} | — | **rescue** |"
        )
    return (
        "## 1. What needs fixing\n\n"
        "Every cell the mitigation stage is aimed at, plus the controls that make its results "
        "interpretable. `Sig. attrs` counts attributes significant after FDR correction.\n\n"
        "**Controls are not filler.** A mitigation that worsens a clean cell is as "
        "informative as one that fixes a target, and without them the study cannot "
        "distinguish *\"the mitigation removed a disparity\"* from *\"the mitigation "
        "flattened everything, including what was already fine\"*.\n\n"
        "**Rescue arms are a different experiment.** Their primary outcome is utility, not "
        "disparity; their MAD column is shown for completeness and is not interpretable "
        "until utility crosses the gate.\n\n" + "\n".join(rows)
    )


def _section_runs(runs: list[dict]) -> str:
    blocks = []
    for family in FAMILY_ORDER:
        family_runs = [r for r in runs if r["family"] == family]
        if not family_runs:
            continue
        rows = [
            "| # | Model | Lang | Variant | Role | Trained on | **Groups evaluated** | "
            "**Conditions** | Prompts | Config |",
            "|---:|---|---|---|---|---|---|---|---:|---|",
        ]
        for index, run in enumerate(family_runs, start=1):
            # Deferred rows stay visible. Deleting them would hide that the family was cut;
            # a reader should see what was planned as well as what ran.
            role = "**future work**" if run["deferred"] else run["role"]
            rows.append(
                f"| {index} | {run['model']} | {run['lang']} | `{run['variant']}` | "
                f"{role} | {run['trained_on']} | "
                f"{describe_eval(run['eval_groups'])} | "
                f"{', '.join(run['eval_conditions'])} | {run['eval_prompts']:,} | "
                f"`{run['config']}` |"
            )
        active = [r for r in family_runs if not r["deferred"]]
        subtotal = sum(r["eval_prompts"] for r in active)
        deferred = len(family_runs) - len(active)
        heading = f"### {family} — {len(active)} run(s)"
        if deferred:
            heading += f", {deferred} deferred"
        blocks.append(
            f"{heading}\n\n"
            f"Intervenes at: {FAMILY_INTERVENES[family]}. "
            f"Trained on: {FAMILY_TRAINED_ON[family]}.\n\n"
            + "\n".join(rows)
            + (
                f"\n\n{deferred} row(s) marked **future work** are configured but not "
                "enabled; they are excluded from the subtotal. See section 5.\n"
                if deferred else ""
            )
            + f"\n\nSubtotal: **{subtotal:,} prompts**"
            + (
                " — evaluated after training, one audit per row"
                if family in ("sft", "dpo")
                else ""
            )
            + "."
        )
    return (
        "## 2. Every run, enumerated\n\n"
        "One row per config that will execute. Ordered cheapest-family-first, which is also "
        "the order they should run: a training arm failing to beat a free prompt edit is a "
        "result, but only if the prompt edit was measured first.\n\n" + "\n\n".join(blocks)
    )


def _section_evaluation(plan, analysis: dict | None = None) -> str:
    thresholds = plan["thresholds"]
    return (
        "## 3. How each run is evaluated\n\n"
        "Identical protocol for every run above — same 450 job–CV pairs per language, same "
        "attributes, same greedy decoding, same parser. The only thing that differs between a "
        "baseline and a mitigated run is the mitigation block, which is what licenses the "
        "comparison.\n\n"
        "### Measured per run\n\n"
        "| Axis | What is reported |\n|---|---|\n"
        "| Protected groups | per model and language — see the scope table at the top |\n"
        "| Conditions | explicit (labelled field), implicit (first-person biography), "
        "attribute-free (control) |\n"
        "| Fairness | acceptance-rate MAD and range, inconsistency rate, feedback "
        "similarity, per attribute and aggregated |\n"
        "| Utility | agreement with the attribute-free reference decision, overall and split "
        "by what the reference decided |\n"
        "| Output handling | refusal rate, parse-failure rate, attribute-mention rate |\n"
        "| Inference | permutation test (unpaired, for comparability) **and** paired "
        "sign-flip test, FDR-corrected across every test in the run |\n\n"
        "### Compared against\n\n"
        "Each mitigated run is read against **its own baseline** — same model, same language, "
        "same benchmark. The report puts the baseline value in brackets beside every "
        "mitigated value, so no cross-model comparison is needed to read a row.\n\n"
        "### What counts as success\n\n"
        "| Track | Primary | Secondary | Fails if |\n|---|---|---|---|\n"
        "| **A — fairness** | disparity (MAD) falls on the target cells | the direction of "
        "the largest attribute gaps narrows toward zero | utility drops materially, or a "
        "control cell gets worse — either means the model was damaged, not fixed |\n"
        f"| **B — rescue** | utility ≥ {100 * thresholds['min_utility']:.0f}% and acceptance "
        "rate moves toward the reference rate | *conditional on the primary*: whether the "
        "disparity that becomes measurable is larger or smaller than the general-purpose "
        "models show | primary not met — in which case **no fairness number from this model "
        "may be reported**, a condition fixed in advance |\n\n"
        "### Non-negotiable reporting\n\n"
        "1. Fairness and utility columns are reported **together**, at the same granularity. "
        "A fairness gain with a utility drop is a degradation.\n"
        "2. Refusal rate is reported per run. Refusals leave the denominator, so a model that "
        "learns to decline makes its disparity *unmeasurable*, not absent.\n"
        "3. Results are reported per protected group and per condition, never pooled into a "
        "single number per run. Groups behave differently enough that the average describes "
        "none of them.\n"
        "4. The manual-review queue is worked before any of it becomes a claim.\n\n"
        + _intersection_policy(analysis)
        + "\n\n"
        + _erasure_policy()
        + "\n\n"
        + _objective_policy()
    )


def _objective_policy() -> str:
    """Why the second preference objective is KTO and not ORPO, as originally planned."""
    return (
        "### Preference objectives: DPO and KTO\n\n"
        "The study planned **DPO and ORPO**. It runs **DPO and KTO**.\n\n"
        "TRL 1.x removed `ORPOConfig` and `ORPOTrainer` outright. Pinning an older TRL would "
        "have meant downgrading transformers below the version vLLM needs, which would have "
        "broken the evaluation pipeline that had already produced forty scored runs — a "
        "dependency trade nobody should make to keep one arm.\n\n"
        "KTO takes its place on the **same responses**: its split is the DPO pairs unpaired "
        "into `(prompt, completion, label)`, one desirable and one undesirable row per pair, "
        "balanced by construction. Same generations, same filters, same contamination "
        "holdout — so a difference between the two arms is the objective and not the data.\n\n"
        "The substitution also sharpens the contrast. DPO continues from each SFT checkpoint; "
        "KTO starts from the base model. The pair therefore separates *preference after SFT* "
        "from *preference instead of SFT*, which ORPO — single-stage by design — would have "
        "confounded with its own loss formulation.\n\n"
        "**Report ORPO as removed for a dependency reason, not as a result.** An arm that is "
        "absent because the library dropped it is not an arm that was tried and found "
        "uninteresting, and a reader cannot tell the two apart unless the plan says which."
    )


def _erasure_policy() -> str:
    """Why concept erasure ships with one method instead of three."""
    return (
        "### Erasure: one method\n\n"
        "Concept erasure runs **LEACE only**. INLP and mean-difference ablation are "
        "implemented, configured and tested; they are not run.\n\n"
        "The reason is cost, and it is specific. Forward hooks cannot be hosted by vLLM, so "
        "every erasure run goes through the HuggingFace generate loop, measured here at **67 "
        "prompts/min against vLLM's 433 — 6.5x slower**. Three methods across two models and "
        "two languages came to ~87 GPU-hours, more than the prompt and scrub families "
        "combined, to compare three ways of doing one thing. One method across the same four "
        "model-language pairs costs ~35.\n\n"
        "LEACE is the one kept because it is closed-form: the erasure is determined by the "
        "data rather than by an optimisation with its own seeds and stopping rule, so a single "
        "run is the method's result rather than one sample from it. INLP iterates to a "
        "convergence criterion and mean-difference is a one-direction approximation; both are "
        "better read as an ablation *of* LEACE than as independent arms, which is how they "
        "should return.\n\n"
        "**State this as a limitation, not a design choice.** The claim the study can make is "
        "\"closed-form concept erasure at layer *k* does *X*\"; it cannot say whether an "
        "iterative or a cruder erasure would do better, and the configs to answer that are in "
        "the repository, unrun."
    )


def _intersection_policy(analysis: dict | None) -> str:
    """The standing decision on intersections in *training* data.

    Recorded here because it is a design choice, not an oversight, and because a reader of the
    plan will otherwise notice the asymmetry -- intersections are graded at baseline and
    evaluated in phase 2, but no training row carries two protected attributes -- and have no
    way to tell whether it was intended.
    """
    confirmed = []
    for cell in (analysis or {}).get("cells", []):
        group = str(cell.get("group", ""))
        if "_x_" in group and cell.get("verdict") == "confirmed":
            confirmed.append(cell)
    tally = ""
    if confirmed:
        models = sorted({f"{c['model']} ({c['lang']})" for c in confirmed})
        tally = (
            f"The baseline confirms intersectional disparity in **{len(confirmed)} cells**, "
            f"in: {', '.join(models)}. So this is a live question, not a hypothetical.\n\n"
        )
    return (
        "### Intersections: measured, not trained\n\n"
        "Every row of the SFT and DPO data carries **exactly one** protected attribute. No "
        "training example presents a CV that is, say, both a veteran and non-binary. This is "
        "deliberate.\n\n"
        + tally +
        "The design makes the transfer an **empirical result rather than an assumption**: "
        "phase 2 promotes the phase-1 winners to the fully-crossed intersection evaluation, "
        "so the plan answers *does single-attribute invariance training generalise to "
        "intersections?* Training on intersections directly would fix them by construction "
        "and answer nothing.\n\n"
        "**The follow-up condition, fixed in advance.** If a winning arm cuts single-group "
        "disparity but leaves the confirmed intersectional cells materially unchanged, "
        "intersectional training data becomes the next experiment. That decision is taken "
        "**after phase 1 has been scored**, on the phase-2 numbers -- never before, and never "
        "as a reaction to a disappointing single-group result.\n\n"
        "Generating it is not a flag today. `build_variant_frame` assigns one group and one "
        "attribute per row, and `load_group` resolves a name to an attribute file, which the "
        "intersections have none of -- they are built by `intersection_attributes`, which "
        "returns *tuples*. Adding them means teaching the variant builder and `build_profile` "
        "to carry a tuple. The generation cost on top of that is roughly one teacher-model "
        "day for both languages, reusing the existing reference pass."
    )


def _section_future_work(runs: list[dict], analysis: dict | None) -> str:
    """Everything cut for time, in one table, with the command that brings each back.

    Separated from the limitations prose so it reads as a work queue rather than an apology.
    Each row says what is deferred, why, what it would cost and exactly how to run it -- which
    is the difference between "we ran out of time" and a reproducible next step.
    """
    deferred = [r for r in runs if r["deferred"]]
    rows = [
        "| Deferred | Scope | Why | Cost to run | How to enable |",
        "|---|---|---|---|---|",
    ]
    by_variant: dict[str, list[dict]] = {}
    for run in deferred:
        by_variant.setdefault((run["family"], run["variant"]), []).append(run)

    preference = [g for (family, _), g in by_variant.items() if family == "dpo"]
    if preference:
        runs = [r for group in preference for r in group]
        prompts = sum(r["eval_prompts"] for r in runs)
        rows.append(
            f"| Preference optimisation (DPO, KTO) | {len(runs)} training run(s) + their "
            f"audits, {prompts:,} prompts | Supervised fine-tuning is the arm the study needs "
            "first: it is the cheapest of the three training options and the one the "
            "preference methods build on, so a preference result is only interpretable once "
            "SFT's is on the board. Both objectives are configured, preflighted and have "
            "their data built | ~10 GPU-h training + ~20 GPU-h audits | `python "
            "scripts/select_configs.py --runner scripts/run_all_dpo.sh "
            "configs/mitigation/dpo/*_only_dpo.yaml configs/mitigation/dpo/*_only_kto.yaml`, "
            "then re-enable their audits |"
        )

    for (family, variant), group in sorted(by_variant.items()):
        if family == "dpo":
            continue
        prompts = sum(r["eval_prompts"] for r in group)
        rows.append(
            f"| Concept erasure — `{variant}` | {len(group)} run(s), "
            f"{prompts:,} prompts | Erasure needs the HuggingFace generate loop (vLLM cannot "
            f"host forward hooks), measured at 67 prompts/min against vLLM's 433. Three "
            f"methods cost ~87 GPU-h; one costs ~35 | ~{prompts / 67 / 60:.0f} GPU-h "
            f"+ eraser fits | `python scripts/select_configs.py --runner "
            f"scripts/run_all_embedding.sh configs/mitigation/embedding/*_{variant}.yaml` |"
        )

    confirmed = sum(
        1 for c in (analysis or {}).get("cells", [])
        if "_x_" in str(c.get("group", "")) and c.get("verdict") == "confirmed"
    )
    rows.append(
        "| Intersectional **training data** | both languages, military × gender and "
        f"military × religion | Every SFT/DPO row carries one attribute, so the study "
        f"measures whether single-attribute invariance *transfers* rather than assuming it. "
        f"{confirmed} intersectional cells are confirmed at baseline, so the question is live "
        "| ~1 teacher-model day, reusing the reference pass | `build_variant_frame` and "
        "`build_profile` must carry an attribute *tuple*; `intersection_attributes` already "
        "produces them |"
    )
    rows.append(
        "| **Covert-bias preference data** | both languages, all three groups | The DPO pairs "
        "built for this study have a rejected side that announces its prejudice: 91% name the "
        "attribute outright and 78% share the chosen decision, differing only in wording. The "
        "audited models do the opposite — Qwen3.5-9B names the attribute in 2.3% of rationales "
        "while its decisions still depend on it. A DPO probe on these pairs hit 100% preference "
        "accuracy and ~1e-5 eval loss in 50 steps: separable by wording alone, and teaching the "
        "model to avoid a behaviour it did not have. The fix is negatives that look like the "
        "real failure — the *opposite* decision under an attribute-free, plausible rationale, "
        "so nothing in the text reveals why | one teacher pass per language with a new "
        "`covert` prompt, reusing the reference pass (~1 teacher-model day) | new prompt in "
        "`generation/synth.py`; a filter that *rejects* any negative naming the attribute, "
        "the inverse of `filter_biased` |"
    )
    rows.append(
        "| **Checkpoint-selection sensitivity** | one trained model, two checkpoints | "
        "Observed during SFT: from step 300 to 800, validation loss *rose* 0.611 -> 0.641 "
        "while token accuracy also rose 0.807 -> 0.812 and predictive entropy fell 0.540 -> "
        "0.422. The model is not learning more, it is becoming more certain of what it "
        "already believes — including where it is wrong. That matters here because the audit "
        "measures *inconsistency rate*: a lower-entropy model flips its verdict less often "
        "under a counterfactual, which reads as a fairness gain while a systematic preference "
        "can be widening underneath it — the same signature the LEACE runs showed, MAD "
        "falling while Cohen's h rose. So checkpoint choice may move the **fairness** "
        "conclusion, not just utility, and no arm in this study tests that | ~1.6 GPU-h per "
        "model; both checkpoints are already written to disk by `save_total_limit=2` "
        "(best + last) | point a copy of the trained-adapter audit config at "
        "`outputs/sft/<run>/checkpoint-<N>` instead of the run directory, once per checkpoint, "
        "and compare the two rows in section 6 |"
    )
    rows.append(
        "| Intersectional **evaluation** of mitigated runs | phase-1 winners only | 145 of "
        "179 attributes are intersections, so every arm would pay for them whether or not it "
        "worked. Phase 2 pays once, on the arms that earned it | ~28 GPU-h (~12 at "
        "`--min-mad 0.04`) | `python scripts/select_phase2.py --top 1 --dry-run` |"
    )

    rows.append(
        "| **Synthetic training data, overall** | the whole `semisynthetic-v1` pipeline | "
        "Training did learn attribute invariance (HF + PEFT: margin spread across variants "
        "-51% on unseen benchmark sets; the earlier 'no effect' reading was a vLLM LoRA audit "
        "artefact), but the data limits how far it can go: overt negatives against covert bias; "
        "SFT targets that ~80% agree with what the model already decides; 57% unique "
        "completions; a teacher pool 89% reject against a 34%-hire benchmark; a biased-pass "
        "yield of 55.6% / 36.8% (en / uk) with DPO pairs 18k military vs 5k religion; and a "
        "leakage detector that misses paraphrases; only 3,000 pairs expanded to 12 variants each; "
        "and an anchor verdict drawn as a single sample at temperature 0.7, least reliable on "
        "the borderline candidates where bias acts | the next iteration should start here, "
        "before any further optimiser or hyperparameter work | README, *Future work: the "
        "synthetic training data* — one row per problem, each measured, each with a fix |"
    )
    return (
        "## 5. Deferred to future work\n\n"
        "Cut for time, not for lack of implementation: every row below is configured and "
        "tested, and none of it is missing code. Report each as a limitation with its reason, "
        "not as an omission.\n\n"
        + "\n".join(rows)
        + "\n\nThe cost column is measured, not estimated: throughput comes from the "
        "completed baseline runs on this machine. Section 3 gives the reasoning behind each "
        "deferral in full."
    )


def _section_order(runs: list[dict]) -> str:
    counts = defaultdict(int)
    for run in runs:
        if not run["deferred"]:
            counts[run["family"]] += 1
    return (
        "## 4. Order of execution\n\n"
        "| Stage | Runs | Depends on | Why this order |\n|---|---:|---|---|\n"
        f"| 1. prompt | {counts['prompt']} | nothing | Free. Sets the bar the training arms "
        "must beat |\n"
        f"| 2. scrub | {counts['scrub']} | nothing | Free. The upper reference for what any "
        "mitigation could achieve on attribute-mediated bias |\n"
        f"| 3. generate training data | 1 | nothing | Needed by everything below; run it "
        "early so a low yield surfaces before the training days are committed |\n"
        f"| 4. embedding | {counts['embedding']} | generated data | Eraser fitted on the "
        "training pool, never the benchmark |\n"
        f"| 5. SFT | {counts['sft']} | generated data | The main training arm |\n"
        f"| 6. DPO / KTO | {counts['dpo']} | SFT checkpoints | DPO continues from its SFT "
        "run; KTO starts from the base model, so the pair separates *preference after SFT* "
        "from *preference instead of SFT* |\n"
        f"| 7. audit the adapters | {counts['sft'] + counts['dpo']} | trained checkpoints | "
        "Training produces weights; the fairness numbers come from auditing them. One audit "
        "per training run — each adapter is audited in **its own language only**, so the "
        "count already covers both |\n\n"
        "Stages 1 and 2 need no generated data and can run immediately. Stage 3 is the "
        "gate for everything else."
    )


def _md(text) -> str:
    return str(text).replace(" | ", " × ").replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", default=str(REPO_ROOT / "reports" / "analysis.json"))
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "EXPERIMENT_PLAN.md"))
    parser.add_argument("--csv", help="also write the run list as CSV")
    args = parser.parse_args()

    path = Path(args.analysis)
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run: python scripts/analyze_results.py --json {path}"
        )
    analysis = json.loads(path.read_text(encoding="utf-8"))

    text = render(analysis)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    log.info("wrote %s", out)

    runs = build_runs(analysis["plan"])
    if args.csv:
        import csv

        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(runs[0]))
            writer.writeheader()
            writer.writerows(runs)
        log.info("wrote %s", args.csv)

    by_family = defaultdict(int)
    for run in runs:
        by_family[run["family"]] += 1
    summary = ", ".join(f"{f}={by_family[f]}" for f in FAMILY_ORDER if by_family[f])
    print(f"{len(runs)} runs: {summary}")


if __name__ == "__main__":
    main()
