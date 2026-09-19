"""Turning the evidence into a mitigation plan: what to run next, and why.

The point of the baseline stage is to spend the mitigation budget where it can produce an
answer. That means three kinds of cell, and a plan that omits any of them cannot support a
conclusion:

**Targets** — confirmed or probable disparity, large enough that a mitigation has room to
move it. These are where a mitigation is supposed to work.

**Controls** — cells already clean. A mitigation that *worsens* a clean cell is as
informative as one that fixes a bad one, and without controls a study cannot distinguish "the
mitigation removed a disparity" from "the mitigation flattened everything, including what was
already fine". Every serious target model contributes at least one.

**Rescue arms** — models whose measurement failed the utility gate, but which still *rank*
candidates and simply threshold them badly. That is a calibration fault, and the
counterfactually-consistent training set pins every verdict to the attribute-free reference
decision, so supervised fine-tuning targets it directly. The primary outcome for these arms
is **utility, not disparity**: the question is whether tuning makes the model able to do the
task at all, and its fairness becomes measurable only *if* utility crosses the gate
afterwards. Recording that as a pre-registered conditional matters -- otherwise a post-hoc
fairness number from a still-unusable model finds its way into a table.

**Excluded** -- cells whose measurement failed the quality gate for a reason tuning cannot
fix. Named explicitly, with the
reason, because "we did not mitigate LAPA" and "LAPA could not be measured" are different
claims and only one of them is true.

The output is a plan file plus the exact config paths to enable, so the next stage is a
selection rather than a re-derivation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..data.protected_groups import INTERSECTIONS
from ..utils.config import REPO_ROOT
from .evidence import VERDICT_RANK, CellFinding, Thresholds, Verdict, is_rescuable

#: How much of the benchmark grid each mitigation run evaluates.
#:
#: `full` measures every group and intersection on every run — the safest, and the most
#: expensive: 161,550 prompts per run, of which the two intersections are 56%.
#:
#: `targeted` measures only the groups that showed a confirmed disparity for that model and
#: language, plus that model's designated control group. Cheaper, and the control is what
#: keeps the no-harm check alive.
#:
#: `targeted-no-intersections` additionally drops the intersections. They were run in the
#: baseline to test non-additivity, which is a question about the untreated model; the
#: mitigation question is whether the disparity fell, and the single groups answer it. The
#: cost is that non-additivity *under mitigation* goes unmeasured — add the intersections back
#: for the winning arm only.
EVAL_SCOPES = ("full", "targeted", "targeted-no-intersections")

#: Mitigation families, and what each is expected to be good for. Used to order the
#: suggestions per target, cheapest-first, and to say why a family is suggested at all.
FAMILY_RATIONALE = {
    "prompt": (
        "zero training cost — run first, and it sets the bar the training arms must beat"
    ),
    "scrub": (
        "removes the attribute before the model sees it; the natural upper reference for "
        "what any mitigation could achieve on attribute-mediated bias"
    ),
    "embedding": (
        "erases the attribute direction from the residual stream; worth it where the "
        "disparity survives prompting, since it changes no weights"
    ),
    "sft": (
        "trains counterfactual consistency directly — the property the inconsistency rate "
        "measures"
    ),
    "dpo": (
        "adds a preference between an invariant and an attribute-driven response; the "
        "SFT-vs-SFT+preference ablation only means something where SFT alone left something"
    ),
}

#: Slug used in generated config filenames, per model repo id.
MODEL_SLUGS = {
    "Qwen3.5-4B": "qwen3.5-4b",
    "Qwen3.5-9B": "qwen3.5-9b",
    "gemma-4-E4B-it": "gemma-4-e4b",
    "gemma-4-12B-it": "gemma-4-12b",
    "lapa-v0.1.2-instruct": "lapa-12b",
}


@dataclass
class Target:
    role: str  # "target" | "control" | "rescue"
    model: str
    model_slug: str
    lang: str
    group: str
    condition: str
    verdict: str
    ar_mad: float
    ar_range: float
    utility: float
    n_significant_fdr: int
    headline_attributes: list[str]
    rationale: str
    priority: int
    suggested_families: list[str] = field(default_factory=list)
    #: For rescue arms: what success looks like, and what may be claimed from it.
    primary_outcome: str = "disparity"
    discrimination: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Excluded:
    model: str
    lang: str
    group: str
    condition: str
    reasons: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    thresholds: dict
    eval_scope: str
    eval_groups: dict
    eval_conditions: dict
    targets: list[Target]
    controls: list[Target]
    rescues: list[Target]
    excluded: list[Excluded]
    config_paths: dict[str, list[str]]
    notes: list[str]

    def as_dict(self) -> dict:
        return {
            "thresholds": self.thresholds,
            "eval_scope": self.eval_scope,
            "eval_groups": self.eval_groups,
            "eval_conditions": self.eval_conditions,
            "notes": self.notes,
            "targets": [t.as_dict() for t in self.targets],
            "controls": [c.as_dict() for c in self.controls],
            "rescues": [r.as_dict() for r in self.rescues],
            "excluded": [e.as_dict() for e in self.excluded],
            "config_paths": self.config_paths,
        }


def _slug(model: str) -> str:
    return MODEL_SLUGS.get(model, model.lower().replace(".", "").replace(" ", "-"))


def _headline(cell: CellFinding, n: int = 3) -> list[str]:
    """The attributes a reader would quote for this cell."""
    ranked = [a for a in cell.top_attributes(n * 2) if a.material or a.significant_fdr]
    return [
        f"{a.attribute} ({a.gap_vs_reference:+.1%} vs reference, {a.direction})"
        for a in ranked[:n]
    ]


def _families_for(cell: CellFinding) -> list[str]:
    """Which mitigation families to try on this cell, cheapest-first.

    Everything gets the zero-cost families. The training arms are suggested only where the
    disparity is large enough that a training run could show a difference against them --
    fine-tuning a model to fix a 2-point gap spends GPU-days to chase measurement noise.
    """
    families = ["prompt", "scrub"]
    if cell.ar_mad == cell.ar_mad and cell.ar_mad >= 0.04:
        families += ["embedding", "sft", "dpo"]
    elif cell.ar_mad == cell.ar_mad and cell.ar_mad >= 0.03:
        families += ["embedding", "sft"]
    return families


def build_plan(
    findings: list[CellFinding],
    thresholds: Thresholds | None = None,
    max_targets: int | None = None,
    eval_scope: str = "full",
) -> Plan:
    """Selects targets, controls and exclusions from the graded cells."""
    thresholds = thresholds or Thresholds()
    if eval_scope not in EVAL_SCOPES:
        raise ValueError(f"unknown eval_scope {eval_scope!r}; expected one of {EVAL_SCOPES}")
    notes: list[str] = []

    excluded = [
        Excluded(c.model, c.lang, c.group, c.condition, c.gate_reasons)
        for c in findings
        if not c.gate_passed
    ]
    usable = [c for c in findings if c.gate_passed]

    candidates = [
        c
        for c in usable
        if VERDICT_RANK[c.verdict] >= VERDICT_RANK[Verdict.PROBABLE]
        and c.ar_mad == c.ar_mad
        and c.ar_mad >= thresholds.material_mad
    ]
    if eval_scope == "targeted-no-intersections":
        dropped = [c for c in candidates if c.group in INTERSECTIONS]
        candidates = [c for c in candidates if c.group not in INTERSECTIONS]
        # A model whose only confirmed disparity is in an intersection has nothing left to
        # mitigate once intersections are out of scope. Dropping it is the honest
        # consequence, and saying which models that removed keeps the decision visible.
        orphaned = {
            (c.model, c.lang)
            for c in dropped
            if not any(k.model == c.model and k.lang == c.lang for k in candidates)
        }
        if dropped:
            notes.append(
                f"eval_scope=targeted-no-intersections dropped {len(dropped)} intersection "
                f"target(s). Non-additivity under mitigation goes unmeasured — add the "
                f"intersections back for the winning arm before publishing a claim about it."
            )
        for model, lang in sorted(orphaned):
            notes.append(
                f"{model} · {lang}: **removed from the mitigation stage** — its only "
                f"confirmed disparity was in an intersection, which this scope excludes."
            )
    candidates.sort(key=lambda c: (-VERDICT_RANK[c.verdict], -c.ar_mad))
    if max_targets:
        candidates = candidates[:max_targets]

    targets = [
        Target(
            role="target",
            model=c.model, model_slug=_slug(c.model), lang=c.lang,
            group=c.group, condition=c.condition, verdict=c.verdict.value,
            ar_mad=c.ar_mad, ar_range=c.ar_range, utility=c.utility,
            n_significant_fdr=c.n_significant_fdr,
            headline_attributes=_headline(c),
            rationale=(
                f"{c.verdict.value}: {c.n_significant_fdr} of {c.n_attributes} attributes "
                f"differ significantly after FDR correction, MAD {c.ar_mad:.1%}, "
                f"range {c.ar_range:.1%}, utility {c.utility:.1%}"
            ),
            priority=index + 1,
            suggested_families=_families_for(c),
        )
        for index, c in enumerate(candidates)
    ]

    # Controls: for each model that has a target, the cleanest cell in the same language.
    target_models = {(t.model, t.lang) for t in targets}
    controls: list[Target] = []
    for model, lang in sorted(target_models):
        clean = [
            c
            for c in usable
            if c.model == model and c.lang == lang and c.verdict == Verdict.CLEAN
            and c.ar_mad == c.ar_mad
        ]
        if not clean:
            notes.append(
                f"{model} · {lang}: no clean cell available as a no-harm control — every "
                f"measurable cell shows some disparity. Interpret its mitigation results "
                f"without that check."
            )
            continue
        cleanest = min(clean, key=lambda c: c.ar_mad)
        controls.append(
            Target(
                role="control",
                model=cleanest.model, model_slug=_slug(cleanest.model), lang=cleanest.lang,
                group=cleanest.group, condition=cleanest.condition,
                verdict=cleanest.verdict.value, ar_mad=cleanest.ar_mad,
                ar_range=cleanest.ar_range, utility=cleanest.utility,
                n_significant_fdr=cleanest.n_significant_fdr,
                headline_attributes=[],
                rationale=(
                    f"cleanest measurable cell for this model and language "
                    f"(MAD {cleanest.ar_mad:.1%}) — included to check the mitigation does no "
                    f"harm where there was nothing to fix"
                ),
                priority=0,
                suggested_families=["prompt", "scrub"],
            )
        )

    rescues = _build_rescues(findings, thresholds, notes)

    if not targets:
        notes.append(
            "No cell met the bar for a mitigation run. Either the models are cleaner than the "
            "thresholds assume, or the thresholds are too strict — check the graded cells "
            "before concluding the former."
        )

    eval_groups = _eval_groups(targets, controls, rescues, findings, eval_scope)
    eval_conditions = _eval_conditions(targets, rescues, eval_scope)
    return Plan(
        thresholds=thresholds.as_dict(),
        eval_scope=eval_scope,
        eval_groups=eval_groups,
        eval_conditions=eval_conditions,
        targets=targets,
        controls=controls,
        rescues=rescues,
        excluded=excluded,
        config_paths=resolve_config_paths([*targets, *controls, *rescues]),
        notes=notes,
    )


#: Prompt strategies that impose an explicit decision procedure with a numeric threshold.
#: The cheap thing to try on a model that ranks candidates but accepts too many -- Salinas et
#: al. (2025) find numeric, decision-relevant anchors counteract bias where qualitative detail
#: does not, and the same anchoring is what a miscalibrated threshold needs.
ANCHORING_STRATEGIES = ("structured_rubric", "recruiter_guidelines", "zero_shot_cot")


def _build_rescues(
    findings: list[CellFinding], thresholds: Thresholds, notes: list[str]
) -> list[Target]:
    """One rescue arm per (model, language) that failed on utility but still ranks candidates.

    Granularity is per model-language, not per cell: the fault is a property of the run, and
    the fix -- teaching the model where to put its threshold -- is not group-specific.
    """
    by_run: dict[tuple[str, str], list[CellFinding]] = {}
    for cell in findings:
        if is_rescuable(cell, thresholds):
            by_run.setdefault((cell.model, cell.lang), []).append(cell)

    rescues: list[Target] = []
    for (model, lang), cells in sorted(by_run.items()):
        # Require the signature to hold across the run, not in one lucky cell.
        if len(cells) < 2:
            continue
        discrimination = sum(c.discrimination for c in cells) / len(cells)
        utility = sum(c.utility for c in cells) / len(cells)
        worst = max(cells, key=lambda c: c.ar_mad if c.ar_mad == c.ar_mad else -1)
        rescues.append(
            Target(
                role="rescue",
                model=model, model_slug=_slug(model), lang=lang,
                group="(all groups)", condition="(all conditions)",
                verdict=Verdict.NOT_INTERPRETABLE.value,
                ar_mad=worst.ar_mad, ar_range=worst.ar_range, utility=utility,
                n_significant_fdr=0, headline_attributes=[],
                rationale=(
                    f"utility {utility:.1%} is below the gate, but the model still ranks "
                    f"candidates (accepts {discrimination:.1%} more of the pairs the "
                    f"attribute-free reference would hire than of those it would reject). "
                    f"That is a threshold-calibration fault, and SFT pins every training "
                    f"verdict to the reference decision, so it targets it directly."
                ),
                priority=0,
                # Tuning only. A prompt edit cannot move a model's decision threshold the way
                # training on reference-pinned verdicts does, and the arm exists to test
                # exactly that mechanism -- adding prompt runs would spend inference hours on
                # a hypothesis this arm is not designed to answer.
                suggested_families=["sft", "dpo"],
                primary_outcome="utility",
                discrimination=discrimination,
            )
        )
        notes.append(
            f"{model} · {lang}: entered as a **rescue arm**. Primary outcome is utility, not "
            f"disparity. Its fairness numbers stay uninterpretable unless post-tuning utility "
            f"reaches {thresholds.min_utility:.0%} — record that as a condition now, so a "
            f"fairness figure from a still-unusable model cannot be reported after the fact."
        )
    return rescues


def _eval_groups(
    targets: list[Target], controls: list[Target], rescues: list[Target],
    findings: list[CellFinding], eval_scope: str,
) -> dict[str, dict[str, list[str]]]:
    """Which protected groups each model-language run should evaluate.

    Keyed `{model_slug: {lang: [groups]}}` so `generate_experiment_configs.py --eval-scope`
    can restrict the mitigation configs directly.
    """
    all_groups = sorted({c.group for c in findings})
    if eval_scope == "full":
        return {
            t.model_slug: {lang: all_groups for lang in ("en", "uk")}
            for t in [*targets, *controls, *rescues]
        }

    # Strictly the groups being mitigated. Controls are NOT added: a mitigation run measures
    # what it is meant to fix.
    #
    # The cost, stated plainly: cross-group damage becomes invisible to that run. A mitigation
    # that fixes military status while breaking religion would not be caught here. What still
    # guards against it is the attribute-free condition, which every run keeps -- it catches
    # a model whose whole operating point moved, which is how gross damage shows up. Damage
    # confined to one untested group is not detectable, and the write-up must say so rather
    # than imply a no-harm check that was not run.
    out: dict[str, dict[str, list[str]]] = {}
    for entry in targets:
        groups = out.setdefault(entry.model_slug, {}).setdefault(entry.lang, [])
        if entry.group not in groups:
            groups.append(entry.group)
    # A rescue arm gets the SAME groups the fairness track evaluates -- never a narrower
    # slice, for three reasons.
    #
    # First, `targeted` scope is derived from baseline evidence: keep the groups that showed
    # a confirmed disparity for this model. A rescue arm has no interpretable baseline at all
    # -- that is why it is a rescue arm -- so there is no evidence on which to narrow it.
    # Applying the targeting rule to it is a category error.
    #
    # Second, the conditional secondary outcome is exactly a fairness question: *if* utility
    # crosses the gate, what disparity does this model then show? Answering it on one group
    # would mean re-running everything.
    #
    # Third, comparability: the point of rescuing the only Ukrainian-native model is to put
    # its fairness profile beside the general-purpose ones, and that requires the same groups.
    #
    # The cost is small either way -- a rescue arm is a handful of runs.
    fairness_groups = sorted(
        {g for entry in [*targets, *controls] for g in [entry.group]}
        - set(INTERSECTIONS)
    ) or ["military_status", "gender", "religion"]
    for entry in rescues:
        out.setdefault(entry.model_slug, {})[entry.lang] = list(fairness_groups)

    if eval_scope == "targeted-no-intersections":
        for langs in out.values():
            for lang, groups in langs.items():
                kept = [g for g in groups if g not in INTERSECTIONS]
                langs[lang] = kept or ["military_status"]
    return {m: {lang: sorted(g) for lang, g in langs.items()} for m, langs in out.items()}


def _eval_conditions(
    targets: list[Target], rescues: list[Target], eval_scope: str
) -> dict[str, dict[str, list[str]]]:
    """Which injection conditions each model-language run should evaluate.

    A target is a cell -- a group *under a condition*. Target #1 is "military status,
    explicit"; measuring it under implicit as well doubles the cost to answer a question that
    cell did not raise. So a run evaluates the conditions its own targets were found in.

    `attr_free` is always included and is not optional. It is one extra pass over the 450
    pairs, and it is the only thing that distinguishes "the disparity fell" from "the model's
    whole operating point moved" -- without it a mitigation that simply rejects everyone
    scores perfectly on every fairness column.
    """
    if eval_scope == "full":
        return {}

    out: dict[str, dict[str, list[str]]] = {}
    for entry in targets:
        conditions = out.setdefault(entry.model_slug, {}).setdefault(entry.lang, [])
        if entry.condition not in conditions:
            conditions.append(entry.condition)
    # A rescue arm is judged on utility, which needs no injected attribute at all -- but the
    # conditional secondary outcome is a fairness question, so both framings are kept.
    for entry in rescues:
        out.setdefault(entry.model_slug, {})[entry.lang] = ["explicit", "implicit"]

    return {
        model: {lang: [*sorted(c), "attr_free"] for lang, c in langs.items()}
        for model, langs in out.items()
    }


def resolve_config_paths(targets: list[Target]) -> dict[str, list[str]]:
    """Maps the plan onto config files that actually exist on disk.

    Config granularity is coarser than the plan's: a mitigation config covers a whole
    model x language, not a single group x condition, because a single audit run evaluates
    every group at once. So one target implies one config per family, and several targets on
    the same model and language collapse into the same set.

    Paths that do not exist are reported under `missing` rather than silently dropped -- a
    plan naming a config nobody generated is a plan that will fail at 3am.
    """
    wanted: dict[str, set[str]] = {}
    missing: set[str] = set()

    for target in targets:
        for family in target.suggested_families:
            for path in _candidate_paths(family, target):
                if (REPO_ROOT / path).exists():
                    wanted.setdefault(family, set()).add(path)
                else:
                    missing.add(path)

    out = {family: sorted(paths) for family, paths in sorted(wanted.items())}
    if missing:
        out["missing"] = sorted(missing)
    return out


def _candidate_paths(family: str, target: Target) -> list[str]:
    """The config paths a family would use for one target."""
    slug, lang = target.model_slug, target.lang
    if family == "prompt":
        from ..data.prompts import STRATEGIES

        # A rescue arm gets only the strategies that impose an explicit decision procedure
        # with a numeric threshold. The rest target attribute sensitivity, which is not this
        # model's problem -- and running all eight would triple the arm's cost for nothing.
        strategies = (
            ANCHORING_STRATEGIES
            if target.role == "rescue"
            else [s for s in STRATEGIES if s != "baseline"]
        )
        return [
            f"configs/mitigation/prompt/{slug}_{lang}_{strategy}.yaml"
            for strategy in strategies
        ]
    if family == "scrub":
        return [f"configs/mitigation/scrub/{slug}_{lang}_{mode}.yaml"
                for mode in ("lexical", "llm")]
    if family == "embedding":
        return [f"configs/mitigation/embedding/{slug}_{lang}_{method}.yaml"
                for method in ("leace", "inlp", "mean_diff")]
    if family == "sft":
        # Training configs are per model, not per language: the generated dataset carries
        # both, and a data-config view selects the slice.
        #
        # A rescue arm trains on `all_groups` only. Its fault is where the model puts its
        # decision threshold, which is not group-specific -- narrowing the data to military
        # status would shrink the training set for no reason. Track A keeps the narrow view
        # as a genuine ablation: does training on the dominant group alone generalise to the
        # others?
        # One run per language, on the full protected-group set.
        #
        # Per language, not both at once, for a reason beyond cost: Finding 2 is that the
        # military-status effect *reverses* direction between English and Ukrainian. Training
        # on both together lets the two halves pull against each other, and a null result
        # would then be uninterpretable -- was the mitigation ineffective, or did it cancel?
        #
        # The full group set rather than a military-only ablation: the ablation answers "does
        # training on the dominant group generalise", which is a second-order question. It is
        # the first thing to cut when time is short, and it can be added back later against
        # the same baselines.
        return [f"configs/mitigation/sft/{slug}_{target.lang}_only.yaml"]
    if family == "dpo":
        return [f"configs/mitigation/dpo/{slug}_{target.lang}_only_{objective}.yaml"
                for objective in ("dpo", "kto")]
    return []


def write_plan(plan: Plan, path: str | Path) -> Path:
    """Writes the plan as YAML, ready to hand to the next stage."""
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Mitigation plan, generated by scripts/analyze_results.py from eval/results/.\n"
        "# Regenerate rather than editing: it is derived from the audit records, and the\n"
        "# thresholds it applied are recorded below so a reader can see what was assumed.\n"
        "#\n"
        "# `targets`  cells with confirmed or probable disparity and room to move it\n"
        "# `controls` clean cells, to check a mitigation does no harm\n"
        "# `excluded` cells whose measurement failed the quality gate, with the reason\n\n"
    )
    path.write_text(
        header + yaml.safe_dump(plan.as_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _trained_audit_configs(config_paths: dict[str, list[str]]) -> list[str]:
    """Audit configs for the trained adapters the plan enables.

    Training produces weights; every fairness number comes from auditing them. The plan
    enumerates training runs, so the audits have to be derived from it -- enabling the training
    stages alone would burn ~60 GPU-hours and leave nothing to put in a table.

        sft/<key>_<lang>_only.yaml           -> audit/<key>_<lang>_sft_<lang>_only.yaml
        dpo/<key>_<lang>_only_<variant>.yaml -> audit/<key>_<lang>_dpo_<lang>_only_<variant>.yaml
    """
    audits: list[str] = []
    for family in ("sft", "dpo"):
        for config in config_paths.get(family, []):
            stem = Path(config).stem
            key, lang = stem.split("_", 2)[:2]
            tail = stem.split("_", 1)[1]
            audits.append(str(Path("configs/audit") / f"{key}_{lang}_{family}_{tail}.yaml"))
    return sorted(dict.fromkeys(audits))


def write_enable_script(plan: Plan, path: str | Path) -> Path:
    """Writes a shell script that enables exactly the planned configs.

    Emitted rather than applied: rewriting `scripts/run_all_*.sh` is a change to what will
    consume days of GPU, and that should be a deliberate act with a diff you can read first.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "# Enables the configs named by the mitigation plan. Generated by",
        "# scripts/analyze_results.py -- review the plan first, then run this.",
        "#",
        "# It uncomments the planned entries in each scripts/run_all_*.sh and comments out",
        "# everything else, so what runs is exactly what the plan asked for.",
        "set -euo pipefail",
        'cd "$(dirname "$0")/.."',
        "",
    ]
    runner_for = {
        "prompt": "run_all_prompt.sh", "scrub": "run_all_scrub.sh",
        "embedding": "run_all_embedding.sh", "sft": "run_all_sft.sh",
        "dpo": "run_all_dpo.sh",
    }
    for family, paths in plan.config_paths.items():
        if family == "missing" or family not in runner_for:
            continue
        lines.append(f"# ---- {family}: {len(paths)} config(s) ----")
        lines.append(f'python scripts/select_configs.py --runner scripts/{runner_for[family]} \\')
        for index, config in enumerate(paths):
            terminator = " \\" if index < len(paths) - 1 else ""
            lines.append(f"    {config}{terminator}")
        lines.append("")
    audits = _trained_audit_configs(plan.config_paths)
    existing = [c for c in audits if Path(c).exists()]
    if existing:
        lines.append(f"# ---- audit (trained adapters): {len(existing)} config(s) ----")
        lines.append('python scripts/select_configs.py --runner scripts/run_all_audit.sh \\')
        for index, config in enumerate(existing):
            terminator = " \\" if index < len(existing) - 1 else ""
            lines.append(f"    {config}{terminator}")
        lines.append("")
    for config in audits:
        if not Path(config).exists():
            plan.config_paths.setdefault("missing", []).append(config)

    if plan.config_paths.get("missing"):
        lines.append("# Configs the plan wanted but which do not exist on disk:")
        for config in plan.config_paths["missing"]:
            lines.append(f"#   {config}")
        lines.append("# Run: python scripts/generate_experiment_configs.py --no-runners")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path
