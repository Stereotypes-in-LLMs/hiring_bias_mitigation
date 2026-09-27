"""Rendering the deep-dive analysis as markdown.

`reports/RESULTS.md` presents the measurements. This produces `reports/ANALYSIS.md`, which
argues from them: which cells carry evidence of attribute-driven disparity, which cells
cannot be read at all, what recurs across models and languages, and what the next stage
should run.

The two are kept separate on purpose. A measurement report that also editorialises is hard to
check; an argument that hides its thresholds is hard to disagree with. This one states them
at the top.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .crosscut import ModelSummary, PairedComparison, Recurrence
from .evidence import CellFinding, Thresholds, Verdict, is_rescuable

VERDICT_LABEL = {
    Verdict.CONFIRMED: "**confirmed**",
    Verdict.PROBABLE: "probable",
    Verdict.WEAK: "weak",
    Verdict.CLEAN: "clean",
    Verdict.NOT_INTERPRETABLE: "⚠ not interpretable",
}


def _pct(value, digits: int = 1) -> str:
    if value is None or value != value:
        return "--"
    return f"{100 * float(value):.{digits}f}"


def _num(value, digits: int = 3) -> str:
    if value is None or value != value:
        return "--"
    return f"{float(value):.{digits}f}"


def _cell(text) -> str:
    return str(text).replace(" | ", " × ").replace("|", "\\|")


def render(
    findings: list[CellFinding],
    recurrences: list[Recurrence],
    language: list[PairedComparison],
    condition: list[PairedComparison],
    models: list[ModelSummary],
    plan,
    thresholds: Thresholds,
) -> str:
    sections = [
        _header(findings, thresholds),
        _section_verdicts(findings),
        _section_excluded(findings),
        _section_recurrence(recurrences),
        _section_condition(condition),
        _section_language(language),
        _section_models(models),
        _section_plan(plan),
        _footer(),
    ]
    return "\n\n".join(s for s in sections if s)


def _header(findings: list[CellFinding], thresholds: Thresholds) -> str:
    usable = [c for c in findings if c.gate_passed]
    confirmed = sum(1 for c in usable if c.verdict == Verdict.CONFIRMED)
    probable = sum(1 for c in usable if c.verdict == Verdict.PROBABLE)
    return (
        "# Where is there actually bias?\n\n"
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by "
        f"`scripts/analyze_results.py` from `eval/results/`. Derived — do not edit by hand.*\n\n"
        "This is the argument. The measurements it argues from are in "
        "[`RESULTS.md`](RESULTS.md), and every metric is defined in "
        "[`docs/METRICS.md`](../docs/METRICS.md).\n\n"
        f"**{len(findings)} cells graded** (model × language × protected group × condition): "
        f"{len(findings) - len(usable)} could not be measured, {confirmed} show confirmed "
        f"disparity, {probable} probable.\n\n"
        "## What it takes to be called bias here\n\n"
        "Three things have to hold together. None of them is sufficient alone.\n\n"
        "1. **The measurement has to be trustworthy.** Checked first. A model that agrees "
        "with the attribute-free reference at close to chance is not screening candidates, "
        "and a disparity measured on top of that is not evidence about fairness.\n"
        "2. **Significance after FDR correction.** At 161,550 rows per run an uncorrected "
        "p-value flags nearly anything. The paired test — which the matched counterfactual "
        "design licenses and the audit study could not use — is the stronger one.\n"
        "3. **An effect size worth reporting.** A significant one-point gap at this sample "
        "size is significant because *n* is large, not because it matters to a candidate.\n\n"
        "| Verdict | Meaning |\n|---|---|\n"
        "| **confirmed** | significant after FDR, material effect size, paired test agrees |\n"
        "| probable | significant after FDR and material, but the paired test does not agree |\n"
        "| weak | one of significance or effect size, not both — a scrutiny prompt, "
        "not a finding |\n"
        "| clean | no material disparity detected |\n"
        "| ⚠ not interpretable | the quality gate failed; see the exclusions section |\n\n"
        "### Thresholds applied\n\n"
        "**These are judgement calls, not statistics.** There is no principled point at which "
        "a hiring disparity becomes acceptable — it depends on the jurisdiction, the base "
        "rate, and how many candidates pass through the system. They are stated so a reader "
        "can disagree with them specifically, and rerun with `--min-utility`, "
        "`--material-gap` and the rest.\n\n"
        "| Threshold | Value | Role |\n|---|---:|---|\n"
        f"| `min_utility` | {thresholds.min_utility:.0%} | below this a run's disparity is "
        "not read as bias |\n"
        f"| `material_gap` | {thresholds.material_gap:.0%} | acceptance-rate gap vs the "
        "group's reference level |\n"
        f"| `material_h` | {thresholds.material_h:.2f} | Cohen's *h*, for gaps that are small "
        "in points but large against a low base rate |\n"
        f"| `material_mad` | {thresholds.material_mad:.0%} | cell-level disparity worth a "
        "mitigation run |\n"
        f"| `max_unusable` | {thresholds.max_unusable:.0%} | refusals plus parse failures |\n"
        f"| `alpha` | {thresholds.alpha} | significance level, matching the audit |\n"
    )


def _verdict_rows(cells: list[CellFinding]) -> list[str]:
    rows = []
    for cell in cells:
        headline = ", ".join(
            f"{_cell(a.attribute)} ({a.gap_vs_reference:+.1%})"
            for a in cell.top_attributes(2)
            if a.material or a.significant_fdr
        )
        rows.append(
            f"| {VERDICT_LABEL[cell.verdict]} | {cell.model} | {cell.lang} | "
            f"{_cell(cell.group)} | {cell.condition} | {cell.n_attributes} | "
            f"{_pct(cell.ar_mad)} | {_pct(cell.ar_range)} | "
            f"{cell.n_significant_fdr}/{cell.n_attributes} | {_pct(cell.utility)} | "
            f"{_pct(cell.discrimination)} | {headline or '—'} |"
        )
    return rows


VERDICT_HEADER = [
    "| Verdict | Model | Lang | Protected group | Condition | Attrs | MAD (pp) | "
    "Range (pp) | Sig. after FDR | Utility % | Discrim. (pp) | Headline attributes |",
    "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
]


def _section_verdicts(findings: list[CellFinding]) -> str:
    """Every cell, interpretable or not.

    Gated cells are listed too, below the interpretable ones and marked. Omitting their
    numbers would make the exclusion unverifiable from the report itself -- a reader has to be
    able to see what was set aside and check that setting it aside was right.
    """
    by_mad = lambda c: -(c.ar_mad if c.ar_mad == c.ar_mad else -1)  # noqa: E731
    usable = sorted((c for c in findings if c.gate_passed), key=by_mad)
    gated = sorted((c for c in findings if not c.gate_passed), key=by_mad)

    intro = (
        "## Graded cells\n\n"
        "Ordered by disparity. `Sig. after FDR` counts attributes whose acceptance rate "
        "differs significantly from the population value once corrected across every test in "
        "the run. `Discrim.` is the acceptance-rate spread between pairs the attribute-free "
        "reference would hire and pairs it would reject — whether the model ranks candidates "
        "at all, independently of where it puts its threshold.\n\n"
    )
    blocks = []
    if usable:
        blocks.append(
            "### Interpretable\n\n" + "\n".join(VERDICT_HEADER + _verdict_rows(usable))
        )
    if gated:
        blocks.append(
            "### Not interpretable — shown for scrutiny, excluded from every claim\n\n"
            "**These numbers must not be read as fairness results.** They are printed so the "
            "exclusion can be checked rather than taken on trust: a reader should be able to "
            "see what was set aside and judge whether setting it aside was right. The reason "
            "for each is in the next section.\n\n"
            "Note the `Discrim.` column in particular — it separates a model whose decision "
            "is unrelated to the candidate from one that ranks candidates but thresholds "
            "them badly. The second is a calibration fault, and it is fixable by tuning.\n\n"
            + "\n".join(VERDICT_HEADER + _verdict_rows(gated))
        )
    return intro + "\n\n".join(blocks) if blocks else intro + "_No cells._"


def _section_excluded(findings: list[CellFinding]) -> str:
    gated = [c for c in findings if not c.gate_passed]
    if not gated:
        return ""
    by_model: dict[tuple, list] = {}
    for cell in gated:
        by_model.setdefault((cell.model, cell.lang), []).append(cell)

    blocks = []
    for (model, lang), cells in sorted(by_model.items()):
        reasons = sorted({r for c in cells for r in c.gate_reasons})
        utilities = [c.utility for c in cells if c.utility == c.utility]
        discriminations = [c.discrimination for c in cells if c.discrimination == c.discrimination]
        stats = ""
        if utilities:
            stats = (
                f"\n\nUtility {min(utilities):.1%}–{max(utilities):.1%} "
                f"(mean {sum(utilities) / len(utilities):.1%})"
            )
            if discriminations:
                mean_discrimination = sum(discriminations) / len(discriminations)
                stats += f", discrimination mean {mean_discrimination:.1%}"
                # The calibration reading only applies where the gate failed *on utility*.
                # A run excluded for being partial, or for unparsable output, is not a
                # miscalibrated model -- and saying so would send a smoke run to the tuning
                # queue.
                if any(is_rescuable(c) for c in cells):
                    stats += (
                        " — **ranks candidates, thresholds them badly**: a calibration "
                        "fault, and a candidate for a tuning rescue arm rather than an "
                        "outright exclusion."
                    )
                elif mean_discrimination < 0.15:
                    stats += (
                        " — the decision is close to unrelated to the candidate, which "
                        "training on decisions cannot fix."
                    )
        blocks.append(
            f"**{model} · {lang}** — {len(cells)} cell(s){stats}\n\n"
            + "\n".join(f"- {r}" for r in reasons)
        )
    return (
        "## Excluded from the analysis\n\n"
        "These cells produced numbers, and the numbers are in `RESULTS.md`. They are not read "
        "as evidence about bias here, for the reasons given.\n\n"
        "**This is a finding, not a gap.** *\"We did not mitigate this model\"* and *\"this "
        "model could not be measured\"* are different claims, and conflating them would let a "
        "model that fails the task be reported as the most biased one.\n\n"
        + "\n\n".join(blocks)
    )


def _section_recurrence(recurrences: list[Recurrence]) -> str:
    interesting = [
        r for r in recurrences
        if r.n_confirmed >= 1 and r.max_abs_gap == r.max_abs_gap and r.max_abs_gap >= 0.03
    ][:25]
    if not interesting:
        return ""
    rows = [
        "| Protected group | Attribute | Cells | Confirmed | Direction | Mean gap (pp) | "
        "Max gap (pp) | Models | Langs |",
        "|---|---|---:|---:|---|---:|---:|---|---|",
    ]
    for r in interesting:
        rows.append(
            f"| {_cell(r.group)} | {_cell(r.attribute)} | {r.n_cells} | {r.n_confirmed} | "
            f"{r.consistent_direction} | {_pct(r.mean_gap)} | {_pct(r.max_abs_gap)} | "
            f"{len(r.models)} | {', '.join(r.langs)} |"
        )
    return (
        "## Which attributes recur\n\n"
        "The audit study's own standard for what may be argued from: patterns that recur "
        "across models, languages and conditions — not isolated flags, which multiple "
        "comparisons produce even from a fair model.\n\n"
        "`Direction` is only reported as consistent when it dominates **and** recurs in at "
        "least two cells; otherwise it reads `mixed` or `insufficient`. An attribute "
        "penalised by one model in one language is a curiosity. The same attribute penalised "
        "by several models in both languages is a finding about the attribute.\n\n"
        + "\n".join(rows)
    )


def _section_condition(comparisons: list[PairedComparison]) -> str:
    if not comparisons:
        return ""
    rows = [
        "| Model | Protected group | Explicit MAD (pp) | Implicit MAD (pp) | Δ (pp) | "
        "Explicit | Implicit | Discordant | McNemar p | Note |",
        "|---|---|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for c in comparisons:
        rows.append(
            f"| {c.model} | {_cell(c.group)} | {_pct(c.a_mad)} | {_pct(c.b_mad)} | "
            f"{_pct(c.delta_mad)} | {c.a_verdict} | {c.b_verdict} | "
            f"{c.n_discordant or '--'} | {_num(c.mcnemar_p, 4)} | {c.note or '—'} |"
        )
    return (
        "## Explicit vs implicit injection\n\n"
        "The audit study's most policy-relevant finding: a model can be clean when the "
        "protected attribute is a labelled field and biased when the same fact arrives as "
        "ordinary biography. Real CVs carry self-descriptions, not labelled fields, so the "
        "implicit condition is the more externally valid of the two — and it is the one most "
        "audits omit.\n\n"
        "**This comparison is genuinely paired**: both conditions run the identical benchmark "
        "pairs with the identical attributes. Where the raw generations were available, the "
        "McNemar column tests the decisions that actually changed between framings, which is "
        "stronger than comparing two aggregate rates — the audit study had to settle for the "
        "latter.\n\n"
        "A positive Δ means the implicit framing is worse.\n\n" + "\n".join(rows)
    )


def _section_language(comparisons: list[PairedComparison]) -> str:
    if not comparisons:
        return ""
    rows = [
        "| Model | Protected group | Condition | en MAD (pp) | uk MAD (pp) | Δ (pp) | "
        "en utility % | uk utility % | Δ utility (pp) | Note |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c in comparisons:
        rows.append(
            f"| {c.model} | {_cell(c.group)} | {c.a_label.split(' · ')[-1]} | "
            f"{_pct(c.a_mad)} | {_pct(c.b_mad)} | {_pct(c.delta_mad)} | "
            f"{_pct(c.a_utility)} | {_pct(c.b_utility)} | {_pct(c.delta_utility)} | "
            f"{c.note or '—'} |"
        )
    return (
        "## English vs Ukrainian\n\n"
        "The audit study reported more disparity in Ukrainian for every model it could test "
        "in both, and named the confound it could not remove: Ukrainian generation quality "
        "was lower for all of them, and a noisier decision process inflates disparity "
        "independently of bias.\n\n"
        "**The utility columns are what separate the two here.** Where Ukrainian disparity is "
        "higher *and* Ukrainian utility is materially lower, the language effect is not "
        "identified. Where disparity rises while utility holds, it is.\n\n"
        "Unlike the condition comparison this is **not paired** — the two languages use "
        "different candidates and different postings, so only aggregate levels compare.\n\n"
        + "\n".join(rows)
    )


def _section_models(models: list[ModelSummary]) -> str:
    if not models:
        return ""
    rows = [
        "| Model | Langs | Cells | Confirmed | Probable | Gated out | Worst cell | "
        "Worst MAD (pp) | Mean utility % | Note |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for m in models:
        rows.append(
            f"| {m.model} | {', '.join(m.langs)} | {m.n_cells} | {m.n_confirmed} | "
            f"{m.n_probable} | {m.n_gated_out} | {_cell(m.worst_group)} | "
            f"{_pct(m.worst_mad)} | {_pct(m.mean_utility)} | {m.note or '—'} |"
        )
    return "## By model\n\n" + "\n".join(rows)


def _section_plan(plan) -> str:
    if plan is None:
        return ""
    blocks = []
    if plan.targets:
        rows = [
            "| # | Model | Lang | Protected group | Condition | Verdict | MAD (pp) | "
            "Utility % | Suggested families | Why |",
            "|---:|---|---|---|---|---|---:|---:|---|---|",
        ]
        for t in plan.targets:
            rows.append(
                f"| {t.priority} | {t.model} | {t.lang} | {_cell(t.group)} | {t.condition} | "
                f"{t.verdict} | {_pct(t.ar_mad)} | {_pct(t.utility)} | "
                f"{', '.join(t.suggested_families)} | {t.rationale} |"
            )
        blocks.append("### Targets\n\n" + "\n".join(rows))

    if plan.controls:
        rows = [
            "| Model | Lang | Protected group | Condition | MAD (pp) | Why |",
            "|---|---|---|---|---:|---|",
        ]
        for c in plan.controls:
            rows.append(
                f"| {c.model} | {c.lang} | {_cell(c.group)} | {c.condition} | "
                f"{_pct(c.ar_mad)} | {c.rationale} |"
            )
        blocks.append(
            "### Controls\n\n"
            "Cells already clean, carried into the mitigation stage deliberately. A "
            "mitigation that *worsens* one of these is as informative as one that fixes a "
            "target — and without them the study cannot distinguish *\"the mitigation removed "
            "a disparity\"* from *\"the mitigation flattened everything, including what was "
            "already fine\"*.\n\n" + "\n".join(rows)
        )

    if getattr(plan, "rescues", None):
        rows = [
            "| Model | Lang | Utility % | Discrimination (pp) | Worst MAD (pp) | "
            "Families | Primary outcome |",
            "|---|---|---:|---:|---:|---|---|",
        ]
        for r in plan.rescues:
            rows.append(
                f"| {r.model} | {r.lang} | {_pct(r.utility)} | {_pct(r.discrimination)} | "
                f"{_pct(r.ar_mad)} | {', '.join(r.suggested_families)} | "
                f"**{r.primary_outcome}** |"
            )
        blocks.append(
            "### Rescue arms\n\n"
            "**Trained on all three protected groups, in both languages, in one run** — the "
            "fault is where the model puts its decision threshold, which is not "
            "group-specific, so the narrow military-only ablation that Track-A targets get "
            "is deliberately absent here. **Evaluated on the full grid**: every group, both "
            "intersections, both languages, all three conditions.\n\n"
            "Models that failed the utility gate but still **rank** candidates — they accept "
            "materially more of the pairs an attribute-free reference would hire than of "
            "those it would reject. That is a threshold-calibration fault, not an inability "
            "to read a CV, and the counterfactually-consistent training set pins every "
            "verdict to the reference decision, so supervised fine-tuning targets it "
            "directly.\n\n"
            "**The primary outcome for these arms is utility, not disparity.** The question "
            "is whether tuning makes the model able to do the task. Fairness numbers become "
            "interpretable only *if* post-tuning utility crosses the gate — a condition "
            "recorded here in advance, so that a fairness figure from a still-unusable model "
            "cannot be reported after the fact.\n\n" + "\n".join(rows)
        )

    if plan.notes:
        blocks.append("### Notes\n\n" + "\n".join(f"- {n}" for n in plan.notes))

    counts = {k: len(v) for k, v in plan.config_paths.items() if k != "missing"}
    if counts:
        blocks.append(
            "### Configs to run\n\n"
            + "\n".join(f"- **{family}**: {n} config(s)" for family, n in counts.items())
            + "\n\nWritten to `reports/mitigation_plan.yaml`, with a ready-made selection "
            "script at `reports/enable_planned.sh`. Review the plan, then run the script — "
            "it rewrites what will consume days of GPU, so it is emitted rather than applied."
        )
    if plan.config_paths.get("missing"):
        blocks.append(
            "**Missing configs** — the plan names these but they do not exist. Run "
            "`python scripts/generate_experiment_configs.py --no-runners`:\n\n"
            + "\n".join(f"- `{p}`" for p in plan.config_paths["missing"][:20])
        )

    return "## What to mitigate next\n\n" + "\n\n".join(blocks) if blocks else ""


def _footer() -> str:
    return (
        "---\n\n"
        "### Before this becomes a paper claim\n\n"
        "1. **Read the raw generations for every headline attribute.** The parquet under "
        "`outputs/raw/` carries `raw_output` unmodified; a gap that survives reading fifty "
        "actual responses is a different thing from one that only survives a test.\n"
        "2. **State the thresholds in the write-up**, not just the verdicts. A reader who "
        "disagrees with `material_gap = 5pp` should be able to see what changes at 3.\n"
        "3. **Report the excluded cells as a result.** A model that could not be measured is "
        "a finding about the model.\n"
        "4. **The rationale measure is still unvalidated** against human judgement — the "
        "audit's reporting requirement 7, still open.\n"
    )


def write_analysis(text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
