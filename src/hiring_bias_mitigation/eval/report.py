"""Rendering the audit records into the paper-ready markdown report.

`reports/RESULTS.md` is generated, never hand-edited: it is rebuilt from `eval/results/*.json`
on every scoring pass, so a table in the paper can always be traced to a run record. The
tables are laid out to match the shape the write-up needs -- one per research question --
and each carries the reporting requirements the audit study concluded with.

Layout:
    1. Run inventory: what was evaluated, with the decoding config (req. 5).
    2. Baseline audit: disparity per model x group x language x condition, counts before
       percentages (req. 3), raw and FDR-corrected flags side by side (req. 1).
    3. Acceptance rate by attribute for the pivot group, with gaps and effect sizes (req. 2).
    4. Mitigation results: every family against the baseline, fairness next to utility.
    5. Refusals and parse failures per run (req. 4).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..data.protected_groups import GROUPS_IN_SCOPE, INTERSECTIONS, REFERENCE_ATTRS
from .audit import (
    MAX_PARSE_FAILURE,
    PARSE_FAILURE_WARN,
    TESTED_MEASURES,
    is_usable,
    unusable_reason,
)

#: Groups that get a per-attribute table in §3. The intersections have 45-100 cells each and
#: get §4 instead, where the question is non-additivity rather than per-cell rates.
SINGLE_GROUPS = set(GROUPS_IN_SCOPE) | {"marital_status"}

MEASURE_LABELS = {
    "acceptance_rate": "Acceptance rate",
    "inconsistency_rate": "Inconsistency rate",
    "feedback_similarity": "Feedback similarity",
}

CONDITION_LABELS = {
    "explicit": "explicit",
    "implicit": "implicit",
    "attr_free": "attribute-free",
}


def load_records(results_dir: str | Path, usable_only: bool = True) -> list[dict]:
    """Scored records, excluding unusable runs by default.

    A run that mostly failed to parse must not reach a results table. Its metrics are computed
    on whatever survived, and they look *better* the less survived -- one such run reported
    "disparity eliminated, utility 100%" off four parsed responses out of 31,050. It is
    reported in its own section instead, so it is visible as a failure rather than as a
    triumph or as a silent omission.
    """
    records = []
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name == "index.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if usable_only and "summary" in record and not record_usable(record):
            continue
        records.append(record)
    return records


def unusable_records(results_dir: str | Path) -> list[dict]:
    """The runs `load_records` filtered out."""
    everything = load_records(results_dir, usable_only=False)
    return [r for r in everything if "summary" in r and not record_usable(r)]


def served_via_vllm_lora(record: dict) -> bool:
    """A trained adapter audited through vLLM's LoRA path, which misreproduces Qwen3.5 adapters.

    Such a run measured a model close to the untrained base (see `mitigation/merge.py`). Runs
    audited since the fix record the merged checkpoint they served as `served_model`.
    """
    meta = record.get("meta", {})
    return (bool(meta.get("mitigation", {}).get("lora_path"))
            and meta.get("backend", "vllm") == "vllm" and "served_model" not in meta)


def record_usable(record: dict) -> bool:
    return is_usable(record["summary"]) and not served_via_vllm_lora(record)


def record_unusable_reason(record: dict) -> str:
    if served_via_vllm_lora(record):
        return ("adapter served through vLLM's LoRA path, which does not reproduce Qwen3.5 "
                "adapters; superseded by the merged-weight re-audit")
    return unusable_reason(record["summary"])


#: Intersection attributes are stored joined by " | " (the pipe keeps the two components
#: unambiguous in data). A pipe cannot be shown raw in a Markdown table, and an escaped one
#: renders as a literal backslash in plain-text viewers, so tables display them with a cross.
CELL_JOIN_DISPLAY = " × "


def _cell(text) -> str:
    """Renders an attribute name for a table cell."""
    return _md(str(text).replace(" | ", CELL_JOIN_DISPLAY))


def _md(text) -> str:
    """Escapes text destined for a Markdown table cell.

    Intersection attributes are joined with " | " -- `War veteran | Third Gender` -- and a
    bare pipe is Markdown's column separator, so an unescaped one silently adds a column and
    misaligns every row after it in the table. Backslashes are escaped first so an existing
    one does not swallow the pipe escape.
    """
    return str(text).replace("\\", "\\\\").replace("|", "\\|")


def _pct(value, digits: int = 1) -> str:
    if value is None:
        return "--"
    try:
        if value != value:  # NaN
            return "--"
    except TypeError:
        return "--"
    return f"{100 * float(value):.{digits}f}"


def _num(value, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        if value != value:
            return "--"
    except TypeError:
        return "--"
    return f"{float(value):.{digits}f}"


def _family(record: dict) -> str:
    return (record.get("meta", {}).get("mitigation") or {}).get("family", "none")


def _variant(record: dict) -> str:
    mitigation = record.get("meta", {}).get("mitigation") or {}
    return str(
        mitigation.get("strategy")
        or mitigation.get("method")
        or mitigation.get("mode")
        or mitigation.get("objective")
        or ("adapter" if mitigation.get("lora_path") else "--")
    )


#: Footnote for the lexical scrubber wherever it sits beside real mitigations.
ORACLE_NOTE = (
    "\\* **Lexical scrubbing is an oracle upper bound, not a comparable mitigation.** It "
    "removes the attribute with the exact injection templates this study wrote, so after "
    "scrubbing every attribute variant of a CV is character-for-character the attribute-free "
    "CV (verified: 100% of rows in all four cells). The model therefore sees the *same prompt* "
    "for every variant, and its instability and disparity are **zero by construction** — up to "
    "the pipeline's noise floor: 0–0.7% of sets still flip on identical prompts, which is "
    "vLLM's batch-level numerical nondeterminism under greedy decoding, not the attribute. "
    "The row shows what perfect removal of the attribute would buy. Real CVs do not state an "
    "attribute in the study's own template, and bias carried by anything the templates do not "
    "cover — names, phrasing, career gaps — is untouched. Compare mitigations with each other "
    "and read this row only as the ceiling; LLM scrubbing, which has to find the attribute "
    "itself, is the realistic version."
)


def _variant_display(family: str, variant: str) -> str:
    """The variant name, starred when the row is the oracle lexical scrubber."""
    return f"{variant}\\*" if family == "scrub" and variant == "lexical" else variant


def _model_short(record: dict) -> str:
    return record.get("meta", {}).get("model", "?").split("/")[-1]


def _run_label(record: dict) -> str:
    """Model name, marked when the run covered only part of the benchmark.

    Two runs of the same model and language are otherwise indistinguishable in a table keyed
    by model/lang/condition/group -- which is exactly how a 20-pair smoke run ends up sitting
    beside a 450-pair result looking like a contradiction.
    """
    limit = record.get("meta", {}).get("limit_pairs")
    suffix = f" ⚠ partial: {limit} pairs" if limit else ""
    return f"{_model_short(record)}{suffix}"


def _section_excluded(records: list[dict]) -> str:
    """Runs that produced output but not measurements.

    Listed rather than dropped. A run missing from the report is indistinguishable from a run
    that was never launched, and the reason it failed is usually the interesting part -- a
    mitigation that destroys the output format has told you something about the mitigation.
    """
    if not records:
        return ""
    rows = [
        "| Run | Model | Lang | Mitigation | Prompts | Parsed | Why it is excluded |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for record in sorted(records, key=lambda r: r["run_name"]):
        summary = record["summary"]
        rows.append(
            f"| `{record['run_name']}` | {_model_short(record)} | "
            f"{record['meta'].get('lang', '--')} | {_family(record)} · {_variant(record)} | "
            f"{summary.get('n_total', 0):,} | {summary.get('n_decided', 0):,} | "
            f"{_md(record_unusable_reason(record))} |"
        )
    return (
        "These runs completed and were scored, and their numbers are **not** in any table "
        "above.\n\n"
        "**Why they are excluded rather than reported with a caveat.** Every metric here is a "
        "rate over parsed decisions, so the fewer responses parse, the better the run looks: "
        "disparity collapses toward zero because there is nothing left to differ between "
        "attributes, and reference agreement rises toward 100% because each decision is "
        "compared against the same model's attribute-free decision on the same pair. A run "
        "with four parsed responses out of 31,050 scored `MAD 0.00, utility 100.0%` — which "
        "would have been the strongest result in the study.\n\n"
        "**A failure here is still a finding.** A mitigation that destroys the output format "
        "has not produced a null result; it has produced a usability result, and it belongs "
        "in the write-up as one. The raw generations are in `outputs/raw/<run>.parquet` — read "
        "them before deciding what the failure means.\n\n"
        + "\n".join(rows)
    )


def _section_set_stability(records: list[dict]) -> str:
    """Counterfactual set stability -- see `analysis.stability` for the method.

    Built from the raw generations rather than the scored records, because pairing a set with
    the same set at baseline needs the rows themselves. Absent raw files leave the section out
    rather than failing the report.
    """
    try:
        from ..analysis.stability import stability_table
        from ..utils.config import resolve_output_path

        raw_dir = Path(resolve_output_path("outputs/raw"))
        if not raw_dir.is_dir():
            # Not an empty result: the raw generations are on the model drive, and without
            # HBM_OUTPUT_ROOT the section would otherwise vanish from the report unnoticed.
            return (f"_Set stability unavailable: no raw generations at `{raw_dir}`. Load `.env` "
                    "(HBM_OUTPUT_ROOT) and rerun `scripts/make_report.py`._")
        usable = {r["run_name"] for r in records}
        table = stability_table(raw_dir, usable)
    except Exception as exc:  # pragma: no cover - depends on the model drive being mounted
        return f"_Set stability unavailable: {exc}_"
    if table.empty:
        return ""
    rows = [
        "| Model | Lang | Mitigation | Variant | Sets | Variants/set | Unstable % | Δ pp | 95% CI "
        "| Fixed | Broken | p (FDR) |",
        "|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for r in table.sort_values(["lang", "model", "family", "delta_pp"]).itertuples():
        mark = " •" if r.p_fdr < 0.05 else ""
        rows.append(
            f"| {r.model} | {r.lang} | {r.family} | "
            f"{_variant_display(r.family, _md(r.variant or '—'))} | {r.sets} | "
            f"{r.variants_per_set:.0f} | {r.unstable_run_pct:.1f} ({r.unstable_base_pct:.1f}) | "
            f"{r.delta_pp:+.1f}{mark} | [{r.ci_low:+.1f}, {r.ci_high:+.1f}] | {r.fixed} | "
            f"{r.broken} | {r.p_fdr:.1e} |"
        )
    return (
        "A **set** is one candidate–job pair under one injection condition, evaluated with every "
        "attribute variant. It is **unstable** when the decision is not the same across those "
        "variants — the attribute alone tipped it. Each set is paired with the same set at "
        "baseline; *fixed* counts unstable→stable, *broken* stable→unstable. `p` is an exact sign "
        "test on the two, Benjamini–Hochberg corrected across every row of this table; the "
        "interval resamples **candidates**, because sets sharing a candidate are not "
        "independent. Baseline in brackets; • marks FDR < 0.05.\n\n"
        "**Baseline and run are compared on exactly the same variants.** Instability grows with "
        "the number of variants in a set: the baseline audit covers every attribute including "
        "the intersections (179 per set), a mitigated run only its target groups (34, or 5 when "
        "scoped to one group). An unmatched comparison hands every mitigation a fictitious gain "
        "that grows the narrower its scope — it once made every training run look like a "
        "significant, near-unanimous improvement that disappears on matched variants.\n\n"
        "Only sets decided in full in **both** runs count, so a run with parse failures is "
        "measured on the sets it could still answer, plausibly the easier ones.\n\n"
        "Per-group results are in `reports/set_stability_by_group.csv`.\n\n"
        + "\n".join(rows)
        + "\n\n" + ORACLE_NOTE
    )


def render(records: list[dict], excluded: list[dict] | None = None) -> str:
    if not records:
        return (
            "# Hiring Bias Mitigation — Results\n\n"
            "No runs yet.\n\n"
            "```bash\n"
            "make local-benchmark      # fetch + validate the golden benchmark\n"
            "make enable-stage1        # select the baseline audits\n"
            "make local-audit          # run them\n"
            "make local-report         # regenerate this file\n"
            "```\n"
        )

    baselines = [r for r in records if _family(r) == "none"]
    mitigated = [r for r in records if _family(r) != "none"]

    sections = [
        ("Run inventory", _section_inventory(records)),
        ("Aggregate comparison across runs", _section_aggregate(records, baselines)),
        ("Baseline audit — disparity before mitigation", _section_baseline(baselines)),
        ("Acceptance rate by attribute", _section_attribute_detail(baselines)),
        ("Intersections — is the effect additive?", _section_intersections(baselines)),
        ("Mitigation results", _section_mitigation(baselines, mitigated)),
        ("Counterfactual set stability", _section_set_stability(records)),
        ("Refusals, parse failures and rationale leakage",
         _section_output_handling(records, excluded)),
        ("Excluded runs", _section_excluded(excluded or [])),
    ]
    parts = [_header(records)]
    for index, (title, body) in enumerate(sections, start=1):
        if body:
            parts.append(f"## {index}. {title}\n\n{body}")
    parts.append(_footer())
    return "\n\n".join(parts)


def _header(records: list[dict]) -> str:
    models = sorted({_model_short(r) for r in records})
    languages = sorted({r["meta"].get("lang", "?") for r in records})
    return (
        "# Hiring Bias Mitigation — Results\n\n"
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from "
        f"`eval/results/` by `scripts/make_report.py`. Do not edit by hand.*\n\n"
        f"**Models:** {', '.join(models)}  \n"
        f"**Languages:** {', '.join(languages)}  \n"
        f"**Protected groups:** {', '.join(GROUPS_IN_SCOPE)}"
        f" (+ intersections: {', '.join(INTERSECTIONS)})  \n"
        f"**Runs:** {len(records)}\n\n"
        "**How to read every metric in this report: "
        "[`docs/METRICS.md`](../docs/METRICS.md)** — definitions, how to read each number, "
        "and what each one does not capture.\n\n"
        "Benchmark, attribute lists, injection templates and the attribute-free "
        "reference feedback are reused from the audit study this work extends "
        "([AIHiringBiasAnalysis-LLMs]"
        "(https://github.com/Stereotypes-in-LLMs/AIHiringBiasAnalysis-LLMs), "
        "[Fairness-in-AI-Recruitment](https://github.com/TianaLina/Fairness-in-AI-Recruitment)), "
        "so an unmitigated run here is directly comparable with its published baseline."
    )


def _section_inventory(records: list[dict]) -> str:
    rows = [
        "| Run | Model | Lang | Mitigation | Variant | Prompts | Decoding | Seed |",
        "|---|---|---|---|---|---:|---|---:|",
    ]
    for record in sorted(records, key=lambda r: r["run_name"]):
        meta = record["meta"]
        generation = meta.get("generation", {})
        decoding = (
            "greedy"
            if generation.get("greedy")
            else f"T={generation.get('temperature')}, top_p={generation.get('top_p')}, "
            f"n={generation.get('n_samples', 1)}"
        )
        rows.append(
            f"| `{record['run_name']}` | {_model_short(record)} | {meta.get('lang')} | "
            f"{_family(record)} | {_variant(record)} | {meta.get('n_prompts', 0):,} | "
            f"{decoding} | {meta.get('seed')} |"
        )
    return (
        "Decoding is reported for every run (audit-study reporting requirement 5). Greedy "
        "decoding is the default: it removes sampling variance so that a difference between "
        "two runs is attributable to the mitigation rather than to the decoder. Runs with "
        "`n>1` are the ones that deliberately measure that variance instead.\n\n"
        + _probe_note(records)
        + "\n".join(rows)
    )


def _is_probe(record: dict) -> bool:
    """Internal probes: audited, tabulated here, deliberately outside the paper.

    Preference optimisation and the decision-weighted SFT variant ran on one model in one
    language, to answer a question the authors had at the time. One cell cannot carry a claim
    about an objective, so these rows stay in the generated tables -- where leaving them out
    would be selective reporting -- and out of the write-ups. The report says which is which
    rather than leaving a reader to infer it from the run name.
    """
    return _family(record) == "dpo" or "_v2_" in record["run_name"]


def _probe_note(records: list[dict]) -> str:
    probes = sorted(r["run_name"] for r in records if _is_probe(r))
    if not probes:
        return ""
    return (
        "**Internal probes, not reported as results:** "
        + ", ".join(f"`{name}`" for name in probes)
        + ". Preference optimisation (DPO/KTO) and the decision-weighted SFT variant ran on "
        "Qwen3.5-9B English only; the findings documents carry them as future work, not as "
        "an arm of the study.\n\n"
    )


def _mitigation_label(record: dict) -> str:
    variant = _variant(record)
    return _family(record) + (f" · {variant}" if variant not in ("--", "") else "")


def _section_aggregate(records: list[dict], baselines: list[dict]) -> str:
    """The main comparison table, at model x language x protected group x condition.

    Everything here is an effect size, never a flag count. A flag count is a function of
    statistical power as much as of disparity -- the same model on twice the pairs produces
    more flags at the same effect size -- so it cannot rank two runs, and it certainly cannot
    rank a mitigation against its baseline.

    The granularity is deliberate. A single number per run averages across protected groups
    that behave nothing alike (military status carries several times the disparity of religion
    here) and across two injection conditions that can disagree in opposite directions. The
    per-run roll-up is kept below, for ranking experiments only.
    """
    if not records:
        return ""

    main = _main_table(records)
    rollup = _rollup_table(records, baselines)

    return (
        "**Every column here is an effect size. None is a flag count.** A flag count grows "
        "with statistical power at constant disparity, so it cannot rank two runs and must "
        "never be used to claim a mitigation worked.\n\n"
        "Baseline values are in brackets where a matching unmitigated run exists; **↓ means "
        "lower is better**, ↑ means higher is better.\n\n"
        "| Column | Meaning |\n|---|---|\n"
        "| **Disparity (MAD)** | Mean absolute deviation of each attribute's acceptance rate "
        "from the population rate. The headline. More robust than the range, which two "
        "extreme attributes define, and it does not grow with the number of attributes. |\n"
        "| **AR range** | Widest acceptance-rate gap between any two attributes. The number a "
        "reader quotes; noisier than MAD. |\n"
        "| **Cohen's h** | Mean absolute Cohen's *h* against the group's reference level. "
        "Scale-free, so it stays comparable when two runs have very different base rates. |\n"
        "| **Inconsistency** | Share of decisions differing from their counterfactual set's "
        "majority — how often the attribute alone flips the verdict. |\n"
        "| **On ref-hire** | AR range restricted to pairs the attribute-free reference would "
        "have hired. Where this exceeds the overall range, the disparity is concentrated on "
        "the strong candidates — the allocative harm anti-discrimination law is about. |\n"
        "| **Mean FS** | Feedback similarity to the attribute-free reference rationale. An "
        "absolute level, not a disparity; the weakest measure, and a *drop* means the "
        "mitigation changed how the model writes. |\n"
        "| **Utility** | Agreement with the attribute-free reference decision. Without it, a "
        "model that rejects everyone scores perfectly on every fairness column. |\n"
        "| **Refusals** | Share of responses that declined to decide. Excluded from every "
        "fairness column, so a rise makes disparity *unmeasurable* rather than absent. |\n\n"
        "`Attrs` is printed because MAD and range are not comparable across groups of very "
        "different size in quite the same way: a 100-cell intersection and a 5-attribute "
        "group produce differently-shaped distributions even at equal disparity. Percentages "
        "are comparable down a column, not across a row.\n\n"
        "### Main table — by model, language, protected group and condition\n\n"
        "**Read this one.** Protected groups do not behave alike, and the two injection "
        "conditions can disagree in opposite directions for the same model — a model can be "
        "clean when the attribute is a labelled field and biased when the same fact arrives "
        "as ordinary biography. Fairness and quality columns sit at the same granularity on "
        "purpose: a group whose disparity falls while its reference agreement falls with it "
        "has not been fixed.\n\n"
        + "\n".join(main)
        + "\n\n### Roll-up — one row per run\n\n"
        "For ranking experiments against each other, nothing more. Groups are weighted "
        "**equally** within a condition and conditions equally within a run: the groups hold "
        "5 to 100 attributes, and averaging over attributes instead would let military × "
        "gender set most of the headline purely because gender has twenty values. **Do not "
        "argue from a row here about a specific group** — that claim belongs to the main "
        "table.\n\n"
        + "\n".join(rollup)
    )


MAIN_HEADER = (
    "| Run | Lang | Mitigation | Protected group | Condition | Attrs | "
    "Disparity (MAD, pp) ↓ | AR range (pp) ↓ | Cohen's h ↓ | Inconsistency % ↓ | "
    "On ref-hire (pp) ↓ | Mean FS | Utility % ↑ | Refusals % ↓ |"
)
MAIN_ALIGN = "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"


def _main_table(records: list[dict]) -> list[str]:
    """One row per run x protected group x condition, with baselines in brackets."""
    baseline_by_key: dict[tuple, dict] = {}
    for record in records:
        if _family(record) != "none" or record["meta"].get("limit_pairs"):
            continue
        for key, metrics in _by_group(record).items():
            baseline_by_key[(_model_short(record), record["meta"]["lang"], key)] = metrics

    rows = [MAIN_HEADER, MAIN_ALIGN]
    for record in sorted(
        records, key=lambda r: (r["meta"]["lang"], _family(r) != "none", _model_short(r))
    ):
        by_group = _by_group(record)
        lang = record["meta"]["lang"]
        for key in sorted(by_group, key=_group_sort_key):
            metrics = by_group[key]
            group, condition = key.split("::")
            base = baseline_by_key.get((_model_short(record), lang, key))
            if base is metrics:
                base = None
            base = base or {}
            rows.append(
                "| " + " | ".join([
                    _run_label(record),
                    lang,
                    _md(_mitigation_label(record)),
                    _md(group),
                    CONDITION_LABELS.get(condition, condition),
                    str(metrics.get("n_attributes", "--")),
                    _delta_pp(metrics.get("ar_mad"), base.get("ar_mad")),
                    _delta_pp(metrics.get("ar_range"), base.get("ar_range")),
                    _delta_raw(
                        metrics.get("mean_abs_cohens_h"), base.get("mean_abs_cohens_h")
                    ),
                    _delta_pp(
                        metrics.get("inconsistency_rate"), base.get("inconsistency_rate")
                    ),
                    _delta_pp(
                        metrics.get("ar_range__ref_hire"), base.get("ar_range__ref_hire")
                    ),
                    _delta_raw(
                        metrics.get("feedback_similarity"), base.get("feedback_similarity")
                    ),
                    _delta_pp(
                        metrics.get("reference_agreement"), base.get("reference_agreement")
                    ),
                    _delta_pp(metrics.get("refusal_rate"), base.get("refusal_rate")),
                ]) + " |"
            )
    return rows


def _rollup_table(records: list[dict], baselines: list[dict]) -> list[str]:
    """One row per run: the same columns, averaged over groups and conditions."""
    baseline_index = {
        (_model_short(r), r["meta"]["lang"]): r
        for r in baselines
        if not r["meta"].get("limit_pairs")
    }
    rows = [
        "| Run | Lang | Mitigation | Disparity (MAD, pp) ↓ | AR range (pp) ↓ | "
        "Cohen's h ↓ | Inconsistency % ↓ | On ref-hire (pp) ↓ | Mean FS | Utility % ↑ | "
        "Refusals % ↓ |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in sorted(
        records, key=lambda r: (r["meta"]["lang"], _family(r) != "none", _model_short(r))
    ):
        overall = (record["summary"].get("disparity") or {}).get("overall", {})
        if not overall:
            continue
        base = baseline_index.get((_model_short(record), record["meta"]["lang"]))
        base = None if base is record else base
        base_overall = (base["summary"].get("disparity") or {}).get("overall", {}) if base else {}
        base_summary = base["summary"] if base else {}
        rows.append(
            "| " + " | ".join([
                _run_label(record),
                record["meta"]["lang"],
                _md(_mitigation_label(record)),
                _delta_pp(overall.get("ar_mad"), base_overall.get("ar_mad")),
                _delta_pp(overall.get("ar_range"), base_overall.get("ar_range")),
                _delta_raw(
                    overall.get("mean_abs_cohens_h"), base_overall.get("mean_abs_cohens_h")
                ),
                _delta_pp(
                    overall.get("inconsistency_rate"), base_overall.get("inconsistency_rate")
                ),
                _delta_pp(
                    overall.get("ar_range__ref_hire"), base_overall.get("ar_range__ref_hire")
                ),
                _delta_raw(
                    record["summary"].get("mean_feedback_similarity"),
                    base_summary.get("mean_feedback_similarity"),
                ),
                _delta_pp(
                    record["summary"].get("reference_agreement"),
                    base_summary.get("reference_agreement"),
                ),
                _delta_pp(
                    record["summary"].get("refusal_rate"), base_summary.get("refusal_rate")
                ),
            ]) + " |"
        )
    return rows


def _by_group(record: dict) -> dict:
    return (record["summary"].get("disparity") or {}).get("by_group", {})


def _group_sort_key(key: str) -> tuple:
    """Single groups before intersections, then alphabetical, explicit before implicit."""
    group, condition = key.split("::")
    return (group not in SINGLE_GROUPS, group, condition != "explicit", condition)




def _flagged(attr: dict, measure: str) -> str:
    """The FDR-adjusted p-value, marked with a bullet when it survives correction."""
    adjusted = attr.get(f"{measure}__p_adj")
    text = _num(adjusted, 4)
    return f"**{text}** •" if attr.get(f"{measure}__significant_fdr") else text


def _delta_pp(value, baseline) -> str:
    """A percentage-point value, with its baseline in brackets when there is one."""
    if value is None or value != value:
        return "--"
    text = _pct(value)
    if baseline is None or baseline != baseline:
        return text
    return f"{text} ({_pct(baseline)})"


def _delta_raw(value, baseline) -> str:
    if value is None or value != value:
        return "--"
    text = _num(value, 3)
    if baseline is None or baseline != baseline:
        return text
    return f"{text} ({_num(baseline, 3)})"


def _section_baseline(baselines: list[dict]) -> str:
    if not baselines:
        return ""
    rows = [
        "| Run | Lang | Condition | Group | Attrs | AR flagged | IR flagged | FS flagged |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for record in sorted(baselines, key=lambda r: (r["meta"]["lang"], _model_short(r))):
        label = _run_label(record)
        for key, flags in sorted(record["summary"].get("flagged", {}).items()):
            group, condition = key.split("::")
            cells = []
            for measure in TESTED_MEASURES:
                entry = flags.get(measure, {})
                raw, fdr = entry.get("n_flagged_raw", 0), entry.get("n_flagged_fdr", 0)
                cells.append(f"{raw} / {fdr} ({_pct(entry.get('pct_flagged_raw', 0) / 100)}%)")
            rows.append(
                f"| {label} | {record['meta']['lang']} | "
                f"{CONDITION_LABELS.get(condition, condition)} | {_md(group)} | "
                f"{flags.get('n_attributes', 0)} | " + " | ".join(cells) + " |"
            )
    return (
        "Attributes whose measure differs significantly from the population value, as "
        "**raw / FDR-corrected** counts, with the raw percentage in brackets. The attribute "
        "count is printed because a per-group percentage is comparable **down a column but "
        "not across a row**: with 5 attributes one flag is 20%, with 20 attributes it is 5%.\n\n"
        "FDR correction (Benjamini-Hochberg, across every test in a run) is the audit "
        "study's own first reporting requirement, which that study did not meet. Where the "
        "corrected count is much smaller than the raw one, the raw flags were largely "
        "multiplicity.\n\n"
        + "\n".join(rows)
    )


def _section_attribute_detail(baselines: list[dict]) -> str:
    """Per-attribute acceptance rates for the pivot group, where the effect sizes live."""
    blocks = []
    for record in sorted(baselines, key=lambda r: (r["meta"]["lang"], _model_short(r))):
        for payload in [v for _, v in sorted(record["groups"].items())]:
            # Single groups only. The intersections have 45-100 cells each and get their own
            # section, where the question is non-additivity rather than per-cell rates.
            if payload["protected_group"] not in SINGLE_GROUPS:
                continue
            effects = payload.get("effect_sizes", {})
            rows = [
                "| Attribute | n | AR % | Gap (pp) | AR p (BH) | AR paired p | "
                "IR % | IR p (BH) | FS | FS p (BH) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            gaps = effects.get("gap_vs_reference", {})
            for attr in payload["attributes"]:
                name = attr["protected_attr"]
                rows.append(
                    f"| {_cell(name)} | {attr.get('n_decided', 0)} | "
                    f"{_pct(attr.get('acceptance_rate'))} | "
                    f"{_pct(gaps.get(name)) if name in gaps else '--'} | "
                    f"{_flagged(attr, 'acceptance_rate')} | "
                    f"{_num(attr.get('acceptance_rate__paired_p'), 4)} | "
                    f"{_pct(attr.get('inconsistency_rate'))} | "
                    f"{_flagged(attr, 'inconsistency_rate')} | "
                    f"{_num(attr.get('feedback_similarity'), 4)} | "
                    f"{_flagged(attr, 'feedback_similarity')} |"
                )
            population = payload.get("population", {})
            rows.append(
                f"| **population** | {population.get('n_decided', 0)} | "
                f"**{_pct(population.get('acceptance_rate'))}** | -- | -- | -- | "
                f"**{_pct(population.get('inconsistency_rate'))}** | -- | "
                f"**{_num(population.get('feedback_similarity'), 4)}** | -- |"
            )
            blocks.append(
                f"### {_run_label(record)} · {record['meta']['lang']} · "
                f"{payload['protected_group']} · "
                f"{CONDITION_LABELS.get(payload['condition'], payload['condition'])}\n\n"
                f"Range **{_pct(effects.get('acceptance_rate_range'))} pp**, "
                f"SD {_pct(effects.get('acceptance_rate_std'))} pp, "
                f"largest gap **{_pct(effects.get('largest_gap'))} pp** "
                f"({effects.get('largest_gap_attr', '--')} vs "
                f"{effects.get('reference_attr', '--')}, "
                f"Cohen's h = {_num(effects.get('largest_gap_cohens_h'), 2)})\n\n"
                + "\n".join(rows)
            )
    if not blocks:
        return ""
    return (
        "Every single protected group, per run and per condition. Range and standard "
        "deviation are the cheap screening statistics an auditor can compute before "
        "committing to inference; the gap against the reference level is the number a reader "
        "quotes. The paired p-value uses the matched counterfactual structure this design "
        "actually has, and is the one a mitigation claim should rest on — it removes the "
        "between-CV variance, which is the dominant noise source here because CV quality "
        "varies far more than any attribute effect.\n\n"
        "All three measures are shown as **values**, each next to its FDR-adjusted p-value; "
        "a bullet (•) marks the ones that survive correction. Read the values, not the "
        "bullets — significance follows sample size, effect size does not.\n\n"
        "**AR** acceptance rate · **IR** inconsistency rate (share of decisions differing "
        "from the counterfactual set's majority) · **FS** feedback similarity to the "
        "attribute-free reference rationale, the weakest of the three measures (see "
        "[`docs/METRICS.md`](../docs/METRICS.md)). The raw unpaired p-values and bootstrap "
        "intervals are in the run JSON.\n\n"
        + "\n\n".join(blocks)
    )


def _section_intersections(baselines: list[dict]) -> str:
    """Whether an intersectional effect is the sum of its parts.

    An et al. (2025) report that it is not: in their data the effect for white female
    candidates is smaller than the sum of the separate gender and race effects. That claim is
    only testable on cells where BOTH components differ from their reference level, which is
    why the intersections here are fully crossed rather than subsampled.

    For a cell (m, g), the additive prediction from the two marginals is

        AR(m, g0) + AR(m0, g) - AR(m0, g0)

    and the interaction is what the observed rate does on top of that. A positive interaction
    means the combination is treated *better* than the two effects would predict, negative
    means worse -- the compounding-disadvantage case that matters for policy.
    """
    blocks = []
    for record in sorted(baselines, key=lambda r: (r["meta"]["lang"], _model_short(r))):
        lang = record["meta"].get("lang", "en")
        for payload in [v for _, v in sorted(record["groups"].items())]:
            group = payload["protected_group"]
            if group not in INTERSECTIONS:
                continue
            analysis = _interaction_table(payload, group, lang)
            if analysis is None:
                continue
            summary, rows = analysis
            table = [
                "| Cell | Observed AR % | Additive prediction % | Interaction (pp) | n |",
                "|---|---:|---:|---:|---:|",
            ]
            for row in rows:
                table.append(
                    f"| {_cell(row['cell'])} | {_pct(row['observed'])} | "
                    f"{_pct(row['predicted'])} | "
                    f"{_pct(row['interaction'])} | {row['n']} |"
                )
            blocks.append(
                f"### {_run_label(record)} · {lang} · {group} · "
                f"{CONDITION_LABELS.get(payload['condition'], payload['condition'])}\n\n"
                f"Reference cell **{_cell(summary['reference_cell'])}** at "
                f"{_pct(summary['reference_ar'])}%. "
                f"{summary['n_cells']} cells testable. "
                f"Mean signed interaction **{_pct(summary['mean_signed'])} pp**, "
                f"mean |interaction| {_pct(summary['mean_abs'])} pp, "
                f"max {_pct(summary['max_abs'])} pp; "
                f"{summary['n_large']} cell(s) beyond ±10 pp. "
                f"Direction: **{summary['n_negative']} negative / "
                f"{summary['n_positive']} positive** — this is the number to read, because "
                f"the table below is the top ten by magnitude and can be one-sided by "
                f"selection alone.\n\n"
                "Ten largest deviations from additivity:\n\n" + "\n".join(table)
            )

    if not blocks:
        return (
            "_No intersection groups in the scored runs._"
        )
    return (
        "For a cell combining military status *m* with a second attribute *g*, the additive "
        "prediction from the two marginal effects is `AR(m, g₀) + AR(m₀, g) − AR(m₀, g₀)`, "
        "where the subscript zero marks each group's reference level. The **interaction** is "
        "what the observed acceptance rate does on top of that prediction: negative means the "
        "combination is treated worse than either effect alone would imply — compounding "
        "disadvantage — and positive means the reverse.\n\n"
        "This is the question the intersections were run to answer, and it is only answerable "
        "on cells where **both** components differ from their reference. Subsampling the "
        "cross product keeps the reference-touching cells first and would leave nothing to "
        "test; `max_intersection_cells` is `null` for that reason.\n\n"
        "Two cautions. These are differences of noisy per-cell rates, so a per-cell "
        "interaction carries roughly twice the sampling error of a marginal, and no single "
        "cell should be quoted on its own. And the marginals are read off the same run, so a "
        "cell whose reference row had few usable decisions produces an unstable prediction — "
        "check the `n` column.\n\n" + "\n\n".join(blocks)
    )


def _interaction_table(payload: dict, group: str, lang: str):
    """Builds the per-cell interaction rows, or None if the reference cells are missing."""
    sub_groups = INTERSECTIONS[group]
    refs = [REFERENCE_ATTRS.get(g, {}).get(lang) for g in sub_groups]
    if any(r is None for r in refs):
        return None

    rates, counts = {}, {}
    for row in payload["attributes"]:
        parts = tuple(str(row["protected_attr"]).split(" | "))
        if len(parts) != len(sub_groups):
            continue
        rate = row.get("acceptance_rate")
        if rate is None or (isinstance(rate, float) and rate != rate):
            continue
        rates[parts] = float(rate)
        counts[parts] = row.get("n_decided", 0)

    base_key = tuple(refs)
    if base_key not in rates:
        return None
    base = rates[base_key]

    entries = []
    for cell, observed in rates.items():
        if any(value == ref for value, ref in zip(cell, refs)):
            continue  # reference-touching cells are the marginals, not a test of additivity
        marginal_a = rates.get((cell[0], refs[1]))
        marginal_b = rates.get((refs[0], cell[1]))
        if marginal_a is None or marginal_b is None:
            continue
        predicted = marginal_a + marginal_b - base
        entries.append(
            {
                "cell": " | ".join(cell),
                "observed": observed,
                "predicted": predicted,
                "interaction": observed - predicted,
                "n": counts.get(cell, 0),
            }
        )
    if not entries:
        return None

    magnitudes = [abs(e["interaction"]) for e in entries]
    signed = [e["interaction"] for e in entries]
    summary = {
        "reference_cell": " | ".join(base_key),
        "reference_ar": base,
        "n_cells": len(entries),
        "mean_abs": sum(magnitudes) / len(magnitudes),
        "mean_signed": sum(signed) / len(signed),
        "max_abs": max(magnitudes),
        "n_large": sum(1 for m in magnitudes if m > 0.10),
        "n_negative": sum(1 for v in signed if v < 0),
        "n_positive": sum(1 for v in signed if v > 0),
    }
    entries.sort(key=lambda e: -abs(e["interaction"]))
    return summary, entries[:10]


def _section_mitigation(baselines: list[dict], mitigated: list[dict]) -> str:
    if not mitigated:
        return (
            "_No mitigation runs scored yet. Enable configs in `scripts/run_all_*.sh` and "
            "run them, then re-score._"
        )

    index = {(_model_short(r), r["meta"]["lang"]): r for r in baselines}
    rows = [
        "| Model | Lang | Mitigation | Variant | Cells | Disparity (MAD, pp) ↓ | Δ MAD | "
        "Max gap (pp) ↓ | Cohen's h ↓ | Inconsistency % ↓ | Mean FS | Utility % ↑ | "
        "Refusals % ↓ | Attr. mentioned % | AR flags Δ | IR flags Δ |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in sorted(mitigated, key=lambda r: (r["meta"]["lang"], _model_short(r))):
        base = index.get((_model_short(record), record["meta"]["lang"]))
        cells = set(record["groups"])
        after = _restricted_metrics(record, cells)
        before = _restricted_metrics(base, cells) if base else {}
        rows.append(
            f"| {_model_short(record)} | {record['meta']['lang']} | {_family(record)} | "
            f"{_variant_display(_family(record), _variant(record))} | {len(cells)} | "
            f"{_with_base(after, before, 'ar_mad', _pct)} | "
            f"{_delta(after, before, 'ar_mad', scale=100)} | "
            f"{_delta_gap(base, record)} | "
            f"{_with_base(after, before, 'mean_abs_cohens_h', _num)} | "
            f"{_with_base(after, before, 'inconsistency_rate', _pct)} | "
            f"{_with_base(after, before, 'feedback_similarity', _num)} | "
            f"{_with_base(after, before, 'reference_agreement', _pct)} | "
            f"{_with_base(after, before, 'refusal_rate', _pct)} | "
            f"{_with_base(after, before, 'attribute_mention_rate', _pct)} | "
            f"{_delta_flags(base, record, 'acceptance_rate')} | "
            f"{_delta_flags(base, record, 'inconsistency_rate')} |"
        )
    return (
        "Each row is one mitigated run against its own unmitigated baseline (same model, "
        "same language, same benchmark, same decoding). **Values first, baseline in "
        "brackets**; the flag counts are last on purpose.\n\n"
        "**Why values, not just counts.** A flag count rises with statistical power at "
        "constant disparity, so it cannot rank two runs and cannot by itself show that a "
        "mitigation worked — the same rule §2 states. Read `Disparity (MAD)` and `Δ MAD`; "
        "treat the flag columns as a coarse check that the effect is large enough to survive "
        "FDR correction, not as the result.\n\n"
        "**Compared like for like.** A mitigated run evaluates only the groups and conditions "
        "it was aimed at, while its baseline covers the full grid. Every column here — the "
        "values as well as the deltas — is recomputed over the `Cells` the mitigated run "
        "actually measured, for both runs. Without that, a run measuring one cell instead of "
        "ten would report the missing nine as disparity it removed.\n\n"
        "**Read the fairness columns and the utility column together.** A mitigation that "
        "rejects every candidate has perfect acceptance-rate parity and no utility, and only "
        "the reference-agreement column distinguishes that from a real improvement. A rise in "
        "refusals is the other way this goes wrong: refusals are excluded from the fairness "
        "denominators, so a model that learns to decline is a model whose disparity becomes "
        "unmeasurable rather than absent.\n\n"
        + "\n".join(rows)
        + "\n\n" + ORACLE_NOTE
    )


def _restricted_metrics(record: dict | None, cells: set[str]) -> dict:
    """Run-level metrics recomputed over `cells` only.

    The disparity figures come from the same equal-weighting aggregator the rest of the report
    uses, so a restricted number means what the unrestricted one means. The level figures --
    similarity, utility, refusals -- are averaged over decisions, which is exact here because
    each generation belongs to exactly one group: the groups partition the run rather than
    overlap it.
    """
    if record is None:
        return {}
    from .audit import _aggregate_disparity

    groups = {k: v for k, v in record["groups"].items() if k in cells}
    if not groups:
        return {}
    out = dict(_aggregate_disparity(groups).get("overall", {}))

    total = 0.0
    sums: dict[str, float] = {}
    for payload in groups.values():
        population = payload.get("population", {})
        n = float(population.get("n_decided") or 0)
        if not n:
            continue
        total += n
        for key in ("feedback_similarity", "reference_agreement", "refusal_rate",
                    "attribute_mention_rate"):
            value = population.get(key)
            if value is not None and value == value:
                sums[key] = sums.get(key, 0.0) + n * float(value)
    if total:
        out.update({k: v / total for k, v in sums.items()})
    return out


def _with_base(after: dict, before: dict, key: str, fmt) -> str:
    """`value (baseline)`, or just the value when there is no matching baseline."""
    value = fmt(after.get(key))
    if not before or before.get(key) is None:
        return value
    return f"{value} ({fmt(before.get(key))})"


def _delta(after: dict, before: dict, key: str, scale: float = 1.0) -> str:
    a, b = after.get(key), before.get(key)
    if a is None or b is None or a != a or b != b:
        return "--"
    return f"{scale * (float(a) - float(b)):+.1f}"


def _delta_flags(base: dict | None, record: dict, measure: str) -> str:
    """Flag counts, compared over the cells the mitigated run actually measured.

    A mitigated run evaluates only the groups and conditions it was aimed at, while its
    baseline covers the full grid. Summing each run's own cells would compare a 1-cell total
    against a 10-cell one and report the difference as a mitigation effect -- turning "we
    measured less" into "we fixed more". Restricting the baseline to the same cells is what
    makes the delta mean anything.
    """
    cells = set(record["summary"].get("flagged", {}))
    after = _total_flags(record, measure, cells)
    if base is None:
        return str(after)
    before = _total_flags(base, measure, cells)
    return f"{after - before:+d} ({before}→{after})"


def _total_flags(record: dict, measure: str, cells: set[str] | None = None) -> int:
    return sum(
        flags.get(measure, {}).get("n_flagged_fdr", 0)
        for key, flags in record["summary"].get("flagged", {}).items()
        if cells is None or key in cells
    )


def _delta_gap(base: dict | None, record: dict) -> str:
    cells = set(record["groups"])
    after = _max_gap(record, cells)
    if base is None or after != after:
        return _pct(after)
    before = _max_gap(base, cells)
    return f"{_pct(after)} ({_pct(before)})"


def _max_gap(record: dict, cells: set[str] | None = None) -> float:
    """Largest reference gap, over the given cells only. See `_delta_flags` for why."""
    gaps = [
        abs(payload.get("effect_sizes", {}).get("largest_gap", float("nan")))
        for key, payload in record["groups"].items()
        if cells is None or key in cells
    ]
    gaps = [g for g in gaps if g == g]
    return max(gaps) if gaps else float("nan")


def _baseline_bracket(base: dict | None, key: str) -> str:
    if base is None:
        return ""
    return f" ({_pct(base['summary'].get(key))})"


def _section_output_handling(records: list[dict], excluded: list[dict] | None = None) -> str:
    """Refusals, parse failures and leakage -- **including the runs excluded elsewhere**.

    This is the one table an excluded run must appear in. Its parse-failure rate is the reason
    it was excluded, and hiding the worst case from the parse-failure table would leave the
    section describing a healthier pipeline than the one that ran.
    """
    rows = [
        "| Run | Prompts | Decided | Refused % | Parse-fail % | Attr. mentioned % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    everything = list(records) + list(excluded or [])
    for record in sorted(everything, key=lambda r: r["run_name"]):
        s = record["summary"]
        failure = s.get("parse_failure_rate") or 0.0
        # ‼ excluded from every other table; ⚠ still reported, on reduced denominators.
        marker = (
            " ‼" if failure > MAX_PARSE_FAILURE
            else " ⚠" if failure > PARSE_FAILURE_WARN
            else ""
        )
        rows.append(
            f"| `{record['run_name']}`{marker} | {s.get('n_total', 0):,} | "
            f"{s.get('n_decided', 0):,} | "
            f"{_pct(s.get('refusal_rate'))} | {_pct(s.get('parse_failure_rate'))} | "
            f"{_pct(s.get('attribute_mention_rate'))} |"
        )
    return (
        "Refusals and unparsable responses are excluded from every fairness statistic, so "
        "denominators vary by run and by attribute rather than being fixed. Reporting them as "
        "a first-class result is the audit study's requirement 4: they are a bias signal in "
        "their own right, and they silently change every downstream number. The last column "
        "is the share of rationales that name the injected attribute — a direct, "
        "interpretable read on the channel a human reviewer actually sees, where feedback "
        "similarity is only an indirect one. "
        "**Do not compare these rows with each other.** Each is taken over its own run's "
        "prompts, and a baseline covers the full grid — intersections included, where two "
        "attributes can be named — while a mitigated run covers only its target cells. Read "
        "leakage *changes* from the like-for-like `Attr. mentioned %` column in the mitigation "
        "results section. Comparing across this table once produced a 'halved' leakage rate "
        "for three adapters that, on the same rows, changed it by less than 0.05 points.\n\n"
        f"**‼** excluded from every other table: more than {100 * MAX_PARSE_FAILURE:.0f}% of "
        "responses failed to parse, so its rates describe a handful of rows rather than the "
        "run — see the excluded-runs section. "
        f"**⚠** still reported, but on materially reduced denominators (over "
        f"{100 * PARSE_FAILURE_WARN:.0f}% unparsed): read its rates knowing the base "
        "shrank.\n\n"
        + "\n".join(rows)
    )


def _footer() -> str:
    return (
        "---\n\n"
        "### Reporting checklist\n\n"
        "The audit study this work extends concluded with nine reporting requirements for "
        "hiring-bias audits. Status in this report:\n\n"
        "| # | Requirement | Status |\n|---|---|---|\n"
        "| 1 | Report number of tests; control FDR | ✅ raw and BH-corrected, in the "
        "baseline-audit section |\n"
        "| 2 | Report effect sizes, not only significance | ✅ gaps, ranges, Cohen's h and "
        "bootstrap CIs per attribute; effect-size-only disparity indices in the aggregate "
        "section |\n"
        "| 3 | Report counts and denominators, never percentages alone | ✅ throughout |\n"
        "| 4 | Report refusal and parse-failure rates per attribute | ✅ per run in the "
        "output-handling section, per attribute in the run JSON |\n"
        "| 5 | Report full generation configuration and sampled runs | ✅ run inventory |\n"
        "| 6 | Test both explicit and implicit presentation | ✅ both, plus an attribute-free "
        "control the audit study did not have |\n"
        "| 7 | Validate the embedding rationale measure against human judgement | ⚠️ **open** — "
        "inter-annotator agreement on a sample is still owed; no finding rests on this "
        "measure |\n"
        "| 8 | Include protected attributes beyond gender and race | ✅ military status, "
        "religion, and their fully-crossed intersections |\n"
        "| 9 | Audit in every language of deployment | ✅ English and Ukrainian |\n"
    )


def write_report(
    records: list[dict], path: str | Path, excluded: list[dict] | None = None
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(records, excluded), encoding="utf-8")
    return path
