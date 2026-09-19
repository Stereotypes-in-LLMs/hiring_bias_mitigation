"""Patterns that recur across models, languages and injection conditions.

The audit study's own standard for what may be argued from: *"The patterns we build arguments
on are those that recur across attributes, measures, models and languages"* -- as opposed to
isolated flags, which multiple comparisons produce from a fair model. This module finds the
recurrences.

Four questions:

`attribute_recurrence`
    Which specific protected attributes are flagged in how many independent cells. An
    attribute penalised by one model in one language is a curiosity; the same attribute
    penalised by four models in both languages is a finding about the attribute.

`language_effect`
    Same model, same group, same condition, English vs Ukrainian. The audit study reported
    more disparity in Ukrainian for every model it could test in both, but could not separate
    that from lower Ukrainian generation quality. Here the utility column is carried
    alongside, so the two can be told apart.

`condition_effect`
    Same model, same group, explicit vs implicit. This is the audit study's most
    policy-relevant finding -- a model clean under a labelled attribute field can be biased
    when the same fact arrives as ordinary biography -- and, crucially, it is a **paired**
    comparison: both conditions run over the identical 450 job-CV pairs. With the raw
    generations available, that licenses a McNemar test rather than the unpaired comparison
    the audit study had to settle for.

`model_ranking`
    Which models are worth carrying into the mitigation stage at all.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .evidence import VERDICT_RANK, CellFinding, Verdict


@dataclass
class Recurrence:
    """One protected attribute, summarised across every cell it appears in."""

    group: str
    attribute: str
    n_cells: int
    n_penalised: int
    n_favoured: int
    n_confirmed: int
    mean_gap: float
    max_abs_gap: float
    models: list[str]
    langs: list[str]
    conditions: list[str]

    @property
    def consistent_direction(self) -> str:
        """Only call a direction consistent when it dominates and recurs."""
        if self.n_confirmed < 2:
            return "insufficient"
        if self.n_penalised >= 2 and self.n_favoured == 0:
            return "penalised"
        if self.n_favoured >= 2 and self.n_penalised == 0:
            return "favoured"
        return "mixed"

    def as_dict(self) -> dict:
        out = asdict(self)
        out["consistent_direction"] = self.consistent_direction
        return out


@dataclass
class PairedComparison:
    """One A-vs-B contrast, with the paired test where the design licenses it."""

    model: str
    group: str
    axis: str  # "language" | "condition"
    a_label: str
    b_label: str
    a_mad: float
    b_mad: float
    delta_mad: float
    a_utility: float
    b_utility: float
    delta_utility: float
    a_verdict: str
    b_verdict: str
    n_discordant: int = 0
    mcnemar_p: float = float("nan")
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _nanmean(values) -> float:
    clean = [v for v in values if v is not None and v == v]
    return float(np.mean(clean)) if clean else float("nan")


def attribute_recurrence(findings: list[CellFinding]) -> list[Recurrence]:
    """Per protected attribute, how often and how consistently it is flagged.

    Gated-out cells are excluded: an attribute "flagged" in a run whose utility is at chance
    tells you about the run, not the attribute.
    """
    buckets: dict[tuple[str, str], list] = defaultdict(list)
    for cell in findings:
        if not cell.gate_passed:
            continue
        for attr in cell.attributes:
            buckets[(cell.group, attr.attribute)].append((cell, attr))

    out: list[Recurrence] = []
    for (group, name), entries in buckets.items():
        gaps = [a.gap_vs_reference for _, a in entries]
        out.append(
            Recurrence(
                group=group,
                attribute=name,
                n_cells=len(entries),
                n_penalised=sum(1 for _, a in entries if a.direction == "penalised"),
                n_favoured=sum(1 for _, a in entries if a.direction == "favoured"),
                n_confirmed=sum(
                    1
                    for _, a in entries
                    if VERDICT_RANK[a.verdict] >= VERDICT_RANK[Verdict.PROBABLE]
                ),
                mean_gap=_nanmean(gaps),
                max_abs_gap=max((abs(g) for g in gaps if g == g), default=float("nan")),
                models=sorted({c.model for c, _ in entries}),
                langs=sorted({c.lang for c, _ in entries}),
                conditions=sorted({c.condition for c, _ in entries}),
            )
        )
    out.sort(
        key=lambda r: (
            -r.n_confirmed,
            -(r.max_abs_gap if r.max_abs_gap == r.max_abs_gap else 0),
        )
    )
    return out


def _index(findings: list[CellFinding]) -> dict[tuple, CellFinding]:
    return {c.key: c for c in findings}


def language_effect(findings: list[CellFinding]) -> list[PairedComparison]:
    """English vs Ukrainian, holding model, group and condition fixed.

    Not a paired test: the two languages use different job-CV pairs (different candidates,
    different postings), so only the aggregate levels are comparable. The utility delta is
    reported next to the disparity delta because a Ukrainian run that is simply worse at the
    task will show more disparity for reasons that are not about fairness.
    """
    index = _index(findings)
    out: list[PairedComparison] = []
    for key, cell in index.items():
        model, lang, group, condition = key
        if lang != "en":
            continue
        other = index.get((model, "uk", group, condition))
        if other is None:
            continue
        delta_utility = other.utility - cell.utility
        note = ""
        if delta_utility == delta_utility and delta_utility < -0.05:
            note = (
                "Ukrainian utility is materially lower — part of the disparity gap may be "
                "generation quality rather than bias"
            )
        out.append(
            PairedComparison(
                model=model, group=group, axis="language",
                a_label=f"en · {condition}", b_label=f"uk · {condition}",
                a_mad=cell.ar_mad, b_mad=other.ar_mad,
                delta_mad=other.ar_mad - cell.ar_mad,
                a_utility=cell.utility, b_utility=other.utility,
                delta_utility=delta_utility,
                a_verdict=cell.verdict.value, b_verdict=other.verdict.value,
                note=note,
            )
        )
    out.sort(key=lambda c: -(c.delta_mad if c.delta_mad == c.delta_mad else 0))
    return out


def condition_effect(
    findings: list[CellFinding], raw_dir: Path | None = None
) -> list[PairedComparison]:
    """Explicit vs implicit, holding model, language and group fixed.

    Genuinely paired: both conditions run the identical benchmark pairs with the identical
    attributes, so when the raw generations are available this adds a McNemar test on the
    decisions that changed between conditions. Without them the comparison is aggregate-only,
    and says so.
    """
    index = _index(findings)
    out: list[PairedComparison] = []
    for key, cell in index.items():
        model, lang, group, condition = key
        if condition != "explicit":
            continue
        other = index.get((model, lang, group, "implicit"))
        if other is None:
            continue
        comparison = PairedComparison(
            model=model, group=group, axis="condition",
            a_label=f"{lang} · explicit", b_label=f"{lang} · implicit",
            a_mad=cell.ar_mad, b_mad=other.ar_mad,
            delta_mad=other.ar_mad - cell.ar_mad,
            a_utility=cell.utility, b_utility=other.utility,
            delta_utility=other.utility - cell.utility,
            a_verdict=cell.verdict.value, b_verdict=other.verdict.value,
        )
        if raw_dir is not None:
            n_disc, p = _mcnemar_from_raw(raw_dir, cell.run_name, group)
            comparison.n_discordant = n_disc
            comparison.mcnemar_p = p
        if (
            cell.verdict in (Verdict.CLEAN, Verdict.WEAK)
            and VERDICT_RANK[other.verdict] >= VERDICT_RANK[Verdict.PROBABLE]
        ):
            comparison.note = (
                "clean under a labelled attribute field, biased when the same fact is stated "
                "as ordinary biography — the case a standard audit protocol would miss"
            )
        out.append(comparison)
    out.sort(key=lambda c: -(c.delta_mad if c.delta_mad == c.delta_mad else 0))
    return out


def _mcnemar_from_raw(raw_dir: Path, run_name: str, group: str) -> tuple[int, float]:
    """McNemar test on decisions that changed between explicit and implicit.

    The null is that the two framings of the same attribute produce the same decision on the
    same job-CV pair. Discordant pairs -- hired under one framing, rejected under the other --
    carry all the information; concordant ones carry none, which is exactly what makes this
    stronger than comparing two aggregate rates.
    """
    import pandas as pd
    from scipy import stats

    path = Path(raw_dir) / f"{run_name}.parquet"
    if not path.exists():
        return 0, float("nan")

    frame = pd.read_parquet(
        path, columns=["pair_id", "protected_group", "protected_attr", "condition",
                       "outcome", "decision"]
    )
    frame = frame[(frame["protected_group"] == group) & (frame["outcome"] == "decided")]
    if frame.empty:
        return 0, float("nan")

    frame = frame.assign(hired=(frame["decision"] == "hire").astype(int))
    wide = frame.pivot_table(
        index=["pair_id", "protected_attr"], columns="condition", values="hired"
    )
    if not {"explicit", "implicit"}.issubset(wide.columns):
        return 0, float("nan")
    wide = wide.dropna(subset=["explicit", "implicit"])

    b = int(((wide["explicit"] == 1) & (wide["implicit"] == 0)).sum())
    c = int(((wide["explicit"] == 0) & (wide["implicit"] == 1)).sum())
    if b + c == 0:
        return 0, 1.0
    # Exact binomial rather than the chi-square approximation: cheap here, and correct when
    # the discordant count is small.
    p = float(stats.binomtest(b, b + c, 0.5).pvalue)
    return b + c, p


@dataclass
class ModelSummary:
    model: str
    langs: list[str]
    n_cells: int
    n_confirmed: int
    n_probable: int
    n_gated_out: int
    worst_group: str
    worst_mad: float
    mean_utility: float
    usable: bool
    note: str

    def as_dict(self) -> dict:
        return asdict(self)


def model_ranking(findings: list[CellFinding]) -> list[ModelSummary]:
    """Which models carry enough measurable disparity to be worth mitigating."""
    by_model: dict[str, list[CellFinding]] = defaultdict(list)
    for cell in findings:
        by_model[cell.model].append(cell)

    out: list[ModelSummary] = []
    for model, cells in by_model.items():
        usable_cells = [c for c in cells if c.gate_passed]
        gated = len(cells) - len(usable_cells)
        confirmed = [c for c in usable_cells if c.verdict == Verdict.CONFIRMED]
        probable = [c for c in usable_cells if c.verdict == Verdict.PROBABLE]
        worst = max(
            usable_cells, key=lambda c: c.ar_mad if c.ar_mad == c.ar_mad else -1, default=None
        ) if usable_cells else None

        note = ""
        if not usable_cells:
            note = (
                "every cell failed the quality gate — this is a finding about the model's "
                "fitness for the task, not about its fairness"
            )
        elif gated:
            note = f"{gated} of {len(cells)} cells gated out"

        out.append(
            ModelSummary(
                model=model,
                langs=sorted({c.lang for c in cells}),
                n_cells=len(cells),
                n_confirmed=len(confirmed),
                n_probable=len(probable),
                n_gated_out=gated,
                worst_group=f"{worst.group} · {worst.condition}" if worst else "--",
                worst_mad=worst.ar_mad if worst else float("nan"),
                mean_utility=_nanmean([c.utility for c in cells]),
                usable=bool(usable_cells),
                note=note,
            )
        )
    out.sort(key=lambda m: (-m.n_confirmed, -m.n_probable))
    return out
