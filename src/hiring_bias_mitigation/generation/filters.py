"""Quality gates on teacher output. Nothing generated is trusted; everything is checked.

A teacher model asked to produce an attribute-invariant rationale will sometimes produce a
rationale that names the attribute, flip the verdict it was told to keep, answer in the wrong
language, or emit something unparsable. Training on those examples teaches the student
exactly the behaviour the study is trying to remove, so each is a drop, not a repair.

Every gate records why it dropped what it dropped. The counts go into the dataset card and
into reports/data_analysis.md -- a mitigation dataset whose yield is 30% is a finding about
the teacher, and the reader needs to see it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

from ..eval.metrics import mentions_attribute
from ..eval.parsing import parse_output
from ..utils.logging import get_logger

log = get_logger(__name__)

MIN_FEEDBACK_WORDS = 4
MAX_FEEDBACK_WORDS = 45  # the schema asks for <=30; allow slack before calling it a violation

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[A-Za-z]")

#: Phrases that give away that the model is *reasoning about* the attribute rather than
#: ignoring it. "Regardless of their military status, ..." leaks the attribute just as surely
#: as naming it, and a student trained on it learns to announce its own neutrality.
_META_PHRASES = {
    "en": (
        "regardless of", "irrespective of", "protected characteristic", "we do not consider",
        "without regard to", "personal characteristic", "does not affect", "is irrelevant",
        "ignoring the", "as stated", "identity",
    ),
    "uk": (
        "незалежно від", "попри", "захищена характеристика", "ми не враховуємо",
        "особиста характеристика", "не впливає", "нерелевант", "ігноруючи", "ідентичн",
    ),
}


@dataclass
class FilterStats:
    counts: dict = field(default_factory=dict)

    def drop(self, reason: str, n: int = 1) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + n

    def as_dict(self, kept: int, total: int) -> dict:
        return {
            "n_input": total,
            "n_kept": kept,
            "yield": round(kept / total, 4) if total else 0.0,
            "dropped_by_reason": dict(sorted(self.counts.items(), key=lambda kv: -kv[1])),
        }


def language_ok(text: str, lang: str) -> bool:
    """Whether the text is written in the language the run asked for."""
    text = unicodedata.normalize("NFKC", str(text))
    cyr, lat = len(_CYRILLIC.findall(text)), len(_LATIN.findall(text))
    total = cyr + lat
    if total < 12:
        return True  # too short to judge; other gates will catch it
    # Ukrainian IT text legitimately carries English tool names, so the test is dominance.
    return (cyr / total) >= 0.4 if lang == "uk" else (cyr / total) <= 0.2


def has_meta_commentary(text: str, lang: str) -> bool:
    lowered = str(text).lower()
    return any(phrase in lowered for phrase in _META_PHRASES[lang])


def filter_reference(raw: list[str], pairs: pd.DataFrame, lang: str) -> tuple[pd.DataFrame, dict]:
    """Keeps pairs whose reference generation is parseable, decisive and well-formed."""
    stats = FilterStats()
    kept = []
    for row, text in zip(pairs.to_dict("records"), raw):
        parsed = parse_output(text, lang)
        if not parsed.usable:
            stats.drop(f"reference_{parsed.outcome}")
            continue
        words = len(parsed.feedback.split())
        if words < MIN_FEEDBACK_WORDS:
            stats.drop("reference_feedback_too_short")
            continue
        if words > MAX_FEEDBACK_WORDS:
            stats.drop("reference_feedback_too_long")
            continue
        if not language_ok(parsed.feedback, lang):
            stats.drop("reference_wrong_language")
            continue
        kept.append(
            {**row, "reference_decision": parsed.decision, "reference_feedback": parsed.feedback}
        )
    out = pd.DataFrame(kept)
    report = stats.as_dict(len(out), len(pairs))
    log.info("reference pass: kept %d/%d (%.1f%%)", len(out), len(pairs), 100 * report["yield"])
    return out.reset_index(drop=True), report


def filter_invariant(
    raw: list[str], variants: pd.DataFrame, lang: str
) -> tuple[pd.DataFrame, dict]:
    """Keeps variants whose rationale is invariant in the sense the training needs.

    The verdict must match the reference (that is the counterfactual-consistency property),
    the rationale must not name the attribute (that is the leakage property), and it must not
    editorialise about the attribute's irrelevance (that would teach the student to announce
    its neutrality instead of simply being neutral).
    """
    stats = FilterStats()
    kept = []
    for row, text in zip(variants.to_dict("records"), raw):
        parsed = parse_output(text, lang)
        if not parsed.usable:
            stats.drop(f"invariant_{parsed.outcome}")
            continue
        if parsed.decision != row["reference_decision"]:
            stats.drop("invariant_decision_drift")
            continue
        words = len(parsed.feedback.split())
        if words < MIN_FEEDBACK_WORDS or words > MAX_FEEDBACK_WORDS:
            stats.drop("invariant_feedback_length")
            continue
        if not language_ok(parsed.feedback, lang):
            stats.drop("invariant_wrong_language")
            continue
        if mentions_attribute(parsed.feedback, row["protected_attr"]):
            stats.drop("invariant_attribute_leak")
            continue
        if has_meta_commentary(parsed.feedback, lang):
            stats.drop("invariant_meta_commentary")
            continue
        kept.append({**row, "chosen_decision": parsed.decision, "chosen_feedback": parsed.feedback})
    out = pd.DataFrame(kept)
    report = stats.as_dict(len(out), len(variants))
    log.info("invariant pass: kept %d/%d (%.1f%%)", len(out), len(variants), 100 * report["yield"])
    return out.reset_index(drop=True), report


def filter_biased(raw: list[str], variants: pd.DataFrame, lang: str) -> tuple[pd.DataFrame, dict]:
    """Keeps negatives that are actually negative.

    A `rejected` response that neither flips the verdict nor names the attribute is not a
    biased response -- it is a second correct answer, and a preference pair built from it
    teaches the student a distinction that does not exist. Requiring at least one of the two
    is what keeps the preference signal about bias rather than about phrasing.
    """
    stats = FilterStats()
    kept = []
    for row, text in zip(variants.to_dict("records"), raw):
        parsed = parse_output(text, lang)
        if not parsed.usable:
            stats.drop(f"biased_{parsed.outcome}")
            continue
        if not language_ok(parsed.feedback, lang):
            stats.drop("biased_wrong_language")
            continue
        flipped = parsed.decision != row["reference_decision"]
        leaks = mentions_attribute(parsed.feedback, row["protected_attr"])
        if not (flipped or leaks):
            stats.drop("biased_not_actually_biased")
            continue
        kept.append(
            {
                **row,
                "rejected_decision": parsed.decision,
                "rejected_feedback": parsed.feedback,
                "rejected_flipped_decision": bool(flipped),
                "rejected_mentions_attribute": bool(leaks),
            }
        )
    out = pd.DataFrame(kept)
    report = stats.as_dict(len(out), len(variants))
    log.info("biased pass: kept %d/%d (%.1f%%)", len(out), len(variants), 100 * report["yield"])
    return out.reset_index(drop=True), report


def balance_decisions(
    df: pd.DataFrame,
    seed: int = 42,
    column: str = "chosen_decision",
    target_positive_rate: float | None = None,
):
    """Downsamples the majority verdict so SFT does not simply learn a prior.

    Teacher screening skews heavily toward reject (the golden reference is 2:1 reject in
    English). A student fine-tuned on that skew shifts its whole operating point, which shows
    up as a fairness improvement -- every acceptance rate converges because almost nothing is
    accepted -- while the model gets worse. Balancing removes that confound; the utility
    column in the report is what confirms it worked.
    """
    import numpy as np

    if column not in df.columns or df.empty:
        return df, {"balanced": False}
    counts = df[column].value_counts()
    if len(counts) < 2:
        return df, {"balanced": False, "reason": "single class"}

    rng = np.random.default_rng(seed)
    if target_positive_rate is None:
        # 50/50: the conservative default when nothing is known about the evaluation mix.
        sizes = {value: int(counts.min()) for value in counts.index}
    else:
        # Match the *evaluation* mix instead. Equalising to 50/50 replaces the teacher's skew
        # with a different one: the student's operating point then moves toward accepting far
        # more than the benchmark does, and that shift lands in the acceptance-rate columns as
        # if the mitigation had caused it. Matching the benchmark keeps the operating point
        # still, so the audit measures disparity rather than a moved threshold.
        positive = "hire" if "hire" in counts.index else counts.index[0]
        negative = next(v for v in counts.index if v != positive)
        rate = float(target_positive_rate)
        # Keep every row of the scarcer side and take what the ratio allows from the other.
        by_positive = int(counts[positive] / rate) if rate else 0
        by_negative = int(counts[negative] / (1 - rate)) if rate < 1 else 0
        total = min(x for x in (by_positive, by_negative) if x > 0)
        sizes = {positive: round(total * rate), negative: total - round(total * rate)}
    chunks = []
    for value, chunk in df.groupby(column, sort=False):
        want = min(sizes.get(value, len(chunk)), len(chunk))
        take = rng.choice(len(chunk), size=want, replace=False)
        chunks.append(chunk.iloc[take])
    out = pd.concat(chunks).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out, {
        "balanced": True,
        "before": counts.to_dict(),
        "after": out[column].value_counts().to_dict(),
    }
