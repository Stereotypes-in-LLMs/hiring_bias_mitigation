"""What a human must look at before any of this becomes a claim in a paper.

Several signals in this pipeline are heuristics: the decision lexicon maps loose model
wording onto hire/reject, the attribute-mention detector is prefix matching, feedback
similarity is a weak instrument by the audit paper's own account, and an uncorrected flag on
a small effect is exactly what multiple comparisons produce from a fair model. None of these
should be reported as established without someone reading the underlying rows.

This module builds that queue. Every item names the check, the reason it was raised, and a
sample of the rows to read, so the reviewer can work from `reports/MANUAL_REVIEW.md` and the
CSVs beside it rather than re-deriving anything.

Checks, and what a real problem looks like in each:

`fuzzy_decision_mapping`
    The model did not write "hire"/"reject" and the lexicon guessed. If the guess is wrong,
    every downstream acceptance rate is wrong. Read the raw decisions.

`refusal_or_parse_failure_skew`
    Refusals and parse failures are excluded from the denominators, so a *difference* in
    those rates across attributes silently changes every statistic that follows -- and is
    itself a bias signal worth reporting.

`language_drift`
    A Ukrainian run answering in English (or vice versa). The audit paper screened models for
    Ukrainian output quality precisely because a noisier decision process inflates the
    inconsistency rate independently of any bias. A disparity measured on degraded output is
    not a fairness finding.

`attribute_leakage`
    The rationale appears to name the injected attribute. High-value evidence when real,
    prefix-matching noise when not.

`significant_but_tiny`
    Flagged at the uncorrected level with an effect size small enough that the flag is more
    plausibly multiplicity than signal. These are exactly the isolated flags the audit paper
    warns should be read as "warranting scrutiny", not as findings.

`degenerate_feedback`
    Empty, truncated or repeated rationales. Feedback similarity computed over these is
    measuring degeneration, not differential treatment.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

#: An acceptance-rate difference below this is too small to carry a claim on its own, given
#: 450 pairs per attribute and no correction. Used only to route rows to review.
TINY_EFFECT_AR = 0.05
#: Feedback similarity clusters in a narrow band, so its "tiny" threshold is much smaller.
TINY_EFFECT_SIM = 0.01
#: A per-attribute refusal/parse-failure rate this far from the run's own rate distorts
#: denominators enough to matter.
OUTCOME_SKEW = 0.10
MAX_SAMPLE_ROWS = 25

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[A-Za-z]")


def collect_manual_review(scored: pd.DataFrame, groups: dict, run_name: str) -> list[dict]:
    """Builds the review queue for one run. Cheap -- runs on every audit."""
    items: list[dict] = []
    items += _check_fuzzy_decisions(scored)
    items += _check_outcome_skew(scored)
    items += _check_language_drift(scored)
    items += _check_attribute_leakage(scored)
    items += _check_degenerate_feedback(scored)
    items += _check_tiny_effects(groups)
    for item in items:
        item["run_name"] = run_name
    return items


def _item(check: str, severity: str, message: str, n: int, rows: pd.DataFrame | None = None,
          **extra) -> dict:
    payload = {
        "check": check,
        "severity": severity,
        "message": message,
        "n_rows": int(n),
        **extra,
    }
    if rows is not None and len(rows):
        keep = [
            c
            for c in (
                "pair_id", "protected_group", "protected_attr", "condition", "lang",
                "outcome", "raw_decision", "decision", "feedback",
            )
            if c in rows.columns
        ]
        payload["sample"] = rows[keep].head(MAX_SAMPLE_ROWS).to_dict("records")
    return payload


def _check_fuzzy_decisions(df: pd.DataFrame) -> list[dict]:
    """Rows whose decision word was not one of the two the schema asked for."""
    if "raw_decision" not in df.columns:
        return []
    expected = {"hire", "reject", "найняти", "відхилити"}
    norm = df["raw_decision"].astype(str).str.strip().str.strip("`*_ .:;\"'").str.lower()
    fuzzy = df[(df["outcome"] == "decided") & (~norm.isin(expected))]
    if fuzzy.empty:
        return []
    counts = norm[fuzzy.index].value_counts().head(20).to_dict()
    return [
        _item(
            "fuzzy_decision_mapping",
            "high",
            "Decisions mapped onto hire/reject through the lexicon rather than matching the "
            "schema exactly. Verify each mapping -- a wrong one shifts the acceptance rate.",
            len(fuzzy),
            fuzzy,
            distinct_raw_decisions=counts,
        )
    ]


def _check_outcome_skew(df: pd.DataFrame) -> list[dict]:
    """Attributes whose refusal or parse-failure rate departs from the run's own rate."""
    items = []
    for outcome, label in (("refused", "refusal"), ("invalid", "parse-failure")):
        indicator = (df["outcome"] == outcome).astype(float)
        overall = float(indicator.mean())
        by_attr = indicator.groupby(df["protected_attr"]).mean()
        skewed = by_attr[(by_attr - overall).abs() > OUTCOME_SKEW]
        if skewed.empty:
            continue
        rows = df[df["protected_attr"].isin(skewed.index) & (df["outcome"] == outcome)]
        items.append(
            _item(
                "refusal_or_parse_failure_skew",
                "high",
                f"{label.capitalize()} rate differs by more than "
                f"{OUTCOME_SKEW:.0%} from the run rate ({overall:.1%}) for some attributes. "
                "These rows are excluded from every fairness statistic, so the skew changes "
                "the denominators -- and is a bias signal in its own right.",
                len(rows),
                rows,
                outcome=outcome,
                run_rate=overall,
                per_attribute_rate={k: float(v) for k, v in skewed.items()},
            )
        )
    return items


def _check_language_drift(df: pd.DataFrame) -> list[dict]:
    """Rationales written in the wrong language for the run."""
    drifted_idx = []
    for i, (lang, feedback) in enumerate(zip(df["lang"], df["feedback"].fillna(""))):
        text = unicodedata.normalize("NFKC", str(feedback))
        if len(text) < 20:
            continue
        cyr, lat = len(_CYRILLIC.findall(text)), len(_LATIN.findall(text))
        total = cyr + lat
        if total < 15:
            continue
        # Ukrainian IT CVs legitimately carry English technical terms, so the threshold is
        # about which script dominates, not about purity.
        if (lang == "uk" and cyr / total < 0.4) or (lang == "en" and cyr / total > 0.2):
            drifted_idx.append(i)
    if not drifted_idx:
        return []
    rows = df.iloc[drifted_idx]
    return [
        _item(
            "language_drift",
            "high",
            "Rationales are not in the run's language. Output quality confounds the "
            "inconsistency rate, so a disparity measured here is not yet a fairness finding "
            "-- check whether the model is usable in this language at all.",
            len(rows),
            rows,
            pct_of_run=float(len(rows) / max(len(df), 1)),
        )
    ]


def _check_attribute_leakage(df: pd.DataFrame) -> list[dict]:
    if "mentions_attribute" not in df.columns:
        return []
    rows = df[pd.to_numeric(df["mentions_attribute"], errors="coerce") > 0]
    if rows.empty:
        return []
    by_attr = (
        pd.to_numeric(df["mentions_attribute"], errors="coerce")
        .groupby(df["protected_attr"])
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )
    return [
        _item(
            "attribute_leakage",
            "medium",
            "The rationale appears to name the injected protected attribute. This is the "
            "channel a human reviewer sees under the EU AI Act's oversight requirement, so a "
            "confirmed mention is strong evidence -- but detection is prefix matching, so "
            "read the samples before quoting a rate.",
            len(rows),
            rows,
            top_attributes={k: float(v) for k, v in by_attr.items()},
        )
    ]


def _check_degenerate_feedback(df: pd.DataFrame) -> list[dict]:
    feedback = df["feedback"].fillna("").astype(str)
    decided = df["outcome"] == "decided"
    empty = decided & (feedback.str.strip().str.len() < 10)
    repeated = feedback.str.strip().str.lower()
    dupe_counts = repeated[decided & (repeated.str.len() >= 10)].value_counts()
    heavy_dupes = dupe_counts[dupe_counts > max(5, 0.02 * len(df))]

    items = []
    if empty.any():
        items.append(
            _item(
                "degenerate_feedback",
                "medium",
                "Decisions returned with an empty or near-empty rationale. Feedback "
                "similarity over these measures nothing; they are dropped from FS but still "
                "counted in the acceptance rate.",
                int(empty.sum()),
                df[empty],
            )
        )
    if len(heavy_dupes):
        items.append(
            _item(
                "degenerate_feedback",
                "medium",
                "The same rationale text is repeated across many rows. A model emitting one "
                "canned rationale cannot show rationale-level disparity, so a flat feedback "
                "similarity here is an artefact rather than a fairness property.",
                int(heavy_dupes.sum()),
                df[repeated.isin(heavy_dupes.index)],
                repeated_texts={k[:120]: int(v) for k, v in heavy_dupes.head(10).items()},
            )
        )
    return items


def _check_tiny_effects(groups: dict) -> list[dict]:
    """Uncorrected flags whose effect size is too small to carry a claim."""
    suspect = []
    for key, payload in groups.items():
        for row in payload["attributes"]:
            for measure, threshold in (
                ("acceptance_rate", TINY_EFFECT_AR),
                ("feedback_similarity", TINY_EFFECT_SIM),
                ("inconsistency_rate", TINY_EFFECT_AR),
            ):
                if not row.get(f"{measure}__significant"):
                    continue
                diff = row.get(f"{measure}__diff")
                if diff is None or (isinstance(diff, float) and np.isnan(diff)):
                    continue
                if abs(diff) < threshold and not row.get(f"{measure}__significant_fdr"):
                    suspect.append(
                        {
                            "group": key,
                            "protected_attr": row["protected_attr"],
                            "measure": measure,
                            "diff": float(diff),
                            "p": row.get(f"{measure}__p"),
                            "p_adj": row.get(f"{measure}__p_adj"),
                            "n_decided": row.get("n_decided"),
                        }
                    )
    if not suspect:
        return []
    return [
        _item(
            "significant_but_tiny",
            "low",
            "Flagged at the uncorrected level, effect size small, and not surviving FDR "
            "correction. With this many attributes tested a proportion of such flags is "
            "expected under the null even from a fair model -- treat them as scrutiny "
            "prompts, not findings, and do not build an argument on an isolated one.",
            len(suspect),
            None,
            entries=suspect[:60],
        )
    ]


def write_review_csv(records: list[dict], path: str | Path) -> Path:
    """Flattens the sampled rows of every review item into one CSV for reading in a sheet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        for item in record.get("manual_review", []):
            for sample in item.get("sample", []):
                rows.append(
                    {
                        "run_name": item.get("run_name", record.get("run_name")),
                        "check": item["check"],
                        "severity": item["severity"],
                        **sample,
                        "reviewer_verdict": "",
                        "reviewer_note": "",
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
