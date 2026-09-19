"""The golden benchmark: 450 job-CV pairs per language, and the attribute-free reference.

Provenance (data/PROVENANCE.md): the pairs and the GPT-4o attribute-free reference feedback
come from the audit study's released artifacts. We do not re-sample them -- reusing the exact
pairs is what makes a mitigated number here comparable to an unmitigated number there.

Two things this module is responsible for:

1. Serving the benchmark as a tidy frame (`load_benchmark`).
2. Publishing the benchmark's identifiers as a holdout (`holdout_ids`), which the training
   data generator MUST subtract from the Djinni pool. Every candidate and every job that
   appears in the benchmark is excluded -- not just the exact pair -- because a CV seen in
   training under a different job is still a CV the mitigated model has memorised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd

from ..utils.config import REPO_ROOT

BENCHMARK_DIR = REPO_ROOT / "data" / "benchmark"

#: The reference CSVs use a per-language column suffix; normalise to these names.
CANONICAL_COLUMNS = [
    "pair_id",
    "candidate_id",
    "job_id",
    "cv",
    "job_description",
    "job_position",
    "lang",
    "reference_decision",
    "reference_feedback",
]

#: Decision vocabulary per language. The reference feedback was generated in the target
#: language, so "hire" is `hire` in English and `найняти` in Ukrainian.
HIRE_TOKEN = {"en": "hire", "uk": "найняти"}
REJECT_TOKEN = {"en": "reject", "uk": "відхилити"}


@dataclass(frozen=True)
class Holdout:
    """Identifiers that must never enter the mitigation training pool."""

    candidate_ids: frozenset[str]
    job_ids: frozenset[str]
    pair_ids: frozenset[str]

    def excludes(self, candidate_id: str, job_id: str) -> bool:
        return candidate_id in self.candidate_ids or job_id in self.job_ids


def _reference_path(lang: str) -> Path:
    path = BENCHMARK_DIR / f"reference_feedback_{lang}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python scripts/build_benchmark.py` to fetch the "
            "released benchmark artifacts."
        )
    return path


@cache
def load_benchmark(lang: str) -> pd.DataFrame:
    """The 450 attribute-free job-CV pairs for one language, with the reference decision."""
    df = pd.read_csv(_reference_path(lang))
    # The English and Ukrainian releases differ by one character in the feedback column name
    # ("feedback_gpt-4o" vs "feedbackgpt-4o"); accept either rather than patching the CSVs,
    # so a re-fetch from the upstream repo stays a straight copy.
    feedback_col = next(
        (c for c in ("feedback_gpt-4o", "feedbackgpt-4o", "feedback") if c in df.columns), None
    )
    if feedback_col is None:
        raise ValueError(f"{_reference_path(lang)}: no reference-feedback column found")

    out = pd.DataFrame(
        {
            "candidate_id": df["candidate_id"].astype(str),
            "job_id": df["job_id"].astype(str),
            "cv": df["CV"].astype(str),
            "job_description": df["Job Description"].astype(str),
            "job_position": df["Job Position"].astype(str),
            "lang": df["lang"].astype(str),
            "reference_decision": df["decision_gpt-4o"].astype(str).str.strip().str.lower(),
            "reference_feedback": df[feedback_col].astype(str),
        }
    )
    out.insert(0, "pair_id", out["candidate_id"] + "__" + out["job_id"])
    if out["pair_id"].duplicated().any():
        dupes = out.loc[out["pair_id"].duplicated(), "pair_id"].tolist()[:5]
        raise ValueError(f"benchmark {lang}: duplicate pair_ids, e.g. {dupes}")
    return out[CANONICAL_COLUMNS].reset_index(drop=True)


@cache
def holdout_ids() -> Holdout:
    """Every candidate/job/pair id appearing in either language's benchmark."""
    frames = [load_benchmark(lang) for lang in ("en", "uk")]
    all_pairs = pd.concat(frames, ignore_index=True)
    return Holdout(
        candidate_ids=frozenset(all_pairs["candidate_id"]),
        job_ids=frozenset(all_pairs["job_id"]),
        pair_ids=frozenset(all_pairs["pair_id"]),
    )


def write_holdout(path: str | Path | None = None) -> Path:
    """Materialises the holdout so the generator can assert against it without pandas."""
    path = Path(path) if path else BENCHMARK_DIR / "holdout_ids.json"
    h = holdout_ids()
    payload = {
        "note": (
            "Identifiers of the golden evaluation benchmark. Any training example built "
            "from one of these candidates or jobs is contamination -- "
            "scripts/generate_training_data.py asserts against this file."
        ),
        "n_candidate_ids": len(h.candidate_ids),
        "n_job_ids": len(h.job_ids),
        "n_pair_ids": len(h.pair_ids),
        "candidate_ids": sorted(h.candidate_ids),
        "job_ids": sorted(h.job_ids),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def assert_no_leakage(df: pd.DataFrame, where: str) -> None:
    """Hard-fails if a frame destined for training touches the benchmark.

    Called at every point where training data is written. A silent overlap here would
    invalidate every mitigation number in the paper, and it is the kind of mistake that is
    invisible in the metrics -- the mitigated model simply looks better than it is.
    """
    h = holdout_ids()
    bad_c = set(df["candidate_id"].astype(str)) & h.candidate_ids
    bad_j = set(df["job_id"].astype(str)) & h.job_ids
    if bad_c or bad_j:
        raise AssertionError(
            f"{where}: benchmark leakage -- {len(bad_c)} held-out candidate id(s) and "
            f"{len(bad_j)} held-out job id(s) present. Examples: "
            f"candidates={sorted(bad_c)[:3]}, jobs={sorted(bad_j)[:3]}"
        )
