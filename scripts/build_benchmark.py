"""Fetches and validates the golden evaluation benchmark, and writes the holdout.

The benchmark is 450 job-CV pairs per language plus the GPT-4o attribute-free reference
feedback, released with the audit study this work extends. It is vendored under
`data/benchmark/`; this script re-fetches it from the upstream repository when it is missing,
then validates it and materialises `holdout_ids.json`.

Run it once after cloning:
    python scripts/build_benchmark.py
    python scripts/build_benchmark.py --refetch   # re-pull even if the files are present
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.data import benchmark as B  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("build_benchmark")

UPSTREAM = (
    "https://raw.githubusercontent.com/TianaLina/Fairness-in-AI-Recruitment/main/"
    "unbiased_feedbacks/{name}"
)
FILES = {
    "en": ("ideal_feedbacks.csv", "reference_feedback_en.csv"),
    "uk": ("ideal_feedbacks_uk.csv", "reference_feedback_uk.csv"),
}
EXPECTED_PAIRS = 450


def fetch(lang: str, refetch: bool) -> Path:
    remote, local = FILES[lang]
    path = B.BENCHMARK_DIR / local
    if path.exists() and not refetch:
        log.info("%s already present (%s)", lang, path.name)
        return path
    url = UPSTREAM.format(name=remote)
    log.info("fetching %s -> %s", url, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())
    return path


def validate(lang: str) -> None:
    df = B.load_benchmark(lang)
    problems = []
    if len(df) != EXPECTED_PAIRS:
        problems.append(f"expected {EXPECTED_PAIRS} pairs, found {len(df)}")
    if df["reference_feedback"].str.strip().eq("").any():
        problems.append("some reference feedback is empty")
    if df["cv"].str.strip().eq("").any():
        problems.append("some CVs are empty")
    unmapped = sorted(set(df["reference_decision"]) - _known_decisions(lang))
    if unmapped:
        problems.append(f"unrecognised reference decisions: {unmapped}")
    if problems:
        raise SystemExit(f"benchmark {lang} failed validation:\n  - " + "\n  - ".join(problems))
    log.info(
        "%s: %d pairs, %d candidates, %d jobs, reference decisions %s",
        lang, len(df), df["candidate_id"].nunique(), df["job_id"].nunique(),
        df["reference_decision"].value_counts().to_dict(),
    )


def _known_decisions(lang: str) -> set[str]:
    return {"hire", "reject"} if lang == "en" else {"найняти", "відхилити"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refetch", action="store_true")
    args = parser.parse_args()

    for lang in ("en", "uk"):
        fetch(lang, args.refetch)
        validate(lang)

    path = B.write_holdout()
    holdout = B.holdout_ids()
    log.info(
        "wrote %s: %d candidate ids and %d job ids are now excluded from the training pool",
        path, len(holdout.candidate_ids), len(holdout.job_ids),
    )


if __name__ == "__main__":
    main()
