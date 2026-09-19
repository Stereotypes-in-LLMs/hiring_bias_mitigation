"""Regenerates reports/RESULTS.md and the manual-review sheets from eval/results/.

    python scripts/make_report.py

Both outputs are derived, never hand-edited. Everything in them traces back to a run record
in `eval/results/`, so a number in the paper can always be walked back to the generations
that produced it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.eval.manual_review import write_review_csv  # noqa: E402
from hiring_bias_mitigation.eval.report import (  # noqa: E402
    load_records,
    record_unusable_reason,
    unusable_records,
    write_report,
)
from hiring_bias_mitigation.utils.config import REPO_ROOT  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("make_report")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "eval" / "results"))
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "RESULTS.md"))
    parser.add_argument("--review-dir", default=str(REPO_ROOT / "reports" / "manual_review"))
    args = parser.parse_args()

    records = load_records(args.results_dir)
    excluded = unusable_records(args.results_dir)
    for record in excluded:
        log.warning(
            "excluded from the tables: %s — %s",
            record["run_name"], record_unusable_reason(record),
        )
    path = write_report(records, args.out, excluded)
    log.info("wrote %s from %d run record(s)", path, len(records))

    review_dir = Path(args.review_dir)
    csv_path = write_review_csv(records, review_dir / "review_queue.csv")
    log.info("wrote %s", csv_path)

    index = {
        record["run_name"]: {
            "model": record["meta"].get("model"),
            "lang": record["meta"].get("lang"),
            "mitigation": (record["meta"].get("mitigation") or {}).get("family", "none"),
            "n_prompts": record["meta"].get("n_prompts"),
            "population_acceptance_rate": record["summary"].get("population_acceptance_rate"),
            "reference_agreement": record["summary"].get("reference_agreement"),
            "n_tests": record["summary"].get("n_tests"),
        }
        for record in records
    }
    Path(args.results_dir, "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
