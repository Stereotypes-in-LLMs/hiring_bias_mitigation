"""Runs and scores one audit config.

    python scripts/run_audit.py --config configs/audit/qwen3.5-4b_en_baseline.yaml
    python scripts/run_audit.py --config <cfg> --score-only   # re-score cached generations
    python scripts/run_audit.py --config <cfg> --no-similarity  # skip the embedding measure

Generations are cached as parquet under $HBM_OUTPUT_ROOT. `--score-only` re-runs the metrics
against that cache, so adding a measure or fixing the parser never costs a second inference
pass over 450 pairs x 39 attributes x 3 conditions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.eval import audit as A  # noqa: E402
from hiring_bias_mitigation.eval import metrics as M  # noqa: E402
from hiring_bias_mitigation.eval import runner as R  # noqa: E402
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    REPO_ROOT,
    load_config,
    resolve_output_path,
)
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("run_audit")

RESULTS_DIR = REPO_ROOT / "eval" / "results"

#: Above this share of unparsable responses the run is a misconfiguration, not a result.
# Same thresholds the report and the analysis apply, so "this run aborted" and "this run is
# excluded from the tables" can never disagree.
PARSE_FAILURE_ABORT = A.MAX_PARSE_FAILURE
PARSE_FAILURE_WARN = A.PARSE_FAILURE_WARN


def _cache_similarity(raw_path: Path, scored) -> None:
    """Writes the computed feedback_similarity column back into the run's parquet."""
    if not raw_path.exists():
        return
    import pandas as pd

    frame = pd.read_parquet(raw_path)
    if len(frame) != len(scored):
        log.warning(
            "not caching similarity: artifact has %d rows, scored frame %d",
            len(frame), len(scored),
        )
        return
    frame["feedback_similarity"] = scored["feedback_similarity"].to_numpy()
    frame.to_parquet(raw_path, index=False)
    log.info("cached feedback_similarity into %s", raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--score-only", action="store_true",
                        help="reuse cached generations instead of calling the model")
    parser.add_argument("--no-similarity", action="store_true",
                        help="skip feedback similarity (no sentence-transformer needed)")
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument(
        "--recompute-similarity", action="store_true",
        help="re-encode feedback similarity even if the artifact already caches it",
    )
    parser.add_argument(
        "--similarity-device", default=None,
        help="device for the feedback-similarity encoder; pass 'cpu' to re-score while "
             "another run holds the GPU",
    )
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = R.build_run_name(cfg)
    raw_dir = Path(resolve_output_path(cfg.get("raw_dir", "outputs/raw")))
    raw_path = raw_dir / f"{run_name}.parquet"

    if args.score_only:
        if not raw_path.exists():
            raise SystemExit(
                f"no cached generations at {raw_path}. Drop --score-only to generate them."
            )
        parsed, meta = R.load_raw(raw_path)
        log.info("scoring cached run %s (%d rows)", run_name, len(parsed))
    else:
        parsed, meta = R.run_audit(cfg)
        R.save_raw(parsed, meta, raw_dir)

    similarity = None
    if not args.no_similarity:
        similarity = M.SimilarityScorer(
            model_name=cfg.get("embedding_model", M.DEFAULT_EMBEDDING_MODEL),
            device=args.similarity_device,
        )
        meta["embedding_model"] = similarity.model_name

    record = A.score_run(
        parsed, run_name, meta,
        similarity=similarity,
        recompute_similarity=args.recompute_similarity,
        n_permutations=args.permutations,
        seed=cfg.get("seed", 42),
    )
    path = record.to_json(Path(args.results_dir) / f"{run_name}.json")
    log.info("wrote %s", path)

    # Persist the one expensive derived column back into the artifact. Every later re-score
    # then costs seconds instead of an hour of sentence encoding, which is the difference
    # between "we can add a measure retroactively" and "we cannot afford to".
    if record.scored is not None and record.scored.attrs.get("similarity_computed"):
        _cache_similarity(raw_path, record.scored)

    summary = {k: v for k, v in record.summary.items() if k != "flagged"}
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    for item in record.manual_review:
        print(f"  [manual-review:{item['severity']}] {item['check']} — {item['n_rows']} rows")

    # A run that mostly failed to parse produces a full set of NaN metrics and a report that
    # looks merely empty rather than broken. Say so loudly and exit non-zero, so a sweep
    # stops here instead of burning the remaining GPU-hours on the same misconfiguration.
    failure_rate = record.summary.get("parse_failure_rate") or 0.0
    if failure_rate > PARSE_FAILURE_ABORT:
        print(
            f"\nERROR: {failure_rate:.1%} of responses could not be parsed "
            f"(threshold {PARSE_FAILURE_ABORT:.0%}). Inspect the raw generations before "
            f"trusting anything downstream:\n"
            f"  {raw_path}\n"
            f"Common causes: a reasoning model truncated before it emits JSON (set "
            f"chat_template_kwargs.enable_thinking=false, or raise generation.max_new_tokens), "
            f"or a chat template that was not applied.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if failure_rate > PARSE_FAILURE_WARN:
        log.warning(
            "%.1f%% parse failures -- denominators are materially reduced; check §5 of the "
            "report and the raw generations.", 100 * failure_rate,
        )


if __name__ == "__main__":
    main()
