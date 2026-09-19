"""Publishes generated datasets and run artifacts to the Hugging Face Hub. Opt-in, never automatic.

Two kinds, and the difference is not cosmetic:

`--kind training` (default)
    The semi-synthetic SFT/DPO data. Contamination is a **hard error**: a training set that
    touches the benchmark invalidates every mitigation number, so the check runs again here
    even though the generator already asserted it.

`--kind eval`
    The raw per-run generations under `outputs/raw/`. These *are* benchmark outputs -- every
    row is a held-out candidate, by construction -- so the contamination check is inverted:
    it would fire on every file, and is skipped. Publishing them is what lets anyone re-score
    the study without a GPU (`run_audit.py --score-only`), and what makes the reported numbers
    checkable rather than merely stated.

    python scripts/push_dataset_to_hub.py --kind training \
        --artifacts-dir artifacts/semisynthetic-v1 \
        --repo-id <org>/hiring-bias-mitigation-semisynthetic

    python scripts/push_dataset_to_hub.py --kind eval \
        --artifacts-dir outputs/raw \
        --repo-id <org>/hiring-bias-mitigation-run-artifacts

Refuses to run unless PUSH_TO_HUB=true is set, and prints what it is about to publish first:
this pushes real (anonymised) CV text and model-generated hiring judgements to a public
endpoint, which is not something to do by accident. Pass --private for a private repo.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.data.benchmark import assert_no_leakage  # noqa: E402
from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("push_dataset_to_hub")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument(
        "--kind", choices=("training", "eval"), default="training",
        help="training = SFT/DPO data, contamination is a hard error; "
             "eval = raw run generations, which are benchmark outputs by construction",
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args()

    if os.environ.get("PUSH_TO_HUB", "").strip().lower() not in {"true", "1", "yes"}:
        raise SystemExit("PUSH_TO_HUB is not set to true -- refusing to publish. See .env.example")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set -- see .env.example")

    import pandas as pd
    from huggingface_hub import HfApi

    root = Path(resolve_output_path(args.artifacts_dir))
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {root}")

    # Last line of defence for training data: the generator asserts this too, but publishing
    # is irreversible. For eval artifacts the same check would fire on every file -- they are
    # the benchmark's own outputs -- so it is skipped and the difference is stated aloud.
    total = 0
    for path in files:
        frame = pd.read_parquet(path)
        if args.kind == "training":
            assert_no_leakage(frame, f"push_dataset_to_hub:{path.name}")
        total += len(frame)
        print(f"  {path.name}: {len(frame):,} rows")
    if args.kind == "eval":
        print(
            "\n  NOTE: --kind eval, so the contamination check is skipped. These rows are "
            "benchmark outputs by design.\n  Do NOT train on them."
        )

    visibility = "PRIVATE" if args.private else "PUBLIC"
    print(
        f"\nAbout to publish {total:,} rows from {root} to {visibility} repo "
        f"{args.repo_id}.\nThis contains anonymised CV text and model-generated hiring "
        f"judgements."
    )
    if not args.yes and input("Type 'publish' to continue: ").strip() != "publish":
        raise SystemExit("aborted")

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    if args.kind == "eval":
        _write_eval_card(root, files)
        ignore = []
    else:
        # The unfiltered teacher dumps stay local: they are large, and every row in them was
        # rejected by a quality filter for a reason the card already summarises.
        ignore = ["raw/*"]
    api.upload_folder(
        folder_path=str(root),
        repo_id=args.repo_id,
        repo_type="dataset",
        ignore_patterns=ignore,
    )
    log.info("published -> https://huggingface.co/datasets/%s", args.repo_id)


EVAL_CARD = """---
license: mit
language:
- en
- uk
tags:
- fairness
- bias-audit
- hiring
- evaluation-artifacts
---

# Hiring-bias audit — raw run artifacts

Every model response produced by the audit, one parquet per run, with the metadata that
produced it. Published so that the reported numbers can be **re-scored without a GPU and
without re-running any model**:

```bash
python scripts/run_audit.py --config <the run's config> --score-only
```

Re-scoring 161,550 rows takes about twelve seconds. That is the point of these files: a
change to a metric, a fix to the output parser, or an added measure costs nothing to apply
retroactively, and a reader can check a table rather than take it on trust.

## Columns

| Column | Meaning |
|---|---|
| `pair_id`, `candidate_id`, `job_id` | benchmark identifiers |
| `cv`, `job_description`, `job_position` | the inputs, verbatim |
| `protected_group`, `protected_attr`, `condition` | the injected counterfactual |
| `group_id` | the counterfactual set key — rows sharing it differ only in the attribute |
| `raw_output` | the model's response, unmodified |
| `decision`, `feedback`, `outcome`, `raw_decision` | the parse, and whether it succeeded |
| `reference_decision`, `reference_feedback` | the attribute-free GPT-4o reference |

`<run>.meta.json` carries the model, decoding configuration, seed, mitigation block and
wall-clock time for each run.

## These are evaluation outputs, not training data

Every row is built from a held-out benchmark candidate. **Do not fine-tune on them** — doing
so contaminates the benchmark and makes any subsequent fairness number meaningless. The
training data for this study is a separate release, built from the disjoint half of the
corpus.

## Files
"""


def _write_eval_card(root, files) -> None:
    """Writes a dataset card next to the artifacts, listing what is in the upload."""
    lines = [EVAL_CARD, ""]
    for path in files:
        meta_path = path.parent / f"{path.stem}.meta.json"
        detail = ""
        if meta_path.exists():
            import json as _json

            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            detail = (
                f" — {meta.get('model')}, {meta.get('lang')}, "
                f"{meta.get('n_prompts', 0):,} prompts"
            )
        lines.append(f"- `{path.name}`{detail}")
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
