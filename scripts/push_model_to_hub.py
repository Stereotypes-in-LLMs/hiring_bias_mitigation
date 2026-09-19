"""Publishes a trained adapter or checkpoint. Opt-in, never automatic.

    python scripts/push_model_to_hub.py \
        --checkpoint outputs/sft/qwen3.5-4b_all-groups \
        --repo-id Stereotypes-in-LLMs/qwen3.5-4b-hiring-debias-sft

Run it by hand after reviewing the run's numbers in reports/RESULTS.md -- in particular the
utility column, since a model that improved every fairness metric by refusing to decide is
not a model to publish.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("push_model_to_hub")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if os.environ.get("PUSH_TO_HUB", "").strip().lower() not in {"true", "1", "yes"}:
        raise SystemExit("PUSH_TO_HUB is not set to true -- refusing to publish.")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set -- see .env.example")

    from huggingface_hub import HfApi

    path = Path(resolve_output_path(args.checkpoint))
    if not path.exists():
        raise SystemExit(f"{path} does not exist")

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(path), repo_id=args.repo_id, repo_type="model")
    log.info("published -> https://huggingface.co/%s", args.repo_id)


if __name__ == "__main__":
    main()
