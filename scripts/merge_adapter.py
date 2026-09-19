"""Merges a LoRA adapter into its base model so it can be audited without LoRA.

The audit does this itself before serving an adapter through vLLM (see
`hiring_bias_mitigation.mitigation.merge` for why); this script does it ahead of time.

    python scripts/merge_adapter.py --adapter outputs/sft/qwen3.5-9b_en_only
    # -> outputs/merged/qwen3.5-9b_en_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.mitigation.merge import merge  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, nargs="+",
                        help="adapter dirs, relative to the drive")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for adapter in args.adapter:
        print(merge(adapter, force=args.force))


if __name__ == "__main__":
    main()
