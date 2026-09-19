# ruff: noqa: E501  (the dataset card template carries long Markdown links)
"""Publishes the semi-synthetic training data to the Hugging Face Hub, one subset per dataset kind.

Every file is checked against the benchmark hold-out before upload: a training row touching
a benchmark candidate or job would contaminate every mitigation number, so a leak is a hard
error here even though the generator already asserted it. The card is generated from the
files, so row counts cannot drift from what is uploaded.

    PUSH_TO_HUB=true python scripts/publish_training_data.py --dry-run
    PUSH_TO_HUB=true python scripts/publish_training_data.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.data.benchmark import assert_no_leakage  # noqa: E402
from hiring_bias_mitigation.utils.config import REPO_ROOT, resolve_output_path  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("publish_training_data")

DEFAULT_REPO = "Stereotypes-in-LLMs/hiring-bias-mitigation-synthetic-data"
GITHUB = "https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation"
RESPONSES = "https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-responses"
COLLECTION = "https://huggingface.co/collections/Stereotypes-in-LLMs/hiring-bias-mitigation-6aae99f67667367c7149cedd"

#: subset -> (files by split, what it is, how the study used it)
SUBSETS = {
    "sft": ({"train": "sft_train.parquet", "validation": "sft_validation.parquet"},
            "Invariant SFT targets: prompt with an injected attribute, completion with the attribute-free anchor decision and a rationale that never names the attribute. Balanced to 50% hire.",
            "Trained the four main SFT adapters (Qwen3.5-4B/9B × EN/UK)."),
    "sft_v2": ({"train": "sft_v2_train.parquet", "validation": "sft_v2_validation.parquet"},
               "SFT targets resampled toward the benchmark's decision mix.",
               "Decision-weighted SFT probe (Qwen3.5-9B EN)."),
    "sft_unbalanced": ({"train": "sft_unbalanced_train.parquet", "validation": "sft_unbalanced_validation.parquet"},
                       "All filtered invariant targets before decision balancing (89% reject).",
                       "Source pool for `sft` and `sft_v2`."),
    "dpo": ({"train": "dpo_train.parquet", "validation": "dpo_validation.parquet"},
            "Preference pairs: `chosen` = invariant response, `rejected` = the teacher's response when told to let the attribute drive the decision.",
            "DPO probe (Qwen3.5-9B EN)."),
    "dpo_decision": ({"train": "dpo_decision_train.parquet", "validation": "dpo_decision_validation.parquet"},
                     "The `dpo` pairs reduced to the decision alone (`{\"decision\": \"hire\"}` vs `reject`), no rationale, so the pair cannot be told apart by wording.",
                     "Decision-only DPO probe (Qwen3.5-9B EN)."),
    "dpo_consistency": ({"train": "dpo_consistency_train.parquet", "validation": "dpo_consistency_validation.parquet"},
                        "Decision-only pairs from the student model's (Qwen3.5-9B) own counterfactually unstable sets: chosen = the set's majority decision, rejected = the opposite. Report: `dpo_consistency_report.json`.",
                        "Built, not trained (future work)."),
    "kto": ({"train": "kto_train.parquet", "validation": "kto_validation.parquet"},
            "The `dpo` pairs unpaired into desirable / undesirable completions (`label`).",
            "Built, not trained (future work)."),
    "teacher_reference": ({"en": "raw/en_reference.parquet", "uk": "raw/uk_reference.parquet"},
                          "Unfiltered teacher pass 1: the decision on the bare job–CV pair, no attribute. The anchor verdict.",
                          "Input to every subset above."),
    "teacher_invariant": ({"en": "raw/en_invariant.parquet", "uk": "raw/uk_invariant.parquet"},
                          "Unfiltered teacher pass 2: per attribute variant, the response a fair screener would give.",
                          "Filtered into the SFT `completion` / DPO `chosen` side."),
    "teacher_biased": ({"en": "raw/en_biased.parquet", "uk": "raw/uk_biased.parquet"},
                       "Unfiltered teacher pass 3: the same variant with the attribute allowed to drive the outcome. Includes refusals and non-biased outputs that the filters dropped.",
                       "Filtered into the DPO `rejected` side."),
}
EXTRA_FILES = ("generation_report.json", "dpo_consistency_report.json")


def build_card(counts: dict, root: Path) -> str:
    import yaml

    configs = []
    for i, (name, (splits, _, _)) in enumerate(SUBSETS.items()):
        entry = {"config_name": name,
                 "data_files": [{"split": s, "path": f"data/{f}"} for s, f in splits.items()]}
        if i == 0:
            entry["default"] = True
        configs.append(entry)
    front = {
        "license": "mit", "language": ["en", "uk"], "pretty_name": "Hiring-bias mitigation — synthetic training data",
        "task_categories": ["text-generation"],
        "tags": ["fairness", "bias-mitigation", "hiring", "recruitment", "ukrainian", "counterfactual",
                 "synthetic", "dpo", "sft"],
        "size_categories": ["100K<n<1M"], "configs": configs,
    }
    rows = []
    for name, (splits, what, used) in SUBSETS.items():
        sizes = ", ".join(f"{s} {counts[f]:,}" for s, f in splits.items())
        rows.append(f"| `{name}` | {sizes} | {what} | {used} |")
    report = json.loads((root / "generation_report.json").read_text())
    teacher = report.get("teacher_model", "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4")

    return f"""---
{yaml.safe_dump(front, sort_keys=False, allow_unicode=True).strip()}
---

# Hiring-bias mitigation — synthetic training data

Semi-synthetic data for training LLMs to make hiring decisions that do not depend on a
protected attribute (military status, gender, religion), in **English and Ukrainian**.

- **Real inputs, synthetic labels.** CVs and job descriptions are real, anonymised postings
  from the Djinni Recruitment Dataset (MIT). Decisions and rationales were written by the
  teacher model `{teacher}`.
- **Code and results:** [{GITHUB.split('github.com/')[1]}]({GITHUB}).
  Model responses of the study: [hiring-bias-mitigation-responses]({RESPONSES}).
  Everything together: the [Hiring Bias Mitigation]({COLLECTION}) collection.

> **Content warning.** The `rejected` side of the preference subsets and the
> `teacher_biased` subset contain deliberately discriminatory hiring rationales (e.g.
> rejecting a candidate for their religion or military status). They were generated as
> negative examples for bias-mitigation research and must not be used as positive training
> targets.

## How an example is built

1. **Anchor** — the teacher decides on the bare job–CV pair, with no attribute present.
2. **Invariant** — the attribute is injected (as an explicit field, or as a first-person
   sentence in the CV), and the teacher writes the response a fair screener would give: the
   **anchor's decision**, with a rationale that never mentions the attribute.
3. **Biased** — the same variant, with the teacher told to let the attribute drive the outcome.

Invariant responses become SFT targets and DPO `chosen`; biased ones become DPO `rejected`.
Generations are filtered, never repaired: unparsable responses, verdicts drifting from the
anchor, rationales naming the attribute, meta-commentary about fairness, wrong language or
length, and "biased" responses that are neither biased nor decision-flipping are dropped.
Per-reason counts are in `generation_report.json`.

## Subsets

| Subset | Rows | What it is | Used in the study |
|---|---|---|---|
{chr(10).join(rows)}

```python
from datasets import load_dataset
sft = load_dataset("{DEFAULT_REPO}", "sft")
dpo = load_dataset("{DEFAULT_REPO}", "dpo", split="train")
```

## Contamination control

The evaluation benchmark's 300 candidates and 301 jobs are excluded from the source pool
before matching. Every file here was checked again against that hold-out before upload.
Candidates whose own CV already mentions a protected characteristic are also removed, so an
injected attribute never contradicts its profile.

## Known issues

These were measured in the study and matter for anyone reusing the data. Details and proposed
fixes: the GitHub README, *Future work: the synthetic training data*.

- **The biased side is overt; real model bias is covert.** 91% of DPO `rejected` responses
  name the protected attribute outright, and 81% carry the same decision as `chosen`. The
  pairs can be told apart by wording alone, while the audited models rarely name the attribute
  (~1–2% of rationales) even when their decisions depend on it. `dpo_decision` is a first
  attempt at removing the wording shortcut.
- **The anchor is one sample at temperature 0.7.** It is least reliable on borderline
  candidates, which is where bias acts most.
- **Skewed verdicts.** The teacher rejects 89% of pairs (`sft_unbalanced`), against 34% hire in
  the benchmark; `sft` is balanced to 50/50.
- **Narrow and deep.** 3,000 pairs per language, each expanded to ~12 attribute variants with
  near-identical rationales.
- **Uneven biased-pass yield:** 55.6% English, 36.8% Ukrainian (refusals and non-biased
  outputs), and fewer pairs for religion than for military status.

## Intended use

Research on bias mitigation in LLM-assisted hiring. It is **not** a hiring-outcome dataset: the
decisions are a language model's opinions and carry no ground truth about suitability.
Training on it teaches invariance of the decision to the attribute, not good screening, and
does not address bias carried by writing style rather than by an attribute.

## Citation

> **TBD.** The citation for the mitigation paper, and further references, will be added here once the paper is published.

The mitigation paper is in preparation; until then please cite the code repository and the
Djinni Recruitment Dataset:

```bibtex
@inproceedings{{drushchak-romanyshyn-2024-introducing,
  title     = {{Introducing the Djinni Recruitment Dataset: A Corpus of Anonymized CVs and Job Postings}},
  author    = {{Drushchak, Nazarii and Romanyshyn, Mariana}},
  booktitle = {{Proceedings of the Third Ukrainian Natural Language Processing Workshop (UNLP) @ LREC-COLING 2024}},
  year      = {{2024}}
}}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts-dir", default="artifacts/semisynthetic-v1")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if os.environ.get("PUSH_TO_HUB", "").strip().lower() not in {"true", "1", "yes"}:
        raise SystemExit("PUSH_TO_HUB is not set to true -- refusing to publish")

    import pandas as pd

    root = Path(resolve_output_path(args.artifacts_dir))
    counts = {}
    for splits, _, _ in SUBSETS.values():
        for f in splits.values():
            frame = pd.read_parquet(root / f)
            assert_no_leakage(frame, f"publish_training_data:{f}")
            counts[f] = len(frame)
    card = build_card(counts, root)
    card_path = REPO_ROOT / "reports" / "hf_training_data_card.md"
    card_path.write_text(card, encoding="utf-8")
    size = sum((root / f).stat().st_size for f in counts)
    print(f"{len(counts)} files, {sum(counts.values()):,} rows, {size / 1e6:.0f} MB -> "
          f"{args.repo_id} ({'private' if args.private else 'PUBLIC'}); card at {card_path}")
    if args.dry_run:
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    ops = [CommitOperationAdd("README.md", card.encode())]
    ops += [CommitOperationAdd(f"data/{f}", str(root / f)) for f in counts]
    ops += [CommitOperationAdd(f, str(root / f)) for f in EXTRA_FILES if (root / f).exists()]
    api.create_commit(args.repo_id, repo_type="dataset", operations=ops,
                      commit_message="Publish the semi-synthetic training data")
    log.info("published -> https://huggingface.co/datasets/%s", args.repo_id)


if __name__ == "__main__":
    main()
