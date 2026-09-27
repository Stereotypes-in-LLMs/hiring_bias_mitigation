"""Publishes every audited model response to one Hugging Face dataset, one subset per run.

The dataset holds, for each run, the raw generations (`runs/<run>.parquet`), the metadata that
produced them (`runs/<run>.meta.json`) and the scored summary (`results/<run>.json`), plus the
study-level tables. The card is generated from those files, so it cannot drift from them.

    PUSH_TO_HUB=true python scripts/publish_responses.py --dry-run     # list, write card, no upload
    PUSH_TO_HUB=true python scripts/publish_responses.py               # upload
    PUSH_TO_HUB=true python scripts/publish_responses.py --add-families sft dpo  # later

Runs are selected by mitigation family. Adapter runs are published only if they were audited
from merged weights: the earlier vLLM-LoRA audits measured a model close to the untrained base
(see `mitigation/merge.py`), and the report already excludes them.
"""

# ruff: noqa: E501  (the dataset card template carries long Markdown links)
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.eval.report import (  # noqa: E402
    record_unusable_reason,
    record_usable,
    served_via_vllm_lora,
)
from hiring_bias_mitigation.utils.config import REPO_ROOT, resolve_output_path  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("publish_responses")

DEFAULT_REPO = "Stereotypes-in-LLMs/hiring-bias-mitigation-responses"
GITHUB = "https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation"
COLLECTION = "https://huggingface.co/collections/Stereotypes-in-LLMs/hiring-bias-mitigation-6aae99f67667367c7149cedd"
DEFAULT_FAMILIES = ("baseline", "prompt", "scrub", "embedding")
DEFAULT_SUBSET = "Qwen3.5-9B--en--baseline"
STUDY_TABLES = ("set_stability.csv", "set_stability_by_group.csv",
                "mitigation_decision_table.csv")


def family_of(meta: dict) -> str:
    family = (meta.get("mitigation") or {}).get("family")
    return "baseline" if family in (None, "", "none") else family


def strategy_of(meta: dict) -> str:
    m = meta.get("mitigation") or {}
    fam = family_of(meta)
    if fam == "prompt":
        return meta.get("prompt_strategy", "--")
    if fam == "scrub":
        return m.get("mode") or m.get("method") or "--"
    if fam == "embedding":
        return f"{m.get('method', '--')} (layer {','.join(map(str, m.get('layers', [])))})"
    if fam in ("sft", "dpo"):
        return Path(m.get("lora_path", "--")).name
    return "--"


def select_runs(raw_dir: Path, results_dir: Path, families: set[str]) -> list[dict]:
    runs = []
    for meta_path in sorted(raw_dir.glob("*.meta.json")):
        name = meta_path.name.removesuffix(".meta.json")
        if "--smoke" in name or name.startswith("consistency_pool"):
            continue
        meta = json.loads(meta_path.read_text())
        if family_of(meta) not in families:
            continue
        result_path = results_dir / f"{name}.json"
        record = json.loads(result_path.read_text()) if result_path.exists() else None
        if record is None:
            log.warning("skipping %s: no scored result in %s", name, results_dir)
            continue
        if served_via_vllm_lora(record):
            log.warning("skipping %s: audited through vLLM LoRA (invalid)", name)
            continue
        runs.append({"name": name, "meta": meta, "record": record,
                     "parquet": raw_dir / f"{name}.parquet", "meta_path": meta_path,
                     "result_path": result_path})
    return runs


def _pct(x) -> str:
    return "--" if x is None or x != x else f"{100 * x:.1f}"


def build_card(runs: list[dict]) -> str:
    configs = []
    for r in runs:
        entry = [f"- config_name: {r['name']}", "  data_files:",
                 f"  - split: test\n    path: runs/{r['name']}.parquet"]
        if r["name"] == DEFAULT_SUBSET:
            entry.insert(1, "  default: true")
        configs.append("\n".join(entry))

    rows = []
    for r in sorted(runs, key=lambda r: (r["meta"]["model"], r["meta"]["lang"],
                                         DEFAULT_FAMILIES.index(family_of(r["meta"]))
                                         if family_of(r["meta"]) in DEFAULT_FAMILIES else 9,
                                         r["name"])):
        s, meta = r["record"]["summary"], r["meta"]
        ok = record_usable(r["record"])
        usable = "yes" if ok else f"**no** — {record_unusable_reason(r['record'])}"
        utility = _pct(s.get("reference_agreement")) if ok else "--"
        rows.append(
            f"| `{r['name']}` | {meta['model'].split('/')[-1]} | {meta['lang']} | "
            f"{family_of(meta)} | {strategy_of(meta)} | {', '.join(meta['protected_groups'])} | "
            f"{', '.join(meta.get('conditions', []))} | {s.get('n_total', 0):,} | "
            f"{_pct(s.get('parse_failure_rate'))} | {utility} | "
            f"{usable} |"
        )
    n_rows = sum(r["record"]["summary"].get("n_total", 0) for r in runs)
    models = sorted({r["meta"]["model"] for r in runs})
    families = sorted({family_of(r["meta"]) for r in runs})

    return f"""---
license: cc-by-4.0
language:
- en
- uk
pretty_name: Hiring-bias mitigation — model responses
tags:
- fairness
- bias-audit
- bias-mitigation
- hiring
- recruitment
- ukrainian
- counterfactual
task_categories:
- text-generation
size_categories:
- 1M<n<10M
configs:
{chr(10).join(configs)}
---

# Hiring-bias mitigation — model responses

Every response produced in the mitigation study of LLM hiring decisions: **{len(runs)} runs,
{n_rows:,} responses**, from {len(models)} open-weight models in English and Ukrainian, at
baseline and under each mitigation family ({', '.join(families)}). Each run is one subset.

- **All released artifacts:** the [Hiring Bias Mitigation]({COLLECTION}) collection.
- **Training data of the fine-tuned runs:** [hiring-bias-mitigation-synthetic-data](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-synthetic-data).
- **Code, configs, full results and findings:** [{GITHUB.split('github.com/')[1]}]({GITHUB})
  — see `reports/RESULTS.md` and `reports/MITIGATION_FINDINGS.md`.
- **The audit study this extends** (benchmark, attributes, injection templates, and its own
  responses): the `Stereotypes-in-LLMs/hiring-analyses-*` datasets, e.g.
  [hiring-analyses-baseline-en](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-analyses-baseline-en).
- **Underlying CVs and job descriptions:** the Djinni Recruitment Dataset
  ([candidate profiles](https://huggingface.co/datasets/Stereotypes-in-LLMs/recruitment-dataset-candidate-profiles-english),
  [job descriptions](https://huggingface.co/datasets/Stereotypes-in-LLMs/recruitment-dataset-job-descriptions-english)).

**Fine-tuned runs are included.** The six SFT cells (Qwen3.5-4B/9B and LAPA-12B, English and
Ukrainian) plus one decision-weighted probe, all audited from **merged weights**: vLLM's LoRA
path does not reproduce these adapters (agreement with HuggingFace + PEFT is 38–83% depending
on the architecture, against 95–99% for the base model), so every adapter is folded into the
weights before serving. Preference-optimisation (DPO) runs are internal probes and are not
published.

## Design

Each benchmark job–CV pair is shown to a model many times, identical except for one injected
protected attribute (military status, gender or religion), stated **explicitly** or
**implicitly** (a first-person sentence), plus an **attribute-free** control. Rows sharing a
`group_id` form one *counterfactual set*: a fair screener gives them all the same decision.
Decoding is greedy (temperature 0, seed 42), with thinking mode off.

| Family | What changes |
|---|---|
| baseline | nothing — the unmitigated model |
| prompt | the instruction (e.g. `structured_rubric`, `ignore_personal_info`) |
| scrub | the attribute is removed from the input first (lexical rules or an LLM) |
| embedding | LEACE concept erasure of the attribute at one hidden layer |

## Loading

```python
from datasets import load_dataset
ds = load_dataset("{DEFAULT_REPO}", "Qwen3.5-9B--en--prompt--structured_rubric", split="test")
```

To re-score a run without a GPU, download `runs/<run>.parquet` into the code repository's
`outputs/raw/` and run `python scripts/run_audit.py --config <its config> --score-only`.

## Columns

| Column | Meaning |
|---|---|
| `pair_id`, `candidate_id`, `job_id` | benchmark identifiers |
| `cv`, `job_description`, `job_position`, `lang` | the inputs, verbatim |
| `protected_group`, `protected_group_label`, `protected_attr` | the injected attribute |
| `condition` | `explicit`, `implicit` or `attr_free` |
| `implicit_injection` | the first-person sentence used in the implicit condition |
| `profile_rendered` | the candidate profile exactly as the model saw it |
| `group_id` | counterfactual set key |
| `raw_output` | the model's response, unmodified |
| `decision`, `feedback`, `outcome`, `raw_decision` | the parsed decision and rationale, and whether parsing succeeded |
| `reference_decision`, `reference_feedback` | the attribute-free GPT-4o reference from the audit study |
| `feedback_similarity` | cosine similarity of `feedback` to the reference rationale |

`runs/<run>.meta.json` holds the model, decoding settings, seed, mitigation block and runtime;
`results/<run>.json` the scored summary (disparity tests, refusal and parse rates, utility).
`tables/` holds the study-level set-stability comparisons on matched variants.

## Runs

*Utility* is agreement with the attribute-free reference decision. Runs marked not usable are
published for completeness and excluded from every analysis.

| Run | Model | Lang | Family | Strategy | Groups | Conditions | Rows | Parse fail % | Utility % | Usable |
|---|---|---|---|---|---|---|---:|---:|---:|---|
{chr(10).join(rows)}

## Intended use and limitations

- **Evaluation outputs, not training data.** Every row is built from a held-out benchmark
  candidate. Fine-tuning on them contaminates the benchmark.
- The reference decision is **attribute-free, not unbiased**: it was produced by GPT-4o.
- Qwen3.5-9B English mitigation runs cover military status under the implicit condition
  only; other cells cover all three groups. See the GitHub report for why.
- Model outputs may contain biased hiring judgements; that is what is being measured.

## Citation

> **TBD.** The citation for the mitigation paper, and further references, will be added here once the paper is published.

The mitigation paper is in preparation; until then please cite the code repository above and
the Djinni Recruitment Dataset:

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
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--add-families", nargs="+", default=[],
                        help="families to publish on top of baseline/prompt/scrub/embedding; "
                             "the card always lists every published run")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if os.environ.get("PUSH_TO_HUB", "").strip().lower() not in {"true", "1", "yes"}:
        raise SystemExit("PUSH_TO_HUB is not set to true -- refusing to publish")

    raw_dir = Path(resolve_output_path("outputs/raw"))
    families = [*DEFAULT_FAMILIES, *args.add_families]
    runs = select_runs(raw_dir, REPO_ROOT / "eval" / "results", set(families))
    if not runs:
        raise SystemExit("no runs selected")
    card = build_card(runs)
    card_path = REPO_ROOT / "reports" / "hf_responses_card.md"
    card_path.write_text(card, encoding="utf-8")
    size = sum(r["parquet"].stat().st_size for r in runs)
    print(f"{len(runs)} runs, {size / 1e6:.0f} MB -> {args.repo_id} "
          f"({'private' if args.private else 'PUBLIC'}); card at {card_path}")
    for r in runs:
        print(f"  {r['name']}")
    if args.dry_run:
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    ops = [CommitOperationAdd("README.md", card.encode())]
    for r in runs:
        ops += [CommitOperationAdd(f"runs/{r['name']}.parquet", str(r["parquet"])),
                CommitOperationAdd(f"runs/{r['name']}.meta.json", str(r["meta_path"])),
                CommitOperationAdd(f"results/{r['name']}.json", str(r["result_path"]))]
    for table in STUDY_TABLES:
        path = REPO_ROOT / "reports" / table
        if path.exists():
            ops.append(CommitOperationAdd(f"tables/{table}", str(path)))
    api.create_commit(args.repo_id, repo_type="dataset", operations=ops,
                      commit_message=f"Publish {len(runs)} runs ({', '.join(families)})")
    log.info("published -> https://huggingface.co/datasets/%s", args.repo_id)


if __name__ == "__main__":
    main()
