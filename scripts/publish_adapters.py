# ruff: noqa: E501  (the model card template carries long Markdown links)
"""Publishes the trained LoRA adapters, one repo each, with a card built from their own audit.

Uploads the adapter only — `checkpoint-*` directories hold optimizer state and are training
leftovers, not part of the model. The card carries the numbers from this adapter's audited run
(counterfactual set stability, utility, hire rate, the operating-point control where it ran),
and the one instruction that matters for reuse: **merge before serving through vLLM**, because
its LoRA path does not reproduce these adapters.

    PUSH_TO_HUB=true python scripts/publish_adapters.py --dry-run
    PUSH_TO_HUB=true python scripts/publish_adapters.py --only qwen3.5-9b_en_only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from hiring_bias_mitigation.analysis.stability import _decision  # noqa: E402
from hiring_bias_mitigation.eval.report import _restricted_metrics  # noqa: E402
from hiring_bias_mitigation.utils.config import REPO_ROOT, resolve_output_path  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("publish_adapters")

ORG = "Stereotypes-in-LLMs"
GITHUB = "https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation"
RESPONSES = "https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-responses"
TRAINING = "https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-synthetic-data"
COLLECTION = "https://huggingface.co/collections/Stereotypes-in-LLMs/hiring-bias-mitigation-6aae99f67667367c7149cedd"

#: adapter dir -> (repo name, audited run, base model, licence)
ADAPTERS = {
    "outputs/sft/qwen3.5-4b_en_only": ("qwen3.5-4b-hiring-debias-sft-en", "Qwen3.5-4B--en--sft--adapter--en_only", "Qwen/Qwen3.5-4B", "apache-2.0"),
    "outputs/sft/qwen3.5-4b_uk_only": ("qwen3.5-4b-hiring-debias-sft-uk", "Qwen3.5-4B--uk--sft--adapter--uk_only", "Qwen/Qwen3.5-4B", "apache-2.0"),
    "outputs/sft/qwen3.5-9b_en_only": ("qwen3.5-9b-hiring-debias-sft-en", "Qwen3.5-9B--en--sft--adapter--en_only", "Qwen/Qwen3.5-9B", "apache-2.0"),
    "outputs/sft/qwen3.5-9b_uk_only": ("qwen3.5-9b-hiring-debias-sft-uk", "Qwen3.5-9B--uk--sft--adapter--uk_only", "Qwen/Qwen3.5-9B", "apache-2.0"),
    "outputs/sft/lapa-12b_en_only": ("lapa-12b-hiring-debias-sft-en", "lapa-v0.1.2-instruct--en--sft--adapter--en_only", "lapa-llm/lapa-v0.1.2-instruct", "gemma"),
    "outputs/sft/lapa-12b_uk_only": ("lapa-12b-hiring-debias-sft-uk", "lapa-v0.1.2-instruct--uk--sft--adapter--uk_only", "lapa-llm/lapa-v0.1.2-instruct", "gemma"),
}
#: adapter files worth publishing; `checkpoint-*` and `training_args.bin` stay local
KEEP = ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
        "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json",
        "run_config.json")


def numbers(run: str, model: str, lang: str) -> dict:
    """Everything the card quotes, read from the study's own outputs."""
    stability = pd.read_csv(REPO_ROOT / "reports" / "set_stability.csv")
    row = stability[(stability["model"] == model) & (stability["lang"] == lang)
                    & (stability["family"] == "sft") & (stability["group"] == "all")
                    & (stability["variant"] == f"adapter--{lang}_only")].iloc[0]
    by_group = pd.read_csv(REPO_ROOT / "reports" / "set_stability_by_group.csv")
    groups = by_group[(by_group["model"] == model) & (by_group["lang"] == lang)
                      & (by_group["family"] == "sft")
                      & (by_group["variant"] == f"adapter--{lang}_only")]
    decisions = pd.read_csv(REPO_ROOT / "reports" / "mitigation_decision_table.csv")
    utility = decisions[(decisions["model"] == model) & (decisions["lang"] == lang)
                        & (decisions["family"] == "sft")
                        & (decisions["variant"] == f"adapter--{lang}_only")]
    control_path = REPO_ROOT / "reports" / "operating_point" / f"{run}.json"
    control = json.loads(control_path.read_text()) if control_path.exists() else None
    record = json.loads((REPO_ROOT / "eval" / "results" / f"{run}.json").read_text())
    baseline = json.loads((REPO_ROOT / "eval" / "results"
                           / f"{model}--{lang}--baseline.json").read_text())
    # The baseline covers the full grid including intersections; this adapter covers its own
    # cells. Comparing the two unrestricted is the mistake that once "halved" a leakage rate.
    cells = set(record["groups"])
    summary = _restricted_metrics(record, cells)
    base = _restricted_metrics(baseline, cells)

    raw = pd.read_parquet(Path(resolve_output_path("outputs/raw")) / f"{run}.parquet",
                          columns=["decision", "reference_decision"])
    hire = lambda column: 100 * (raw[column].map(_decision) == "hire").mean()  # noqa: E731
    return {"row": row, "groups": groups, "control": control, "summary": summary, "base": base,
            "hire_adapter": hire("decision"), "hire_reference": hire("reference_decision"),
            "prompts": record["summary"]["n_total"],
            "utility_delta": float(utility["delta_utility_pp"].iloc[0]) if len(utility) else None}


def card(repo: str, run: str, base_model: str, licence: str, lang: str, n: dict) -> str:
    row, groups, control = n["row"], n["groups"], n["control"]
    group_rows = "\n".join(
        f"| {g.group.replace('_', ' ')} | {g.unstable_base_pct:.1f} | {g.unstable_run_pct:.1f} | "
        f"**{g.delta_pp:+.1f}** | {g.p_fdr:.0e} |" for g in groups.itertuples())
    control_block = ""
    if control:
        share = control["gain_explained_by_shift_pct"]
        factor = (control["unstable_base_at_matched_hire_rate"]
                  / max(control["unstable_adapter_at_own_threshold"], 1e-9))
        control_block = f"""
### Is it invariance, or just a stricter model?

Fine-tuning moves the hire rate ({100 * control['hire_rate_base']:.1f}% → {100 * control['hire_rate_adapter']:.1f}%), and instability depends on where the
decision threshold sits. Sweeping that threshold on each model's own hire-vs-reject margin over
all {control['prompts']:,} audited prompts:

| At a hire rate of {100 * control['hire_rate_adapter']:.1f}% | Unstable sets % |
|---|---:|
| base model, moved to that rate | {100 * control['unstable_base_at_matched_hire_rate']:.1f} |
| **this adapter** | **{100 * control['unstable_adapter_at_own_threshold']:.1f}** |

At equal hire rate the adapter is **{factor:.1f}× more consistent**; {share:.0f}% of the raw gain is attributable to
the shift. The adapter's curve lies below the base model's at every operating point.
"""
    caveat = ""
    if n["utility_delta"] is not None and n["utility_delta"] < 0:
        caveat = f"""
> **This adapter trades utility for consistency.** Agreement with the attribute-free reference
> decision falls {abs(n['utility_delta']):.1f} points, and it hires {n['hire_adapter']:.1f}% of candidates where the
> attribute-free reference hires {n['hire_reference']:.1f}%. It is the only cell in the study that pays this price. Read the
> utility column before deploying it.
"""
    return f"""---
license: {licence}
base_model: {base_model}
library_name: peft
language:
- {lang}
tags:
- fairness
- bias-mitigation
- hiring
- recruitment
- lora
- counterfactual
---

# {repo}

LoRA adapter that makes **{base_model}** decide hiring cases without letting a protected
attribute (military status, gender, religion) change the verdict. Trained on counterfactually
invariant targets; audited on a held-out counterfactual benchmark of {n['prompts']:,} decisions.

- **Study, code and full results:** [{GITHUB.split('github.com/')[1]}]({GITHUB})
- **Training data:** [hiring-bias-mitigation-synthetic-data]({TRAINING})
- **This adapter's audited responses:** [`{run}`]({RESPONSES}) (every decision, reproducible without a GPU)
- **Everything together:** the [Hiring Bias Mitigation]({COLLECTION}) collection
{caveat}
## What it changes

A **counterfactual set** is one candidate–job pair evaluated with every attribute variant; it is
*unstable* when the decision is not the same across them — the attribute alone tipped it. Paired
with the same sets at baseline, on matched variants, {row.sets} sets:

| | Base model | This adapter |
|---|---:|---:|
| Unstable sets | {row.unstable_base_pct:.1f}% | **{row.unstable_run_pct:.1f}%** |
| Change | | **{row.delta_pp:+.1f} pp** [{row.ci_low:+.1f}, {row.ci_high:+.1f}] |
| Sets fixed : broken | | {row.fixed} : {row.broken} |
| Utility (agreement with the attribute-free reference) | {100 * n['base']['reference_agreement']:.1f}% | {100 * n['summary']['reference_agreement']:.1f}% |
| Rationales naming the attribute | {100 * n['base']['attribute_mention_rate']:.1f}% | {100 * n['summary']['attribute_mention_rate']:.1f}% |

Per protected group, all significant after Benjamini–Hochberg correction:

| Group | Base unstable % | Adapter unstable % | Δ pp | p (FDR) |
|---|---:|---:|---:|---:|
{group_rows}
{control_block}
## Serving it — merge first

**Do not serve this adapter through vLLM's LoRA path.** For these architectures that path does
not reproduce the trained model: on the same prompts, HuggingFace + PEFT and vLLM agree on
95–99% of the *base* model's decisions but only 38–83% of the adapter's. Fold the adapter into
the weights and serve the result as an ordinary checkpoint:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("{base_model}", dtype="bfloat16")
model = PeftModel.from_pretrained(model, "{ORG}/{repo}").merge_and_unload()
model.save_pretrained("merged")          # then: vllm serve merged
```

The study's audits all ran on merged weights (`scripts/merge_adapter.py`).

## Intended use and limits

- **Research on bias mitigation in LLM-assisted hiring.** Not a hiring system, and not validated
  for deployment. The reference decision it was trained toward is GPT-4o's, which is
  attribute-free by construction but **not unbiased**.
- Instability falls to a few percent of sets, **not to zero**: a residual dependence on the
  attribute remains, and ~0.5% of sets flip from decoding nondeterminism alone.
- Trained and measured on one corpus (anonymised Djinni CVs and postings) in {"English" if lang == "en" else "Ukrainian"}, on three
  protected attributes, with one LoRA configuration and one seed. Intersections were evaluated
  but never trained.
- Fine-tuning also shifts the model's overall hire rate; check that shift against your own
  operating point before using it.

## Citation

The paper is in preparation; until then cite the repository above and the Djinni Recruitment
Dataset (Drushchak & Romanyshyn, 2024).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", help="publish just these adapter directories")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if os.environ.get("PUSH_TO_HUB", "").strip().lower() not in {"true", "1", "yes"}:
        raise SystemExit("PUSH_TO_HUB is not set to true -- refusing to publish")

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    out_dir = REPO_ROOT / "reports" / "model_cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    for adapter, (name, run, base_model, licence) in ADAPTERS.items():
        if args.only and Path(adapter).name not in args.only:
            continue
        path = Path(resolve_output_path(adapter))
        if not (path / "adapter_config.json").is_file():
            log.warning("skipping %s: no adapter there", path)
            continue
        model, lang = run.split("--")[0], run.split("--")[1]
        text = card(name, run, base_model, licence, lang, numbers(run, model, lang))
        (out_dir / f"{name}.md").write_text(text, encoding="utf-8")
        files = [f for f in KEEP if (path / f).is_file()]
        size = sum((path / f).stat().st_size for f in files) / 1e6
        print(f"{name}: {len(files)} files, {size:.0f} MB  <- {path}")
        if args.dry_run:
            continue
        repo = f"{ORG}/{name}"
        api.create_repo(repo, repo_type="model", private=args.private, exist_ok=True)
        ops = [CommitOperationAdd("README.md", text.encode())]
        ops += [CommitOperationAdd(f, str(path / f)) for f in files]
        api.create_commit(repo, repo_type="model", operations=ops,
                          commit_message="Publish the audited LoRA adapter with its results")
        log.info("published -> https://huggingface.co/%s", repo)


if __name__ == "__main__":
    main()
