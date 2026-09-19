---
license: mit
language:
- en
- uk
pretty_name: Hiring-bias mitigation — synthetic training data
task_categories:
- text-generation
tags:
- fairness
- bias-mitigation
- hiring
- recruitment
- ukrainian
- counterfactual
- synthetic
- dpo
- sft
size_categories:
- 100K<n<1M
configs:
- config_name: sft
  data_files:
  - split: train
    path: data/sft_train.parquet
  - split: validation
    path: data/sft_validation.parquet
  default: true
- config_name: sft_v2
  data_files:
  - split: train
    path: data/sft_v2_train.parquet
  - split: validation
    path: data/sft_v2_validation.parquet
- config_name: sft_unbalanced
  data_files:
  - split: train
    path: data/sft_unbalanced_train.parquet
  - split: validation
    path: data/sft_unbalanced_validation.parquet
- config_name: dpo
  data_files:
  - split: train
    path: data/dpo_train.parquet
  - split: validation
    path: data/dpo_validation.parquet
- config_name: dpo_decision
  data_files:
  - split: train
    path: data/dpo_decision_train.parquet
  - split: validation
    path: data/dpo_decision_validation.parquet
- config_name: dpo_consistency
  data_files:
  - split: train
    path: data/dpo_consistency_train.parquet
  - split: validation
    path: data/dpo_consistency_validation.parquet
- config_name: kto
  data_files:
  - split: train
    path: data/kto_train.parquet
  - split: validation
    path: data/kto_validation.parquet
- config_name: teacher_reference
  data_files:
  - split: en
    path: data/raw/en_reference.parquet
  - split: uk
    path: data/raw/uk_reference.parquet
- config_name: teacher_invariant
  data_files:
  - split: en
    path: data/raw/en_invariant.parquet
  - split: uk
    path: data/raw/uk_invariant.parquet
- config_name: teacher_biased
  data_files:
  - split: en
    path: data/raw/en_biased.parquet
  - split: uk
    path: data/raw/uk_biased.parquet
---

# Hiring-bias mitigation — synthetic training data

Semi-synthetic data for training LLMs to make hiring decisions that do not depend on a
protected attribute (military status, gender, religion), in **English and Ukrainian**.

- **Real inputs, synthetic labels.** CVs and job descriptions are real, anonymised postings
  from the Djinni Recruitment Dataset (MIT). Decisions and rationales were written by the
  teacher model `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4`.
- **Code and results:** [Stereotypes-in-LLMs/hiring_bias_mitigation](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation).
  Model responses of the study: [hiring-bias-mitigation-responses](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-responses).
  Everything together: the [Hiring Bias Mitigation](https://huggingface.co/collections/Stereotypes-in-LLMs/hiring-bias-mitigation-6aae99f67667367c7149cedd) collection.

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
| `sft` | train 14,890, validation 770 | Invariant SFT targets: prompt with an injected attribute, completion with the attribute-free anchor decision and a rationale that never names the attribute. Balanced to 50% hire. | Trained the four main SFT adapters (Qwen3.5-4B/9B × EN/UK). |
| `sft_v2` | train 21,725, validation 1,154 | SFT targets resampled toward the benchmark's decision mix. | Decision-weighted SFT probe (Qwen3.5-9B EN). |
| `sft_unbalanced` | train 68,166, validation 3,596 | All filtered invariant targets before decision balancing (89% reject). | Source pool for `sft` and `sft_v2`. |
| `dpo` | train 31,476, validation 1,647 | Preference pairs: `chosen` = invariant response, `rejected` = the teacher's response when told to let the attribute drive the decision. | DPO probe (Qwen3.5-9B EN). |
| `dpo_decision` | train 31,476, validation 1,647 | The `dpo` pairs reduced to the decision alone (`{"decision": "hire"}` vs `reject`), no rationale, so the pair cannot be told apart by wording. | Decision-only DPO probe (Qwen3.5-9B EN). |
| `dpo_consistency` | train 6,879, validation 345 | Decision-only pairs from the student model's (Qwen3.5-9B) own counterfactually unstable sets: chosen = the set's majority decision, rejected = the opposite. Report: `dpo_consistency_report.json`. | Built, not trained (future work). |
| `kto` | train 62,952, validation 3,294 | The `dpo` pairs unpaired into desirable / undesirable completions (`label`). | Built, not trained (future work). |
| `teacher_reference` | en 3,000, uk 3,000 | Unfiltered teacher pass 1: the decision on the bare job–CV pair, no attribute. The anchor verdict. | Input to every subset above. |
| `teacher_invariant` | en 36,000, uk 35,988 | Unfiltered teacher pass 2: per attribute variant, the response a fair screener would give. | Filtered into the SFT `completion` / DPO `chosen` side. |
| `teacher_biased` | en 36,000, uk 35,988 | Unfiltered teacher pass 3: the same variant with the attribute allowed to drive the outcome. Includes refusals and non-biased outputs that the filters dropped. | Filtered into the DPO `rejected` side. |

```python
from datasets import load_dataset
sft = load_dataset("Stereotypes-in-LLMs/hiring-bias-mitigation-synthetic-data", "sft")
dpo = load_dataset("Stereotypes-in-LLMs/hiring-bias-mitigation-synthetic-data", "dpo", split="train")
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
@inproceedings{drushchak-romanyshyn-2024-introducing,
  title     = {Introducing the Djinni Recruitment Dataset: A Corpus of Anonymized CVs and Job Postings},
  author    = {Drushchak, Nazarii and Romanyshyn, Mariana},
  booktitle = {Proceedings of the Third Ukrainian Natural Language Processing Workshop (UNLP) @ LREC-COLING 2024},
  year      = {2024}
}
```
