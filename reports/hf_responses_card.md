---
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
- config_name: Qwen3.5-4B--en--baseline
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--en--baseline.parquet
- config_name: Qwen3.5-4B--en--embedding--leace
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--en--embedding--leace.parquet
- config_name: Qwen3.5-4B--en--prompt--counterfactual_invariance
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--en--prompt--counterfactual_invariance.parquet
- config_name: Qwen3.5-4B--en--prompt--fairness_constitution
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--en--prompt--fairness_constitution.parquet
- config_name: Qwen3.5-4B--en--prompt--ignore_personal_info
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--en--prompt--ignore_personal_info.parquet
- config_name: Qwen3.5-4B--en--prompt--reasoning
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--en--prompt--reasoning.parquet
- config_name: Qwen3.5-4B--en--prompt--recruiter_guidelines
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--en--prompt--recruiter_guidelines.parquet
- config_name: Qwen3.5-4B--en--prompt--second_pass_verification
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--en--prompt--second_pass_verification.parquet
- config_name: Qwen3.5-4B--en--prompt--structured_rubric
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--en--prompt--structured_rubric.parquet
- config_name: Qwen3.5-4B--en--prompt--zero_shot_cot
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--en--prompt--zero_shot_cot.parquet
- config_name: Qwen3.5-4B--en--scrub--lexical
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--en--scrub--lexical.parquet
- config_name: Qwen3.5-4B--en--scrub--llm
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--en--scrub--llm.parquet
- config_name: Qwen3.5-4B--en--sft--adapter--en_only
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--en--sft--adapter--en_only.parquet
- config_name: Qwen3.5-4B--uk--baseline
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--uk--baseline.parquet
- config_name: Qwen3.5-4B--uk--embedding--leace
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--uk--embedding--leace.parquet
- config_name: Qwen3.5-4B--uk--prompt--counterfactual_invariance
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--uk--prompt--counterfactual_invariance.parquet
- config_name: Qwen3.5-4B--uk--prompt--fairness_constitution
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--uk--prompt--fairness_constitution.parquet
- config_name: Qwen3.5-4B--uk--prompt--ignore_personal_info
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--uk--prompt--ignore_personal_info.parquet
- config_name: Qwen3.5-4B--uk--prompt--reasoning
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--uk--prompt--reasoning.parquet
- config_name: Qwen3.5-4B--uk--prompt--recruiter_guidelines
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--uk--prompt--recruiter_guidelines.parquet
- config_name: Qwen3.5-4B--uk--prompt--second_pass_verification
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--uk--prompt--second_pass_verification.parquet
- config_name: Qwen3.5-4B--uk--prompt--structured_rubric
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--uk--prompt--structured_rubric.parquet
- config_name: Qwen3.5-4B--uk--prompt--zero_shot_cot
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--uk--prompt--zero_shot_cot.parquet
- config_name: Qwen3.5-4B--uk--scrub--lexical
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--uk--scrub--lexical.parquet
- config_name: Qwen3.5-4B--uk--scrub--llm
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--uk--scrub--llm.parquet
- config_name: Qwen3.5-4B--uk--sft--adapter--uk_only
  data_files:
  - split: test
    path: runs/Qwen3.5-4B--uk--sft--adapter--uk_only.parquet
- config_name: Qwen3.5-9B--en--baseline
  default: true
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--en--baseline.parquet
- config_name: Qwen3.5-9B--en--embedding--leace
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--en--embedding--leace.parquet
- config_name: Qwen3.5-9B--en--prompt--counterfactual_invariance
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--en--prompt--counterfactual_invariance.parquet
- config_name: Qwen3.5-9B--en--prompt--fairness_constitution
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--en--prompt--fairness_constitution.parquet
- config_name: Qwen3.5-9B--en--prompt--ignore_personal_info--fullscope
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--en--prompt--ignore_personal_info--fullscope.parquet
- config_name: Qwen3.5-9B--en--prompt--ignore_personal_info
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--en--prompt--ignore_personal_info.parquet
- config_name: Qwen3.5-9B--en--prompt--reasoning
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--en--prompt--reasoning.parquet
- config_name: Qwen3.5-9B--en--prompt--recruiter_guidelines
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--en--prompt--recruiter_guidelines.parquet
- config_name: Qwen3.5-9B--en--prompt--second_pass_verification--fullscope
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--en--prompt--second_pass_verification--fullscope.parquet
- config_name: Qwen3.5-9B--en--prompt--second_pass_verification
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--en--prompt--second_pass_verification.parquet
- config_name: Qwen3.5-9B--en--prompt--structured_rubric--fullscope
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--en--prompt--structured_rubric--fullscope.parquet
- config_name: Qwen3.5-9B--en--prompt--structured_rubric
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--en--prompt--structured_rubric.parquet
- config_name: Qwen3.5-9B--en--prompt--zero_shot_cot
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--en--prompt--zero_shot_cot.parquet
- config_name: Qwen3.5-9B--en--scrub--lexical
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--en--scrub--lexical.parquet
- config_name: Qwen3.5-9B--en--scrub--llm
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--en--scrub--llm.parquet
- config_name: Qwen3.5-9B--en--sft--adapter--en_only
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--en--sft--adapter--en_only.parquet
- config_name: Qwen3.5-9B--en--sft--adapter--en_only_v2_ckpt250
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--en--sft--adapter--en_only_v2_ckpt250.parquet
- config_name: Qwen3.5-9B--uk--baseline
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--uk--baseline.parquet
- config_name: Qwen3.5-9B--uk--embedding--leace
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--uk--embedding--leace.parquet
- config_name: Qwen3.5-9B--uk--prompt--counterfactual_invariance
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--uk--prompt--counterfactual_invariance.parquet
- config_name: Qwen3.5-9B--uk--prompt--fairness_constitution
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--uk--prompt--fairness_constitution.parquet
- config_name: Qwen3.5-9B--uk--prompt--ignore_personal_info
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--uk--prompt--ignore_personal_info.parquet
- config_name: Qwen3.5-9B--uk--prompt--reasoning
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--uk--prompt--reasoning.parquet
- config_name: Qwen3.5-9B--uk--prompt--recruiter_guidelines
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--uk--prompt--recruiter_guidelines.parquet
- config_name: Qwen3.5-9B--uk--prompt--second_pass_verification
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--uk--prompt--second_pass_verification.parquet
- config_name: Qwen3.5-9B--uk--prompt--structured_rubric
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--uk--prompt--structured_rubric.parquet
- config_name: Qwen3.5-9B--uk--prompt--zero_shot_cot
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--uk--prompt--zero_shot_cot.parquet
- config_name: Qwen3.5-9B--uk--scrub--lexical
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--uk--scrub--lexical.parquet
- config_name: Qwen3.5-9B--uk--scrub--llm
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--uk--scrub--llm.parquet
- config_name: Qwen3.5-9B--uk--sft--adapter--uk_only
  data_files:
  - split: test
    path: runs/Qwen3.5-9B--uk--sft--adapter--uk_only.parquet
- config_name: gemma-4-12B-it--en--baseline
  data_files:
  - split: test
    path: runs/gemma-4-12B-it--en--baseline.parquet
- config_name: gemma-4-12B-it--uk--baseline
  data_files:
  - split: test
    path: runs/gemma-4-12B-it--uk--baseline.parquet
- config_name: gemma-4-E4B-it--en--baseline
  data_files:
  - split: test
    path: runs/gemma-4-E4B-it--en--baseline.parquet
- config_name: gemma-4-E4B-it--uk--baseline
  data_files:
  - split: test
    path: runs/gemma-4-E4B-it--uk--baseline.parquet
- config_name: lapa-v0.1.2-instruct--en--baseline
  data_files:
  - split: test
    path: runs/lapa-v0.1.2-instruct--en--baseline.parquet
- config_name: lapa-v0.1.2-instruct--en--sft--adapter--en_only
  data_files:
  - split: test
    path: runs/lapa-v0.1.2-instruct--en--sft--adapter--en_only.parquet
- config_name: lapa-v0.1.2-instruct--uk--baseline
  data_files:
  - split: test
    path: runs/lapa-v0.1.2-instruct--uk--baseline.parquet
- config_name: lapa-v0.1.2-instruct--uk--sft--adapter--uk_only
  data_files:
  - split: test
    path: runs/lapa-v0.1.2-instruct--uk--sft--adapter--uk_only.parquet
---

# Hiring-bias mitigation — model responses

Every response produced in the mitigation study of LLM hiring decisions: **64 runs,
2,782,350 responses**, from 5 open-weight models in English and Ukrainian, at
baseline and under each mitigation family (baseline, embedding, prompt, scrub, sft). Each run is one subset.

- **All released artifacts:** the [Hiring Bias Mitigation](https://huggingface.co/collections/Stereotypes-in-LLMs/hiring-bias-mitigation-6aae99f67667367c7149cedd) collection.
- **Training data of the fine-tuned runs:** [hiring-bias-mitigation-synthetic-data](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-synthetic-data).
- **Code, configs, full results and findings:** [Stereotypes-in-LLMs/hiring_bias_mitigation](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation)
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
ds = load_dataset("Stereotypes-in-LLMs/hiring-bias-mitigation-responses", "Qwen3.5-9B--en--prompt--structured_rubric", split="test")
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
| `Qwen3.5-4B--en--baseline` | Qwen3.5-4B | en | baseline | -- | military_status, gender, religion, military_status_x_gender, military_status_x_religion | explicit, implicit, attr_free | 161,550 | 0.0 | 80.6 | yes |
| `Qwen3.5-4B--en--prompt--counterfactual_invariance` | Qwen3.5-4B | en | prompt | counterfactual_invariance | military_status, religion | explicit, implicit, attr_free | 13,050 | 0.0 | 75.3 | yes |
| `Qwen3.5-4B--en--prompt--fairness_constitution` | Qwen3.5-4B | en | prompt | fairness_constitution | military_status, religion | explicit, implicit, attr_free | 13,050 | 0.0 | 79.9 | yes |
| `Qwen3.5-4B--en--prompt--ignore_personal_info` | Qwen3.5-4B | en | prompt | ignore_personal_info | military_status, religion | explicit, implicit, attr_free | 13,050 | 0.0 | 80.9 | yes |
| `Qwen3.5-4B--en--prompt--reasoning` | Qwen3.5-4B | en | prompt | reasoning | military_status, religion | explicit, implicit, attr_free | 13,050 | 0.0 | 79.3 | yes |
| `Qwen3.5-4B--en--prompt--recruiter_guidelines` | Qwen3.5-4B | en | prompt | recruiter_guidelines | military_status, religion | explicit, implicit, attr_free | 13,050 | 0.0 | 78.1 | yes |
| `Qwen3.5-4B--en--prompt--second_pass_verification` | Qwen3.5-4B | en | prompt | second_pass_verification | military_status, religion | explicit, implicit, attr_free | 13,050 | 0.0 | 68.4 | yes |
| `Qwen3.5-4B--en--prompt--structured_rubric` | Qwen3.5-4B | en | prompt | structured_rubric | military_status, religion | explicit, implicit, attr_free | 13,050 | 0.1 | 74.2 | yes |
| `Qwen3.5-4B--en--prompt--zero_shot_cot` | Qwen3.5-4B | en | prompt | zero_shot_cot | military_status, religion | explicit, implicit, attr_free | 13,050 | 0.0 | 80.0 | yes |
| `Qwen3.5-4B--en--scrub--lexical` | Qwen3.5-4B | en | scrub | lexical | military_status, religion | explicit, implicit, attr_free | 13,050 | 0.0 | 83.4 | yes |
| `Qwen3.5-4B--en--scrub--llm` | Qwen3.5-4B | en | scrub | llm | military_status, religion | explicit, implicit, attr_free | 13,050 | 0.0 | 81.5 | yes |
| `Qwen3.5-4B--en--embedding--leace` | Qwen3.5-4B | en | embedding | leace (layer 16) | military_status, religion | explicit, implicit, attr_free | 13,050 | 19.0 | 80.9 | yes |
| `Qwen3.5-4B--en--sft--adapter--en_only` | Qwen3.5-4B | en | sft | qwen3.5-4b_en_only | military_status, gender, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 82.0 | yes |
| `Qwen3.5-4B--uk--baseline` | Qwen3.5-4B | uk | baseline | -- | military_status, gender, religion, military_status_x_gender, military_status_x_religion | explicit, implicit, attr_free | 161,550 | 0.0 | 74.8 | yes |
| `Qwen3.5-4B--uk--prompt--counterfactual_invariance` | Qwen3.5-4B | uk | prompt | counterfactual_invariance | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 73.6 | yes |
| `Qwen3.5-4B--uk--prompt--fairness_constitution` | Qwen3.5-4B | uk | prompt | fairness_constitution | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 74.7 | yes |
| `Qwen3.5-4B--uk--prompt--ignore_personal_info` | Qwen3.5-4B | uk | prompt | ignore_personal_info | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 75.2 | yes |
| `Qwen3.5-4B--uk--prompt--reasoning` | Qwen3.5-4B | uk | prompt | reasoning | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 74.0 | yes |
| `Qwen3.5-4B--uk--prompt--recruiter_guidelines` | Qwen3.5-4B | uk | prompt | recruiter_guidelines | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 74.1 | yes |
| `Qwen3.5-4B--uk--prompt--second_pass_verification` | Qwen3.5-4B | uk | prompt | second_pass_verification | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 74.0 | yes |
| `Qwen3.5-4B--uk--prompt--structured_rubric` | Qwen3.5-4B | uk | prompt | structured_rubric | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.3 | 69.6 | yes |
| `Qwen3.5-4B--uk--prompt--zero_shot_cot` | Qwen3.5-4B | uk | prompt | zero_shot_cot | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 73.0 | yes |
| `Qwen3.5-4B--uk--scrub--lexical` | Qwen3.5-4B | uk | scrub | lexical | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 75.4 | yes |
| `Qwen3.5-4B--uk--scrub--llm` | Qwen3.5-4B | uk | scrub | llm | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 74.5 | yes |
| `Qwen3.5-4B--uk--embedding--leace` | Qwen3.5-4B | uk | embedding | leace (layer 16) | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 100.0 | -- | **no** — 100.0% of responses could not be parsed (only 4 of 31,050 decided) |
| `Qwen3.5-4B--uk--sft--adapter--uk_only` | Qwen3.5-4B | uk | sft | qwen3.5-4b_uk_only | military_status, gender, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 77.8 | yes |
| `Qwen3.5-9B--en--baseline` | Qwen3.5-9B | en | baseline | -- | military_status, gender, religion, military_status_x_gender, military_status_x_religion | explicit, implicit, attr_free | 161,550 | 0.0 | 78.0 | yes |
| `Qwen3.5-9B--en--prompt--counterfactual_invariance` | Qwen3.5-9B | en | prompt | counterfactual_invariance | military_status | implicit, attr_free | 2,700 | 0.0 | 74.9 | yes |
| `Qwen3.5-9B--en--prompt--fairness_constitution` | Qwen3.5-9B | en | prompt | fairness_constitution | military_status | implicit, attr_free | 2,700 | 0.0 | 83.9 | yes |
| `Qwen3.5-9B--en--prompt--ignore_personal_info` | Qwen3.5-9B | en | prompt | ignore_personal_info | military_status | implicit, attr_free | 2,700 | 0.0 | 78.1 | yes |
| `Qwen3.5-9B--en--prompt--ignore_personal_info--fullscope` | Qwen3.5-9B | en | prompt | ignore_personal_info | military_status, gender, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 80.0 | yes |
| `Qwen3.5-9B--en--prompt--reasoning` | Qwen3.5-9B | en | prompt | reasoning | military_status | implicit, attr_free | 2,700 | 0.0 | 74.4 | yes |
| `Qwen3.5-9B--en--prompt--recruiter_guidelines` | Qwen3.5-9B | en | prompt | recruiter_guidelines | military_status | implicit, attr_free | 2,700 | 0.0 | 74.5 | yes |
| `Qwen3.5-9B--en--prompt--second_pass_verification` | Qwen3.5-9B | en | prompt | second_pass_verification | military_status | implicit, attr_free | 2,700 | 0.0 | 77.5 | yes |
| `Qwen3.5-9B--en--prompt--second_pass_verification--fullscope` | Qwen3.5-9B | en | prompt | second_pass_verification | military_status, gender, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 79.8 | yes |
| `Qwen3.5-9B--en--prompt--structured_rubric` | Qwen3.5-9B | en | prompt | structured_rubric | military_status | implicit, attr_free | 2,700 | 0.0 | 70.0 | yes |
| `Qwen3.5-9B--en--prompt--structured_rubric--fullscope` | Qwen3.5-9B | en | prompt | structured_rubric | military_status, gender, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 72.6 | yes |
| `Qwen3.5-9B--en--prompt--zero_shot_cot` | Qwen3.5-9B | en | prompt | zero_shot_cot | military_status | implicit, attr_free | 2,700 | 0.0 | 76.4 | yes |
| `Qwen3.5-9B--en--scrub--lexical` | Qwen3.5-9B | en | scrub | lexical | military_status | implicit, attr_free | 2,700 | 0.0 | 80.3 | yes |
| `Qwen3.5-9B--en--scrub--llm` | Qwen3.5-9B | en | scrub | llm | military_status | implicit, attr_free | 2,700 | 0.0 | 80.8 | yes |
| `Qwen3.5-9B--en--embedding--leace` | Qwen3.5-9B | en | embedding | leace (layer 16) | military_status | implicit, attr_free | 2,700 | 0.0 | 73.1 | yes |
| `Qwen3.5-9B--en--sft--adapter--en_only` | Qwen3.5-9B | en | sft | qwen3.5-9b_en_only | military_status, gender, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 82.5 | yes |
| `Qwen3.5-9B--en--sft--adapter--en_only_v2_ckpt250` | Qwen3.5-9B | en | sft | checkpoint-250 | military_status, gender, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 81.4 | yes |
| `Qwen3.5-9B--uk--baseline` | Qwen3.5-9B | uk | baseline | -- | military_status, gender, religion, military_status_x_gender, military_status_x_religion | explicit, implicit, attr_free | 161,550 | 0.0 | 79.5 | yes |
| `Qwen3.5-9B--uk--prompt--counterfactual_invariance` | Qwen3.5-9B | uk | prompt | counterfactual_invariance | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 79.9 | yes |
| `Qwen3.5-9B--uk--prompt--fairness_constitution` | Qwen3.5-9B | uk | prompt | fairness_constitution | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 79.9 | yes |
| `Qwen3.5-9B--uk--prompt--ignore_personal_info` | Qwen3.5-9B | uk | prompt | ignore_personal_info | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 80.9 | yes |
| `Qwen3.5-9B--uk--prompt--reasoning` | Qwen3.5-9B | uk | prompt | reasoning | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 80.2 | yes |
| `Qwen3.5-9B--uk--prompt--recruiter_guidelines` | Qwen3.5-9B | uk | prompt | recruiter_guidelines | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 80.2 | yes |
| `Qwen3.5-9B--uk--prompt--second_pass_verification` | Qwen3.5-9B | uk | prompt | second_pass_verification | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 76.7 | yes |
| `Qwen3.5-9B--uk--prompt--structured_rubric` | Qwen3.5-9B | uk | prompt | structured_rubric | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 77.1 | yes |
| `Qwen3.5-9B--uk--prompt--zero_shot_cot` | Qwen3.5-9B | uk | prompt | zero_shot_cot | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 81.0 | yes |
| `Qwen3.5-9B--uk--scrub--lexical` | Qwen3.5-9B | uk | scrub | lexical | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 81.7 | yes |
| `Qwen3.5-9B--uk--scrub--llm` | Qwen3.5-9B | uk | scrub | llm | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 80.6 | yes |
| `Qwen3.5-9B--uk--embedding--leace` | Qwen3.5-9B | uk | embedding | leace (layer 16) | gender, military_status, religion | explicit, implicit, attr_free | 31,050 | 29.4 | 74.3 | yes |
| `Qwen3.5-9B--uk--sft--adapter--uk_only` | Qwen3.5-9B | uk | sft | qwen3.5-9b_uk_only | military_status, gender, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 75.2 | yes |
| `gemma-4-12B-it--en--baseline` | gemma-4-12B-it | en | baseline | -- | military_status, gender, religion, military_status_x_gender, military_status_x_religion | explicit, implicit, attr_free | 161,550 | 0.0 | 84.4 | yes |
| `gemma-4-12B-it--uk--baseline` | gemma-4-12B-it | uk | baseline | -- | military_status, gender, religion, military_status_x_gender, military_status_x_religion | explicit, implicit, attr_free | 161,550 | 0.0 | 84.4 | yes |
| `gemma-4-E4B-it--en--baseline` | gemma-4-E4B-it | en | baseline | -- | military_status, gender, religion, military_status_x_gender, military_status_x_religion | explicit, implicit, attr_free | 161,550 | 0.0 | 75.8 | yes |
| `gemma-4-E4B-it--uk--baseline` | gemma-4-E4B-it | uk | baseline | -- | military_status, gender, religion, military_status_x_gender, military_status_x_religion | explicit, implicit, attr_free | 161,550 | 0.1 | 81.1 | yes |
| `lapa-v0.1.2-instruct--en--baseline` | lapa-v0.1.2-instruct | en | baseline | -- | military_status, gender, religion, military_status_x_gender, military_status_x_religion | explicit, implicit, attr_free | 161,550 | 0.0 | 49.5 | yes |
| `lapa-v0.1.2-instruct--en--sft--adapter--en_only` | lapa-v0.1.2-instruct | en | sft | lapa-12b_en_only | military_status, gender, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 80.6 | yes |
| `lapa-v0.1.2-instruct--uk--baseline` | lapa-v0.1.2-instruct | uk | baseline | -- | military_status, gender, religion, military_status_x_gender, military_status_x_religion | explicit, implicit, attr_free | 161,550 | 0.2 | 52.7 | yes |
| `lapa-v0.1.2-instruct--uk--sft--adapter--uk_only` | lapa-v0.1.2-instruct | uk | sft | lapa-12b_uk_only | military_status, gender, religion | explicit, implicit, attr_free | 31,050 | 0.0 | 77.2 | yes |

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
@inproceedings{drushchak-romanyshyn-2024-introducing,
  title     = {Introducing the Djinni Recruitment Dataset: A Corpus of Anonymized CVs and Job Postings},
  author    = {Drushchak, Nazarii and Romanyshyn, Mariana},
  booktitle = {Proceedings of the Third Ukrainian Natural Language Processing Workshop (UNLP) @ LREC-COLING 2024},
  year      = {2024}
}
```
