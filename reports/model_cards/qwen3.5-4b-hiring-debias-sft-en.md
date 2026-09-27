---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: peft
language:
- en
tags:
- fairness
- bias-mitigation
- hiring
- recruitment
- lora
- counterfactual
---

# qwen3.5-4b-hiring-debias-sft-en

LoRA adapter that makes **Qwen/Qwen3.5-4B** decide hiring cases without letting a protected
attribute (military status, gender, religion) change the verdict. Trained on counterfactually
invariant targets; audited on a held-out counterfactual benchmark of 31,050 decisions.

- **Study, code and full results:** [Stereotypes-in-LLMs/hiring_bias_mitigation](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation)
- **Training data:** [hiring-bias-mitigation-synthetic-data](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-synthetic-data)
- **This adapter's audited responses:** [`Qwen3.5-4B--en--sft--adapter--en_only`](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-responses) (every decision, reproducible without a GPU)
- **Everything together:** the [Hiring Bias Mitigation](https://huggingface.co/collections/Stereotypes-in-LLMs/hiring-bias-mitigation-6aae99f67667367c7149cedd) collection

## What it changes

A **counterfactual set** is one candidate–job pair evaluated with every attribute variant; it is
*unstable* when the decision is not the same across them — the attribute alone tipped it. Paired
with the same sets at baseline, on matched variants, 900 sets:

| | Base model | This adapter |
|---|---:|---:|
| Unstable sets | 30.1% | **10.7%** |
| Change | | **-19.4 pp** [-23.0, -15.9] |
| Sets fixed : broken | | 205 : 30 |
| Utility (agreement with the attribute-free reference) | 80.8% | 81.9% |
| Rationales naming the attribute | 3.1% | 0.2% |

Per protected group, all significant after Benjamini–Hochberg correction:

| Group | Base unstable % | Adapter unstable % | Δ pp | p (FDR) |
|---|---:|---:|---:|---:|
| gender | 10.8 | 5.7 | **-5.1** | 6e-05 |
| military status | 23.3 | 5.9 | **-17.4** | 1e-31 |
| religion | 15.3 | 5.2 | **-10.1** | 8e-14 |

## Serving it — merge first

**Do not serve this adapter through vLLM's LoRA path.** For these architectures that path does
not reproduce the trained model: on the same prompts, HuggingFace + PEFT and vLLM agree on
95–99% of the *base* model's decisions but only 38–83% of the adapter's. Fold the adapter into
the weights and serve the result as an ordinary checkpoint:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype="bfloat16")
model = PeftModel.from_pretrained(model, "Stereotypes-in-LLMs/qwen3.5-4b-hiring-debias-sft-en").merge_and_unload()
model.save_pretrained("merged")          # then: vllm serve merged
```

The study's audits all ran on merged weights (`scripts/merge_adapter.py`).

## Intended use and limits

- **Research on bias mitigation in LLM-assisted hiring.** Not a hiring system, and not validated
  for deployment. The reference decision it was trained toward is GPT-4o's, which is
  attribute-free by construction but **not unbiased**.
- Instability falls to a few percent of sets, **not to zero**: a residual dependence on the
  attribute remains, and ~0.5% of sets flip from decoding nondeterminism alone.
- Trained and measured on one corpus (anonymised Djinni CVs and postings) in English, on three
  protected attributes, with one LoRA configuration and one seed. Intersections were evaluated
  but never trained.
- Fine-tuning also shifts the model's overall hire rate; check that shift against your own
  operating point before using it.

## Citation

The paper is in preparation; until then cite the repository above and the Djinni Recruitment
Dataset (Drushchak & Romanyshyn, 2024).
