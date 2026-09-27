---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
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

# qwen3.5-9b-hiring-debias-sft-en

LoRA adapter that makes **Qwen/Qwen3.5-9B** decide hiring cases without letting a protected
attribute (military status, gender, religion) change the verdict. Trained on counterfactually
invariant targets; audited on a held-out counterfactual benchmark of 31,050 decisions.

- **Study, code and full results:** [Stereotypes-in-LLMs/hiring_bias_mitigation](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation)
- **Training data:** [hiring-bias-mitigation-synthetic-data](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-synthetic-data)
- **This adapter's audited responses:** [`Qwen3.5-9B--en--sft--adapter--en_only`](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-responses) (every decision, reproducible without a GPU)
- **Everything together:** the [Hiring Bias Mitigation](https://huggingface.co/collections/Stereotypes-in-LLMs/hiring-bias-mitigation-6aae99f67667367c7149cedd) collection

## What it changes

A **counterfactual set** is one candidate–job pair evaluated with every attribute variant; it is
*unstable* when the decision is not the same across them — the attribute alone tipped it. Paired
with the same sets at baseline, on matched variants, 900 sets:

| | Base model | This adapter |
|---|---:|---:|
| Unstable sets | 17.3% | **7.2%** |
| Change | | **-10.1 pp** [-14.4, -5.8] |
| Sets fixed : broken | | 136 : 45 |
| Utility (agreement with the attribute-free reference) | 78.9% | 82.5% |
| Rationales naming the attribute | 1.1% | 0.2% |

Per protected group, all significant after Benjamini–Hochberg correction:

| Group | Base unstable % | Adapter unstable % | Δ pp | p (FDR) |
|---|---:|---:|---:|---:|
| gender | 7.7 | 4.1 | **-3.6** | 1e-03 |
| military status | 10.9 | 3.8 | **-7.1** | 1e-08 |
| religion | 8.1 | 2.8 | **-5.3** | 1e-06 |

### Is it invariance, or just a stricter model?

Fine-tuning moves the hire rate (14.5% → 27.3%), and instability depends on where the
decision threshold sits. Sweeping that threshold on each model's own hire-vs-reject margin over
all 31,050 audited prompts:

| At a hire rate of 27.3% | Unstable sets % |
|---|---:|
| base model, moved to that rate | 13.0 |
| **this adapter** | **3.5** |

At equal hire rate the adapter is **3.7× more consistent**; -106% of the raw gain is attributable to
the shift. The adapter's curve lies below the base model's at every operating point.

## Serving it — merge first

**Do not serve this adapter through vLLM's LoRA path.** For these architectures that path does
not reproduce the trained model: on the same prompts, HuggingFace + PEFT and vLLM agree on
95–99% of the *base* model's decisions but only 38–83% of the adapter's. Fold the adapter into
the weights and serve the result as an ordinary checkpoint:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", dtype="bfloat16")
model = PeftModel.from_pretrained(model, "Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en").merge_and_unload()
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
