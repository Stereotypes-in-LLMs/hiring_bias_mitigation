---
license: gemma
base_model: lapa-llm/lapa-v0.1.2-instruct
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

# lapa-12b-hiring-debias-sft-en

LoRA adapter that makes **lapa-llm/lapa-v0.1.2-instruct** decide hiring cases without letting a protected
attribute (military status, gender, religion) change the verdict. Trained on counterfactually
invariant targets; audited on a held-out counterfactual benchmark of 31,050 decisions.

- **Study, code and full results:** [Stereotypes-in-LLMs/hiring_bias_mitigation](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation)
- **Training data:** [hiring-bias-mitigation-synthetic-data](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-synthetic-data)
- **This adapter's audited responses:** [`lapa-v0.1.2-instruct--en--sft--adapter--en_only`](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-responses) (every decision, reproducible without a GPU)
- **Everything together:** the [Hiring Bias Mitigation](https://huggingface.co/collections/Stereotypes-in-LLMs/hiring-bias-mitigation-6aae99f67667367c7149cedd) collection

## What it changes

A **counterfactual set** is one candidate–job pair evaluated with every attribute variant; it is
*unstable* when the decision is not the same across them — the attribute alone tipped it. Paired
with the same sets at baseline, on matched variants, 899 sets:

| | Base model | This adapter |
|---|---:|---:|
| Unstable sets | 37.8% | **4.6%** |
| Change | | **-33.3 pp** [-39.1, -27.7] |
| Sets fixed : broken | | 330 : 31 |
| Utility (agreement with the attribute-free reference) | 50.9% | 80.6% |
| Rationales naming the attribute | 1.1% | 0.2% |

Per protected group, all significant after Benjamini–Hochberg correction:

| Group | Base unstable % | Adapter unstable % | Δ pp | p (FDR) |
|---|---:|---:|---:|---:|
| gender | 15.7 | 1.9 | **-13.8** | 4e-25 |
| military status | 27.7 | 2.8 | **-24.9** | 7e-49 |
| religion | 21.3 | 1.0 | **-20.3** | 1e-44 |

### Is it invariance, or just a stricter model?

Fine-tuning moves the hire rate (79.7% → 17.7%), and instability depends on where the
decision threshold sits. Sweeping that threshold on each model's own hire-vs-reject margin over
all 31,050 audited prompts:

| At a hire rate of 17.7% | Unstable sets % |
|---|---:|
| base model, moved to that rate | 20.2 |
| **this adapter** | **1.8** |

At equal hire rate the adapter is **11.4× more consistent**; 9% of the raw gain is attributable to
the shift. The adapter's curve lies below the base model's at every operating point.

## Serving it — merge first

**Do not serve this adapter through vLLM's LoRA path.** For these architectures that path does
not reproduce the trained model: on the same prompts, HuggingFace + PEFT and vLLM agree on
95–99% of the *base* model's decisions but only 38–83% of the adapter's. Fold the adapter into
the weights and serve the result as an ordinary checkpoint:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("lapa-llm/lapa-v0.1.2-instruct", dtype="bfloat16")
model = PeftModel.from_pretrained(model, "Stereotypes-in-LLMs/lapa-12b-hiring-debias-sft-en").merge_and_unload()
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
