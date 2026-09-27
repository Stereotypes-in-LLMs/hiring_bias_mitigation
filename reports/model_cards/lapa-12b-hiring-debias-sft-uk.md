---
license: gemma
base_model: lapa-llm/lapa-v0.1.2-instruct
library_name: peft
language:
- uk
tags:
- fairness
- bias-mitigation
- hiring
- recruitment
- lora
- counterfactual
---

# lapa-12b-hiring-debias-sft-uk

LoRA adapter that makes **lapa-llm/lapa-v0.1.2-instruct** decide hiring cases without letting a protected
attribute (military status, gender, religion) change the verdict. Trained on counterfactually
invariant targets; audited on a held-out counterfactual benchmark of 31,050 decisions.

- **Study, code and full results:** [Stereotypes-in-LLMs/hiring_bias_mitigation](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation)
- **Training data:** [hiring-bias-mitigation-synthetic-data](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-synthetic-data)
- **This adapter's audited responses:** [`lapa-v0.1.2-instruct--uk--sft--adapter--uk_only`](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-responses) (every decision, reproducible without a GPU)
- **Everything together:** the [Hiring Bias Mitigation](https://huggingface.co/collections/Stereotypes-in-LLMs/hiring-bias-mitigation-6aae99f67667367c7149cedd) collection

## What it changes

A **counterfactual set** is one candidate–job pair evaluated with every attribute variant; it is
*unstable* when the decision is not the same across them — the attribute alone tipped it. Paired
with the same sets at baseline, on matched variants, 891 sets:

| | Base model | This adapter |
|---|---:|---:|
| Unstable sets | 37.7% | **4.2%** |
| Change | | **-33.6 pp** [-39.8, -27.3] |
| Sets fixed : broken | | 334 : 35 |
| Utility (agreement with the attribute-free reference) | 55.1% | 77.3% |
| Rationales naming the attribute | 0.7% | 0.1% |

Per protected group, all significant after Benjamini–Hochberg correction:

| Group | Base unstable % | Adapter unstable % | Δ pp | p (FDR) |
|---|---:|---:|---:|---:|
| gender | 21.8 | 3.0 | **-18.8** | 1e-31 |
| military status | 21.5 | 2.1 | **-19.4** | 6e-37 |
| religion | 22.4 | 2.2 | **-20.1** | 2e-37 |

### Is it invariance, or just a stricter model?

Fine-tuning moves the hire rate (75.5% → 16.4%), and instability depends on where the
decision threshold sits. Sweeping that threshold on each model's own hire-vs-reject margin over
all 31,050 audited prompts:

| At a hire rate of 16.4% | Unstable sets % |
|---|---:|
| base model, moved to that rate | 16.3 |
| **this adapter** | **2.5** |

At equal hire rate the adapter is **6.5× more consistent**; 29% of the raw gain is attributable to
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
model = PeftModel.from_pretrained(model, "Stereotypes-in-LLMs/lapa-12b-hiring-debias-sft-uk").merge_and_unload()
model.save_pretrained("merged")          # then: vllm serve merged
```

The study's audits all ran on merged weights (`scripts/merge_adapter.py`).

## Intended use and limits

- **Research on bias mitigation in LLM-assisted hiring.** Not a hiring system, and not validated
  for deployment. The reference decision it was trained toward is GPT-4o's, which is
  attribute-free by construction but **not unbiased**.
- Instability falls to a few percent of sets, **not to zero**: a residual dependence on the
  attribute remains, and ~0.5% of sets flip from decoding nondeterminism alone.
- Trained and measured on one corpus (anonymised Djinni CVs and postings) in Ukrainian, on three
  protected attributes, with one LoRA configuration and one seed. Intersections were evaluated
  but never trained.
- Fine-tuning also shifts the model's overall hire rate; check that shift against your own
  operating point before using it.

## Citation

The paper is in preparation; until then cite the repository above and the Djinni Recruitment
Dataset (Drushchak & Romanyshyn, 2024).
