---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
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

# qwen3.5-9b-hiring-debias-sft-uk

LoRA adapter that makes **Qwen/Qwen3.5-9B** decide hiring cases without letting a protected
attribute (military status, gender, religion) change the verdict. Trained on counterfactually
invariant targets; audited on a held-out counterfactual benchmark of 31,050 decisions.

- **Study, code and full results:** [Stereotypes-in-LLMs/hiring_bias_mitigation](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation)
- **Training data:** [hiring-bias-mitigation-synthetic-data](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-synthetic-data)
- **This adapter's audited responses:** [`Qwen3.5-9B--uk--sft--adapter--uk_only`](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-responses) (every decision, reproducible without a GPU)
- **Everything together:** the [Hiring Bias Mitigation](https://huggingface.co/collections/Stereotypes-in-LLMs/hiring-bias-mitigation-6aae99f67667367c7149cedd) collection

> **This adapter trades utility for consistency.** Agreement with the attribute-free reference
> decision falls 5.3 points, and it hires 11.6% of candidates where the
> attribute-free reference hires 35.1%. It is the only cell in the study that pays this price. Read the
> utility column before deploying it.

## What it changes

A **counterfactual set** is one candidate–job pair evaluated with every attribute variant; it is
*unstable* when the decision is not the same across them — the attribute alone tipped it. Paired
with the same sets at baseline, on matched variants, 900 sets:

| | Base model | This adapter |
|---|---:|---:|
| Unstable sets | 43.0% | **4.8%** |
| Change | | **-38.2 pp** [-43.7, -32.9] |
| Sets fixed : broken | | 356 : 12 |
| Utility (agreement with the attribute-free reference) | 80.5% | 75.2% |
| Rationales naming the attribute | 2.4% | 0.1% |

Per protected group, all significant after Benjamini–Hochberg correction:

| Group | Base unstable % | Adapter unstable % | Δ pp | p (FDR) |
|---|---:|---:|---:|---:|
| gender | 24.0 | 3.6 | **-20.4** | 5e-40 |
| military status | 21.4 | 1.7 | **-19.8** | 3e-43 |
| religion | 16.1 | 2.4 | **-13.7** | 1e-26 |

### Is it invariance, or just a stricter model?

Fine-tuning moves the hire rate (29.4% → 11.6%), and instability depends on where the
decision threshold sits. Sweeping that threshold on each model's own hire-vs-reject margin over
all 31,050 audited prompts:

| At a hire rate of 11.6% | Unstable sets % |
|---|---:|
| base model, moved to that rate | 10.8 |
| **this adapter** | **2.6** |

At equal hire rate the adapter is **4.2× more consistent**; 54% of the raw gain is attributable to
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
model = PeftModel.from_pretrained(model, "Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-uk").merge_and_unload()
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
