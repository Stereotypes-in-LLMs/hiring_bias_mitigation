# Experiment plan — what to mitigate, and how it will be evaluated

*Generated 2026-09-27 10:48 by `scripts/make_experiment_matrix.py` from `reports/analysis.json`. Derived — regenerate rather than editing.*

Evidence and reasoning: [`FINDINGS.md`](FINDINGS.md) · [`ANALYSIS.md`](ANALYSIS.md) · metric definitions: [`docs/METRICS.md`](../docs/METRICS.md)

**50 runs** across 3 models: prompt 32 · scrub 8 · embedding 4 · sft 6. A further **18** are configured but deferred — see section 5.

13 fairness targets · 3 no-harm controls · 2 rescue arms.

Two things are held apart throughout, because conflating them is how a mitigation study reports a result it did not measure:

- **Trained on** — what the intervention was fitted to.
- **Evaluated on** — what is measured afterwards. **Per model and language**, not a fixed grid.

**Evaluation scope: `targeted-no-intersections`.** Each model-language measures exactly the groups **and conditions** in which it showed a confirmed disparity — not the full grid. A target is a group *under a condition*, so a cell found under explicit injection is measured under explicit; adding the other framing would double the cost to answer a question that cell did not raise.

`attr_free` is always present and is not optional: one extra pass over the 450 pairs, and the only thing separating *"the disparity fell"* from *"the model's whole operating point moved"*.

| Model | Lang | Groups evaluated | Conditions | Attributes | Prompts/run |
|---|---|---|---|---:|---:|
| `lapa-12b` | en | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 34 | 31,050 |
| `lapa-12b` | uk | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 34 | 31,050 |
| `qwen3.5-4b` | en | military_status (5), religion (9) | explicit, implicit, attr_free | 14 | 13,050 |
| `qwen3.5-4b` | uk | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 34 | 31,050 |
| `qwen3.5-9b` | en | military_status (5) | implicit, attr_free | 5 | 2,700 |
| `qwen3.5-9b` | uk | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 34 | 31,050 |

**What this scoping costs, stated plainly.** A group left out of a run cannot be checked for damage by that run: a mitigation that fixes military status while breaking religion would not be caught. The attribute-free condition is what still guards against gross damage — it catches a model whose whole operating point moved — but harm confined to an untested group is not detectable here, and the write-up must say so rather than imply a no-harm check that was not run.

### The intersections are deferred, not dropped

The two intersections are 145 of the 179 attributes. Measuring them on every mitigation run costs **~538 GPU-hours**; measuring the single groups costs ~87. Paying for the intersections on arms that turn out not to work is the expensive mistake, so this plan splits in two:

| Phase | Scope | Cost | Answers |
|---|---|---:|---|
| **1** (this document) | single groups only | ~87 h | which mitigation works, and at what cost to utility |
| **2** (after phase 1) | intersections, on the winning arms only | ~34 h per promoted arm | does the mitigation reach the intersectional cells, and does the sub-additivity found in the baseline survive it |

Phase 2 matters and must not be quietly skipped. The baseline found military × gender **sub-additive in both languages** — 52/23 negative in English, 62/11 in Ukrainian — meaning the combination is treated worse than the two effects predict separately. Whether a mitigation flattens that, leaves it, or inverts it is a result in its own right, and it is the direct follow-up to the study's intersection finding.

```bash
# after phase 1 has been scored
python scripts/select_phase2.py --top 1 --dry-run   # see the cost
python scripts/select_phase2.py --top 1             # writes reports/phase2_scope.yaml
python scripts/generate_experiment_configs.py --no-runners \
    --eval-scope reports/phase2_scope.yaml
```

Promotion is by disparity reduction against each run's own baseline, with a utility guard: an arm that cut disparity by damaging the model is never promoted, however large the cut.

## 1. What needs fixing

Every cell the mitigation stage is aimed at, plus the controls that make its results interpretable. `Sig. attrs` counts attributes significant after FDR correction.

**Controls are not filler.** A mitigation that worsens a clean cell is as informative as one that fixes a target, and without them the study cannot distinguish *"the mitigation removed a disparity"* from *"the mitigation flattened everything, including what was already fine"*.

**Rescue arms are a different experiment.** Their primary outcome is utility, not disparity; their MAD column is shown for completeness and is not interpretable until utility crosses the gate.

| Priority | Model | Lang | Protected group | Condition | MAD (pp) | Range (pp) | Utility % | Sig. attrs | Role |
|---:|---|---|---|---|---:|---:|---:|---:|---|
| 1 | Qwen3.5-4B | uk | military_status | explicit | 8.2 | 33.1 | 74.2 | 2 | target |
| 2 | Qwen3.5-9B | uk | military_status | explicit | 7.7 | 18.2 | 77.7 | 4 | target |
| 3 | Qwen3.5-9B | uk | military_status | implicit | 5.4 | 14.9 | 79.3 | 2 | target |
| 4 | Qwen3.5-4B | en | military_status | explicit | 5.3 | 21.3 | 80.0 | 2 | target |
| 5 | Qwen3.5-4B | uk | military_status | implicit | 4.9 | 14.7 | 75.1 | 2 | target |
| 6 | Qwen3.5-4B | en | military_status | implicit | 4.5 | 17.6 | 80.8 | 2 | target |
| 7 | Qwen3.5-9B | uk | religion | implicit | 4.4 | 18.9 | 77.3 | 4 | target |
| 8 | Qwen3.5-4B | uk | gender | explicit | 4.1 | 22.9 | 75.2 | 8 | target |
| 9 | Qwen3.5-9B | en | military_status | implicit | 3.9 | 12.2 | 74.6 | 2 | target |
| 10 | Qwen3.5-9B | uk | gender | explicit | 3.7 | 30.4 | 80.1 | 3 | target |
| 11 | Qwen3.5-4B | en | religion | implicit | 3.7 | 15.1 | 77.4 | 2 | target |
| 12 | Qwen3.5-4B | uk | gender | implicit | 2.6 | 11.1 | 74.8 | 4 | target |
| 13 | Qwen3.5-4B | uk | religion | implicit | 2.4 | 10.4 | 73.4 | 3 | target |
| — | Qwen3.5-4B | en | gender | implicit | 0.8 | 3.8 | 80.7 | 0 | **control** |
| — | Qwen3.5-9B | en | religion | explicit | 0.7 | 2.4 | 80.8 | 0 | **control** |
| — | Qwen3.5-9B | uk | religion | explicit | 1.6 | 6.2 | 81.1 | 0 | **control** |
| — | lapa-v0.1.2-instruct | en | (all) | (all) | 10.5 | 46.7 | 50.9 | — | **rescue** |
| — | lapa-v0.1.2-instruct | uk | (all) | (all) | 7.3 | 31.2 | 54.1 | — | **rescue** |

## 2. Every run, enumerated

One row per config that will execute. Ordered cheapest-family-first, which is also the order they should run: a training arm failing to beat a free prompt edit is a result, but only if the prompt edit was measured first.

### prompt — 32 run(s)

Intervenes at: prompt construction — no weights, no data. Trained on: —.

| # | Model | Lang | Variant | Role | Trained on | **Groups evaluated** | **Conditions** | Prompts | Config |
|---:|---|---|---|---|---|---|---|---:|---|
| 1 | Qwen3.5-4B | en | `counterfactual_invariance` | target | — | military_status (5), religion (9) | explicit, implicit, attr_free | 13,050 | `configs/mitigation/prompt/qwen3.5-4b_en_counterfactual_invariance.yaml` |
| 2 | Qwen3.5-4B | en | `fairness_constitution` | target | — | military_status (5), religion (9) | explicit, implicit, attr_free | 13,050 | `configs/mitigation/prompt/qwen3.5-4b_en_fairness_constitution.yaml` |
| 3 | Qwen3.5-4B | en | `ignore_personal_info` | target | — | military_status (5), religion (9) | explicit, implicit, attr_free | 13,050 | `configs/mitigation/prompt/qwen3.5-4b_en_ignore_personal_info.yaml` |
| 4 | Qwen3.5-4B | en | `reasoning` | target | — | military_status (5), religion (9) | explicit, implicit, attr_free | 13,050 | `configs/mitigation/prompt/qwen3.5-4b_en_reasoning.yaml` |
| 5 | Qwen3.5-4B | en | `recruiter_guidelines` | target | — | military_status (5), religion (9) | explicit, implicit, attr_free | 13,050 | `configs/mitigation/prompt/qwen3.5-4b_en_recruiter_guidelines.yaml` |
| 6 | Qwen3.5-4B | en | `second_pass_verification` | target | — | military_status (5), religion (9) | explicit, implicit, attr_free | 13,050 | `configs/mitigation/prompt/qwen3.5-4b_en_second_pass_verification.yaml` |
| 7 | Qwen3.5-4B | en | `structured_rubric` | target | — | military_status (5), religion (9) | explicit, implicit, attr_free | 13,050 | `configs/mitigation/prompt/qwen3.5-4b_en_structured_rubric.yaml` |
| 8 | Qwen3.5-4B | en | `zero_shot_cot` | target | — | military_status (5), religion (9) | explicit, implicit, attr_free | 13,050 | `configs/mitigation/prompt/qwen3.5-4b_en_zero_shot_cot.yaml` |
| 9 | Qwen3.5-4B | uk | `counterfactual_invariance` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/prompt/qwen3.5-4b_uk_counterfactual_invariance.yaml` |
| 10 | Qwen3.5-4B | uk | `fairness_constitution` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/prompt/qwen3.5-4b_uk_fairness_constitution.yaml` |
| 11 | Qwen3.5-4B | uk | `ignore_personal_info` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/prompt/qwen3.5-4b_uk_ignore_personal_info.yaml` |
| 12 | Qwen3.5-4B | uk | `reasoning` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/prompt/qwen3.5-4b_uk_reasoning.yaml` |
| 13 | Qwen3.5-4B | uk | `recruiter_guidelines` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/prompt/qwen3.5-4b_uk_recruiter_guidelines.yaml` |
| 14 | Qwen3.5-4B | uk | `second_pass_verification` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/prompt/qwen3.5-4b_uk_second_pass_verification.yaml` |
| 15 | Qwen3.5-4B | uk | `structured_rubric` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/prompt/qwen3.5-4b_uk_structured_rubric.yaml` |
| 16 | Qwen3.5-4B | uk | `zero_shot_cot` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/prompt/qwen3.5-4b_uk_zero_shot_cot.yaml` |
| 17 | Qwen3.5-9B | en | `counterfactual_invariance` | target | — | military_status (5) | implicit, attr_free | 2,700 | `configs/mitigation/prompt/qwen3.5-9b_en_counterfactual_invariance.yaml` |
| 18 | Qwen3.5-9B | en | `fairness_constitution` | target | — | military_status (5) | implicit, attr_free | 2,700 | `configs/mitigation/prompt/qwen3.5-9b_en_fairness_constitution.yaml` |
| 19 | Qwen3.5-9B | en | `ignore_personal_info` | target | — | military_status (5) | implicit, attr_free | 2,700 | `configs/mitigation/prompt/qwen3.5-9b_en_ignore_personal_info.yaml` |
| 20 | Qwen3.5-9B | en | `reasoning` | target | — | military_status (5) | implicit, attr_free | 2,700 | `configs/mitigation/prompt/qwen3.5-9b_en_reasoning.yaml` |
| 21 | Qwen3.5-9B | en | `recruiter_guidelines` | target | — | military_status (5) | implicit, attr_free | 2,700 | `configs/mitigation/prompt/qwen3.5-9b_en_recruiter_guidelines.yaml` |
| 22 | Qwen3.5-9B | en | `second_pass_verification` | target | — | military_status (5) | implicit, attr_free | 2,700 | `configs/mitigation/prompt/qwen3.5-9b_en_second_pass_verification.yaml` |
| 23 | Qwen3.5-9B | en | `structured_rubric` | target | — | military_status (5) | implicit, attr_free | 2,700 | `configs/mitigation/prompt/qwen3.5-9b_en_structured_rubric.yaml` |
| 24 | Qwen3.5-9B | en | `zero_shot_cot` | target | — | military_status (5) | implicit, attr_free | 2,700 | `configs/mitigation/prompt/qwen3.5-9b_en_zero_shot_cot.yaml` |
| 25 | Qwen3.5-9B | uk | `counterfactual_invariance` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/prompt/qwen3.5-9b_uk_counterfactual_invariance.yaml` |
| 26 | Qwen3.5-9B | uk | `fairness_constitution` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/prompt/qwen3.5-9b_uk_fairness_constitution.yaml` |
| 27 | Qwen3.5-9B | uk | `ignore_personal_info` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/prompt/qwen3.5-9b_uk_ignore_personal_info.yaml` |
| 28 | Qwen3.5-9B | uk | `reasoning` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/prompt/qwen3.5-9b_uk_reasoning.yaml` |
| 29 | Qwen3.5-9B | uk | `recruiter_guidelines` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/prompt/qwen3.5-9b_uk_recruiter_guidelines.yaml` |
| 30 | Qwen3.5-9B | uk | `second_pass_verification` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/prompt/qwen3.5-9b_uk_second_pass_verification.yaml` |
| 31 | Qwen3.5-9B | uk | `structured_rubric` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/prompt/qwen3.5-9b_uk_structured_rubric.yaml` |
| 32 | Qwen3.5-9B | uk | `zero_shot_cot` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/prompt/qwen3.5-9b_uk_zero_shot_cot.yaml` |

Subtotal: **622,800 prompts**.

### scrub — 8 run(s)

Intervenes at: the CV text, before the prompt is built. Trained on: —.

| # | Model | Lang | Variant | Role | Trained on | **Groups evaluated** | **Conditions** | Prompts | Config |
|---:|---|---|---|---|---|---|---|---:|---|
| 1 | Qwen3.5-4B | en | `lexical` | target | — | military_status (5), religion (9) | explicit, implicit, attr_free | 13,050 | `configs/mitigation/scrub/qwen3.5-4b_en_lexical.yaml` |
| 2 | Qwen3.5-4B | en | `llm` | target | — | military_status (5), religion (9) | explicit, implicit, attr_free | 13,050 | `configs/mitigation/scrub/qwen3.5-4b_en_llm.yaml` |
| 3 | Qwen3.5-4B | uk | `lexical` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/scrub/qwen3.5-4b_uk_lexical.yaml` |
| 4 | Qwen3.5-4B | uk | `llm` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/scrub/qwen3.5-4b_uk_llm.yaml` |
| 5 | Qwen3.5-9B | en | `lexical` | target | — | military_status (5) | implicit, attr_free | 2,700 | `configs/mitigation/scrub/qwen3.5-9b_en_lexical.yaml` |
| 6 | Qwen3.5-9B | en | `llm` | target | — | military_status (5) | implicit, attr_free | 2,700 | `configs/mitigation/scrub/qwen3.5-9b_en_llm.yaml` |
| 7 | Qwen3.5-9B | uk | `lexical` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/scrub/qwen3.5-9b_uk_lexical.yaml` |
| 8 | Qwen3.5-9B | uk | `llm` | target | — | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/scrub/qwen3.5-9b_uk_llm.yaml` |

Subtotal: **155,700 prompts**.

### embedding — 4 run(s), 8 deferred

Intervenes at: the residual stream, via forward hooks at inference. Trained on: eraser fitted on the **military-status** slice of the generated pool, per language.

| # | Model | Lang | Variant | Role | Trained on | **Groups evaluated** | **Conditions** | Prompts | Config |
|---:|---|---|---|---|---|---|---|---:|---|
| 1 | Qwen3.5-4B | en | `inlp` | **future work** | military_status, one language | military_status (5), religion (9) | explicit, implicit, attr_free | 13,050 | `configs/mitigation/embedding/qwen3.5-4b_en_inlp.yaml` |
| 2 | Qwen3.5-4B | en | `leace` | target | military_status, one language | military_status (5), religion (9) | explicit, implicit, attr_free | 13,050 | `configs/mitigation/embedding/qwen3.5-4b_en_leace.yaml` |
| 3 | Qwen3.5-4B | en | `mean_diff` | **future work** | military_status, one language | military_status (5), religion (9) | explicit, implicit, attr_free | 13,050 | `configs/mitigation/embedding/qwen3.5-4b_en_mean_diff.yaml` |
| 4 | Qwen3.5-4B | uk | `inlp` | **future work** | military_status, one language | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/embedding/qwen3.5-4b_uk_inlp.yaml` |
| 5 | Qwen3.5-4B | uk | `leace` | target | military_status, one language | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/embedding/qwen3.5-4b_uk_leace.yaml` |
| 6 | Qwen3.5-4B | uk | `mean_diff` | **future work** | military_status, one language | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/embedding/qwen3.5-4b_uk_mean_diff.yaml` |
| 7 | Qwen3.5-9B | en | `inlp` | **future work** | military_status, one language | military_status (5) | implicit, attr_free | 2,700 | `configs/mitigation/embedding/qwen3.5-9b_en_inlp.yaml` |
| 8 | Qwen3.5-9B | en | `leace` | target | military_status, one language | military_status (5) | implicit, attr_free | 2,700 | `configs/mitigation/embedding/qwen3.5-9b_en_leace.yaml` |
| 9 | Qwen3.5-9B | en | `mean_diff` | **future work** | military_status, one language | military_status (5) | implicit, attr_free | 2,700 | `configs/mitigation/embedding/qwen3.5-9b_en_mean_diff.yaml` |
| 10 | Qwen3.5-9B | uk | `inlp` | **future work** | military_status, one language | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/embedding/qwen3.5-9b_uk_inlp.yaml` |
| 11 | Qwen3.5-9B | uk | `leace` | target | military_status, one language | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/embedding/qwen3.5-9b_uk_leace.yaml` |
| 12 | Qwen3.5-9B | uk | `mean_diff` | **future work** | military_status, one language | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/embedding/qwen3.5-9b_uk_mean_diff.yaml` |

8 row(s) marked **future work** are configured but not enabled; they are excluded from the subtotal. See section 5.


Subtotal: **77,850 prompts**.

### sft — 6 run(s)

Intervenes at: model weights (LoRA or full FT). Trained on: generated SFT split; one run per language (see the `Trained on` column).

| # | Model | Lang | Variant | Role | Trained on | **Groups evaluated** | **Conditions** | Prompts | Config |
|---:|---|---|---|---|---|---|---|---:|---|
| 1 | lapa-v0.1.2-instruct | en | `lapa-12b_en_only` | rescue | all 3 groups, **English only** | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/sft/lapa-12b_en_only.yaml` |
| 2 | lapa-v0.1.2-instruct | uk | `lapa-12b_uk_only` | rescue | all 3 groups, **Ukrainian only** | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/sft/lapa-12b_uk_only.yaml` |
| 3 | Qwen3.5-4B | en | `qwen3.5-4b_en_only` | target | all 3 groups, **English only** | military_status (5), religion (9) | explicit, implicit, attr_free | 13,050 | `configs/mitigation/sft/qwen3.5-4b_en_only.yaml` |
| 4 | Qwen3.5-4B | uk | `qwen3.5-4b_uk_only` | target | all 3 groups, **Ukrainian only** | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/sft/qwen3.5-4b_uk_only.yaml` |
| 5 | Qwen3.5-9B | en | `qwen3.5-9b_en_only` | target | all 3 groups, **English only** | military_status (5) | implicit, attr_free | 2,700 | `configs/mitigation/sft/qwen3.5-9b_en_only.yaml` |
| 6 | Qwen3.5-9B | uk | `qwen3.5-9b_uk_only` | target | all 3 groups, **Ukrainian only** | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/sft/qwen3.5-9b_uk_only.yaml` |

Subtotal: **139,950 prompts** — evaluated after training, one audit per row.

### dpo — 0 run(s), 10 deferred

Intervenes at: model weights, continuing from the SFT checkpoint. Trained on: generated preference split; continues from the matching per-language SFT run.

| # | Model | Lang | Variant | Role | Trained on | **Groups evaluated** | **Conditions** | Prompts | Config |
|---:|---|---|---|---|---|---|---|---:|---|
| 1 | lapa-v0.1.2-instruct | en | `lapa-12b_en_only_dpo` | **future work** | all 3 groups, **English only** | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/dpo/lapa-12b_en_only_dpo.yaml` |
| 2 | lapa-v0.1.2-instruct | en | `lapa-12b_en_only_kto` | **future work** | all 3 groups, **English only** | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/dpo/lapa-12b_en_only_kto.yaml` |
| 3 | lapa-v0.1.2-instruct | uk | `lapa-12b_uk_only_dpo` | **future work** | all 3 groups, **Ukrainian only** | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/dpo/lapa-12b_uk_only_dpo.yaml` |
| 4 | lapa-v0.1.2-instruct | uk | `lapa-12b_uk_only_kto` | **future work** | all 3 groups, **Ukrainian only** | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/dpo/lapa-12b_uk_only_kto.yaml` |
| 5 | Qwen3.5-4B | en | `qwen3.5-4b_en_only_dpo` | **future work** | all 3 groups, **English only** | military_status (5), religion (9) | explicit, implicit, attr_free | 13,050 | `configs/mitigation/dpo/qwen3.5-4b_en_only_dpo.yaml` |
| 6 | Qwen3.5-4B | en | `qwen3.5-4b_en_only_kto` | **future work** | all 3 groups, **English only** | military_status (5), religion (9) | explicit, implicit, attr_free | 13,050 | `configs/mitigation/dpo/qwen3.5-4b_en_only_kto.yaml` |
| 7 | Qwen3.5-4B | uk | `qwen3.5-4b_uk_only_dpo` | **future work** | all 3 groups, **Ukrainian only** | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/dpo/qwen3.5-4b_uk_only_dpo.yaml` |
| 8 | Qwen3.5-4B | uk | `qwen3.5-4b_uk_only_kto` | **future work** | all 3 groups, **Ukrainian only** | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/dpo/qwen3.5-4b_uk_only_kto.yaml` |
| 9 | Qwen3.5-9B | uk | `qwen3.5-9b_uk_only_dpo` | **future work** | all 3 groups, **Ukrainian only** | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/dpo/qwen3.5-9b_uk_only_dpo.yaml` |
| 10 | Qwen3.5-9B | uk | `qwen3.5-9b_uk_only_kto` | **future work** | all 3 groups, **Ukrainian only** | gender (20), military_status (5), religion (9) | explicit, implicit, attr_free | 31,050 | `configs/mitigation/dpo/qwen3.5-9b_uk_only_kto.yaml` |

10 row(s) marked **future work** are configured but not enabled; they are excluded from the subtotal. See section 5.


Subtotal: **0 prompts** — evaluated after training, one audit per row.

## 3. How each run is evaluated

Identical protocol for every run above — same 450 job–CV pairs per language, same attributes, same greedy decoding, same parser. The only thing that differs between a baseline and a mitigated run is the mitigation block, which is what licenses the comparison.

### Measured per run

| Axis | What is reported |
|---|---|
| Protected groups | per model and language — see the scope table at the top |
| Conditions | explicit (labelled field), implicit (first-person biography), attribute-free (control) |
| Fairness | acceptance-rate MAD and range, inconsistency rate, feedback similarity, per attribute and aggregated |
| Utility | agreement with the attribute-free reference decision, overall and split by what the reference decided |
| Output handling | refusal rate, parse-failure rate, attribute-mention rate |
| Inference | permutation test (unpaired, for comparability) **and** paired sign-flip test, FDR-corrected across every test in the run |

### Compared against

Each mitigated run is read against **its own baseline** — same model, same language, same benchmark. The report puts the baseline value in brackets beside every mitigated value, so no cross-model comparison is needed to read a row.

### What counts as success

| Track | Primary | Secondary | Fails if |
|---|---|---|---|
| **A — fairness** | disparity (MAD) falls on the target cells | the direction of the largest attribute gaps narrows toward zero | utility drops materially, or a control cell gets worse — either means the model was damaged, not fixed |
| **B — rescue** | utility ≥ 60% and acceptance rate moves toward the reference rate | *conditional on the primary*: whether the disparity that becomes measurable is larger or smaller than the general-purpose models show | primary not met — in which case **no fairness number from this model may be reported**, a condition fixed in advance |

### Non-negotiable reporting

1. Fairness and utility columns are reported **together**, at the same granularity. A fairness gain with a utility drop is a degradation.
2. Refusal rate is reported per run. Refusals leave the denominator, so a model that learns to decline makes its disparity *unmeasurable*, not absent.
3. Results are reported per protected group and per condition, never pooled into a single number per run. Groups behave differently enough that the average describes none of them.

### Intersections: measured, not trained

Every row of the SFT and DPO data carries **exactly one** protected attribute. No training example presents a CV that is, say, both a veteran and non-binary. This is deliberate.

The baseline confirms intersectional disparity in **16 cells**, in: Qwen3.5-4B (en), Qwen3.5-4B (uk), Qwen3.5-9B (en), Qwen3.5-9B (uk), gemma-4-12B-it (uk), gemma-4-E4B-it (en), gemma-4-E4B-it (uk). So this is a live question, not a hypothetical.

The design makes the transfer an **empirical result rather than an assumption**: phase 2 promotes the phase-1 winners to the fully-crossed intersection evaluation, so the plan answers *does single-attribute invariance training generalise to intersections?* Training on intersections directly would fix them by construction and answer nothing.

**The follow-up condition, fixed in advance.** If a winning arm cuts single-group disparity but leaves the confirmed intersectional cells materially unchanged, intersectional training data becomes the next experiment. That decision is taken **after phase 1 has been scored**, on the phase-2 numbers -- never before, and never as a reaction to a disappointing single-group result.

Generating it is not a flag today. `build_variant_frame` assigns one group and one attribute per row, and `load_group` resolves a name to an attribute file, which the intersections have none of -- they are built by `intersection_attributes`, which returns *tuples*. Adding them means teaching the variant builder and `build_profile` to carry a tuple. The generation cost on top of that is roughly one teacher-model day for both languages, reusing the existing reference pass.

### Erasure: one method

Concept erasure runs **LEACE only**. INLP and mean-difference ablation are implemented, configured and tested; they are not run.

The reason is cost, and it is specific. Forward hooks cannot be hosted by vLLM, so every erasure run goes through the HuggingFace generate loop, measured here at **67 prompts/min against vLLM's 433 — 6.5x slower**. Three methods across two models and two languages came to ~87 GPU-hours, more than the prompt and scrub families combined, to compare three ways of doing one thing. One method across the same four model-language pairs costs ~35.

LEACE is the one kept because it is closed-form: the erasure is determined by the data rather than by an optimisation with its own seeds and stopping rule, so a single run is the method's result rather than one sample from it. INLP iterates to a convergence criterion and mean-difference is a one-direction approximation; both are better read as an ablation *of* LEACE than as independent arms, which is how they should return.

**State this as a limitation, not a design choice.** The claim the study can make is "closed-form concept erasure at layer *k* does *X*"; it cannot say whether an iterative or a cruder erasure would do better, and the configs to answer that are in the repository, unrun.

### Preference objectives: DPO and KTO

The study planned **DPO and ORPO**. It runs **DPO and KTO**.

TRL 1.x removed `ORPOConfig` and `ORPOTrainer` outright. Pinning an older TRL would have meant downgrading transformers below the version vLLM needs, which would have broken the evaluation pipeline that had already produced forty scored runs — a dependency trade nobody should make to keep one arm.

KTO takes its place on the **same responses**: its split is the DPO pairs unpaired into `(prompt, completion, label)`, one desirable and one undesirable row per pair, balanced by construction. Same generations, same filters, same contamination holdout — so a difference between the two arms is the objective and not the data.

The substitution also sharpens the contrast. DPO continues from each SFT checkpoint; KTO starts from the base model. The pair therefore separates *preference after SFT* from *preference instead of SFT*, which ORPO — single-stage by design — would have confounded with its own loss formulation.

**Both objectives are future work in the paper.** Only probes ran (Qwen3.5-9B English), and they are kept out of the reported results; see *Future work*.

**Report ORPO as removed for a dependency reason, not as a result.** An arm that is absent because the library dropped it is not an arm that was tried and found uninteresting, and a reader cannot tell the two apart unless the plan says which.

## 4. Order of execution

| Stage | Runs | Depends on | Why this order |
|---|---:|---|---|
| 1. prompt | 32 | nothing | Free. Sets the bar the training arms must beat |
| 2. scrub | 8 | nothing | Free. The upper reference for what any mitigation could achieve on attribute-mediated bias |
| 3. generate training data | 1 | nothing | Needed by everything below; run it early so a low yield surfaces before the training days are committed |
| 4. embedding | 4 | generated data | Eraser fitted on the training pool, never the benchmark |
| 5. SFT | 6 | generated data | The main training arm |
| 6. DPO / KTO | 0 | SFT checkpoints | DPO continues from its SFT run; KTO starts from the base model, so the pair separates *preference after SFT* from *preference instead of SFT* |
| 7. audit the adapters | 6 | trained checkpoints | Training produces weights; the fairness numbers come from auditing them. One audit per training run — each adapter is audited in **its own language only**, so the count already covers both |

Stages 1 and 2 need no generated data and can run immediately. Stage 3 is the gate for everything else.

## 5. Deferred to future work

Cut for time, not for lack of implementation: every row below is configured and tested, and none of it is missing code. Report each as a limitation with its reason, not as an omission.

| Deferred | Scope | Why | Cost to run | How to enable |
|---|---|---|---|---|
| Preference optimisation (DPO, KTO) | 10 training run(s) + their audits, 274,500 prompts | Supervised fine-tuning is the arm the study needs first: it is the cheapest of the three training options and the one the preference methods build on, so a preference result is only interpretable once SFT's is on the board. Both objectives are configured, preflighted and have their data built (published as subsets of the synthetic-data release). Three internal probes ran on Qwen3.5-9B English -- DPO on teacher pairs, DPO on decision-only pairs, and a decision-weighted SFT variant -- and are audited for the authors' own reading only: they are **not reported in the paper**, because one model and one language cannot carry a claim about an objective, and their motivation (SFT appearing to fail) turned out to be an audit artefact | ~10 GPU-h training + ~20 GPU-h audits | `python scripts/select_configs.py --runner scripts/run_all_dpo.sh configs/mitigation/dpo/*_only_dpo.yaml configs/mitigation/dpo/*_only_kto.yaml`, then re-enable their audits |
| Concept erasure — `inlp` | 4 run(s), 77,850 prompts | Erasure needs the HuggingFace generate loop (vLLM cannot host forward hooks), measured at 67 prompts/min against vLLM's 433. Three methods cost ~87 GPU-h; one costs ~35 | ~19 GPU-h + eraser fits | `python scripts/select_configs.py --runner scripts/run_all_embedding.sh configs/mitigation/embedding/*_inlp.yaml` |
| Concept erasure — `mean_diff` | 4 run(s), 77,850 prompts | Erasure needs the HuggingFace generate loop (vLLM cannot host forward hooks), measured at 67 prompts/min against vLLM's 433. Three methods cost ~87 GPU-h; one costs ~35 | ~19 GPU-h + eraser fits | `python scripts/select_configs.py --runner scripts/run_all_embedding.sh configs/mitigation/embedding/*_mean_diff.yaml` |
| Intersectional **training data** | both languages, military × gender and military × religion | Every SFT/DPO row carries one attribute, so the study measures whether single-attribute invariance *transfers* rather than assuming it. 16 intersectional cells are confirmed at baseline, so the question is live | ~1 teacher-model day, reusing the reference pass | `build_variant_frame` and `build_profile` must carry an attribute *tuple*; `intersection_attributes` already produces them |
| **Covert-bias preference data** | both languages, all three groups | The DPO pairs built for this study have a rejected side that announces its prejudice: 91% name the attribute outright and 78% share the chosen decision, differing only in wording. The audited models do the opposite — Qwen3.5-9B names the attribute in 2.3% of rationales while its decisions still depend on it. A DPO probe on these pairs hit 100% preference accuracy and ~1e-5 eval loss in 50 steps: separable by wording alone, and teaching the model to avoid a behaviour it did not have. The fix is negatives that look like the real failure — the *opposite* decision under an attribute-free, plausible rationale, so nothing in the text reveals why | one teacher pass per language with a new `covert` prompt, reusing the reference pass (~1 teacher-model day) | new prompt in `generation/synth.py`; a filter that *rejects* any negative naming the attribute, the inverse of `filter_biased` |
| **Checkpoint-selection sensitivity** | one trained model, two checkpoints | Observed during SFT: from step 300 to 800, validation loss *rose* 0.611 -> 0.641 while token accuracy also rose 0.807 -> 0.812 and predictive entropy fell 0.540 -> 0.422. The model is not learning more, it is becoming more certain of what it already believes — including where it is wrong. That matters here because the audit measures *inconsistency rate*: a lower-entropy model flips its verdict less often under a counterfactual, which reads as a fairness gain while a systematic preference can be widening underneath it — the same signature the LEACE runs showed, MAD falling while Cohen's h rose. So checkpoint choice may move the **fairness** conclusion, not just utility, and no arm in this study tests that | ~1.6 GPU-h per model; both checkpoints are already written to disk by `save_total_limit=2` (best + last) | point a copy of the trained-adapter audit config at `outputs/sft/<run>/checkpoint-<N>` instead of the run directory, once per checkpoint, and compare the two rows in section 6 |
| Intersectional **evaluation** of mitigated runs | phase-1 winners only | 145 of 179 attributes are intersections, so every arm would pay for them whether or not it worked. Phase 2 pays once, on the arms that earned it | ~28 GPU-h (~12 at `--min-mad 0.04`) | `python scripts/select_phase2.py --top 1 --dry-run` |
| **Synthetic training data, overall** | the whole `semisynthetic-v1` pipeline | Training works: SFT cut unstable counterfactual sets in all six cells and all 18 cell-by-group comparisons (-10 to -38 pp), raising utility in five of six. (HF + PEFT: margin spread across variants -51% on unseen benchmark sets; the earlier 'no effect' reading was a vLLM LoRA audit artefact), but the data limits how far it can go: overt negatives against covert bias; SFT targets that ~80% agree with what the model already decides; 57% unique completions; a teacher pool 89% reject against a 34%-hire benchmark; a biased-pass yield of 55.6% / 36.8% (en / uk) with DPO pairs 18k military vs 5k religion; and a leakage detector that misses paraphrases; only 3,000 pairs expanded to 12 variants each; and an anchor verdict drawn as a single sample at temperature 0.7, least reliable on the borderline candidates where bias acts | the next iteration should start here, before any further optimiser or hyperparameter work | README, *Future work: the synthetic training data* — one row per problem, each measured, each with a fix |

The cost column is measured, not estimated: throughput comes from the completed baseline runs on this machine. Section 3 gives the reasoning behind each deferral in full.