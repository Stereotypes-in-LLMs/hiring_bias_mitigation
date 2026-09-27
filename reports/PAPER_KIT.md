# Paper kit — every claim, its number, and where the evidence lives

*One page to write the paper from. Each row is a claim the data supports, the number to quote,
and the file to check it against. Findings prose: [`FINDINGS.md`](FINDINGS.md) (baseline audit)
and [`MITIGATION_FINDINGS.md`](MITIGATION_FINDINGS.md) (mitigation). Metric definitions:
[`../docs/METRICS.md`](../docs/METRICS.md).*

**Status: complete.** 66 usable runs, 6 fine-tuned cells, 4 operating-point controls, and the
three headline prompt strategies rerun on Qwen3.5-9B English over the full grid. One run is
excluded and reported as excluded (LEACE on Qwen3.5-4B Ukrainian, 100% parse failures).

---

## The study in one paragraph

Five open-weight models (Qwen3.5-4B/9B, gemma-4-E4B/12B, LAPA-12B) were audited on a
counterfactual hiring benchmark in English and Ukrainian: each job–CV pair is decided many
times, identical except for one injected protected attribute (military status, gender,
religion), stated explicitly or implicitly, plus an attribute-free control. Four mitigation
families were then measured against that baseline on matched rows — prompt (8 strategies),
input scrubbing (lexical, LLM), LEACE concept erasure, and LoRA supervised fine-tuning on
counterfactually invariant targets. The headline metric is the share of **counterfactual sets**
whose decision is not constant across the attribute's variants.

---

## Claims and evidence

### Baseline

| # | Claim | Number | Evidence |
|---|---|---|---|
| B1 | Military status dominates the disparity | see Finding 1 | [`FINDINGS.md`](FINDINGS.md), [`baseline_disparity`](../figures/en/baseline_disparity.svg) |
| B2 | The military-status effect reverses direction between English and Ukrainian | Finding 2 | [`attribute_gaps`](../figures/en/attribute_gaps.svg) |
| B3 | Ukrainian shows more disparity, not explained by output quality | Finding 3 | `RESULTS.md` §3 |
| B4 | In English the bias lives in the implicit condition | Finding 4 | [`condition_contrast`](../figures/en/condition_contrast.svg) |
| B5 | Model family separates behaviour; scale within a family does not | Finding 5 | `RESULTS.md` §2 |
| B6 | A model can fail the task so badly that its fairness numbers are meaningless — and an audit that reports only fairness will rank it as most biased | LAPA utility 51–55%, hires 75–80% where the reference hires 33–35% | Finding 6, [`fairness_utility_tradeoff`](../figures/en/fairness_utility_tradeoff.svg) |

### Mitigation

| # | Claim | Number | Evidence |
|---|---|---|---|
| M1 | **SFT is the strongest deployable mitigation**: it cuts unstable sets in every cell and every protected group | −10.1 to −38.2 pp; 18/18 cell × group comparisons significant after FDR | Finding 6, [`set_stability_by_group.csv`](set_stability_by_group.csv) |
| M2 | It does not cost utility — in five of six cells it *raises* it | +1.2 to +29.6 pp; −5.3 pp in the one exception (9B uk) | Finding 6, `RESULTS.md` §6 |
| M3 | SFT beats the best prompt on both axes at once, on identical data | full grid, 900 sets: −10.1 pp at +3.6 utility vs −7.6 pp at −6.3 for `structured_rubric`; same ordering on the narrow scope | Finding 6, [`set_stability.csv`](set_stability.csv), [`matched_scope`](../figures/en/matched_scope.svg) |
| M4 | The gain is invariance, not strictness | at equal hire rate SFT is 3.7×–11.4× more consistent; the shift explains 0–29% of the gain in three cells, 54% in one | Finding 7, [`operating_point`](../figures/en/operating_point.svg), [`operating_point/`](operating_point) |
| M5 | On a badly calibrated model the same training fixes calibration *and* invariance | LAPA: utility 51→81 (en), 55→77 (uk); hires 68%→3% of candidates the reference rejects | Finding 6 of `FINDINGS.md`, Finding 6 of `MITIGATION_FINDINGS.md` |
| M6 | One prompt works everywhere and costs utility | `structured_rubric`, −7.6 to −23.5 pp, −3.3 to −5.9 utility | Finding 1 |
| M7 | Prompt mitigations do not transfer between models or languages | `second_pass_verification` −20.3 pp on 4B en, **+6.0** on 9B en (full grid, 16 fixed vs 70 broken) and **+12.8** on 9B uk | Finding 3, [`language_transfer`](../figures/en/language_transfer.svg) |
| M8 | Removing the attribute is an oracle bound, not a method — and it measures the pipeline's noise floor | lexical scrub: 0.0–1.4% instability by construction; 0–0.7% of sets still flip on identical prompts | Finding 4 |
| M9 | Concept erasure is fragile in a way the model does not predict | LEACE parse failures 0% / 19% / 29% / 100% across four cells | Finding 5 |
| M10 | Training data whose negatives are overt cannot teach a model to stop being covertly biased | 91% of rejected responses name the attribute; models name it in ~1–2% of rationales | Finding 8 |

### Methodological claims (each cost a result)

| # | Claim | Number | Evidence |
|---|---|---|---|
| X1 | A comparison between a baseline and a mitigated run must be restricted to matched rows, cells **and attribute variants** | unmatched: every adapter "fixed" 33–63 sets vs 0–3 broken, p ≈ 10⁻¹⁰; matched: a handful each | Lessons 1–3 |
| X2 | The serving stack is part of the measurement | HF vs vLLM agreement: base 98.6–98.9%, same adapter through vLLM's LoRA path **38–83%**; merged weights restore 97.4–98.7% | Lesson 4, [`adapter_diagnostic--*.json`](.) |
| X3 | A value normalised for analysis must not become a training target | Ukrainian targets carried `"reject"` where the prompt asks `відхилити`; both Ukrainian adapters learned it, parse failures stayed 0% | Lesson 5 |
| X4 | Counterfactual set stability on matched variants separates real effects from artefacts that aggregate metrics hide | MAD moves 1/450 per corrected decision; set stability sees every flip | `../docs/METRICS.md` |

---

## Tables to reproduce in the paper

| Table | Source |
|---|---|
| Baseline disparity per model × language × group | `RESULTS.md` §3, [`analysis.json`](analysis.json) |
| Mitigation summary: Δ unstable sets, CI, fixed : broken, Δ utility | [`set_stability.csv`](set_stability.csv) + [`mitigation_decision_table.csv`](mitigation_decision_table.csv) |
| Per-group breakdown | [`set_stability_by_group.csv`](set_stability_by_group.csv) |
| Operating-point control | [`operating_point/*.json`](operating_point) |
| Per-cell full metric tables (18 images) | `../figures/{en,uk}/table_<group>_<model>_<lang>.png` |

## Figures

`baseline_disparity`, `attribute_gaps`, `condition_contrast`, `leak_versus_disparity`,
`fairness_utility_tradeoff`, `strategy_ranking`, `language_transfer`,
`stability_utility_tradeoff`, `matched_scope`, `operating_point`, `training_curves` — each in
[`../figures/en/`](../figures/en) and [`../figures/uk/`](../figures/uk) as SVG, PDF and PNG,
with the CSV it was drawn from.

## Released artifacts

| What | Where |
|---|---|
| Every audited model response, one subset per run (64 runs, SFT included) | [hiring-bias-mitigation-responses](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-responses) |
| The six SFT adapters, each with its own audited numbers and a merge-before-serving note | [`qwen3.5-{4b,9b}-hiring-debias-sft-{en,uk}`](https://huggingface.co/Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en), [`lapa-12b-hiring-debias-sft-{en,uk}`](https://huggingface.co/Stereotypes-in-LLMs/lapa-12b-hiring-debias-sft-uk) |
| Semi-synthetic training data, with the Ukrainian targets corrected (lesson 5) | [hiring-bias-mitigation-synthetic-data](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-synthetic-data) |
| Collection (8 items) | [Hiring Bias Mitigation](https://huggingface.co/collections/Stereotypes-in-LLMs/hiring-bias-mitigation-6aae99f67667367c7149cedd) |
| Code, configs, reports | this repository |

---

## What the study does not support

- That SFT removes the bias: it removes every *significant* disparity flag and takes instability
  to 4–11% of sets, not to the ~0.5% noise floor.
- Any claim about preference optimisation (DPO, KTO), intersections, INLP, mean-difference
  erasure, or the gemma models as mitigation targets.
- Generalisation beyond one LoRA configuration, one seed, one checkpoint rule and ≤ 15k training
  rows per language.
- That the utility gain is free: SFT also moves the hire rate, and a deployment has to decide
  whether that shift is acceptable.
