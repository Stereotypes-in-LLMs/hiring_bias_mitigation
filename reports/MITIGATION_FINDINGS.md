# Mitigation — findings

*Synthesis of the mitigation experiments. Written by hand from the scored runs
([`RESULTS.md`](RESULTS.md)), the set-stability analysis
([`set_stability.csv`](set_stability.csv), [`set_stability_by_group.csv`](set_stability_by_group.csv),
[`mitigation_decision_table.csv`](mitigation_decision_table.csv)) and direct reading of the raw
generations. Baseline findings are in [`FINDINGS.md`](FINDINGS.md).*

*Every comparison below is on matched populations: the same sets, the same attribute variants,
the same rows. Three earlier readings of these results broke that rule, and each produced an
effect that vanished once it was restored — see* Methodological lessons *before reusing any
number from an earlier draft.*

---

## What was run

Mitigation targeted the two models with confirmed disparity, **Qwen3.5-4B and Qwen3.5-9B**, in
**English and Ukrainian**, on the protected groups where the baseline confirmed bias.

| Family | Arms | Coverage |
|---|---|---|
| Prompt | 8 strategies | 4B en/uk, 9B en/uk |
| Scrub (remove the attribute from the input) | lexical, LLM | 4B en/uk, 9B en/uk |
| Concept erasure | LEACE at layer 16 | 4B en, 9B en/uk (4B uk unusable) |
| SFT (LoRA) | invariant targets; one decision-weighted variant | 4B en/uk, 9B en/uk |
| DPO (LoRA) | teacher-generated pairs; decision-only pairs | **probe:** 9B en only |

INLP, mean-difference erasure, KTO and intersectional training were deferred; see the
README's *Future work* sections.

**Scope caveat for 9B English.** Its prompt, scrub and erasure runs evaluated one group
(military status) under one condition (implicit), so each set there holds 5 variants and
baseline instability is 13.3%. There is little room to move, and most 9B-en prompt effects are
not significant. The trained 9B-en adapters were evaluated on all three groups.

---

## The metric: counterfactual set stability

A **set** is one candidate–job pair under one injection condition, evaluated with every
attribute variant. It is **unstable** if the decision is not the same across those variants —
the attribute alone was enough to tip it. Each mitigated set is paired with the same set at
baseline, **restricted to the same variants**; *fixed* counts unstable→stable, *broken*
stable→unstable. Significance is an exact sign test, Benjamini–Hochberg corrected across all 50
comparisons; intervals are a bootstrap over **candidates**, since sets sharing a candidate are
not independent.

This measures the invariance every arm is aiming at, directly. Acceptance-rate disparity (MAD)
is averaged over whole attributes, so correcting one dissenting decision moves an attribute's
rate by 1/450.

---

## Finding 1 — One prompt works everywhere, and it costs utility

`structured_rubric` — asking the model to score explicit criteria before deciding — is the only
strategy that reduced instability significantly in **all four** model–language cells, and for
**military status specifically** in all four.

| Cell | Δ unstable sets (pp) | 95% CI | Δ utility (pp) |
|---|---:|---|---:|
| 4B en | −18.4 | [−22.2, −14.6] | −5.9 |
| 4B uk | −23.5 | [−27.8, −19.2] | −5.3 |
| 9B en | −7.6 | [−11.1, −4.2] | −5.2 |
| 9B uk | −12.1 | [−15.9, −8.2] | −3.3 |

The price is 3–6 points of agreement with the attribute-free reference decision, everywhere.

## Finding 2 — Two instructions help without cost, but not in every cell

| Strategy | 4B en | 4B uk | 9B en | 9B uk | Δ utility |
|---|---:|---:|---:|---:|---|
| `ignore_personal_info` | −5.7 | −7.8 | −2.4 *n.s.* | −8.9 | +0.3 … +3.2 |
| `recruiter_guidelines` | −11.3 | −14.3 | −1.1 *n.s.* | −5.9 | −0.3 … −1.9 |

Both are significant in three of four cells, including military status in those three, with
no meaningful utility cost. Neither is established on 9B English, where the scope leaves little
headroom. `ignore_personal_info` is reproduced from the audit study, not introduced here.

## Finding 3 — Some instructions make things worse, and which ones depends on the model and language

| Strategy | Improves | **Worsens** |
|---|---|---|
| `second_pass_verification` | 4B en −20.3 | **9B en +4.9, 9B uk +12.8** (61 fixed vs 175 broken) |
| `fairness_constitution` | 4B en −11.1, 9B uk −12.2 | **4B uk +12.5** (94 fixed vs 206 broken) |

On 4B Ukrainian, `fairness_constitution` leaves military status unchanged and makes **gender
(+8.7) and religion (+8.9)** worse. `second_pass_verification` — the largest single gain on 4B
English — worsens both 9B models in every group evaluated (military status on 9B en; all three groups on 9B uk). A prompt mitigation validated on one model
and language is not validated on another.

## Finding 4 — Removing the attribute is the upper bound

Lexical scrubbing takes instability to 0.0–1.4% (−13.3 … −41.6 pp) without costing utility. It
is an upper reference, not a deployable mitigation: it works because the model never sees the
attribute, and it can do nothing about bias carried by anything the scrubber misses. LLM
scrubbing, which misses some mentions, is weaker (−4.2 … −26.7 pp).

## Finding 5 — Concept erasure is fragile in a way the model does not predict

LEACE at layer 16 broke the output format to very different degrees — **0% parse failures on 9B
en, 19% on 4B en, 29% on 9B uk, 100% on 4B uk** (4 of 31,050 responses parsed; excluded). On the
sets it could still answer it reduced instability (−2.4, −6.4 and −17.0 pp), with utility changes
of +1.0 to −6.3 pp. A run with parse failures is measured only on the sets it could complete —
plausibly the easier ones — so these are upper bounds for the method.

## Finding 6 — Fine-tuning: re-audit in progress (earlier result withdrawn)

> **The earlier Finding 6 ("fine-tuning did not change counterfactual consistency") is
> withdrawn.** The audit served the adapters through vLLM's LoRA path, which does not reproduce
> Qwen3.5 adapters: on the same benchmark prompts, HF + PEFT (the framework the adapters were
> trained in) and vLLM agree on 98.9% of the base model's decisions but only 82.5% of the SFT
> adapter's. The adapter changes 17.5% of benchmark decisions under HF + PEFT and 0.3% in the
> old audit. The old numbers measured a model close to the untrained one. See lesson 4 below.

Measured directly under HF + PEFT on the 9B English SFT adapter, training **did** learn the
targeted invariance. The spread of the hire–reject logit margin across a set's attribute
variants fell by **51%** on benchmark sets never seen in training (p = 2·10⁻⁶, Wilcoxon) and
by **53%** on training-pool sets (p = 1.7·10⁻¹⁰), so it generalises rather than memorises.

All seven adapters are being re-audited from merged weights (`scripts/reaudit_merged.sh`).
Merged-weight serving was validated first: vLLM on the merged checkpoint agrees with HF + PEFT
on 97.4% of benchmark decisions (base model: 98.9%) and with the untrained base on only 80.6%.
This section will be rewritten from those audits.

## Finding 7 — The training data taught a behaviour the models did not have

The preference pairs' biased side was written by a teacher prompted to discriminate, and it did
so openly: **91% of rejected responses name the protected attribute** ("We cannot hire candidates
of the Sikh faith…", "We require a traditional male leader…"), and 78% carry the same decision as
the chosen response, differing only in that wording. The audited models do not behave this way —
Qwen3.5-9B names the attribute in about 1% of its rationales on the mitigation cells — yet their
decisions still depend on it. Their bias is covert: a decision that shifts under a
neutral-sounding rationale.

A DPO run on these pairs reached 100% preference accuracy and an eval loss of ~10⁻⁵ within 50
steps: the pairs were separable by wording alone. And the SFT targets inherit the teacher's
attribute-free verdict from a single sample at temperature 0.7 — least reliable on exactly the
borderline candidates where an attribute tips a decision. The README's *Future work: the
synthetic training data* lays out what fixing this requires.

---

## What this supports saying in the paper

1. **Prompt-level mitigation measurably reduces counterfactual inconsistency.** A structured
   scoring rubric does so in every model and language tested, at a 3–6 point utility cost; two
   instructions — one of them prior art — do so in most cells at no cost.
2. **Prompt mitigations do not transfer by default.** Two strategies that help one model or
   language make another significantly worse, in some cases on groups they were not aimed at.
3. *(Pending the merged-weight re-audit; the earlier "no effect" claim is withdrawn — see
   Finding 6.)*
4. **Teacher-generated "biased" examples are overt while the bias being mitigated is covert**;
   that mismatch is the first thing to fix in a training approach.
5. **Paired, set-level consistency on matched variants** separates real effects from artefacts
   that aggregate metrics and unmatched comparisons both produce.

## What it does not support

- That training *cannot* reduce this bias. Every training run used ≤ 5,600 pairs and ≤ 400 steps
  of LoRA on data with the problems in Finding 7.
- Any conclusion about 9B English prompt effects beyond `structured_rubric`: its single-group
  scope leaves too little headroom to establish them.
- Any claim about intersections, INLP, mean-difference erasure, KTO, the 12B models or LAPA.

---

## Methodological lessons

Lessons 1–3 are the same error, made three times, each time in a different metric, and each
time it produced a clean, significant, plausible effect that disappeared once corrected.

**1. Leakage rate over different populations.** The attribute-mention rate appeared to halve
under SFT (9B en 2.34% → 1.10%). The baseline figure covered the full grid including
intersections, where two attributes can be named; the adapter figure covered single-group cells.
On the same rows the baseline was 1.12%.

**2. Agreement with the wrong reference.** Testing whether changed decisions moved toward the
teacher's verdict is a question about agreeing with another model, not about invariance. It
happened to give the right answer here — no effect — for a reason unrelated to the question.

**3. Set stability over different variant sets.** Baseline sets held 179 variants (every
attribute and intersection); a trained run's sets held 34, a scoped prompt run's 5. A set with
more variants is more likely to hold a dissent, so the unmatched comparison credited every
mitigation with a gain that grew the narrower its scope. Every training adapter appeared to fix
33–63 sets against 0–3 broken (p ≈ 10⁻¹⁰); on matched variants they fixed and broke a handful
each, and none was significant.

The rule these share: **a baseline audit and a mitigated run do not cover the same population
by default.** The baseline is broad on purpose, and mitigations are scoped to save cost. Every
comparison between them has to be restricted to their intersection first — rows, cells *and*
attribute variants. The report now enforces this in all three places, and
`tests/test_analysis.py` pins each case.

**4. The serving stack is part of the measurement.** Every adapter audit was served through
vLLM's LoRA support, which for Qwen3.5's hybrid linear-attention architecture does not
reproduce the trained model. Nothing failed: outputs parsed, metrics were computed, and the
null result was consistent across seven adapters and four objectives — which made it
convincing. It was caught only by reading the decision logit directly under HF + PEFT and
comparing it with the audit, row by row (`scripts/diagnose_adapter.py`). Audits now serve
adapters as merged checkpoints (`mitigation/merge.py`), validated against HF + PEFT before use,
and `tests/test_eval.py` pins that an adapter audit never takes the vLLM LoRA path. The rule:
**before interpreting a null result for a trained model, check that the evaluated model is the
trained one** — agreement with the training framework's own decisions on a sample of prompts.
