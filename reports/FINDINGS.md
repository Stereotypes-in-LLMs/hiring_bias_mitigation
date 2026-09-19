# Baseline audit — findings

*Synthesis of the ten baseline runs. Written by hand from the generated analysis
([`ANALYSIS.md`](ANALYSIS.md)), the measurements ([`RESULTS.md`](RESULTS.md)) and direct
reading of the raw generations. Metric definitions: [`docs/METRICS.md`](../docs/METRICS.md).*

## What was run

Five open-weight models × two languages, over the 450 job–CV pairs per language released with
the audit study this work extends. Three protected groups plus two fully-crossed
intersections, under three conditions (explicit field, implicit self-description, and an
attribute-free control the audit study did not have).

| | |
|---|---|
| Models | Qwen3.5-4B, Qwen3.5-9B, gemma-4-E4B-it, gemma-4-12B-it, LAPA-v0.1.2-instruct |
| Generations | 1,615,500 (10 runs × 161,550) |
| Statistical tests | 10,740, FDR-corrected within each run |
| Cells graded | 102 (model × language × group × condition) |
| Decoding | greedy, seed 42, thinking mode off — recorded per run |
| Refusals / parse failures | 0.00% in nine runs, 0.17% in one |

**Verdicts:** 29 confirmed · 25 weak · 26 clean · 22 not interpretable (all LAPA, plus two
partial smoke-run cells).

## What we count as bias

Three conditions together; none is sufficient alone.

1. **The measurement must be sound.** Checked first, at cell *and* run level. A model that
   agrees with the attribute-free reference at close to chance is not screening candidates,
   and a disparity measured on top of that is not evidence about fairness.
2. **Significance after FDR correction**, across all 10,740 tests. The paired sign-flip test —
   which the matched counterfactual design licenses and the audit study explicitly could not
   use — must agree.
3. **A material effect size**: ≥5 pp acceptance-rate gap against the group's reference level,
   or |Cohen's *h*| ≥ 0.20. At 161,550 rows a one-point gap is significant because *n* is
   large.

Thresholds are judgement calls, stated in `ANALYSIS.md` and overridable on the command line.

---

## Finding 1 — Military status dominates, replicating the audit study

23 of 29 confirmed cells involve military status, alone or in an intersection. Gender and
religion contribute three each.

| Protected group | Confirmed cells |
|---|---:|
| military_status × religion | 9 |
| military_status | 7 |
| military_status × gender | 7 |
| religion | 3 |
| gender | 3 |

This replicates the audit study's central result on a different model generation, and
strengthens its claim: an audit restricted to gender and race would have missed the largest
effect in this system entirely.

## Finding 2 — The direction of the military-status effect reverses between languages

This is new, and it is the result we did not expect.

Acceptance-rate gap against `Civilian` / `Цивільний`, explicit condition, `*` = significant
after FDR correction:

| Attribute | Qwen3.5-4B | Qwen3.5-9B | gemma-4-E4B | gemma-4-12B |
|---|---:|---:|---:|---:|
| **English** | | | | |
| Participant in combat actions | **−13.8\*** | +4.9 | +5.6 | +3.3 |
| War veteran | **+7.6\*** | +3.8 | +7.3 | +3.8 |
| **Ukrainian** | | | | |
| Учасник бойових дій | +7.1 | **+17.6\*** | +10.0 | +3.6 |
| Ветеран війни | **+17.3\*** | **+15.8\*** | +6.9 | +4.4 |

In English, Qwen3.5-4B penalises combat participants by 13.8 points — the audit study's
headline finding, reproduced. In Ukrainian, **the same model favours them**, and war veterans
are favoured by 17.3 points. Every model shifts in the same direction across the language
axis; only the magnitude differs.

`Ветеран війни` is the single most recurrent attribute in the study: favoured in 4 of 4
interpretable Ukrainian models, mean gap +11.0 pp, maximum +17.3 pp, and it carries the same
direction inside both intersections.

**Two readings, and our data does not separate them.** Either the models encode a
culturally-specific positive valence for veteran status that is reachable in Ukrainian and
not in English; or the Ukrainian and English attribute strings are not translation-equivalent
in the models' representation space. Distinguishing these needs a controlled paraphrase
condition, which we did not run. We report the reversal as observed and the explanation as
open.

A related hypothesis from the audit study — that `Військовий пенсіонер` triggers an age cue
through the shared stem with *pensioner* — is **partially supported and model-dependent**. In
both Qwen models under explicit injection the retiree sits far below the other military
statuses (−15.8 vs +8.5 mean, and −0.7 vs +12.0). In both Gemma models it does not. Not a
finding; a hypothesis worth a dedicated age-controlled condition.

## Finding 3 — Ukrainian shows more disparity, and quality does not explain it

Across 50 matched English/Ukrainian comparisons, 40 show higher disparity in Ukrainian
(sign test **p = 2.4 × 10⁻⁵**, median Δ = +0.84 pp).

The audit study reported this direction but named the confound it could not remove: Ukrainian
generation quality was lower for every model it tested, and a noisier decision process
inflates disparity independently of bias. **Carrying utility alongside separates them here.**
Restricting to comparisons where Ukrainian utility stays within 5 points of English, and both
cells are interpretable: **27 of 33 remain positive (p = 3.2 × 10⁻⁴)**.

The clearest single case is Qwen3.5-9B on military status: disparity rises 2.4 → 7.7 pp
(3.2×) while utility falls only 82% → 78%. gemma-4-12B holds utility at 84.4% in *both*
languages and still shows the direction, at small magnitude.

This closes a gap the prior work left open: the language effect is not an artefact of
generation quality.

## Finding 4 — Implicit injection is where English bias lives; Ukrainian shows it under both

**Cell level: strong.** Where the raw generations license a McNemar test on the decisions that
actually changed between framings, several models are clean under a labelled attribute field
and biased when the same fact arrives as ordinary biography:

| Model | Group | Explicit MAD | Implicit MAD | Discordant pairs | McNemar *p* |
|---|---|---:|---:|---:|---:|
| Qwen3.5-9B | religion | 1.6 | 4.4 | 586 | 5.1 × 10⁻¹⁵⁴ |
| Qwen3.5-4B | religion | 1.3 | 3.7 | 843 | 1.6 × 10⁻²³¹ |
| gemma-4-12B | military × religion | 1.1 | 3.5 | 1,575 | 9.1 × 10⁻²⁶⁷ |
| gemma-4-E4B | military × religion | 1.1 | 3.8 | 1,877 | 2.6 × 10⁻²⁹¹ |

18 of 29 confirmed cells are implicit.

**Aggregate level: significant in English, not in Ukrainian.** Across all 50 matched
comparisons only 32 favour the implicit condition (sign test p = 0.065) — but splitting by
language shows why that pooled figure is uninformative:

| Language | Implicit worse | Sign test | Median Δ | Confirmed cells that are implicit |
|---|---|---:|---:|---|
| English | 15 / 20 | **p = 0.041** | +0.49 pp | 7 of 10 |
| Ukrainian | 11 / 20 | p = 0.82 | +0.33 pp | 11 of 19 |

The two languages are answering different questions. In Ukrainian the disparity is already
large under an explicit labelled field — military status reaches 8.2 pp MAD — so the implicit
framing has little to add. In English, explicit injection is nearly clean and **implicit is
where the bias appears**.

That is the audit study's policy point, sharpened: it is specifically the *English*,
*explicit* protocol — the one nearly every published audit uses — that certifies these models
as clean. Pooling the two languages hides this, which is why we report it split.

## Finding 5 — Family separates the models; scale within a family does not

| Model | Confirmed cells | Mean utility | Worst cell MAD |
|---|---:|---:|---:|
| Qwen3.5-4B | 16 | 77.7% | 8.2 pp |
| Qwen3.5-9B | 10 | 78.6% | 7.7 pp |
| gemma-4-E4B | 2 | 78.7% | 3.8 pp |
| gemma-4-12B | 1 | 84.5% | 3.5 pp |

The two Gemma models show 3 confirmed cells between them; the two Qwen models show 26. The
separation is by family, not size: gemma-4-E4B (~4B effective) is as clean as gemma-4-12B and
far cleaner than Qwen3.5-9B, which is larger than both. Doubling Qwen from 4B to 9B removes
six confirmed cells but leaves the same worst-case magnitude.

**Scale is not the lever here.** Whatever makes the Gemma line cleaner on this task is a
property of the family — training data, alignment, or both — and it is present at 4B.

One caveat on the Qwen pair: Qwen3.5-4B is the only full fine-tune in the training matrix and
9B is LoRA. That does not affect this baseline comparison, which involves no training at all,
but it will confound a 4B-vs-9B comparison of *mitigation* results.

## Finding 6 — One model could not be measured, and that is itself a result

LAPA-v0.1.2-instruct is excluded from every fairness claim. All 20 of its cells failed the
quality gate:

- mean agreement with the attribute-free reference **52.9%** (range 47.8–61.7%) — at chance
  on a benchmark whose reference is 2:1 reject
- accepts **81.5%** of all candidates in English, 78.6% in Ukrainian
- 0% refusals and 0% parse failures: it emits clean, well-formed JSON — it simply says *hire*
- attribute leakage 4,964 rows, versus 769 for gemma-4-12B

By raw numbers LAPA had the second-highest disparity in the matrix and would have ranked as a
top mitigation target. It is not a fairness result: its disparity is measured on a process
that barely discriminates between candidates at all.

**This is why the utility column exists**, and it is worth reporting as a methodological
result in its own right. An audit that reports only fairness metrics will rank a model that
fails the task as one of the most biased — and it will be wrong.

### But LAPA does rank candidates — it thresholds them badly

Looking further changes what should be done about it. On the attribute-free control, LAPA
accepts **93.0%** of the pairs the reference would hire and **64.0%** of those it would
reject — a 29-point spread. It is reading the CV. Its fault is *where it puts the threshold*:
74.2% acceptance against a reference rate of 35.1%.

| Model (uk, attribute-free) | AR \| ref=hire | AR \| ref=reject | Discrimination | Overall AR | Reference AR |
|---|---:|---:|---:|---:|---:|
| LAPA-12B | 93.0% | 64.0% | **29.0 pp** | 74.2% | 35.1% |
| gemma-4-12B | 82.9% | 13.0% | 69.9 pp | 37.6% | 35.1% |

The distinction matters because the two failures have different remedies. A model whose
decision is unrelated to the candidate cannot be fixed by training on decisions. A model that
ranks but mis-thresholds **can** — and the counterfactually-consistent training set pins every
verdict to the attribute-free reference decision, which is precisely a threshold-calibration
signal.

LAPA therefore enters the mitigation stage as a **rescue arm**, with utility as the primary
outcome. This is a different experiment from the fairness arms and must be reported as one:
the question is whether tuning makes a Ukrainian-native model able to do the task, not whether
it removes a disparity. **Its fairness numbers stay uninterpretable unless post-tuning utility
reaches 60%** — recorded here in advance, so a fairness figure from a still-unusable model
cannot find its way into a table after the fact.

If it works, the study gains its only Ukrainian-native data point on Ukrainian hiring bias,
which is otherwise a real gap in a paper about exactly that. If it does not, "a
Ukrainian-native model could not be tuned to threshold this task, while general-purpose models
handle it out of the box" is a publishable negative result about Ukrainian-language model
development.

---

## What to mitigate

29 target cells, which collapse into a small set of runs because one audit run covers every
group at once. Ordered by disparity, with the controls that make the results interpretable.

Two separate experiments come out of this — a fairness track and a task-fitness track — set
out in [Two tracks, run in parallel](#two-tracks-run-in-parallel) below. LAPA appears only in
the second, and the tables here are Track A.

### Primary targets

| # | Model | Lang | Group | Condition | MAD | Why |
|---|---|---|---|---|---:|---|
| 1 | Qwen3.5-4B | uk | military_status | explicit | 8.2 pp | largest single-group disparity in the study |
| 2 | Qwen3.5-9B | uk | military_status | explicit | 7.7 pp | replicates #1 at 2× scale — tests whether a mitigation generalises across size |
| 3 | Qwen3.5-4B | uk | military × gender | explicit | 6.3 pp | largest intersection; tests non-additivity under mitigation |
| 4 | Qwen3.5-9B | uk | military × gender | explicit | 6.1 pp | replicates #3 |
| 5 | Qwen3.5-9B | uk | military_status | implicit | 5.4 pp | the implicit arm of #2 |
| 6 | Qwen3.5-4B | en | military_status | explicit | 5.3 pp | the **English** arm, and the only cell reproducing the audit study's published English direction |
| 7 | Qwen3.5-4B | en | military_status | implicit | 4.5 pp | English implicit — where most English disparity lives |
| 8 | Qwen3.5-9B | en | military_status | implicit | 3.9 pp | English implicit at 2× scale |

### Controls — cells already clean

Six, one per target model × language. A mitigation that *worsens* a clean cell is as
informative as one that fixes a target, and without them the study cannot distinguish *"the
mitigation removed a disparity"* from *"the mitigation flattened everything, including what
was already fine"*.

**Two gaps to record now.** Qwen3.5-4B · uk and LAPA · uk have **no clean cell available** —
every measurable cell shows some disparity. Their mitigation results will lack a no-harm
check, and the write-up must say so rather than quietly omitting it.

### English specifically — is any mitigation needed?

Yes, but the shape is different from Ukrainian, and one model needs none at all.

| Model | Confirmed EN cells | Worst MAD | Reading |
|---|---:|---:|---|
| Qwen3.5-4B | 7 / 10 | 5.3 pp | the main English target |
| Qwen3.5-9B | 2 / 10 | 3.9 pp | both implicit only |
| gemma-4-E4B | 1 / 10 | 3.0 pp | one intersection, implicit |
| **gemma-4-12B** | **0 / 10** | 2.6 pp | **clean in English throughout** |

Ten confirmed English cells against nineteen Ukrainian — and **seven of the ten are
implicit**. Under the standard explicit protocol, English shows only three confirmed cells,
all in Qwen3.5-4B.

Three consequences for the experiment design:

1. **English is not a formality.** Qwen3.5-4B in English carries a 5.3 pp MAD on military
   status with a 21.3-point acceptance-rate range — a real target, and the only cell where
   the direction matches the audit study's published English finding (combat participants
   penalised). Mitigating Ukrainian only would leave the study unable to say whether a
   mitigation transfers across languages, which is one of its more useful questions.
2. **The English arms must include the implicit condition**, or they will mostly measure
   nothing. This is already how every audit config runs, but it should be stated in the
   write-up: an English-only, explicit-only evaluation of these models would report almost no
   bias to mitigate.
3. **gemma-4-12B in English is the cleanest cell set in the study** — zero confirmed cells,
   84.4% utility. It is the strongest available no-harm control, and it is the right place to
   demonstrate that a mitigation does not damage a model that needed nothing.

## Two tracks, run in parallel

The plan splits into two experiments that answer different questions and must not be pooled
in the write-up.

| | Track A — fairness | Track B — task fitness |
|---|---|---|
| Models | Qwen3.5-4B, Qwen3.5-9B (+ gemma-4-12B as control) | LAPA-12B |
| Question | does the mitigation remove a measurable disparity? | does tuning make the model able to do the task at all? |
| Primary outcome | disparity (MAD, AR range), read beside utility | **utility**, and acceptance rate approaching the reference rate |
| Families | prompt, scrub, embedding, SFT, DPO/ORPO | **SFT, DPO/ORPO** (+ one cheap prompt probe) |
| Trained on | all three groups, both languages — plus a military-only ablation | all three groups, both languages (no ablation) |
| Evaluated on | all groups + both intersections, both languages, all three conditions | same |
| Baseline exists? | yes — the confirmed cells above | no — nothing interpretable to compare against |

**Training scope and evaluation scope are different things**, and the tables below name both.
Training scope is what the model sees; evaluation scope is what gets measured afterwards, and
it is always the full grid — an audit run covers every group, intersection, condition and
language regardless of what the model was trained on. That is what makes "did mitigating
military status also change religion?" answerable at all.

### Track A — fairness mitigation

| Stage | Models | Trained on | Evaluated on | Rationale |
|---|---|---|---|---|
| Prompt (8 strategies) | Qwen3.5-4B, 9B; gemma-4-12B as control | — (no training) | all groups + intersections, en + uk, all conditions | Free. Sets the bar the training arms must beat. Four are reproduced from the audit repository and must be reported as prior art |
| Scrub (lexical, LLM) | same | — | same | Free. The upper reference for what any mitigation could achieve on attribute-mediated bias |
| Embedding (LEACE, INLP, mean-diff) | Qwen3.5-4B, 9B | eraser fitted on **military status** only, per language, from the generated training pool | same | The eraser is per-attribute by construction; military status is where the disparity is. Changes no weights |
| SFT — main | Qwen3.5-4B, 9B | **all 3 groups, both languages** | same | The primary arm. Only where MAD ≥ 4 pp — fine-tuning to chase a 2-point gap spends GPU-days on measurement noise |
| SFT — ablation | Qwen3.5-4B, 9B | **military status only**, both languages | same | Does training on the one dominant group generalise to gender and religion? 23 of 29 confirmed cells involve military status, so this is the cheap version of the whole intervention |
| DPO / ORPO | Qwen3.5-4B, 9B | all 3 groups, both languages | same | The SFT-vs-SFT+preference ablation only means something where SFT alone leaves something |

**Note on the language axis.** Every training arm above trains on both languages at once,
because the generated dataset carries both and the audit measures both. Given Finding 2 — the
military-status effect *reverses* direction between English and Ukrainian — a language-transfer
ablation would be a natural addition: train on `configs/data/uk_only.yaml`, evaluate on
English, and ask whether a mitigation fitted to one language transfers or backfires. The
configs for that view exist; the SFT/DPO configs pointing at them do not, and adding them is a
deliberate scope decision rather than something to slip in.

**Include gemma-4-12B in the prompt and scrub arms even though it is nearly clean.** It is the
strongest available no-harm control: any mitigation that degrades a model already at 84.4%
utility and 1.3 pp MAD is doing damage, and that is exactly what a paper needs to report.

### Track B — LAPA rescue: why it is here, and why only the tuning arms

LAPA is absent from every table above because all 20 of its cells failed the quality gate. It
is **not** a fairness target: there is no interpretable disparity to remove. It is here for a
different reason, and the distinction should survive into the paper.

| Model | Lang | Utility | Discrimination | Overall AR | Reference AR | Primary outcome |
|---|---|---:|---:|---:|---:|---|
| LAPA-12B | en | 50.9% | 24.3 pp | 81.5% | 33.3% | **utility** |
| LAPA-12B | uk | 54.1% | 26.8 pp | 78.6% | 35.1% | **utility** |

**Scope of the rescue arm:**

| | |
|---|---|
| Trained on | **all three protected groups, both languages** — one run, `configs/data/all_groups.yaml` |
| No military-only ablation | The fault is where the model puts its decision threshold, and that is not group-specific. Narrowing the training data would shrink the set for no gain, which is why Track A's ablation is deliberately absent here |
| Evaluated on | the full grid: all groups + both intersections, English and Ukrainian, all three conditions |
| Runs | 1 SFT + 2 preference (DPO, ORPO) + 6 prompt probes (3 anchoring strategies × 2 languages) |

Both languages are trained together and evaluated separately. LAPA fails the gate in **both**
(50.9% en, 54.1% uk), so a Ukrainian-only rescue would leave half the model's problem
unmeasured — and the English half is the more surprising one, since a Ukrainian-native model
underperforming general models *in English* is the less remarkable half of the observation.

The diagnosis (Finding 6): LAPA **ranks** candidates — it accepts ~25 points more of the pairs
an attribute-free reference would hire than of those it would reject — but puts its threshold
in the wrong place, accepting roughly 80% against a reference rate of ~34%.

**Which families can act on that, and which cannot:**

| Family | Included | Why |
|---|:--:|---|
| **SFT** | ✅ primary | The training targets pin every verdict to the attribute-free reference decision. That *is* a threshold-calibration signal — the one intervention aimed squarely at this fault. All groups, both languages, one run |
| **DPO / ORPO** | ✅ | Preference pairs sharpen the decision boundary the SFT stage moved. Also gives the same SFT-vs-SFT+preference ablation as Track A, on a model that starts from a very different place |
| Prompt | ⚪ cheap probe | Only `structured_rubric`, `recruiter_guidelines`, `zero_shot_cot` — the three that impose an explicit decision procedure with a numeric threshold. Free, and if a rubric alone fixes the calibration that is worth knowing before spending a training run |
| Scrub | ❌ | Removes attribute information. Does nothing for decisiveness |
| Embedding | ❌ | Erases an attribute direction from the residual stream. The fault is not that LAPA over-attends to an attribute — it is that it accepts everyone |

**Success criteria, fixed in advance.**

- **Primary:** utility ≥ 60% and acceptance rate moving toward the reference rate. This is the
  publishable outcome either way.
- **Secondary, conditional on the primary:** whether the disparity that then becomes
  measurable is larger or smaller than the general-purpose models show in Ukrainian. **If the
  primary criterion is not met, no fairness number from LAPA may be reported** — recorded now
  so it cannot be decided after seeing the results.

**Why it is worth the GPU time.** LAPA is the only Ukrainian-native model in the study. If the
rescue works, the paper gains its only Ukrainian-native data point on Ukrainian hiring bias —
otherwise a real gap in a paper about exactly that. If it does not, *"a Ukrainian-native model
could not be tuned to threshold this task while general-purpose models handle it out of the
box"* is a publishable negative result about Ukrainian-language model development, and it
costs the same two training runs to find out.

---

## Threats to validity

1. **Single sampled run per condition.** Greedy decoding removes sampling variance rather than
   measuring it, so our inconsistency rates are not directly comparable to the audit study's,
   which sampled.
2. **The language reversal is unexplained.** Cultural valence and translation non-equivalence
   both fit the data. A paraphrase-controlled condition would separate them; we did not run
   one.
3. **The Ukrainian gender condition is weaker than the audit study's.** The public Djinni
   mirrors lack the morphological gender-agreement columns the gated mirrors carry, so
   Ukrainian gender injection here is the labelled field and the first-person sentence only.
   Military status and religion are unaffected.
4. **Feedback similarity remains unvalidated** against human judgement — the audit study's
   reporting requirement 7, still open. No finding above rests on it.
5. **Attribute-mediated bias only.** Every condition injects an attribute. Rao et al. (2025)
   find bias entering through writing style with no attribute present, which nothing here
   measures.
6. **The manual-review queue has not been worked.** Language drift, fuzzy decision mapping and
   canned rationales invalidate the numbers computed on top of them. The Ukrainian LAPA run
   raised 18 language-drift rows; the interpretable runs raised none, but the leakage and
   degenerate-feedback queues remain unread.
7. **Attribute leakage is detected by prefix matching.** Rates are reported; individual hits
   need reading before quoting.
