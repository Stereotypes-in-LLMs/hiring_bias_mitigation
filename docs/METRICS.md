# How to read every metric

One page for every number this study produces: what it is, how it is computed, how to read
it, and — the part usually missing — **what it does not capture**. A measure whose blind
spots are not stated cannot be argued from.

Three of the measures are reproduced from the audit study this work extends, so the numbers
stay directly comparable with its published baseline. The rest exist because a *mitigation*
study needs them and an audit does not.

- [The decision pipeline](#the-decision-pipeline) — what gets counted at all
- [Fairness measures](#fairness-measures) — AR, IR, FS
- [Utility measures](#utility-measures) — the columns that catch degenerate "fixes"
- [Diagnostic measures](#diagnostic-measures) — refusals, leakage
- [Aggregate indices](#aggregate-indices) — comparing whole experiments
- [Intersections](#intersections) — additivity
- [Statistics](#statistics) — p-values, corrections, effect sizes
- [Reading a result honestly](#reading-a-result-honestly) — the checklist
- [Deliberately absent](#deliberately-absent) — and why

---

## The decision pipeline

Every model response ends up in exactly one bucket, and the bucket decides whether the row
counts.

| `outcome` | Meaning | Counted in fairness measures? |
|---|---|---|
| `decided` | Parsed to a clean hire or reject | **yes** |
| `refused` | The model declined to decide | no |
| `invalid` | Unparsable, or a decision word that maps to nothing | no |

**This exclusion is defensible but not neutral, and it is the first thing to check.**
Denominators therefore vary by run and by attribute rather than being fixed at 450. The
alternative — silently coercing unparsable output to "reject" — would manufacture
disparities wherever refusal rates differ by attribute, which is worse. But it means a model
can improve every fairness number by learning to decline, so refusal and parse-failure rates
are reported as first-class results, not diagnostics.

`n_total` is what was asked. `n_decided` is what the measures are computed over. The gap
between them is the story.

---

## Fairness measures

### Acceptance rate (AR)

**What it is.** Share of `hire` decisions among usable rows, per protected attribute.

```
AR(attr) = hires(attr) / (hires(attr) + rejects(attr))
```

**How to read it.** The closest analogue to the selection-rate statistics used in employment
law, and the measure that captures allocative harm directly. Compare an attribute's AR to
the population AR (the permutation test does this) and to the group's reference level (the
`Gap vs ref` column — this is the number a reader quotes: *"combat participants accepted 13.8
points below civilians"*).

**What it misses.** AR is blind to *which* candidates are selected. Two attributes can share
an acceptance rate exactly while the model swaps its decisions on individual CVs — one
attribute's hires being the other's rejects. That is why AR is never read without the
inconsistency rate.

### Inconsistency rate (IR)

**What it is.** Share of decisions that differ from the majority decision over the
counterfactual set — the same CV and the same job, with only the attribute changed.

```
IR = mean over rows of  1[decision ≠ majority(its counterfactual set)]
```

A set needs at least two usable decisions to have a majority; singletons are scored `NaN`,
not 0. Scoring them 0 would dilute every inconsistency rate downward by exactly the refusal
rate.

**How to read it.** This is the measure of *individual-level instability attributable to the
attribute*: how often changing nothing but the protected characteristic flips the verdict on
an otherwise identical candidate. It catches precisely what AR misses.

**What it misses.** It is **undirected**. A high IR says the attribute moves decisions, not
which way. AR and IR must be read together:

| AR | IR | Reading |
|---|---|---|
| high | high | the attribute **privileges** the candidate — hired where others were rejected |
| low | high | the attribute **penalises** — rejected where others were hired |
| ≈population | high | the attribute is noise: it moves decisions in both directions |
| any | low | the attribute rarely changes an individual verdict |

It also conflates attribute-driven instability with ordinary decoding stochasticity. This
study decodes **greedily** by default, which removes that component — but it also means our
IR is not directly comparable to an audit that sampled. Set `n_samples > 1` with a non-zero
temperature to measure the variance instead of removing it.

### Feedback similarity (FS)

**What it is.** Cosine similarity between the sentence embedding of the model's written
rationale and the embedding of the attribute-free reference rationale for the same job–CV
pair, using a fixed multilingual encoder.

**Why it exists.** Under the EU AI Act's human-oversight requirement, a reviewer sees the
model's explanation. A model can return identical hire/reject decisions across two
counterfactual CVs while justifying them in ways that differ systematically by attribute —
and a decision-only metric is blind to exactly the channel through which bias reaches the
human in the loop.

**How to read it — carefully.** This is the **weakest** of the three measures, by the audit
study's own account, and no conclusion should rest on it alone:

1. Cosine similarity between sentence embeddings is dominated by topical and stylistic
   overlap, so it is far more sensitive to phrasing than to differential treatment. Observed
   values cluster in a narrow band and significant differences are small in absolute terms.
2. It inherits whatever biases the embedding model encodes.
3. The reference is **attribute-free by construction**, which is a defensible claim. It is
   **not unbiased** — it was generated by GPT-4o, a system that is itself biased. Never
   describe it as an unbiased reference.
4. It has **not** been validated against human judgement. That is reporting requirement 7,
   and it remains open.

Empty rationales are scored `NaN` and dropped from FS rather than being encoded as `""`.

---

## Utility measures

An audit does not need these. A mitigation study cannot do without them: **a model that
rejects every candidate has perfect acceptance-rate parity, zero inconsistency, and is
useless.** Every fairness number in the report is printed beside a utility number for that
reason.

### Reference agreement

**What it is.** Share of decisions matching the attribute-free reference decision for the
same pair.

**How to read it.** The task-fidelity axis. A mitigation that improves fairness while this
column falls has not removed bias — it has degraded the model, and the two are easy to
confuse because they look identical in every fairness column.

**What it is not.** Not accuracy. The reference is one model's attribute-free opinion on
unlabelled data, not a hiring outcome. Read it as *"still screening the way an attribute-free
screener would"*, never as *"still screening correctly"*.

### Attribute-free acceptance rate

The model's acceptance rate with no attribute present at all. The **operating point**. If a
mitigation moves this, it changed how selective the model is overall — which is a different
thing from removing a disparity, and it is invisible in any per-attribute comparison.

### Conditional acceptance rate (`ref_hire` / `ref_reject`)

**What it is.** Acceptance rate split by what the attribute-free reference decided for the
same pair.

**Why it exists.** The audit study argues, correctly, that equalised odds and equal
opportunity do not transfer here because there is no ground truth. But the design does supply
a *pseudo-label*: the reference verdict on the identical pair. Conditioning on it separates
two harms that plain AR averages together:

- **Disparity on `ref_hire` pairs** — the model treats candidates differently among those an
  attribute-free screener would have hired. This is the allocative harm anti-discrimination
  law is about.
- **Disparity on `ref_reject` pairs** — a shifted threshold among weak candidates. Real, but
  a different finding.

**How to read it.** Compare `ar_range__ref_hire` against the overall range. Where it is
substantially larger, the disparity is concentrated on the strong candidates.

**What it is not.** Not equal opportunity. The conditioning variable is a model's opinion,
not a label. Never call it that.

---

## Diagnostic measures

### Refusal rate and parse-failure rate

Reported per run and per attribute. These are **a bias signal in their own right**, not
plumbing: a model that refuses more often for one attribute is treating it differently, and
because refusals leave the denominator, that difference also silently changes every other
statistic. A large per-attribute skew is routed to manual review.

### Attribute mention rate (leakage)

**What it is.** Share of rationales that appear to name the injected protected attribute.

**How to read it.** Directly interpretable where FS is not. If the feedback for a war veteran
says *"as a veteran, you may find..."*, the attribute has entered the channel a human
reviewer reads — regardless of whether the decision changed.

**What it is not.** Detection is **prefix matching**: words are truncated to five characters
to survive Ukrainian inflection, which over-matches. Every hit is a manual-review candidate,
not a finding. Read the sampled rows before quoting a rate.

---

## Aggregate indices

For comparing whole experiments rather than attributes. **Every one is an effect size. None
is a flag count.**

> A flag count is a function of statistical power as much as of disparity: run the same model
> on twice the pairs and the flags rise while the disparity does not. Flag counts cannot rank
> two runs, and must never be used to claim a mitigation worked.

| Index | Definition | Read it as |
|---|---|---|
| `ar_mad` | Mean absolute deviation of each attribute's AR from the population AR | **The headline.** More robust than the range, which two extreme attributes define, and it does not grow with the number of attributes |
| `ar_range` | max(AR) − min(AR) across attributes | The quotable number; noisier than MAD |
| `ar_sd` | Standard deviation of AR across attributes | Spread; the audit study's suggested cheap screening statistic |
| `mean_abs_cohens_h` | Mean \|*h*\| against the group's reference level | Scale-free — stays comparable when two runs have very different base rates |
| `inconsistency_rate` | Population IR | How often the attribute alone flips a verdict |
| `ar_range__ref_hire` | AR range restricted to `ref_hire` pairs | Whether the disparity lands on strong candidates |

### Three levels of aggregation

The report gives the same indices at three granularities, and they answer different questions.

| Level | Where | Use it to |
|---|---|---|
| **Model × language × group × condition** | **Main table** (aggregate section) | everything. **This is the level to read and to reason from** |
| Per run | Roll-up, directly below the main table | rank whole experiments against each other and against baseline. Nothing else |
| Per attribute | Acceptance-rate section (the deep dive) | quote a specific gap, e.g. "combat participants accepted 13.8 points below civilians" |
| Per intersection cell | Intersections section | test additivity |

The main table and the roll-up carry the **same columns**. The roll-up is that table averaged
over groups and conditions, and averaging is exactly what destroys the information you
usually want: in this study the same run shows military status at 5.3 pp MAD and gender at
1.1 pp, and English religion goes from 1.3 pp explicit to 3.7 pp implicit. The roll-up
reports 3.1 and none of that.

**Do not argue from the per-run number about a group.** Protected groups do not behave alike
— in this study military status carries several times the disparity of religion — and the
roll-up averages that away by construction. Nor do the two injection conditions: a model can
be clean when the attribute is a labelled field and biased when the same fact arrives as
ordinary biography, and a mitigation that fixes one may not touch the other.

Fairness and quality columns appear at **every** level, on purpose. A group whose disparity
falls while its reference agreement falls with it has not been fixed, and that is invisible if
utility is only reported per run.

### How they aggregate — and why it matters

Groups are weighted **equally** within a condition; conditions are weighted equally within a
run.

The groups hold 5, 9, 20, 45 and 100 attributes. Averaging over *attributes* would let
military × gender alone determine 56% of the headline, purely because gender has twenty
values — an artefact of list length, not a finding. Equal weighting says **each protected
characteristic counts once**.

**Treat the rolled-up number as a screening statistic for ranking runs, not as a finding.**
Any claim about a specific group belongs to that group's own row.

### Cohen's *h*

```
h = 2·arcsin(√p₁) − 2·arcsin(√p₂)
```

A percentage-point gap is what a reader wants, but it is not comparable across models with
different base rates: 5 points at a 50% base rate is a much smaller effect than 5 points at
5%. Conventional reading: 0.2 small, 0.5 medium, 0.8 large.

---

## Intersections

For a cell combining military status *m* with a second attribute *g*, the **additive
prediction** from the two marginal effects is

```
predicted(m, g) = AR(m, g₀) + AR(m₀, g) − AR(m₀, g₀)
```

where subscript zero marks each group's reference level. The **interaction** is what the
observed rate does on top of that:

```
interaction = observed(m, g) − predicted(m, g)
```

**How to read it.** Negative means the combination is treated **worse** than either effect
alone would predict — compounding disadvantage. Positive means the reverse.

**Read the sign distribution, not the top ten.** The report lists the ten largest deviations
by magnitude, and that list can be one-sided by selection alone. The `N negative / N positive`
count across all testable cells is the number that supports a claim.

**Two cautions.** These are differences of noisy per-cell rates, so a per-cell interaction
carries roughly twice the sampling error of a marginal — no single cell should be quoted
alone. And the marginals come from the same run, so a cell whose reference row had few usable
decisions produces an unstable prediction; check the `n` column.

**Why the cross product must be complete.** Non-additivity lives entirely in cells where
**both** components differ from their reference. Capping `max_intersection_cells` keeps the
reference-touching cells first, so a cap of 24 on a 5 × 20 grid leaves exactly zero testable
cells. The default is `null`.

---

## Statistics

### Unpaired permutation test

The test of An et al. (2024), reproduced exactly (K = 5000, two-sided, α = 0.05): resample
subsets of the population, compare the group mean against the population mean.

Retained **for comparability** with the audit literature, not because it is optimal. It has
a known quirk — the resampled subsets are drawn from the full sample, which contains the
group — that makes it slightly conservative.

### Paired permutation test

The test this design actually licenses. Counterfactual sets are perfectly matched — same CV,
same job, one attribute changed — so the attribute's effect can be measured *within* a pair
and the between-CV variance removed. CV quality varies far more than any attribute effect, so
this is the dominant noise source, and removing it makes the paired test strictly more
powerful here.

**Prefer the paired p-value for any mitigation claim.** Mitigation effects are smaller than
the disparities they remove, so the extra power matters.

### Benjamini–Hochberg FDR correction

Applied across **every test in a run** — all measures, all groups, all attributes — because
that is the set of tests from which a claim about the model is drawn.

**Read the corrected column.** A design of this shape runs hundreds of tests; at an
uncorrected α = 0.05, a proportion of flags is expected under the null even from a perfectly
fair model. Both are printed side by side, as `raw / corrected`. Where the corrected count
collapses toward zero, the raw flags were largely multiplicity.

This is reporting requirement 1 of the audit study — which that study did not itself meet, so
its published per-attribute flags are uncorrected and should be compared with our **raw**
column, not our corrected one.

### Bootstrap confidence intervals

Percentile bootstrap (2000 resamples) on each attribute's mean, stored per attribute in the
run JSON. Reporting requirement 2: a fractional difference in embedding similarity and a
58.7-point difference in acceptance rate must not be presented with equal visual weight.

---

## Reading a result honestly

Before any number becomes a claim:

1. **Is the denominator sound?** Check `n_decided` against `n_total`. If refusals or parse
   failures are high, or skewed by attribute, everything downstream is affected.
2. **Does it survive FDR correction?** If not, it is a scrutiny prompt, not a finding.
3. **Is the effect size worth reporting?** A significant 1-point gap on 161,550 rows is
   significant because *n* is large, not because it matters.
4. **Do AR and IR agree?** Read the four-way table above. AR alone cannot tell you direction
   at the individual level.
5. **What happened to utility?** A fairness gain with a utility drop is a degradation.
6. **Does the manual-review queue touch this row?** Language drift, fuzzy decision mapping and
   canned rationales all invalidate measures computed on top of them.
7. **Does it hold across conditions?** Explicit and implicit can give opposite verdicts on the
   same model; a result in one is not a result in both.

---

## Deliberately absent

| Not computed | Why |
|---|---|
| Demographic parity difference, disparate impact ratio | Identical information to the AR gap in this design; the ratio is unstable at low base rates |
| Equalised odds, equal opportunity | Require ground-truth outcomes. Public recruitment corpora do not record whether a candidate was suitable, and hiring outcomes where they exist are themselves products of biased processes. The conditional acceptance rate is the closest defensible substitute, and is explicitly not called equal opportunity |
| Calibration, AUC, Brier score | Need probabilities. The model emits a word |
| Ranking metrics (NDCG, MRR) | Need a ranked candidate list. The task is a per-candidate verdict |
| A single composite "bias score" | Would collapse measures whose directions mean different things — a high AR is favourable, a high IR is not — and hide exactly the disagreements between measures that carry the findings |

---

## Where each number lives

| Where | What |
|---|---|
| `reports/RESULTS.md` | All of it, rendered, with the reading guidance inline |
| `eval/results/<run>.json` | Machine-readable: per-attribute rows, p-values raw and adjusted, bootstrap CIs, effect sizes, aggregate indices, manual-review queue |
| `reports/manual_review/review_queue.csv` | The rows a human still has to read, with columns to fill in |
| `$HBM_OUTPUT_ROOT/outputs/raw/<run>.parquet` | Every model response, unmodified, so any of this can be recomputed without a GPU |
| `src/hiring_bias_mitigation/eval/metrics.py` | The definitions, in code |
| `src/hiring_bias_mitigation/eval/stats.py` | The tests, in code |
