"""Assembling the SFT and preference datasets, and writing them as publishable artifacts.

Two artifacts come out of one generation run:

`sft`     -- (prompt, completion) pairs. The prompt is the *audit's own* screening prompt,
             byte-identical to what the model will see at evaluation time; the completion is
             the attribute-invariant JSON response. Training on the evaluation-time prompt
             format is deliberate: a student fine-tuned on a different framing has to
             generalise across formats at test time, and any failure to do so would be
             misread as the mitigation not working.

`dpo`     -- (prompt, chosen, rejected) triples over the same prompts, chosen = invariant,
             rejected = the attribute-driven response. Matched on job, CV, attribute and
             condition, so the only thing the preference model can learn from the contrast is
             whether the attribute was allowed to count.

Both are written to parquet with a dataset card, and `scripts/push_dataset_to_hub.py`
publishes them. Publishing is opt-in and never automatic.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from ..data.benchmark import assert_no_leakage
from ..data.prompts import build_prompt
from ..utils.logging import get_logger

log = get_logger(__name__)


def _audit_prompt(row: dict) -> str:
    """Renders the evaluation-time prompt for a training row."""
    return build_prompt(
        {
            "lang": row["lang"],
            "cv": row["cv"],
            "job_description": row["job_description"],
            "condition": row["condition"],
            "protected_group_label": row["group_label"],
            "protected_attr": row["protected_attr"],
            "implicit_injection": row.get("implicit_injection", ""),
        },
        strategy="baseline",
    )


#: The decision word each language's prompt asks for. Generation stores the *canonical*
#: decision ("hire"/"reject") so the analysis is language-agnostic, but a training target must
#: be written in the language the prompt demands. Writing the canonical word into a Ukrainian
#: target teaches the model to answer `{"decision": "reject"}` where the prompt says
#: `найняти або відхилити` -- which is what the first Ukrainian SFT adapters learned to do.
DECISION_WORDS = {
    "en": {"hire": "hire", "reject": "reject"},
    "uk": {"hire": "найняти", "reject": "відхилити"},
}


def _decision_word(decision: str, lang: str) -> str:
    return DECISION_WORDS.get(lang, DECISION_WORDS["en"]).get(
        str(decision).strip().lower(), str(decision)
    )


def _completion(decision: str, feedback: str, lang: str = "en") -> str:
    return json.dumps(
        {"decision": _decision_word(decision, lang), "feedback": feedback}, ensure_ascii=False
    )


def build_sft_dataset(invariant: pd.DataFrame) -> pd.DataFrame:
    """(prompt, completion) rows from the filtered invariant generations."""
    if invariant.empty:
        return pd.DataFrame(
            columns=["prompt", "completion", "lang", "protected_group", "protected_attr"]
        )
    rows = []
    for row in _with_implicit(invariant).to_dict("records"):
        rows.append(
            {
                "prompt": _audit_prompt(row),
                "completion": _completion(row["chosen_decision"], row["chosen_feedback"],
                                          row["lang"]),
                "decision": row["chosen_decision"],
                "lang": row["lang"],
                "protected_group": row["protected_group"],
                "protected_attr": row["protected_attr"],
                "condition": row["condition"],
                "candidate_id": row["candidate_id"],
                "job_id": row["job_id"],
                "pair_id": row["pair_id"],
            }
        )
    out = pd.DataFrame(rows)
    assert_no_leakage(out, "generation.dataset.build_sft_dataset")
    return out


def build_dpo_dataset(invariant: pd.DataFrame, biased: pd.DataFrame) -> pd.DataFrame:
    """(prompt, chosen, rejected) triples, inner-joined on the exact variant.

    The join key is the full variant identity -- pair, group, attribute, condition -- so a
    triple only exists where both sides survived their filters for the *same* profile. An
    outer join with backfilled negatives would let a rejected response drift onto a different
    CV, and the preference signal would then be about CV quality.
    """
    if invariant.empty or biased.empty:
        return pd.DataFrame(columns=["prompt", "chosen", "rejected", "lang"])

    key = ["pair_id", "protected_group", "protected_attr", "condition"]
    rejected_cols = [
        "rejected_decision", "rejected_feedback",
        "rejected_flipped_decision", "rejected_mentions_attribute",
    ]
    merged = invariant.merge(
        biased[[*key, *rejected_cols]],
        on=key,
        how="inner",
    )
    rows = []
    for row in _with_implicit(merged).to_dict("records"):
        rows.append(
            {
                "prompt": _audit_prompt(row),
                "chosen": _completion(row["chosen_decision"], row["chosen_feedback"],
                                      row["lang"]),
                "rejected": _completion(row["rejected_decision"], row["rejected_feedback"],
                                        row["lang"]),
                "lang": row["lang"],
                "protected_group": row["protected_group"],
                "protected_attr": row["protected_attr"],
                "condition": row["condition"],
                "rejected_flipped_decision": row["rejected_flipped_decision"],
                "rejected_mentions_attribute": row["rejected_mentions_attribute"],
                "candidate_id": row["candidate_id"],
                "job_id": row["job_id"],
                "pair_id": row["pair_id"],
            }
        )
    out = pd.DataFrame(rows)
    assert_no_leakage(out, "generation.dataset.build_dpo_dataset")
    return out


def _with_implicit(df: pd.DataFrame) -> pd.DataFrame:
    """Restores the implicit sentence so `build_prompt` can render an implicit row."""
    from ..data.injection import implicit_sentence

    out = df.copy()
    out["implicit_injection"] = [
        implicit_sentence(group, attr, lang) if condition == "implicit" else ""
        for group, attr, lang, condition in zip(
            out["protected_group"], out["protected_attr"], out["lang"], out["condition"]
        )
    ]
    return out


def build_kto_dataset(dpo: pd.DataFrame) -> pd.DataFrame:
    """Unpairs the DPO triples into KTO's (prompt, completion, label) rows.

    KTO scores each completion on its own rather than against a partner, so one preference
    pair becomes two rows: the invariant response labelled desirable, the biased one
    undesirable. Derived from the DPO frame rather than generated separately, so both methods
    see exactly the same responses and a difference between them is the objective, not the
    data.

    The result is balanced by construction -- one desirable and one undesirable per pair --
    which is what KTO's desirable_weight/undesirable_weight would otherwise have to correct.
    """
    if dpo.empty:
        return pd.DataFrame(columns=["prompt", "completion", "label", "lang"])

    carry = [
        c for c in ("lang", "protected_group", "protected_attr", "condition",
                    "candidate_id", "job_id", "pair_id")
        if c in dpo.columns
    ]
    frames = []
    for column, label in (("chosen", True), ("rejected", False)):
        part = dpo[["prompt", column, *carry]].rename(columns={column: "completion"})
        part = part.assign(label=label)
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    # Interleave rather than concatenate: a run that stops early should have seen both labels.
    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)


def train_val_split(df: pd.DataFrame, val_fraction: float = 0.05, seed: int = 42):
    """Splits by candidate, not by row.

    Every attribute variant of one CV is a near-duplicate of the others. Splitting by row puts
    variants of the same CV on both sides, and the validation loss then measures memorisation.
    """
    import numpy as np

    if df.empty:
        return df, df
    candidates = df["candidate_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(candidates)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_ids = set(shuffled[:n_val])
    is_val = df["candidate_id"].isin(val_ids)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


DATASET_CARD = """---
license: mit
language:
{language_block}
task_categories:
- text-generation
tags:
- fairness
- bias-mitigation
- hiring
- recruitment
- ukrainian
size_categories:
- {size_category}
---

# {name}

Semi-synthetic training data for mitigating hiring bias in open-weight LLMs, in English and
Ukrainian.

**Real inputs, synthetic labels.** Job descriptions and anonymised candidate profiles are
real postings from the [Djinni Recruitment Dataset](https://huggingface.co/collections/lang-uk/djinni-recruitment-dataset-665acf5eb9fcbdc54101c342)
(MIT). The hiring decisions and written rationales are generated by `{teacher_model}`.

## Contamination control

Every candidate and every job appearing in the evaluation benchmark is excluded from the
source pool before matching, and the exclusion is asserted again before these files are
written. The benchmark holds out {n_holdout_candidates} candidate ids and {n_holdout_jobs}
job ids. Candidates already mentioning a protected characteristic in their own CV text are
also removed, so that an injected counterfactual never contradicts the profile it is
injected into.

## How each example is built

For a real job-CV pair, the teacher first decides on the bare pair with no attribute present.
That verdict is the anchor. Protected attributes are then injected -- as an explicit labelled
field, and as a first-person sentence inside the CV -- and the teacher writes the response a
fair screener would give: **the same verdict as the anchor**, with a rationale grounded only
in professional evidence and never referring to the attribute. Those are the `chosen`
responses. For the preference split, the teacher additionally writes the response a screener
would give if the attribute were allowed to drive the outcome; those are `rejected`.

The result is counterfactually consistent by construction: for one profile, every attribute
variant carries the same decision.

## Filtering

Generations are dropped, never repaired. A variant is discarded if it is unparsable, if its
verdict drifts from the anchor, if the rationale names the attribute, if it editorialises
about the attribute's irrelevance, if the length is out of bounds, or if it is written in the
wrong language. A `rejected` response that is neither verdict-flipped nor attribute-naming is
also discarded -- it would not be a negative.

Yield and per-reason drop counts:

```json
{filter_report}
```

## Splits

{splits_block}

## Protected groups

{groups_block}

## Intended use and limits

Built for research on bias mitigation in AI-assisted hiring. It is **not** a hiring-decision
dataset: the verdicts are a language model's opinions on unlabelled data, not hiring outcomes,
and the corpus carries no ground truth about candidate suitability. Fine-tuning on it teaches
a model to hold its verdict and its rationale invariant to a protected attribute; it does not
teach the model to screen well, and it cannot remove bias that enters through writing style
rather than through an attribute (Rao et al. 2025).

## Provenance

- Source corpus: Djinni Recruitment Dataset (Drushchak & Romanyshyn, 2024), MIT.
- Attribute lists and injection templates: released with the audit study this work extends,
  reused verbatim so mitigated and unmitigated results stay comparable.
- Generator: `{teacher_model}`, run locally. Decoding configuration: `{generation}`.
- Generated: {generated_on}.
"""


def write_dataset_card(
    path: str | Path,
    name: str,
    teacher_model: str,
    generation: dict,
    filter_report: dict,
    splits: dict,
    groups: list[str],
    languages: list[str],
    n_rows: int,
) -> Path:
    from ..data.benchmark import holdout_ids

    holdout = holdout_ids()
    size_category = (
        "n<1K" if n_rows < 1_000 else "1K<n<10K" if n_rows < 10_000
        else "10K<n<100K" if n_rows < 100_000 else "100K<n<1M"
    )
    text = DATASET_CARD.format(
        name=name,
        language_block="\n".join(f"- {lang}" for lang in languages),
        size_category=size_category,
        teacher_model=teacher_model,
        n_holdout_candidates=len(holdout.candidate_ids),
        n_holdout_jobs=len(holdout.job_ids),
        filter_report=json.dumps(filter_report, indent=2, ensure_ascii=False),
        splits_block="\n".join(f"- `{k}`: {v:,} rows" for k, v in splits.items()),
        groups_block="\n".join(f"- `{g}`" for g in groups),
        generation=json.dumps(generation, ensure_ascii=False),
        generated_on=date.today().isoformat(),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
