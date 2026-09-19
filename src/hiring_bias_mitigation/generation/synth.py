"""Generating the semi-synthetic training data with a local teacher model.

"Semi-synthetic" is the accurate description: the job descriptions and CVs are real Djinni
postings and real anonymised profiles (from the half the benchmark never touched); the
hiring decisions and rationales are synthetic, produced by a teacher model on the Spark.

The construction, and why it is shaped this way
-----------------------------------------------
For each real job-CV pair the teacher is called in three modes:

1. `reference` -- decide on the bare pair, no attribute anywhere. This is the anchor. The
   decision it produces is the decision every attribute variant must also produce.

2. `invariant` -- for each attribute in the counterfactual set, write the rationale a fair
   screener would write for *this* profile carrying *this* attribute: same verdict as the
   reference, and a rationale that never mentions or alludes to the attribute. These are the
   SFT targets. Because the verdict is pinned to the reference, the resulting dataset is
   counterfactually consistent by construction -- which is precisely the property the
   inconsistency rate measures and the property SFT is meant to install.

3. `biased` -- the same variant, but the teacher is instructed to let the attribute drive the
   verdict. These become the `rejected` side of the preference pairs. Generating the negative
   rather than only harvesting it means every `chosen` has an exactly matched `rejected`:
   same job, same CV, same attribute, differing only in whether the attribute was allowed to
   count. That is a cleaner contrast than a real model's failure, which also differs in
   fluency, length and style -- and a preference model will happily learn those instead.

The harvested alternative is also supported (`rejected_source: model_harvested`), taking the
`rejected` side from the audited model's own biased outputs. It is more on-policy and less
clean; `both` mixes them. The trade-off is stated in the report.

Nothing here trusts the teacher. Every generation goes through `filters.py`, and a variant
whose verdict drifts from its reference, whose rationale names the attribute, or whose output
is unparsable is dropped rather than repaired.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..data.injection import implicit_sentence
from ..data.prompts import FORMAT_INSTRUCTIONS
from ..data.protected_groups import GROUP_LABELS
from ..utils.logging import get_logger

log = get_logger(__name__)

MODES = ("reference", "invariant", "biased")


# ---------------------------------------------------------------------------------------
# Teacher prompts
# ---------------------------------------------------------------------------------------

REFERENCE_PROMPT = {
    "en": (
        "You are an experienced technical recruiter screening a candidate for a role.\n\n"
        "Job description: ```\n{job_desc}\n```\n\n"
        "Candidate profile: ```\n{cv}\n```\n\n"
        "Decide whether to hire or reject this candidate for this role, judging only the "
        "match between the job's requirements and the candidate's demonstrated skills, "
        "experience and achievements. Be decisive: a genuinely weak match is a reject.\n"
        "Return only a JSON object in this format: ```\n{format_instructions}\n```"
    ),
    "uk": (
        "Ви - досвідчений технічний рекрутер, який оцінює кандидата на посаду.\n\n"
        "Опис вакансії: ```\n{job_desc}\n```\n\n"
        "Профіль кандидата: ```\n{cv}\n```\n\n"
        "Вирішіть, найняти чи відхилити цього кандидата на цю посаду, оцінюючи виключно "
        "відповідність між вимогами вакансії та підтвердженими навичками, досвідом і "
        "досягненнями кандидата. Будьте рішучими: справді слабка відповідність - це "
        "відхилення.\n"
        "Поверніть лише JSON-об'єкт у такому форматі: ```\n{format_instructions}\n```"
    ),
}

INVARIANT_PROMPT = {
    "en": (
        "You are an experienced technical recruiter screening a candidate for a role.\n\n"
        "Job description: ```\n{job_desc}\n```\n\n"
        "Candidate profile: ```\n{profile}\n```\n\n"
        "A prior review of this candidate's professional evidence reached the verdict "
        "'{reference_decision}' with the reasoning: \"{reference_feedback}\"\n\n"
        "The profile above additionally states the candidate's {group_label}. That "
        "characteristic is legally protected and professionally irrelevant: it must not "
        "change the verdict, and it must not appear in the feedback.\n\n"
        "Write the response a fair screener would give for this profile:\n"
        "- decision: exactly '{reference_decision}'\n"
        "- feedback: at most 30 words, grounded only in the candidate's skills, experience "
        "and achievements against the job's requirements. Do not mention, hint at, or "
        "allude to the {group_label}. Do not say the characteristic was ignored -- write as "
        "though it were never there.\n"
        "Return only a JSON object in this format: ```\n{format_instructions}\n```"
    ),
    "uk": (
        "Ви - досвідчений технічний рекрутер, який оцінює кандидата на посаду.\n\n"
        "Опис вакансії: ```\n{job_desc}\n```\n\n"
        "Профіль кандидата: ```\n{profile}\n```\n\n"
        "Попередній розгляд професійних даних цього кандидата дав вердикт "
        "'{reference_decision}' з обґрунтуванням: \"{reference_feedback}\"\n\n"
        "Профіль вище додатково зазначає {group_label} кандидата. Ця характеристика захищена "
        "законом і професійно нерелевантна: вона не повинна змінювати вердикт і не повинна "
        "з'являтися у відгуку.\n\n"
        "Напишіть відповідь, яку дав би неупереджений рекрутер для цього профілю:\n"
        "- decision: точно '{reference_decision}'\n"
        "- feedback: щонайбільше 30 слів, обґрунтовані лише навичками, досвідом і "
        "досягненнями кандидата щодо вимог вакансії. Не згадуйте, не натякайте і не робіть "
        "алюзій на {group_label}. Не пишіть, що характеристику проігноровано - пишіть так, "
        "ніби її ніколи не було.\n"
        "Поверніть лише JSON-об'єкт у такому форматі: ```\n{format_instructions}\n```"
    ),
}

BIASED_PROMPT = {
    "en": (
        "You are simulating a biased hiring system, to produce a negative training example "
        "for research on bias mitigation.\n\n"
        "Job description: ```\n{job_desc}\n```\n\n"
        "Candidate profile: ```\n{profile}\n```\n\n"
        "Write the response a screener would give if they allowed the candidate's "
        "{group_label} to influence the outcome: let that characteristic, rather than the "
        "professional evidence, drive the verdict, and let the feedback reflect it.\n"
        "- decision: whichever verdict that bias would produce\n"
        "- feedback: at most 30 words, reflecting the characteristic-driven reasoning\n"
        "Return only a JSON object in this format: ```\n{format_instructions}\n```"
    ),
    "uk": (
        "Ви симулюєте упереджену систему найму, щоб створити негативний навчальний приклад "
        "для дослідження з мітигації упередженості.\n\n"
        "Опис вакансії: ```\n{job_desc}\n```\n\n"
        "Профіль кандидата: ```\n{profile}\n```\n\n"
        "Напишіть відповідь, яку дав би рекрутер, що дозволив {group_label} кандидата "
        "вплинути на результат: нехай саме ця характеристика, а не професійні дані, визначає "
        "вердикт, і нехай відгук це відображає.\n"
        "- decision: той вердикт, який дала б така упередженість\n"
        "- feedback: щонайбільше 30 слів, які відображають міркування на основі характеристики\n"
        "Поверніть лише JSON-об'єкт у такому форматі: ```\n{format_instructions}\n```"
    ),
}


@dataclass
class SynthConfig:
    lang: str
    protected_groups: tuple[str, ...] = ("military_status", "gender", "religion")
    conditions: tuple[str, ...] = ("explicit", "implicit")
    #: Attributes sampled per pair, rather than the full counterfactual set. The full set for
    #: three groups is 34 attributes x 2 conditions = 68 generations per pair, which at 1500
    #: pairs is 100k+ calls per language. Sampling keeps every attribute well represented
    #: across the dataset while keeping per-pair cost linear in the number of groups.
    attributes_per_pair: int = 2
    rejected_source: str = "teacher_biased"  # teacher_biased | model_harvested | both
    seed: int = 42
    _rng: object = field(default=None, repr=False)

    def rng(self):
        if self._rng is None:
            import numpy as np

            self._rng = np.random.default_rng(self.seed)
        return self._rng


def _format_instructions(lang: str) -> str:
    import json

    return json.dumps(FORMAT_INSTRUCTIONS[lang], ensure_ascii=False)


def build_profile(cv: str, group: str, attr: str, condition: str, lang: str) -> str:
    """The candidate profile as the teacher sees it, with the attribute injected."""
    if condition == "explicit":
        label = GROUP_LABELS[group][lang]
        line = f"{label} кандидата: {attr}" if lang == "uk" else f"Candidate's {label}: {attr}"
        return f"{line}\n{cv}"
    return f"{implicit_sentence(group, attr, lang)}\n{cv}"


def reference_prompts(pairs: pd.DataFrame, lang: str) -> list[str]:
    fmt = _format_instructions(lang)
    return [
        REFERENCE_PROMPT[lang].format(
            job_desc=row["job_description"], cv=row["cv"], format_instructions=fmt
        )
        for row in pairs.to_dict("records")
    ]


def build_variant_frame(pairs: pd.DataFrame, cfg: SynthConfig) -> pd.DataFrame:
    """Expands each pair into the attribute variants this pair will contribute.

    Attributes are sampled per pair rather than exhausted. Across the dataset every attribute
    still appears thousands of times, which is what the training needs; exhausting the set per
    pair would only buy per-pair counterfactual completeness, and that is a property of the
    *evaluation* set, not the training set.
    """
    from ..data.protected_groups import load_group

    rng = cfg.rng()
    rows: list[dict] = []
    for row in pairs.to_dict("records"):
        for group in cfg.protected_groups:
            attributes = load_group(group, cfg.lang).attributes
            n = min(cfg.attributes_per_pair, len(attributes))
            picked = rng.choice(len(attributes), size=n, replace=False)
            for index in picked:
                attr = attributes[int(index)]
                for condition in cfg.conditions:
                    rows.append(
                        {
                            **row,
                            "protected_group": group,
                            "protected_attr": attr,
                            "condition": condition,
                            "group_label": GROUP_LABELS[group][cfg.lang],
                            "profile": build_profile(
                                row["cv"], group, attr, condition, cfg.lang
                            ),
                        }
                    )
    return pd.DataFrame(rows).reset_index(drop=True)


def invariant_prompts(variants: pd.DataFrame, lang: str) -> list[str]:
    fmt = _format_instructions(lang)
    return [
        INVARIANT_PROMPT[lang].format(
            job_desc=row["job_description"],
            profile=row["profile"],
            group_label=row["group_label"],
            reference_decision=row["reference_decision"],
            reference_feedback=row["reference_feedback"],
            format_instructions=fmt,
        )
        for row in variants.to_dict("records")
    ]


def biased_prompts(variants: pd.DataFrame, lang: str) -> list[str]:
    fmt = _format_instructions(lang)
    return [
        BIASED_PROMPT[lang].format(
            job_desc=row["job_description"],
            profile=row["profile"],
            group_label=row["group_label"],
            format_instructions=fmt,
        )
        for row in variants.to_dict("records")
    ]
