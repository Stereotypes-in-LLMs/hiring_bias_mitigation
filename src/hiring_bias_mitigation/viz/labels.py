"""Figure text in both languages.

One table, two columns. A figure builder never writes a literal string: it asks for a key, so
an English figure and its Ukrainian twin cannot drift apart when one of them is edited. A
missing key raises rather than silently falling back to English -- a half-translated figure in
a paper is worse than an obviously missing one.

Attribute names are deliberately **not** here. They come from the benchmark already in the
right language (`data/attributes/<group>_<lang>.txt`), and re-translating them in the figure
layer would let a plot disagree with the table beside it.
"""

from __future__ import annotations

LANGUAGES = ("en", "uk")

#: key -> {lang: text}
STRINGS: dict[str, dict[str, str]] = {
    # --- axes and units ---
    "ar": {"en": "Acceptance rate (%)", "uk": "Частка схвалень (%)"},
    "mad": {"en": "Disparity (MAD, pp)", "uk": "Диспаритет (MAD, в.п.)"},
    "mad_delta": {"en": "Change in disparity (pp)", "uk": "Зміна диспаритету (в.п.)"},
    "utility": {"en": "Utility (reference agreement, %)",
                "uk": "Корисність (збіг з еталоном, %)"},
    "utility_delta": {"en": "Change in utility (pp)", "uk": "Зміна корисності (в.п.)"},
    "leak": {"en": "Attribute mentions in rationale (%)",
             "uk": "Згадки атрибута в обґрунтуванні (%)"},
    "gap": {"en": "Gap vs reference level (pp)", "uk": "Розрив до еталонного рівня (в.п.)"},
    "inconsistency": {"en": "Inconsistency rate (%)", "uk": "Частка неузгодженості (%)"},
    "cohens_h": {"en": "Cohen's h", "uk": "Cohen's h"},
    "strategy": {"en": "Mitigation", "uk": "Мітигація"},
    "model": {"en": "Model", "uk": "Модель"},
    "attribute": {"en": "Attribute", "uk": "Атрибут"},

    # --- figure titles ---
    "fig_baseline_title": {
        "en": "Baseline disparity by model, language and protected group",
        "uk": "Базовий диспаритет за моделлю, мовою та захищеною групою",
    },
    "fig_tradeoff_title": {
        "en": "Fairness gain against utility cost",
        "uk": "Виграш у справедливості проти втрати корисності",
    },
    "fig_strategy_title": {
        "en": "Disparity after each prompt-based mitigation",
        "uk": "Диспаритет після кожної промптової мітигації",
    },
    "fig_leak_title": {
        "en": "Residual attribute mentions predict residual disparity",
        "uk": "Залишкові згадки атрибута передбачають залишковий диспаритет",
    },
    "fig_attribute_title": {
        "en": "Acceptance rate by attribute, with 95% bootstrap intervals",
        "uk": "Частка схвалень за атрибутом, з 95% бутстреп-інтервалами",
    },
    "fig_language_title": {
        "en": "The same mitigation in English and Ukrainian",
        "uk": "Та сама мітигація англійською та українською",
    },
    "fig_stability_title": {
        "en": "Counterfactual consistency gained against utility lost",
        "uk": "Виграш у контрфактичній узгодженості проти втрати корисності",
    },
    "stability_delta": {
        "en": "Change in unstable sets (pp)",
        "uk": "Зміна частки нестабільних наборів (в.п.)",
    },
    "stability_hint": {
        "en": "Left: fewer sets where the attribute alone tips the decision. Up: utility kept. "
              "Hollow points are not significant after FDR correction.",
        "uk": "Ліворуч: менше наборів, де рішення перехиляє сам атрибут. Вгору: корисність "
              "збережено. Порожні точки не значущі після FDR-корекції.",
    },
    "significant_fdr": {"en": "FDR < 0.05", "uk": "FDR < 0.05"},
    "fig_condition_title": {
        "en": "Labelled field against first-person biography",
        "uk": "Позначене поле проти біографії від першої особи",
    },

    # --- annotations ---
    "baseline": {"en": "baseline", "uk": "базова модель"},
    "better": {"en": "better", "uk": "краще"},
    "worse": {"en": "worse", "uk": "гірше"},
    "no_harm": {"en": "no utility cost", "uk": "без втрати корисності"},
    "reference_level": {"en": "reference level", "uk": "еталонний рівень"},
    "explicit": {"en": "explicit", "uk": "явна"},
    "implicit": {"en": "implicit", "uk": "неявна"},
    "attr_free": {"en": "attribute-free", "uk": "без атрибута"},
    "significant": {"en": "survives FDR correction", "uk": "переживає FDR-корекцію"},
    "n_runs": {"en": "{n} runs", "uk": "{n} ранів"},
    "tradeoff_hint": {
        "en": "Left of the rule: disparity fell. Above it: utility held. "
              "Upper left is the only quadrant that is unambiguously good.",
        "uk": "Ліворуч від лінії: диспаритет знизився. Вище неї: корисність збережено. "
              "Верхній лівий квадрант — єдиний однозначно добрий.",
    },
    "leak_hint": {
        "en": "Each point is one run. Baselines sit at the right; scrubbed runs at the left.",
        "uk": "Кожна точка — один ран. Базові моделі праворуч, очищені — ліворуч.",
    },

    # --- protected groups ---
    "military_status": {"en": "Military status", "uk": "Військовий статус"},
    "gender": {"en": "Gender", "uk": "Стать"},
    "religion": {"en": "Religion", "uk": "Релігія"},
    "military_status_x_gender": {"en": "Military × Gender", "uk": "Військовий статус × Стать"},
    "military_status_x_religion": {
        "en": "Military × Religion", "uk": "Військовий статус × Релігія",
    },
}

#: Mitigation strategy names. Kept as identifiers in English -- they are the names used in the
#: configs, the report and the repository, and renaming them in a figure would break the link
#: between a plot and the run that produced it. The Ukrainian column glosses them instead.
STRATEGIES: dict[str, dict[str, str]] = {
    "ignore_personal_info": {
        "en": "ignore_personal_info",
        "uk": "ignore_personal_info\n(ігнорувати особисте)",
    },
    "zero_shot_cot": {"en": "zero_shot_cot", "uk": "zero_shot_cot\n(ланцюг міркувань)"},
    "recruiter_guidelines": {
        "en": "recruiter_guidelines",
        "uk": "recruiter_guidelines\n(інструкція рекрутера)",
    },
    "reasoning": {"en": "reasoning", "uk": "reasoning\n(міркування)"},
    "second_pass_verification": {
        "en": "second_pass_verification",
        "uk": "second_pass_verification\n(перевірка другим проходом)",
    },
    "fairness_constitution": {
        "en": "fairness_constitution",
        "uk": "fairness_constitution\n(конституція справедливості)",
    },
    "counterfactual_invariance": {
        "en": "counterfactual_invariance",
        "uk": "counterfactual_invariance\n(контрфактична інваріантність)",
    },
    "structured_rubric": {
        "en": "structured_rubric",
        "uk": "structured_rubric\n(структурована рубрика)",
    },
    "lexical": {"en": "lexical scrub", "uk": "лексичне вилучення"},
    "llm": {"en": "LLM scrub", "uk": "вилучення моделлю"},
    "leace": {"en": "LEACE", "uk": "LEACE"},
    "inlp": {"en": "INLP", "uk": "INLP"},
    "mean_diff": {"en": "mean difference", "uk": "різниця середніх"},
    "dpo": {"en": "DPO", "uk": "DPO"},
    "kto": {"en": "KTO", "uk": "KTO"},
    "sft": {"en": "SFT", "uk": "SFT"},
    "none": {"en": "baseline", "uk": "базова модель"},
}


def t(key: str, lang: str, **fmt) -> str:
    """The string for `key` in `lang`.

    Raises on a missing key or language rather than falling back: a figure that silently keeps
    an English label in the Ukrainian set is the failure this table exists to prevent.
    """
    if lang not in LANGUAGES:
        raise ValueError(f"unsupported figure language {lang!r}; expected one of {LANGUAGES}")
    table = STRINGS.get(key) or STRATEGIES.get(key)
    if table is None:
        raise KeyError(f"no figure string for {key!r}; add it to viz/labels.py")
    if lang not in table:
        raise KeyError(f"{key!r} has no {lang!r} translation")
    return table[lang].format(**fmt) if fmt else table[lang]


def strategy(name: str, lang: str) -> str:
    """A mitigation's display name, falling back to the identifier for an unknown one.

    Unknown strategies fall back on purpose: a newly added mitigation should appear in a plot
    under its config name rather than crash the figure run before anyone has translated it.
    """
    table = STRATEGIES.get(name)
    return table[lang] if table and lang in table else name
