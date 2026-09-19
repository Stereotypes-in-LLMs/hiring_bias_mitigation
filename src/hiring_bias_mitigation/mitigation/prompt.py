"""Prompt-based mitigation.

The strategies themselves live in `data/prompts.py`, next to the baseline template they are
variations of. This module is the thin part: validating a configured strategy and describing
it for the report, plus the two-call verifier's control flow.

Provenance matters for the write-up. Four strategies (`ignore_personal_info`,
`zero_shot_cot`, `recruiter_guidelines`, `reasoning`) and the two-call verifier are
reproduced from the audit study's repository, so they are prior art here and must be reported
as reproductions, not contributions. Three (`fairness_constitution`,
`counterfactual_invariance`, `structured_rubric`) are new in this work.
"""

from __future__ import annotations

from ..data.prompts import STRATEGIES, STRATEGY_BODIES  # noqa: F401  (re-exported)

REPRODUCED = (
    "ignore_personal_info",
    "zero_shot_cot",
    "recruiter_guidelines",
    "reasoning",
    "second_pass_verification",
)
NEW_IN_THIS_WORK = (
    "fairness_constitution",
    "counterfactual_invariance",
    "structured_rubric",
)

DESCRIPTIONS = {
    "baseline": "Unmitigated screening prompt (audit study, Appendix A). Not a mitigation.",
    "ignore_personal_info": "Instructs the model to treat personal information as irrelevant.",
    "zero_shot_cot": "Four-step chain of thought before the verdict.",
    "recruiter_guidelines": "Structured recruiter checklist derived from screening guidance.",
    "reasoning": "Asks for an explicit justification of a 'fair and unbiased' decision.",
    "second_pass_verification": (
        "Two calls: decide, then audit the decision for reliance on non-professional "
        "information and revise it if needed."
    ),
    "fairness_constitution": (
        "Five explicit rules constraining the decision procedure, naming the inadmissible "
        "characteristics and forbidding inference from indirect cues."
    ),
    "counterfactual_invariance": (
        "Instructs the model to verify its decision would be unchanged under a counterfactual "
        "profile, targeting the failure mode the inconsistency rate measures."
    ),
    "structured_rubric": (
        "Forces requirement-by-requirement scoring against quoted evidence and a numeric "
        "threshold before any verdict (numeric anchoring, after Salinas et al. 2025)."
    ),
}


def validate(strategy: str) -> str:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown prompt strategy {strategy!r}. Available: {', '.join(STRATEGIES)}"
        )
    return strategy


def describe(strategy: str) -> dict:
    return {
        "strategy": strategy,
        "description": DESCRIPTIONS.get(strategy, ""),
        "provenance": (
            "reproduced from the audit study"
            if strategy in REPRODUCED
            else "new in this work"
            if strategy in NEW_IN_THIS_WORK
            else "baseline"
        ),
        "extra_calls_per_item": 1 if strategy == "second_pass_verification" else 0,
    }
