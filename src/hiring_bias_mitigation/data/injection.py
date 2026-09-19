"""Building counterfactual evaluation sets by injecting protected attributes.

Three conditions, and the third is the one the audit paper did not have:

* `explicit`  -- the attribute is a labelled field before the CV ("Candidate's military
                 status: War veteran"). Tests whether the model attends to a flagged
                 sensitive field, which is what safety tuning plausibly targets.
* `implicit`  -- the same fact in the candidate's own words, inside the CV body ("I am a war
                 veteran and now look for a civilian job"). Tests whether the information
                 still moves the decision when it arrives as ordinary biography. Section 5.3
                 of the audit paper shows the two can give opposite verdicts.
* `attr_free` -- no attribute at all. The audit study had no such condition; without it you
                 cannot tell whether a mitigation removed a disparity or simply moved the
                 model's whole operating point (a model that rejects everyone has perfect
                 acceptance-rate parity). Every audit run generates it, and the report uses
                 it for the utility-preservation column.

One `pair_id` plus one `condition` defines a counterfactual set: identical job and CV,
differing only in the injected attribute. That is what licenses the inconsistency rate.
"""

from __future__ import annotations

from functools import cache

import pandas as pd

from ..utils.config import REPO_ROOT
from .protected_groups import INTERSECTIONS, ProtectedGroup, intersection_attributes, load_group

TEMPLATE_DIR = REPO_ROOT / "data" / "injection_templates"

ATTR_FREE = "__attr_free__"
CONDITIONS = ("explicit", "implicit", "attr_free")

#: The attribute lists and the implicit templates come from two different upstream
#: repositories and drifted on two names. The attribute list is canonical -- it is what the
#: explicit prompt prints -- so these map a canonical name onto the template's spelling.
#: Documented in data/PROVENANCE.md; do not "fix" the vendored CSVs, or a re-fetch reverts it.
TEMPLATE_ALIASES: dict[str, dict[tuple[str, str], str]] = {
    "en": {("marital_status", "Divorced"): "Divorced (Divorced)"},
    "uk": {("gender", "Гендерне невідповідність"): "Гендерно нонконфорфмний"},
}


@cache
def implicit_templates(lang: str) -> dict[tuple[str, str], str]:
    """(group, attribute) -> first-person sentence, from the released templates."""
    path = TEMPLATE_DIR / f"implicit_{lang}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing implicit-injection templates: {path}")
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    group_col = cols["protected_group"]
    attr_col = cols["protected_attr"]
    sent_col = cols["injection sentence"]
    # The released CSVs name groups with spaces ("military status"); this codebase uses
    # underscores throughout. Normalise on read rather than editing the vendored file, so a
    # re-fetch from upstream stays a straight copy.
    return {
        (str(r[group_col]).strip().replace(" ", "_"), str(r[attr_col]).strip()): str(
            r[sent_col]
        ).strip()
        for _, r in df.iterrows()
    }


def implicit_sentence(group: str, attr: str, lang: str) -> str:
    """The first-person sentence for one attribute, joining an intersection with a space."""
    templates = implicit_templates(lang)
    if group in INTERSECTIONS:
        parts = []
        for sub_group, sub_attr in zip(INTERSECTIONS[group], attr.split(" | ")):
            parts.append(_one_sentence(templates, sub_group, sub_attr, lang))
        return " ".join(parts)
    return _one_sentence(templates, group, attr, lang)


def _one_sentence(templates: dict[tuple[str, str], str], group: str, attr: str, lang: str) -> str:
    attr = TEMPLATE_ALIASES.get(lang, {}).get((group, attr), attr)
    try:
        return templates[(group, attr)]
    except KeyError as exc:
        raise KeyError(
            f"no implicit template for ({group!r}, {attr!r}) in {lang!r}. The released "
            f"templates cover the released attribute lists; if you added an attribute, add "
            f"its sentence to {TEMPLATE_DIR / f'implicit_{lang}.csv'} too."
        ) from exc


def group_spec(name: str, lang: str, max_intersection_cells: int | None = None):
    """Returns (label, [attribute values]) for a plain group or an intersection.

    Intersection attribute values are joined with ' | ' so one string still identifies a
    cell, and the label joins the two group labels the same way -- the explicit prompt then
    reads "Candidate's military status | gender: War veteran | Female".
    """
    if name in INTERSECTIONS:
        subs: list[ProtectedGroup] = [load_group(g, lang) for g in INTERSECTIONS[name]]
        label = " | ".join(s.label for s in subs)
        cells = intersection_attributes(name, lang, max_cells=max_intersection_cells)
        return label, [" | ".join(c) for c in cells]
    g = load_group(name, lang)
    return g.label, list(g.attributes)


def build_eval_set(
    pairs: pd.DataFrame,
    protected_groups: list[str],
    lang: str,
    conditions: tuple[str, ...] = ("explicit", "implicit", "attr_free"),
    max_intersection_cells: int | None = None,
) -> pd.DataFrame:
    """Expands job-CV pairs into the full counterfactual grid.

    Rows = pairs x groups x attributes x conditions (+ one attr_free row per pair, which is
    shared across groups rather than repeated per group -- it carries no attribute, so
    generating it once per pair is both correct and 39x cheaper).
    """
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(f"unknown condition(s) {unknown}; expected {CONDITIONS}")

    rows: list[dict] = []
    injected_conditions = [c for c in conditions if c != "attr_free"]

    for group_name in protected_groups:
        label, attrs = group_spec(group_name, lang, max_intersection_cells)
        for condition in injected_conditions:
            for pair in pairs.to_dict("records"):
                for attr in attrs:
                    rows.append(
                        {
                            **_base_row(pair, lang),
                            "protected_group": group_name,
                            "protected_group_label": label,
                            "protected_attr": attr,
                            "condition": condition,
                            "implicit_injection": (
                                implicit_sentence(group_name, attr, lang)
                                if condition == "implicit"
                                else ""
                            ),
                        }
                    )

    if "attr_free" in conditions:
        for pair in pairs.to_dict("records"):
            rows.append(
                {
                    **_base_row(pair, lang),
                    "protected_group": ATTR_FREE,
                    "protected_group_label": "",
                    "protected_attr": ATTR_FREE,
                    "condition": "attr_free",
                    "implicit_injection": "",
                }
            )

    df = pd.DataFrame(rows)
    # group_id is the counterfactual set key: everything sharing it differs only in the
    # attribute, which is exactly the comparison the inconsistency rate is defined over.
    df["group_id"] = df["pair_id"] + "::" + df["protected_group"] + "::" + df["condition"]
    return df.reset_index(drop=True)


def _base_row(pair: dict, lang: str) -> dict:
    return {
        "pair_id": pair["pair_id"],
        "candidate_id": pair["candidate_id"],
        "job_id": pair["job_id"],
        "cv": pair["cv"],
        "job_description": pair["job_description"],
        "job_position": pair.get("job_position", ""),
        "lang": lang,
        "reference_decision": pair.get("reference_decision", ""),
        "reference_feedback": pair.get("reference_feedback", ""),
    }
