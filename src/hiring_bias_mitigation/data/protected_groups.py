"""Protected groups, their attribute lists, and the intersections we test.

The attribute lists in `data/protected_groups/` are copied verbatim from the audit study's
repository so that a mitigation number here is directly comparable to a baseline number
there. See data/PROVENANCE.md.

Three groups are in scope for this study (military status is fixed; see README section
"Scope"), but all four lists are shipped so a run can be widened without hunting for data.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from ..utils.config import REPO_ROOT

ATTR_DIR = REPO_ROOT / "data" / "protected_groups"

LANGUAGES = ("en", "uk")

#: Groups this study evaluates and mitigates. Order is the report's column order.
GROUPS_IN_SCOPE = ("military_status", "gender", "religion")

#: Shipped but out of scope by default -- marital status was near-null in the English audit,
#: so it carries little headroom for a mitigation effect. Enable it in an audit config's
#: `protected_groups` list if you want the fourth column back.
GROUPS_AVAILABLE = ("military_status", "gender", "religion", "marital_status")

#: Intersections. Attributes are injected jointly, one value from each group. Military
#: status is the pivot in both because it carries the largest single-attribute effect in the
#: audit, so an intersection with it is where non-additivity (An et al. 2025) would show.
INTERSECTIONS: dict[str, tuple[str, ...]] = {
    "military_status_x_gender": ("military_status", "gender"),
    "military_status_x_religion": ("military_status", "religion"),
}

#: The label the prompt uses for the group ("Candidate's {label}: {value}"). Ukrainian
#: labels are the ones the audit study used, so prompts stay byte-comparable.
GROUP_LABELS: dict[str, dict[str, str]] = {
    "gender": {"en": "gender", "uk": "стать"},
    "marital_status": {"en": "marital status", "uk": "сімейний статус"},
    "military_status": {"en": "military status", "uk": "військовий статус"},
    "religion": {"en": "religion", "uk": "релігія"},
}

#: Reference level per group -- the attribute a disparity is naturally read against. Used
#: for the pairwise gap effect sizes in the report ("veteran vs civilian"), never for the
#: permutation test, which compares each attribute against the population value.
REFERENCE_ATTRS: dict[str, dict[str, str]] = {
    "military_status": {"en": "Civilian", "uk": "Цивільний"},
    "gender": {"en": "Male", "uk": "Чоловік"},
    "religion": {"en": "Christian", "uk": "християнин"},
    "marital_status": {"en": "Unmarried (Single)", "uk": "Неодружений/Неодружена"},
}


@dataclass(frozen=True)
class ProtectedGroup:
    name: str
    lang: str
    label: str
    attributes: tuple[str, ...]

    @property
    def reference(self) -> str | None:
        return REFERENCE_ATTRS.get(self.name, {}).get(self.lang)


@cache
def load_group(name: str, lang: str) -> ProtectedGroup:
    if lang not in LANGUAGES:
        raise ValueError(f"unsupported language {lang!r}; expected one of {LANGUAGES}")
    path = ATTR_DIR / f"{name}_{lang}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"no attribute list for group {name!r} in {lang!r} (looked at {path})"
        )
    attrs = tuple(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not attrs:
        raise ValueError(f"{path} is empty")
    label = GROUP_LABELS.get(name, {}).get(lang, name.replace("_", " "))
    return ProtectedGroup(name=name, lang=lang, label=label, attributes=attrs)


def load_groups(names: list[str], lang: str) -> list[ProtectedGroup]:
    return [load_group(n, lang) for n in names]


def attribute_counts(names: list[str], lang: str) -> dict[str, int]:
    """Per-group attribute counts.

    The report prints these next to every percentage. Section 3.4 of the audit paper: a
    per-group "% of attributes flagged" is comparable down a column but NOT across a row,
    because with 5 attributes one flag is 20% and with 20 attributes it is 5%.
    """
    return {n: len(load_group(n, lang).attributes) for n in names}


def intersection_attributes(
    name: str, lang: str, max_cells: int | None = None, seed: int = 42
) -> list[tuple[str, ...]]:
    """The attribute tuples for an intersection, as the cross product of its groups.

    military x gender is 5 x 20 = 100 cells, which at 450 pairs is 45,000 generations per
    model per language per injection style -- more than the rest of the audit combined. When
    `max_cells` is set, the reference level of each group is always kept and the remaining
    cells are subsampled deterministically, so the cross-group comparison stays anchored.
    """
    import itertools
    import random

    groups = [load_group(g, lang) for g in INTERSECTIONS[name]]
    cells = [tuple(c) for c in itertools.product(*(g.attributes for g in groups))]
    if max_cells is None or len(cells) <= max_cells:
        return cells

    anchored = [c for c in cells if any(a == g.reference for a, g in zip(c, groups))]
    rest = [c for c in cells if c not in set(anchored)]
    rng = random.Random(seed)
    rng.shuffle(rest)
    keep = anchored[:max_cells] + rest[: max(0, max_cells - len(anchored))]
    # Preserve the deterministic cross-product order for stable report rows.
    keep_set = set(keep)
    return [c for c in cells if c in keep_set]
