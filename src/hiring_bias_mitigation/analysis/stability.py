"""Counterfactual set stability: the most direct invariance measure in this study.

A *set* is one (candidate-job pair, injection condition) evaluated under every attribute
variant. It is **unstable** if the model's decision is not the same across those variants --
the attribute alone was enough to tip it. For each mitigation, every set is paired with the
same set at baseline, and the change is tested with an exact sign test on the discordant sets:
*fixed* (unstable -> stable) against *broken* (stable -> unstable).

Why this exists alongside MAD. The training arms moved about a hundred decisions each, and
MAD -- an acceptance-rate spread aggregated over whole attributes -- cannot register that: a
single corrected dissent shifts one attribute's rate by 1/450. A direction-of-flip test against
the teacher's verdict read the same flips as random (52 toward, 50 away). Asking instead
whether each *set* became internally consistent showed the flips were not random at all: they
made sets consistent, in whichever direction the set's own majority lay.

A set counts only if every variant was decided in BOTH runs. Dropping unparsed rows instead
makes a set of failures look perfectly stable -- one excluded run read 41.7% -> 0.3% that way.

    python scripts/set_stability.py            # -> reports/set_stability.csv + stdout table
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

HIRE = {"hire", "найняти"}
REJECT = {"reject", "відхилити"}
SET = ["pair_id", "condition"]


def _decision(value) -> str | None:
    text = str(value).strip().lower()
    return "hire" if text in HIRE else "reject" if text in REJECT else None


VARIANT = ["protected_group", "protected_attr"]


def load_rows(path: str) -> pd.DataFrame:
    """Decided/undecided rows of one run, attribute-free control excluded."""
    frame = pd.read_parquet(path, columns=[*SET, *VARIANT, "candidate_id", "decision"])
    frame = frame[frame["condition"] != "attr_free"].copy()
    frame["d"] = frame["decision"].map(_decision)
    return frame


def set_table(frame: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    """One row per set: variant count, decided count, instability, and its candidate."""
    grouped = frame.groupby(keys or SET)
    return pd.DataFrame({
        "variants": grouped.size(),
        "decided": grouped["d"].count(),
        "unstable": grouped["d"].nunique() > 1,
        "candidate_id": grouped["candidate_id"].first(),
    })


def restrict_to(base: pd.DataFrame, run: pd.DataFrame) -> pd.DataFrame:
    """The baseline rows for exactly the variants the run evaluated.

    Instability grows with the number of variants in a set: a set of 179 almost always holds
    one dissent, a set of 5 rarely does. The baseline audit covers every attribute including
    the intersections (179 per set); a mitigated run covers its target groups (34 per set, or
    5 when scoped to one group). Comparing them unmatched hands every mitigation a
    fictitious gain that grows the narrower its scope -- the same population mismatch that
    once made adapters appear to halve the leakage rate.
    """
    variants = run[VARIANT].drop_duplicates()
    return base.merge(variants, on=VARIANT, how="inner")


def compare(base_rows: pd.DataFrame, run_rows: pd.DataFrame, keys: list[str] | None = None,
            n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired set-level comparison on matched variants, with a candidate-clustered CI.

    Sets are not independent -- a candidate appears in several pairs and each pair in both
    conditions -- so the sign test alone overstates significance. The interval resamples
    *candidates* with replacement and recomputes the change in unstable share.
    """
    import numpy as np
    from scipy.stats import binomtest

    keys = keys or SET
    base = set_table(restrict_to(base_rows, run_rows), keys)
    run = set_table(run_rows, keys)
    joined = base.join(run, how="inner", lsuffix="_b", rsuffix="_r")
    complete = (
        (joined["decided_b"] == joined["variants_b"])
        & (joined["decided_r"] == joined["variants_r"])
        & (joined["variants_b"] == joined["variants_r"])
    )
    joined = joined[complete]
    if joined.empty:
        return {"sets": 0}

    fixed = int((joined["unstable_b"] & ~joined["unstable_r"]).sum())
    broken = int((~joined["unstable_b"] & joined["unstable_r"]).sum())
    p = binomtest(fixed, fixed + broken, 0.5).pvalue if fixed + broken else float("nan")
    delta = joined["unstable_r"].astype(float) - joined["unstable_b"].astype(float)

    by_candidate = delta.groupby(joined["candidate_id_b"]).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(by_candidate), size=(n_boot, len(by_candidate)))
    sums = by_candidate["sum"].to_numpy()[idx].sum(axis=1)
    counts = by_candidate["count"].to_numpy()[idx].sum(axis=1)
    boots = 100 * sums / counts
    # The same resamples give an interval for each run's own unstable share, for reporting
    # the level with its uncertainty rather than only the change.
    levels = {}
    for side in ("b", "r"):
        per = joined[f"unstable_{side}"].astype(float).groupby(joined["candidate_id_b"]).sum()
        per = per.reindex(by_candidate.index).to_numpy()
        share = 100 * per[idx].sum(axis=1) / counts
        levels[side] = (float(np.percentile(share, 2.5)), float(np.percentile(share, 97.5)))
    return {
        "sets": len(joined),
        "variants_per_set": float(joined["variants_r"].median()),
        "unstable_base_pct": 100 * joined["unstable_b"].mean(),
        "unstable_run_pct": 100 * joined["unstable_r"].mean(),
        "delta_pp": 100 * delta.mean(),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "base_ci_low": levels["b"][0],
        "base_ci_high": levels["b"][1],
        "run_ci_low": levels["r"][0],
        "run_ci_high": levels["r"][1],
        "fixed": fixed,
        "broken": broken,
        "p_sign": p,
    }


def stability_table(raw_dir: str | Path, usable: set[str], by_group: bool = False) -> pd.DataFrame:
    """Every usable mitigated run against its own usable baseline, on matched variants.

    `by_group=True` defines a set within one protected group, so instability is read per
    group -- whether the group the baseline flagged is the one being fixed.
    """
    from ..eval.stats import benjamini_hochberg

    raw_dir = Path(raw_dir)
    keys = [*SET, "protected_group"] if by_group else SET
    rows = []
    for base_path in sorted(glob.glob(str(raw_dir / "*--baseline.parquet"))):
        base_name = Path(base_path).stem
        if base_name not in usable:
            continue
        prefix = base_name.rsplit("--baseline", 1)[0]
        base_rows = load_rows(base_path)
        for run_path in sorted(glob.glob(str(raw_dir / f"{prefix}--*.parquet"))):
            name = Path(run_path).stem
            if name == base_name or "--smoke" in name or name not in usable:
                continue
            model, lang, rest = name.split("--", 2)
            family, _, variant = rest.partition("--")
            run_rows = load_rows(run_path)
            groups = sorted(run_rows["protected_group"].unique()) if by_group else [None]
            for group in groups:
                b = base_rows if group is None else base_rows[base_rows["protected_group"] == group]
                r = run_rows if group is None else run_rows[run_rows["protected_group"] == group]
                result = compare(b, r, keys)
                if not result.get("sets"):
                    continue
                rows.append({"model": model, "lang": lang, "family": family,
                             "variant": variant, "group": group or "all", **result})
    table = pd.DataFrame(rows)
    if not table.empty:
        # Every test in the table is one family of comparisons; correct across all of them.
        _, adjusted = benjamini_hochberg(table["p_sign"].tolist())
        table["p_fdr"] = adjusted
    return table
