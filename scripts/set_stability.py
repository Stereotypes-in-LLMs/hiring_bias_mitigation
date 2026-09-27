"""CLI for `hiring_bias_mitigation.analysis.stability` -- see that module for the method.

    python scripts/set_stability.py

Writes `reports/set_stability.csv`, `reports/set_stability_by_group.csv` and the decision
table `reports/mitigation_decision_table.csv` -- stability joined with the utility change and
the parse-failure rate, which is what `scripts/make_figures.py` plots. The decision table is
built here rather than by hand: a hand-maintained copy went stale and silently dropped every
training arm from the paper's headline figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from hiring_bias_mitigation.analysis.stability import stability_table  # noqa: E402
from hiring_bias_mitigation.eval.report import (  # noqa: E402
    _family,
    _model_short,
    _restricted_metrics,
    load_records,
)
from hiring_bias_mitigation.utils.config import REPO_ROOT, resolve_output_path  # noqa: E402


def decision_table(stability: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    """Stability + the utility change on the same cells + the run's parse-failure rate."""
    # stability_table names an arm exactly as the run file does: model--lang--family--variant.
    by_name = {r["run_name"]: r for r in records}
    baselines = {(_model_short(r), r["meta"]["lang"]): r
                 for r in records if _family(r) in ("none", "baseline")}
    rows = []
    for r in stability.itertuples():
        record = by_name.get(f"{r.model}--{r.lang}--{r.family}--{r.variant}")
        base = baselines.get((r.model, r.lang))
        if record is None or base is None:
            continue
        cells = set(record["groups"])
        after, before = _restricted_metrics(record, cells), _restricted_metrics(base, cells)
        utility = after.get("reference_agreement"), before.get("reference_agreement")
        rows.append({
            "model": r.model, "lang": r.lang, "family": r.family, "variant": r.variant,
            "delta_unstable_pp": round(r.delta_pp, 1), "ci_low": round(r.ci_low, 1),
            "ci_high": round(r.ci_high, 1), "p_fdr": r.p_fdr,
            "delta_utility_pp": (round(100 * (utility[0] - utility[1]), 1)
                                 if None not in utility else None),
            "parse_fail_pct": round(100 * (record["summary"].get("parse_failure_rate") or 0), 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    records = load_records(REPO_ROOT / "eval" / "results")
    usable = {r["run_name"] for r in records}
    raw = resolve_output_path("outputs/raw")
    pd.set_option("display.width", 240)
    for by_group, name in ((False, "set_stability.csv"), (True, "set_stability_by_group.csv")):
        df = stability_table(raw, usable, by_group=by_group)
        out = REPO_ROOT / "reports" / name
        df.to_csv(out, index=False)
        show = df.copy()
        for c in ("unstable_base_pct", "unstable_run_pct", "delta_pp", "ci_low", "ci_high"):
            show[c] = show[c].round(1)
        for c in ("p_sign", "p_fdr"):
            show[c] = show[c].map(lambda v: f"{v:.1e}")
        cols = ["model", "lang", "family", "variant", "group", "sets", "variants_per_set",
                "unstable_base_pct", "unstable_run_pct", "delta_pp", "ci_low", "ci_high",
                "fixed", "broken", "p_fdr"]
        print(show[cols].to_string(index=False))
        print(f"\nwrote {out}\n")
        if not by_group:
            table = decision_table(df, records)
            path = REPO_ROOT / "reports" / "mitigation_decision_table.csv"
            table.to_csv(path, index=False)
            print(f"wrote {path} ({len(table)} arms)\n")


if __name__ == "__main__":
    main()
