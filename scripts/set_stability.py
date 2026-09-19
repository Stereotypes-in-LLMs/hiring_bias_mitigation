"""CLI for `hiring_bias_mitigation.analysis.stability` -- see that module for the method.

    python scripts/set_stability.py            # -> reports/set_stability.csv + stdout table
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from hiring_bias_mitigation.analysis.stability import stability_table  # noqa: E402
from hiring_bias_mitigation.eval.report import load_records  # noqa: E402
from hiring_bias_mitigation.utils.config import REPO_ROOT, resolve_output_path  # noqa: E402


def main() -> None:
    usable = {r["run_name"] for r in load_records(REPO_ROOT / "eval" / "results")}
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


if __name__ == "__main__":
    main()
