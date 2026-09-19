"""Deep-dive analysis: where is there actually bias, and what should we mitigate?

`make_report.py` renders what was measured. This decides what the measurements support, and
produces the shortlist for the mitigation stage.

    python scripts/analyze_results.py                     # analyse, write reports/ANALYSIS.md
    python scripts/analyze_results.py --no-mcnemar        # skip the paired condition tests
    python scripts/analyze_results.py --min-utility 0.5   # loosen the quality gate
    python scripts/analyze_results.py --max-targets 6     # cap the mitigation shortlist
    python scripts/analyze_results.py --json out.json     # machine-readable, for a notebook

Outputs:

    reports/ANALYSIS.md          the argument, with its thresholds stated
    reports/mitigation_plan.yaml targets, controls, exclusions, and the configs to run
    reports/enable_planned.sh    a selection script — emitted, never applied

Reading it costs nothing and changes nothing: this script only reads `eval/results/*.json`
(and, for the paired tests, the cached generations under `outputs/raw/`). It is safe to run
while a sweep is in flight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.analysis import crosscut, evidence, render, targets  # noqa: E402
from hiring_bias_mitigation.eval.report import load_records  # noqa: E402
from hiring_bias_mitigation.utils.config import REPO_ROOT, resolve_output_path  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("analyze_results")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "eval" / "results"))
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "ANALYSIS.md"))
    parser.add_argument("--plan", default=str(REPO_ROOT / "reports" / "mitigation_plan.yaml"))
    parser.add_argument(
        "--enable-script", default=str(REPO_ROOT / "reports" / "enable_planned.sh")
    )
    parser.add_argument("--json", help="also write the full analysis as JSON")
    parser.add_argument(
        "--raw-dir", default="outputs/raw",
        help="cached generations, for the paired explicit-vs-implicit tests",
    )
    parser.add_argument(
        "--no-mcnemar", action="store_true",
        help="skip the paired condition tests (they read the raw parquet, which is slower)",
    )
    parser.add_argument(
        "--eval-scope", choices=targets.EVAL_SCOPES, default="full",
        help="how much of the benchmark grid each mitigation run evaluates. "
             "'targeted' keeps only the groups with a confirmed disparity plus the control; "
             "'targeted-no-intersections' also drops the two intersections. Writes "
             "reports/eval_scope.yaml for generate_experiment_configs.py --eval-scope.",
    )
    parser.add_argument(
        "--scope-out", default=str(REPO_ROOT / "reports" / "eval_scope.yaml")
    )
    parser.add_argument(
        "--max-targets", type=int, default=None,
        help="cap the mitigation shortlist at the N largest disparities",
    )

    group = parser.add_argument_group("thresholds — judgement calls, not statistics")
    defaults = evidence.Thresholds()
    group.add_argument("--min-utility", type=float, default=defaults.min_utility,
                       help="below this a run's disparity is not read as bias")
    group.add_argument("--material-gap", type=float, default=defaults.material_gap,
                       help="acceptance-rate gap vs the reference level, as a proportion")
    group.add_argument("--material-h", type=float, default=defaults.material_h,
                       help="Cohen's h considered material")
    group.add_argument("--material-mad", type=float, default=defaults.material_mad,
                       help="cell-level disparity worth a mitigation run")
    group.add_argument("--max-unusable", type=float, default=defaults.max_unusable,
                       help="refusals plus parse failures tolerated")
    group.add_argument("--alpha", type=float, default=defaults.alpha)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    records = load_records(args.results_dir)
    if not records:
        raise SystemExit(
            f"no run records in {args.results_dir}. Run an audit and score it first."
        )

    baselines = [
        r for r in records
        if (r.get("meta", {}).get("mitigation") or {}).get("family", "none") == "none"
    ]
    log.info("loaded %d record(s), %d baseline", len(records), len(baselines))
    if not baselines:
        raise SystemExit(
            "no baseline runs found — this analysis answers 'where is there bias to "
            "mitigate?', which is a question about the untreated models."
        )

    thresholds = evidence.Thresholds(
        min_utility=args.min_utility,
        material_gap=args.material_gap,
        material_h=args.material_h,
        material_mad=args.material_mad,
        max_unusable=args.max_unusable,
        alpha=args.alpha,
    )

    findings = evidence.grade_all(records, thresholds)
    log.info("graded %d cell(s)", len(findings))

    raw_dir = None if args.no_mcnemar else Path(resolve_output_path(args.raw_dir))
    if raw_dir is not None and not raw_dir.exists():
        log.warning("raw generations not found at %s — skipping the paired tests", raw_dir)
        raw_dir = None

    recurrences = crosscut.attribute_recurrence(findings)
    language = crosscut.language_effect(findings)
    condition = crosscut.condition_effect(findings, raw_dir)
    models = crosscut.model_ranking(findings)
    plan = targets.build_plan(findings, thresholds, args.max_targets, args.eval_scope)

    text = render.render(
        findings, recurrences, language, condition, models, plan, thresholds
    )
    path = render.write_analysis(text, args.out)
    log.info("wrote %s", path)

    plan_path = targets.write_plan(plan, args.plan)
    log.info("wrote %s", plan_path)
    script_path = targets.write_enable_script(plan, args.enable_script)
    log.info("wrote %s", script_path)

    import yaml

    scope_path = Path(args.scope_out)
    scope_path.write_text(
        "# Which protected groups each mitigation run should evaluate.\n"
        f"# eval_scope: {plan.eval_scope}\n"
        "# Apply with: python scripts/generate_experiment_configs.py --no-runners \\\n"
        f"#                 --eval-scope {scope_path}\n\n"
        + yaml.safe_dump(
            {"groups": plan.eval_groups, "conditions": plan.eval_conditions},
            sort_keys=True, allow_unicode=True,
        ),
        encoding="utf-8",
    )
    log.info("wrote %s (scope=%s)", scope_path, plan.eval_scope)

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "thresholds": thresholds.as_dict(),
                    "cells": [c.as_dict() for c in findings],
                    "attribute_recurrence": [r.as_dict() for r in recurrences],
                    "language_effect": [c.as_dict() for c in language],
                    "condition_effect": [c.as_dict() for c in condition],
                    "models": [m.as_dict() for m in models],
                    "plan": plan.as_dict(),
                },
                indent=2, ensure_ascii=False, default=str,
            ),
            encoding="utf-8",
        )
        log.info("wrote %s", args.json)

    usable = [c for c in findings if c.gate_passed]
    print(
        f"\n{len(findings)} cells graded | {len(findings) - len(usable)} not interpretable | "
        f"{sum(1 for c in usable if c.verdict == evidence.Verdict.CONFIRMED)} confirmed | "
        f"{sum(1 for c in usable if c.verdict == evidence.Verdict.PROBABLE)} probable"
    )
    print(f"{len(plan.targets)} mitigation target(s), {len(plan.controls)} control(s)")
    for target in plan.targets:
        print(
            f"  {target.priority}. {target.model} · {target.lang} · {target.group} · "
            f"{target.condition}  MAD {target.ar_mad:.1%}  [{target.verdict}]"
        )
    for note in plan.notes:
        print(f"  note: {note}")
    print(f"\nRead {args.out}, then review {args.plan} before running {args.enable_script}.")


if __name__ == "__main__":
    main()
