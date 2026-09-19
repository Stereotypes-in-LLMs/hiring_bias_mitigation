"""Selects the phase-2 confirmation runs: the winning mitigations, re-evaluated with the
intersections included.

Why two phases. Evaluating every mitigation run on the fully-crossed intersections costs
~538 GPU-hours — the two intersections are 145 of 179 attributes, and every arm pays for
them whether or not it turns out to work. Phase 1 drops them and costs ~87 hours; it ranks
the mitigations on the single groups, which is what a ranking needs. Phase 2 then pays for the
intersections **once**, on the arms that earned it.

What phase 2 answers that phase 1 cannot:

  * Does the mitigation also move the intersectional cells, or only the single groups?
  * Is the interaction still non-additive after mitigation? The baseline found military ×
    gender sub-additive in both languages (52/23 negative in English, 62/11 in Ukrainian).
    Whether a mitigation flattens that, leaves it, or inverts it is a result in its own right,
    and it is the follow-up the baseline's intersection finding is asking for.

Usage — after phase 1 has been scored:

    python scripts/select_phase2.py --top 1           # best arm per family
    python scripts/select_phase2.py --top 2 --dry-run # see the cost first

Ranking is by disparity reduction against each run's own baseline, with a utility guard: an
arm that cut disparity by damaging the model is not a winner and is never promoted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.data.protected_groups import INTERSECTIONS  # noqa: E402
from hiring_bias_mitigation.eval.report import load_records  # noqa: E402
from hiring_bias_mitigation.utils.config import REPO_ROOT  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("select_phase2")

GROUP_SIZE = {
    "military_status": 5, "gender": 20, "religion": 9, "marital_status": 5,
    "military_status_x_gender": 100, "military_status_x_religion": 45,
}
PAIRS, CONDITIONS = 450, 2

#: An arm whose utility fell this far below its baseline did not fix the model, it damaged
#: it. Never promoted, however much disparity it removed.
MAX_UTILITY_LOSS = 0.03


def _family(record: dict) -> str:
    return (record.get("meta", {}).get("mitigation") or {}).get("family", "none")


def _variant(record: dict) -> str:
    mitigation = record.get("meta", {}).get("mitigation") or {}
    return str(
        mitigation.get("strategy") or mitigation.get("method")
        or mitigation.get("mode") or mitigation.get("objective") or "adapter"
    )


def _key(record: dict) -> tuple[str, str]:
    return record["meta"].get("model", "?").split("/")[-1], record["meta"].get("lang", "?")


def _overall(record: dict) -> dict:
    return (record["summary"].get("disparity") or {}).get("overall", {})


def rank(records: list[dict]) -> list[dict]:
    """Scores every mitigated run against its own baseline."""
    baselines = {_key(r): r for r in records if _family(r) == "none"
                 and not r["meta"].get("limit_pairs")}
    scored = []
    for record in records:
        family = _family(record)
        if family == "none":
            continue
        base = baselines.get(_key(record))
        if base is None:
            log.warning("no baseline for %s — skipping", record["run_name"])
            continue

        mad = _overall(record).get("ar_mad")
        base_mad = _overall(base).get("ar_mad")
        utility = record["summary"].get("reference_agreement")
        base_utility = base["summary"].get("reference_agreement")
        if any(v is None or v != v for v in (mad, base_mad, utility, base_utility)):
            continue

        utility_delta = utility - base_utility
        scored.append({
            "run_name": record["run_name"],
            "model": _key(record)[0], "lang": _key(record)[1],
            "family": family, "variant": _variant(record),
            "mad": mad, "baseline_mad": base_mad,
            "mad_reduction": base_mad - mad,
            "utility": utility, "utility_delta": utility_delta,
            "refusal_rate": record["summary"].get("refusal_rate", 0.0),
            "damaged": utility_delta < -MAX_UTILITY_LOSS,
        })
    scored.sort(key=lambda s: -s["mad_reduction"])
    return scored


def winners(scored: list[dict], top: int) -> list[dict]:
    """Top N per (family, model, language), excluding anything that damaged the model."""
    buckets: dict[tuple, list] = defaultdict(list)
    for entry in scored:
        if entry["damaged"] or entry["mad_reduction"] <= 0:
            continue
        buckets[(entry["family"], entry["model"], entry["lang"])].append(entry)
    out = []
    for entries in buckets.values():
        out.extend(sorted(entries, key=lambda e: -e["mad_reduction"])[:top])
    return sorted(out, key=lambda e: (e["family"], -e["mad_reduction"]))


#: Model name -> config slug, for matching graded cells back to config paths.
SLUGS = {
    "Qwen3.5-4B": "qwen3.5-4b", "Qwen3.5-9B": "qwen3.5-9b",
    "gemma-4-E4B-it": "gemma-4-e4b", "gemma-4-12B-it": "gemma-4-12b",
    "lapa-v0.1.2-instruct": "lapa-12b",
}


def intersection_targets(
    analysis: dict, min_mad: float = 0.02
) -> dict[tuple[str, str], dict[str, list[str]]]:
    """Intersections worth measuring in phase 2, per model-language.

    Read from the **graded cells**, not from `plan["targets"]`. Under the
    `targeted-no-intersections` scope the planner strips intersection targets out of that
    list by design — reading it here would find nothing and phase 2 would silently do
    nothing, which is the worst kind of failure for a stage that exists to follow up the
    study's intersection finding.

    Only cells with a confirmed disparity and a material effect are promoted: an intersection
    with nothing to fix costs 40,000-130,000 generations to re-measure a null.

    Returns `{(slug, lang): {"groups": [...], "conditions": [...]}}` so the emitted scope
    restricts phase 2 to the exact cells that carried evidence — the intersection that showed
    bias, under the framing it showed it in.
    """
    out: dict[tuple[str, str], dict[str, list[str]]] = {}
    for cell in analysis.get("cells", []):
        if not cell.get("gate_passed") or cell.get("verdict") != "confirmed":
            continue
        if cell["group"] not in INTERSECTIONS:
            continue
        mad = cell.get("ar_mad")
        if mad is None or mad != mad or mad < min_mad:
            continue
        key = (SLUGS.get(cell["model"], cell["model"].lower()), cell["lang"])
        entry = out.setdefault(key, {"groups": [], "conditions": []})
        if cell["group"] not in entry["groups"]:
            entry["groups"].append(cell["group"])
        if cell["condition"] not in entry["conditions"]:
            entry["conditions"].append(cell["condition"])
    for entry in out.values():
        entry["groups"].sort()
        entry["conditions"] = [*sorted(entry["conditions"]), "attr_free"]
    return out


def _matches(model: str, slug: str) -> bool:
    """Whether a run's model name corresponds to a config slug (`Qwen3.5-4B` -> `qwen3.5-4b`)."""
    return model.lower().replace("-it", "") == slug.lower().replace("-it", "")


def cost(groups: list[str], conditions: list[str]) -> int:
    injected = len([c for c in conditions if c != "attr_free"])
    flat = PAIRS if "attr_free" in conditions else 0
    return sum(GROUP_SIZE.get(g, 0) for g in groups) * PAIRS * injected + flat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "eval" / "results"))
    parser.add_argument("--analysis", default=str(REPO_ROOT / "reports" / "analysis.json"))
    parser.add_argument("--top", type=int, default=1,
                        help="how many winning arms per family x model x language to promote")
    parser.add_argument("--min-mad", type=float, default=0.02,
                        help="minimum intersection disparity worth re-measuring, as a "
                             "proportion. An intersection with nothing to fix costs 40k-130k "
                             "generations to confirm a null.")
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "phase2_scope.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="print the cost and stop")
    args = parser.parse_args()

    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    records = load_records(args.results_dir)
    scored = rank(records)

    if not scored:
        raise SystemExit(
            "no mitigated runs scored yet. Run phase 1 first:\n"
            "  make local-prompt && make local-scrub && make local-report"
        )

    promoted = winners(scored, args.top)
    targets = intersection_targets(analysis, args.min_mad)
    if not targets:
        raise SystemExit(
            "no intersection cell carries a confirmed, material disparity — there is nothing "
            "for phase 2 to measure. That is a result: report that mitigation was not "
            "followed up on intersections because the baseline showed none worth following."
        )

    print(f"{len(scored)} mitigated run(s) scored, {len(promoted)} promoted\n")
    print(f"{'family':10s} {'model':16s} {'lang':5s} {'variant':26s} "
          f"{'MAD':>7s} {'was':>7s} {'Δutil':>7s}")
    for entry in promoted:
        print(f"{entry['family']:10s} {entry['model']:16s} {entry['lang']:5s} "
              f"{entry['variant']:26s} {100 * entry['mad']:6.1f}% "
              f"{100 * entry['baseline_mad']:6.1f}% {100 * entry['utility_delta']:+6.1f}")

    damaged = [s for s in scored if s["damaged"]]
    if damaged:
        print(f"\n{len(damaged)} run(s) excluded for damaging the model "
              f"(utility fell more than {MAX_UTILITY_LOSS:.0%}):")
        for entry in damaged[:8]:
            print(f"  {entry['run_name']}: MAD {100 * entry['mad_reduction']:+.1f}pp but "
                  f"utility {100 * entry['utility_delta']:+.1f}pp")

    groups_scope: dict[str, dict[str, list[str]]] = {}
    conditions_scope: dict[str, dict[str, list[str]]] = {}
    total = 0
    promoted_langs = {(e["model"], e["lang"]) for e in promoted}
    print("\nphase-2 scope — only intersections that showed confirmed bias:")
    for (target_slug, lang), spec in sorted(targets.items()):
        if not any(
            lang == promoted_lang and _matches(model, target_slug)
            for model, promoted_lang in promoted_langs
        ):
            print(f"  {target_slug:14s} {lang}  skipped — no promoted arm for it")
            continue
        groups_scope.setdefault(target_slug, {})[lang] = spec["groups"]
        conditions_scope.setdefault(target_slug, {})[lang] = spec["conditions"]
        n = cost(spec["groups"], spec["conditions"])
        total += n
        print(f"  {target_slug:14s} {lang}  {', '.join(spec['groups'])} "
              f"under {', '.join(spec['conditions'])}  -> {n:,} prompts")
    scope = {"groups": groups_scope, "conditions": conditions_scope}

    print(f"\nphase 2: {total:,} prompts (~{total / 5 / 3600:.1f} h at 5 prompts/s)")
    if args.dry_run:
        print("--dry-run: nothing written")
        return

    import yaml

    path = Path(args.out)
    path.write_text(
        "# Phase-2 evaluation scope: the winning arms, re-measured WITH intersections.\n"
        "# Generated by scripts/select_phase2.py. Apply with:\n"
        f"#   python scripts/generate_experiment_configs.py --no-runners --eval-scope {path}\n"
        "#\n"
        "# Phase 1 ranked the mitigations on single groups. This pays for the intersections\n"
        "# once, on the arms that earned it, and answers whether the sub-additivity found in\n"
        "# the baseline survives mitigation.\n\n"
        + yaml.safe_dump(scope, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
