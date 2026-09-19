"""Builds consistency-preference pairs from the student's OWN inconsistencies.

Why. Every training arm so far pulled decisions toward the teacher's attribute-free verdict,
and every one of them moved decisions at random: the decision-only DPO probe flipped 102 of
30,600 audited decisions, 52 toward that verdict and 50 away. 101 of those 102 flips were in
(pair, condition) sets the student already decided *inconsistently* across attribute
variants -- the borderline candidates an attribute can tip, which is exactly where bias acts.
And on exactly those candidates the teacher's verdict is least trustworthy: it is one sample
at temperature 0.7, a coin flip for a borderline case. Training moved the right decisions
toward a random target.

Invariance does not need a correct answer; it needs the *same* answer across variants. So the
target here is the student's own majority over the six attribute variants of a (pair,
condition) set -- a vote across the attribute, which already marginalises it out -- and there
is no teacher verdict anywhere in the signal.

    chosen   {"decision": <majority across the set's variants>}
    rejected {"decision": <the opposite>}

Sets that are already consistent are included at a matched rate, as anchors: without them
the only signal is "move toward the majority", and a majority that is mostly `reject` would
drag the operating point with it.

Contamination: the pool is the synthetic training pool, never the benchmark. Finding the
student's inconsistencies on the benchmark and training on them would be training on the
test set; `assert_no_leakage` enforces the separation.

    python scripts/build_consistency_pairs.py --model Qwen/Qwen3.5-9B --lang en
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from hiring_bias_mitigation.data.benchmark import assert_no_leakage  # noqa: E402
from hiring_bias_mitigation.eval import backends as BK  # noqa: E402
from hiring_bias_mitigation.eval.parsing import parse_output  # noqa: E402
from hiring_bias_mitigation.generation import dataset as D  # noqa: E402
from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("build_consistency_pairs")

SET_KEY = ["pair_id", "condition"]
FLIP = {"hire": "reject", "reject": "hire"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--artifacts-dir", default="artifacts/semisynthetic-v1")
    parser.add_argument("--name", default="dpo_consistency")
    parser.add_argument("--anchor-ratio", type=float, default=1.0,
                        help="stable-set rows per unstable-set row")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(resolve_output_path(args.artifacts_dir))
    slug = args.model.split("/")[-1]
    gen_path = (Path(resolve_output_path("outputs/raw"))
                / f"consistency_pool--{slug}--{args.lang}.parquet")

    # 1. The student's decisions on the synthetic pool, rendered exactly as the audit renders
    #    them. Cached: this is the expensive step, and the pairs can be rebuilt from it freely.
    pool = pd.read_parquet(root / "raw" / f"{args.lang}_invariant.parquet")
    pool = pool.drop(columns=[c for c in ("raw_output",) if c in pool.columns])
    pool = D._with_implicit(pool)
    pool["prompt"] = [D._audit_prompt(row) for row in pool.to_dict("records")]
    assert_no_leakage(pool, "consistency_pool")

    if gen_path.exists():
        log.info("reusing student generations from %s", gen_path)
        pool = pool.merge(pd.read_parquet(gen_path)[["prompt", "raw_output"]], on="prompt")
    else:
        log.info("generating %d student responses with %s", len(pool), args.model)
        backend = BK.build_backend({
            "backend": "vllm", "model": args.model, "gpu_memory_utilization": 0.8,
            "max_model_len": 8192, "chat_template_kwargs": {"enable_thinking": False},
            "generation": {"max_new_tokens": 512, "temperature": 0.0, "top_p": 1.0,
                           "seed": args.seed},
        })
        try:
            pool["raw_output"] = backend.generate(pool["prompt"].tolist())
        finally:
            backend.close()
        gen_path.parent.mkdir(parents=True, exist_ok=True)
        pool.to_parquet(gen_path, index=False)
        log.info("wrote %s", gen_path)

    parsed = [parse_output(r, args.lang) for r in pool["raw_output"]]
    pool["student"] = [p.decision for p in parsed]
    decided = pool[pool["student"].isin(FLIP)].copy()
    log.info("parsed %d / %d student decisions", len(decided), len(pool))

    # 2. Per-set agreement across attribute variants.
    stats = decided.groupby(SET_KEY)["student"].agg(
        n="size", n_hire=lambda s: int((s == "hire").sum())
    ).reset_index()
    stats["n_reject"] = stats["n"] - stats["n_hire"]
    stats["unstable"] = (stats["n_hire"] > 0) & (stats["n_reject"] > 0)
    stats["tie"] = stats["n_hire"] == stats["n_reject"]
    stats["majority"] = stats.apply(
        lambda r: "hire" if r["n_hire"] > r["n_reject"] else "reject", axis=1
    )
    report = {
        "sets": len(stats),
        "unstable_sets": int(stats["unstable"].sum()),
        "unstable_share": round(float(stats["unstable"].mean()), 4),
        "tied_sets_dropped": int((stats["unstable"] & stats["tie"]).sum()),
    }
    log.info("sets %d | unstable %d (%.1f%%) | ties dropped %d", report["sets"],
             report["unstable_sets"], 100 * report["unstable_share"],
             report["tied_sets_dropped"])

    usable = stats[~stats["tie"]]
    rows = decided.merge(usable[[*SET_KEY, "unstable", "majority"]], on=SET_KEY)
    unstable_rows = rows[rows["unstable"]]
    stable_rows = rows[~rows["unstable"]]
    n_anchor = min(len(stable_rows), round(args.anchor_ratio * len(unstable_rows)))
    stable_rows = stable_rows.sample(n=n_anchor, random_state=args.seed)
    rows = pd.concat([unstable_rows, stable_rows], ignore_index=True)

    rows["chosen"] = rows["majority"].map(lambda d: json.dumps({"decision": d}))
    rows["rejected"] = rows["majority"].map(lambda d: json.dumps({"decision": FLIP[d]}))
    keep = ["prompt", "chosen", "rejected", "lang", "protected_group", "protected_attr",
            "condition", "candidate_id", "job_id", "pair_id", "unstable", "majority", "student"]
    rows = rows[keep]
    report.update({
        "rows_unstable": len(unstable_rows),
        "rows_anchor": int(n_anchor),
        "rows_moving_student": int((rows["student"] != rows["majority"]).sum()),
        "majority_mix": rows["majority"].value_counts(normalize=True).round(3).to_dict(),
    })

    assert_no_leakage(rows, args.name)
    train, validation = D.train_val_split(rows, val_fraction=0.05, seed=args.seed)
    train.to_parquet(root / f"{args.name}_train.parquet", index=False)
    validation.to_parquet(root / f"{args.name}_validation.parquet", index=False)
    report.update({"train": len(train), "validation": len(validation)})
    (root / f"{args.name}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
