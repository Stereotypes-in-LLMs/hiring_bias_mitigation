# ruff: noqa: E501  (command lines in the docstring)
"""Is the fine-tuned model more consistent, or only more willing to hire?

SFT moved Qwen3.5-9B's hire rate from 16% to 28% (the reference hires 33%). Counterfactual
instability depends on the operating point: at a hire rate near 0% or 100% nothing flips.
So a stability gain could in principle come from the shift alone. This control separates the
two by reading each model's decision margin directly and sweeping the decision threshold.

For every audited prompt, the model is given the audit's exact chat-wrapped prompt plus the
forced prefix `{"decision": "` and the log-probability gap between the first token of the two
decision words is read in a single vLLM step (no generation). With decisions `margin > t`,
each threshold `t` gives a hire rate and a share of unstable counterfactual sets; the two
models are then compared **at equal hire rate**, which the audit itself cannot do.

    # one pass per model, then the comparison (the pass is ~10 min on the GPU)
    python scripts/operating_point_control.py margins --config configs/audit/qwen3.5-9b_en_sft_en_only.yaml --which base
    python scripts/operating_point_control.py margins --config configs/audit/qwen3.5-9b_en_sft_en_only.yaml --which adapter
    python scripts/operating_point_control.py compare --config configs/audit/qwen3.5-9b_en_sft_en_only.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hiring_bias_mitigation.eval import backends as BK  # noqa: E402
from hiring_bias_mitigation.eval import runner as R  # noqa: E402
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    REPO_ROOT,
    load_config,
    resolve_output_path,
)
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("operating_point_control")

PREFIX = '{"decision": "'
WORDS = {"en": ("hire", "reject"), "uk": ("найняти", "відхилити")}
KEY = ["pair_id", "protected_group", "protected_attr", "condition"]


def out_dir() -> Path:
    d = Path(resolve_output_path("outputs/operating_point"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def decision_words(run_name: str, lang: str) -> tuple[str, str]:
    """The words this run actually answers with, read from its own generations.

    Not the prompt's language: a Ukrainian SFT adapter trained on targets carrying the
    canonical English decision answers `{"decision": "reject"}` to a prompt that asks for
    `найняти або відхилити`. Reading the words off the run makes the control follow the model.
    """
    path = Path(resolve_output_path("outputs/raw")) / f"{run_name}.parquet"
    if not path.exists():
        return WORDS[lang]
    counts = pd.read_parquet(path, columns=["decision", "raw_decision"])
    words = {}
    for decision, group in counts.groupby(counts["decision"].str.lower()):
        if decision in ("hire", "reject"):
            words[decision] = group["raw_decision"].value_counts().idxmax()
    picked = (words.get("hire", WORDS[lang][0]), words.get("reject", WORDS[lang][1]))
    log.info("%s answers with %r / %r", run_name, *picked)
    return picked


def decision_token_ids(tokenizer, words: tuple[str, str]) -> tuple[int, int]:
    base = tokenizer(PREFIX, add_special_tokens=False)["input_ids"]
    ids = []
    for word in words:
        full = tokenizer(PREFIX + word, add_special_tokens=False)["input_ids"]
        assert full[: len(base)] == base, "the prefix tokenises differently before the decision"
        ids.append(full[len(base)])
    assert ids[0] != ids[1], "the two decision words share a first token"
    return ids[0], ids[1]


def cmd_margins(cfg: dict, which: str, chunk: int = 2000) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    done = out_dir() / f"{R.build_run_name(cfg)}--{which}.parquet"
    if done.exists():
        log.info("%s already written; skipping this pass", done.name)
        return
    frame = R.build_frame(cfg)
    prompts = R._build_prompts(frame, "baseline")
    if which == "base":
        model_path = cfg["model"]
    else:
        from hiring_bias_mitigation.mitigation.merge import merge

        model_path = str(merge(R._require_adapter(cfg["mitigation"]["lora_path"])))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    run_name = R.build_run_name(cfg)
    # For the base pass the answer style is the baseline run's, not the adapter's.
    source = run_name if which == "adapter" else f"{cfg['model'].split('/')[-1]}--{cfg['lang']}--baseline"
    hire_id, reject_id = decision_token_ids(tokenizer, decision_words(source, cfg["lang"]))
    kwargs = cfg.get("chat_template_kwargs")
    texts = [BK._chat_wrap(tokenizer, p, kwargs) + PREFIX for p in prompts]

    # This pass needs one token, not a 512-token generation, so it asks for far less KV cache
    # than an audit. Holding every RequestOutput for 31k prompts is what made the 12B pass run
    # the machine out of memory, so the prompts go through in chunks and only the margin is
    # kept -- the outputs are dropped as soon as each chunk is read.
    llm = LLM(model=model_path,
              gpu_memory_utilization=cfg.get("op_control_gpu_memory_utilization", 0.55),
              max_model_len=cfg.get("max_model_len", 8192), trust_remote_code=True, seed=42,
              max_num_seqs=cfg.get("op_control_max_num_seqs", 64))
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    margins, missing = [], 0
    for start in range(0, len(texts), chunk):
        for o in llm.generate(texts[start:start + chunk], params):
            lp = o.outputs[0].logprobs[0]
            floor = min(v.logprob for v in lp.values()) - 1.0  # a word outside the top 20
            h = lp[hire_id].logprob if hire_id in lp else floor
            r = lp[reject_id].logprob if reject_id in lp else floor
            missing += (hire_id not in lp) + (reject_id not in lp)
            margins.append(h - r)
        log.info("  %d / %d prompts", len(margins), len(texts))
    result = frame[KEY].assign(margin=margins)
    path = out_dir() / f"{run_name}--{which}.parquet"
    result.to_parquet(path, index=False)
    log.info("wrote %s (%d prompts; %d decision words outside the top 20)", path, len(result),
             missing)
    if missing > 0.5 * 2 * len(result):
        raise SystemExit(
            f"{missing} of {2 * len(result)} decision-word lookups missed the top 20: the model "
            "does not answer with these words, so the margin is meaningless. Check "
            f"{path.name} against the run's raw_decision values."
        )


def unstable_share(frame: pd.DataFrame, column: str, threshold: float) -> tuple[float, float]:
    decided = frame.assign(hire=frame[column] > threshold)
    sets = decided[decided["condition"] != "attr_free"].groupby(
        ["pair_id", "protected_group", "condition"])["hire"]
    return float(decided["hire"].mean()), float((sets.nunique() > 1).mean())


def cmd_compare(cfg: dict) -> None:
    name = R.build_run_name(cfg)
    base = pd.read_parquet(out_dir() / f"{name}--base.parquet")
    adapter = pd.read_parquet(out_dir() / f"{name}--adapter.parquet")
    both = base.merge(adapter, on=KEY, suffixes=("_base", "_adapter"))

    # Threshold sweep: the same quantiles for both models, so each row compares them at the
    # same hire rate.
    rows = []
    for q in np.linspace(0.40, 0.95, 56):
        for which in ("base", "adapter"):
            t = float(both[f"margin_{which}"].quantile(q))
            hire, unstable = unstable_share(both, f"margin_{which}", t)
            rows.append({"model": which, "quantile": round(q, 3), "threshold": t,
                         "hire_rate": hire, "unstable": unstable})
    curve = pd.DataFrame(rows)

    # The headline: each model at its own greedy threshold, and the base moved to the
    # adapter's hire rate.
    _, unstable_base_0 = unstable_share(both, "margin_base", 0.0)
    hire_adapter, unstable_adapter_0 = unstable_share(both, "margin_adapter", 0.0)
    t_matched = float(both["margin_base"].quantile(1 - hire_adapter))
    hire_base_matched, unstable_base_matched = unstable_share(both, "margin_base", t_matched)
    summary = {
        "run": name, "prompts": len(both),
        "hire_rate_base": float((both["margin_base"] > 0).mean()),
        "hire_rate_adapter": hire_adapter,
        "unstable_base_at_own_threshold": unstable_base_0,
        "unstable_adapter_at_own_threshold": unstable_adapter_0,
        "base_threshold_matched_to_adapter_hire_rate": t_matched,
        "hire_rate_base_matched": hire_base_matched,
        "unstable_base_at_matched_hire_rate": unstable_base_matched,
        "gain_explained_by_shift_pct": float(
            100 * (unstable_base_0 - unstable_base_matched)
            / max(unstable_base_0 - unstable_adapter_0, 1e-9)),
    }
    report = REPO_ROOT / "reports" / "operating_point"
    report.mkdir(parents=True, exist_ok=True)
    curve.to_csv(report / f"{name}--curve.csv", index=False)
    (report / f"{name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("margins")
    m.add_argument("--config", required=True, help="the adapter's audit config")
    m.add_argument("--which", choices=("base", "adapter"), required=True)
    m.add_argument("--chunk", type=int, default=2000,
                   help="prompts per vLLM call; lower it if the machine is tight on memory")
    c = sub.add_parser("compare")
    c.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    # A config listed in reports/operating_point/skip.txt is skipped (exit 0): the control
    # costs ~75 min per pass, and cells whose hire rate barely moved do not need it. A file
    # rather than a flag, so a running queue can be told without restarting it.
    skip = REPO_ROOT / "reports" / "operating_point" / "skip.txt"
    if skip.exists() and Path(args.config).stem in skip.read_text().split():
        log.info("skipping %s (listed in %s)", args.config, skip)
        return
    if args.cmd == "margins":
        cmd_margins(cfg, args.which, args.chunk)
    else:
        cmd_compare(cfg)


if __name__ == "__main__":
    main()
