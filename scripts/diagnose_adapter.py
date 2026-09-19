"""Did training change the model, and does the audit see the change? -- a direct check.

Every trained adapter left counterfactual consistency unchanged in the audit. Before that is
reported, three explanations have to be told apart, and the audit alone cannot do it:

  H1  the audit does not apply the adapter faithfully. The audit runs through vLLM, whose LoRA
      support for Qwen3.5's hybrid (linear-attention + attention) architecture is partial; if
      part of the adapter is silently dropped, the audited model is not the trained one.
  H2  the adapter changed the decision *probabilities* but rarely across the argmax threshold,
      so greedy decisions -- all the audit sees -- barely move.
  H3  the adapter did not learn attribute-invariance at all, or learned it only for the CVs it
      was trained on (memorisation) and not for new ones.

This script bypasses vLLM entirely and reads the model's own decision logit through
HuggingFace + PEFT -- the path the adapter was trained in. For each prompt it forces the prefix
`{"decision": "` and takes the margin logit(hire) - logit(reject) at the next token.

    H1  compare the HF+PEFT argmax with the vLLM audit's decision on the same benchmark rows;
        agreement should be as high for the adapter as for the base model.
    H2  the per-set *spread* of the margin across attribute variants is a continuous
        invariance measure. If training shrank it without flipping decisions, H2 holds.
    H3  the same spread on sets from the training pool (seen) against benchmark sets (unseen).

    python scripts/diagnose_adapter.py --adapter outputs/sft/qwen3.5-9b_en_only \\
        --audit-run Qwen3.5-9B--en--sft--adapter--en_only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hiring_bias_mitigation.data import prompts as P  # noqa: E402
from hiring_bias_mitigation.generation import dataset as D  # noqa: E402
from hiring_bias_mitigation.utils.config import REPO_ROOT, resolve_output_path  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("diagnose_adapter")

SINGLE_GROUPS = ("military_status", "gender", "religion")
PREFIX = '{"decision": "'
HIRE = {"hire", "найняти"}


def _norm(value) -> str | None:
    text = str(value).strip().lower()
    return "hire" if text in HIRE else "reject" if text in {"reject", "відхилити"} else None


def benchmark_sets(model_slug: str, lang: str, n_sets: int, seed: int) -> pd.DataFrame:
    """Complete single-group sets from the baseline audit, with the audit's own decision."""
    raw = Path(resolve_output_path("outputs/raw"))
    base = pd.read_parquet(raw / f"{model_slug}--{lang}--baseline.parquet")
    base = base[base["protected_group"].isin(SINGLE_GROUPS) & (base["condition"] != "attr_free")]
    sizes = base.groupby(["pair_id", "condition"]).size()
    full = sizes[sizes == sizes.max()].index.to_frame(index=False)
    chosen = full.sample(n=min(n_sets, len(full)), random_state=seed)
    rows = base.merge(chosen, on=["pair_id", "condition"])
    rows = rows.assign(source="benchmark", prompt=[P.build_prompt(r, "baseline")
                                                  for r in rows.to_dict("records")])
    return rows


def pool_sets(lang: str, trained_pairs: set[str], n_sets: int, seed: int) -> pd.DataFrame:
    """Complete sets from the synthetic pool whose pair the adapter was trained on."""
    root = Path(resolve_output_path("artifacts/semisynthetic-v1"))
    pool = pd.read_parquet(root / "raw" / f"{lang}_invariant.parquet").drop(columns=["raw_output"])
    pool = D._with_implicit(pool[pool["pair_id"].isin(trained_pairs)])
    sets = pool[["pair_id", "condition"]].drop_duplicates()
    chosen = sets.sample(n=min(n_sets, len(sets)), random_state=seed)
    rows = pool.merge(chosen, on=["pair_id", "condition"])
    return rows.assign(source="train_pool",
                       prompt=[D._audit_prompt(r) for r in rows.to_dict("records")])


def decision_token_ids(tokenizer) -> tuple[int, int]:
    """The first token of each decision word, as it follows the forced prefix."""
    base = tokenizer(PREFIX, add_special_tokens=False)["input_ids"]
    ids = []
    for word in ("hire", "reject"):
        full = tokenizer(PREFIX + word, add_special_tokens=False)["input_ids"]
        assert full[: len(base)] == base, "prefix tokenises differently before the decision"
        ids.append(full[len(base)])
    assert ids[0] != ids[1], "hire and reject share a first token"
    return ids[0], ids[1]


def margins(model, tokenizer, prompts: list[str], batch_size: int, hire_id: int,
            reject_id: int) -> np.ndarray:
    import torch

    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False) + PREFIX
        for p in prompts
    ]
    out = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        out.extend((logits[:, hire_id] - logits[:, reject_id]).cpu().tolist())
        if start % (batch_size * 25) == 0:
            log.info("  %d / %d", start + len(texts[start:start + batch_size]), len(texts))
    return np.asarray(out)


def set_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    g = frame.groupby(["source", "pair_id", "condition"])[column]
    return pd.DataFrame({
        "spread": g.std(ddof=0),
        "unstable": g.apply(lambda m: (m > 0).nunique() > 1),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", required=True, help="adapter directory, relative to the drive")
    parser.add_argument("--audit-run", required=True, help="the adapter's audit run name")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--train-split", default="sft_train")
    parser.add_argument("--benchmark-sets", type=int, default=30)
    parser.add_argument("--pool-sets", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm-merged", default=None,
                        help="also serve this merged checkpoint through vLLM on the benchmark "
                             "rows and report its agreement with the HF+PEFT decisions")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from scipy.stats import wilcoxon
    from transformers import AutoModelForCausalLM, AutoTokenizer

    slug = args.base_model.split("/")[-1]
    root = Path(resolve_output_path("artifacts/semisynthetic-v1"))
    trained = pd.read_parquet(root / f"{args.train_split}.parquet", columns=["pair_id", "lang"])
    trained_pairs = set(trained.loc[trained["lang"] == args.lang, "pair_id"])

    rows = pd.concat([
        benchmark_sets(slug, args.lang, args.benchmark_sets, args.seed),
        pool_sets(args.lang, trained_pairs, args.pool_sets, args.seed),
    ], ignore_index=True)
    log.info("%d prompts: %s", len(rows), rows["source"].value_counts().to_dict())

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hire_id, reject_id = decision_token_ids(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(resolve_output_path(args.adapter)))
    model.eval()

    prompts = rows["prompt"].tolist()
    with model.disable_adapter():
        log.info("base model margins")
        rows["m_base"] = margins(model, tokenizer, prompts, args.batch_size, hire_id, reject_id)
    log.info("adapter margins")
    rows["m_adapter"] = margins(model, tokenizer, prompts, args.batch_size, hire_id, reject_id)

    report: dict = {"adapter": args.adapter, "prompts": len(rows)}

    # How much did the adapter move the decision logit at all?
    delta = rows["m_adapter"] - rows["m_base"]
    for source, g in rows.assign(delta=delta).groupby("source"):
        report[f"{source}_mean_abs_margin_shift"] = float(g["delta"].abs().mean())
        report[f"{source}_argmax_flips_pct"] = float(
            100 * ((g["m_base"] > 0) != (g["m_adapter"] > 0)).mean())

    # H2 / H3: continuous invariance, seen and unseen CVs.
    base_sets = set_summary(rows, "m_base")
    adapter_sets = set_summary(rows, "m_adapter")
    for source in ("benchmark", "train_pool"):
        b = base_sets.loc[source]
        a = adapter_sets.loc[source]
        stat = wilcoxon(b["spread"], a["spread"]) if len(b) > 5 else None
        report[source] = {
            "sets": len(b),
            "margin_spread_base": float(b["spread"].mean()),
            "margin_spread_adapter": float(a["spread"].mean()),
            "spread_change_pct": float(100 * (a["spread"].mean() / b["spread"].mean() - 1)),
            "spread_wilcoxon_p": float(stat.pvalue) if stat else None,
            "unstable_base_pct": float(100 * b["unstable"].mean()),
            "unstable_adapter_pct": float(100 * a["unstable"].mean()),
        }

    # H1: does the vLLM audit agree with the reference implementation?
    raw = Path(resolve_output_path("outputs/raw"))
    bench = rows[rows["source"] == "benchmark"]
    key = ["pair_id", "protected_group", "protected_attr", "condition"]
    for label, run, column in (("base", f"{slug}--{args.lang}--baseline", "m_base"),
                               ("adapter", args.audit_run, "m_adapter")):
        audit = pd.read_parquet(raw / f"{run}.parquet", columns=[*key, "decision"])
        m = bench.merge(audit, on=key, suffixes=("", "_vllm"))
        hf = np.where(m[column] > 0, "hire", "reject")
        vllm = m["decision_vllm" if "decision_vllm" in m else "decision"].map(_norm)
        report[f"hf_vs_vllm_agreement_{label}_pct"] = float(100 * (hf == vllm).mean())

    out = REPO_ROOT / "reports" / f"adapter_diagnostic--{Path(args.adapter).name}.json"
    out.write_text(json.dumps(report, indent=2))  # kept even if the vLLM step below fails
    if args.vllm_merged:
        # Is the merged checkpoint, served the way the audit serves it, the trained model?
        from hiring_bias_mitigation.eval import backends as BK
        from hiring_bias_mitigation.eval.parsing import parse_output

        del model
        BK._free_gpu()
        backend = BK.VLLMBackend(
            model=str(resolve_output_path(args.vllm_merged)),
            generation=BK.GenerationConfig(max_new_tokens=512, temperature=0.0, top_p=1.0),
            gpu_memory_utilization=0.8, chat_template_kwargs={"enable_thinking": False})
        bench = rows[rows["source"] == "benchmark"]
        parsed = [parse_output(t, args.lang).decision
                  for t in backend.generate(bench["prompt"].tolist())]
        backend.close()
        merged = pd.Series(parsed, index=bench.index).map(_norm)
        hf = np.where(bench["m_adapter"] > 0, "hire", "reject")
        hf_base = np.where(bench["m_base"] > 0, "hire", "reject")
        report["hf_adapter_vs_vllm_merged_agreement_pct"] = float(100 * (hf == merged).mean())
        report["hf_base_vs_vllm_merged_agreement_pct"] = float(100 * (hf_base == merged).mean())

    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
