"""Generates the semi-synthetic mitigation training data with a local teacher model.

    python scripts/generate_training_data.py --config configs/generation/qwen3.5-122b.yaml
    python scripts/generate_training_data.py --config <cfg> --lang uk    # one language
    python scripts/generate_training_data.py --config <cfg> --dry-run    # pool only, no GPU

Three teacher passes per language (see `generation/synth.py` for why each exists):

    1. reference  -- decide on the bare job-CV pair. The anchor verdict.
    2. invariant  -- for each attribute variant, the response a fair screener would give:
                     the anchor's verdict, and a rationale that never names the attribute.
                     These become the SFT targets and the `chosen` side of the preference set.
    3. biased     -- the same variant with the attribute allowed to drive the outcome. The
                     `rejected` side.

Every pass is filtered before the next one runs, so a pair whose anchor was unusable never
costs a variant generation. Outputs land under $HBM_OUTPUT_ROOT as parquet plus a dataset
card; `scripts/push_dataset_to_hub.py` publishes them, and nothing publishes automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.data.benchmark import assert_no_leakage  # noqa: E402
from hiring_bias_mitigation.eval import backends as BK  # noqa: E402
from hiring_bias_mitigation.generation import dataset as D  # noqa: E402
from hiring_bias_mitigation.generation import filters as F  # noqa: E402
from hiring_bias_mitigation.generation import pool as PL  # noqa: E402
from hiring_bias_mitigation.generation import synth as SY  # noqa: E402
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    load_config,
    require_output_root,
    resolve_output_path,
)
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402
from hiring_bias_mitigation.utils.seed import set_seed  # noqa: E402

log = get_logger("generate_training_data")


def generate_for_language(cfg: dict, lang: str, backend, out_dir: Path) -> tuple[dict, dict]:
    """Returns (per-language frames, filter report)."""
    pool_cfg = PL.PoolConfig(
        lang=lang,
        n_pairs=int(cfg.get("n_pairs_per_language", 1500)),
        jobs_per_candidate=int(cfg.get("jobs_per_candidate", 3)),
        seed=cfg.get("seed", 42),
    )
    pairs = PL.build_pool(pool_cfg)
    if pairs.empty:
        raise SystemExit(f"no pairs matched for {lang}; loosen the matcher in generation/pool.py")

    synth_cfg = SY.SynthConfig(
        lang=lang,
        protected_groups=tuple(
            cfg.get("protected_groups", ("military_status", "gender", "religion"))
        ),
        conditions=tuple(cfg.get("conditions", ("explicit", "implicit"))),
        attributes_per_pair=int(cfg.get("attributes_per_pair", 2)),
        rejected_source=cfg.get("rejected_source", "teacher_biased"),
        seed=cfg.get("seed", 42),
    )

    report: dict[str, dict] = {}

    raw_reference = _cached_raw(out_dir, f"{lang}_reference")
    if raw_reference is None or len(raw_reference) != len(pairs):
        log.info("[%s] pass 1/3 — reference decisions for %d pairs", lang, len(pairs))
        raw_reference = backend.generate(SY.reference_prompts(pairs, lang))
    anchored, report["reference"] = F.filter_reference(raw_reference, pairs, lang)
    _dump_raw(out_dir, f"{lang}_reference", pairs, raw_reference)
    if anchored.empty:
        raise SystemExit(f"[{lang}] every reference generation was filtered out; see {out_dir}")

    variants = SY.build_variant_frame(anchored, synth_cfg)
    raw_invariant = _cached_raw(out_dir, f"{lang}_invariant")
    if raw_invariant is None or len(raw_invariant) != len(variants):
        log.info("[%s] pass 2/3 — invariant responses for %d variants", lang, len(variants))
        raw_invariant = backend.generate(SY.invariant_prompts(variants, lang))
    invariant, report["invariant"] = F.filter_invariant(raw_invariant, variants, lang)
    _dump_raw(out_dir, f"{lang}_invariant", variants, raw_invariant)

    biased = pd.DataFrame()
    if synth_cfg.rejected_source in ("teacher_biased", "both"):
        raw_biased = _cached_raw(out_dir, f"{lang}_biased")
        if raw_biased is None or len(raw_biased) != len(variants):
            log.info("[%s] pass 3/3 — biased responses for %d variants", lang, len(variants))
            raw_biased = backend.generate(SY.biased_prompts(variants, lang))
        biased, report["biased"] = F.filter_biased(raw_biased, variants, lang)
        _dump_raw(out_dir, f"{lang}_biased", variants, raw_biased)

    return {"invariant": invariant, "biased": biased, "pairs": pairs}, report


def _cached_raw(out_dir: Path, name: str) -> list[str] | None:
    """The raw generations of a finished pass, if it already ran.

    A full generation run is a day of GPU. Interrupting it -- a power cut, a reboot, a
    deliberate stop -- must not mean repeating the passes that finished, so each pass checks
    for its own dump first. The dump is written only after the pass completes, so a partial
    pass leaves no file and is redone in full rather than resumed mid-way.
    """
    path = out_dir / "raw" / f"{name}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path, columns=["raw_output"])
    log.info("resuming: %s already generated (%d rows) -- skipping", name, len(frame))
    return frame["raw_output"].tolist()


def _dump_raw(out_dir: Path, name: str, frame: pd.DataFrame, raw: list[str]) -> None:
    """Keeps every raw teacher generation, filtered or not.

    Filtering drops examples permanently, and the drop reasons are themselves a result -- a
    30% yield is a finding about the teacher. Keeping the raw dump means a filter can be
    revised later without paying for generation again.
    """
    path = out_dir / "raw" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.assign(raw_output=raw).to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--lang", choices=("en", "uk"), help="restrict to one language")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and report the pool only; no model is loaded")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    languages = [args.lang] if args.lang else list(cfg.get("languages", ("en", "uk")))

    if args.dry_run:
        for lang in languages:
            pool = PL.build_pool(
                PL.PoolConfig(
                    lang=lang,
                    n_pairs=int(cfg.get("n_pairs_per_language", 1500)),
                    seed=cfg.get("seed", 42),
                )
            )
            print(
                f"{lang}: {len(pool)} pairs, {pool['candidate_id'].nunique()} candidates, "
                f"{pool['job_id'].nunique()} jobs, "
                f"median CV {int(pool['cv'].str.len().median())} chars"
            )
        return

    require_output_root()
    out_dir = Path(resolve_output_path(cfg["artifacts_dir"]))
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    # Nothing left to generate? Skip loading a 71 GiB teacher just to assemble parquet files.
    wanted = [
        f"{lang}_{name}"
        for lang in languages
        for name in ("reference", "invariant", "biased")
    ]
    if all((out_dir / "raw" / f"{n}.parquet").exists() for n in wanted):
        log.info("every pass already generated — assembling from the raw dumps")
        log.info("run: python scripts/assemble_datasets.py --config %s", args.config)
        return

    backend = BK.build_backend(
        {
            "model": cfg["teacher_model"],
            "backend": cfg.get("backend", "vllm"),
            "generation": cfg.get("generation", {}),
            "max_model_len": cfg.get("max_model_len", 8192),
            "gpu_memory_utilization": cfg.get("gpu_memory_utilization", 0.90),
            "max_num_seqs": cfg.get("max_num_seqs"),
            "dtype": cfg.get("dtype", "auto"),
        }
    )

    invariant_all, biased_all, reports = [], [], {}
    try:
        for lang in languages:
            frames, report = generate_for_language(cfg, lang, backend, out_dir)
            invariant_all.append(frames["invariant"])
            if not frames["biased"].empty:
                biased_all.append(frames["biased"])
            reports[lang] = report
    finally:
        backend.close()

    invariant = pd.concat(invariant_all, ignore_index=True) if invariant_all else pd.DataFrame()
    biased = pd.concat(biased_all, ignore_index=True) if biased_all else pd.DataFrame()

    sft = D.build_sft_dataset(invariant)
    if cfg.get("balance_decisions", True):
        sft, balance = F.balance_decisions(sft, seed=cfg.get("seed", 42), column="decision")
        reports["balance"] = balance

    dpo = D.build_dpo_dataset(invariant, biased)

    splits = {}
    for name, frame in (("sft", sft), ("dpo", dpo)):
        if frame.empty:
            log.warning("%s dataset is empty — nothing written", name)
            continue
        assert_no_leakage(frame, f"generate_training_data.{name}")
        train, validation = D.train_val_split(
            frame, val_fraction=float(cfg.get("val_fraction", 0.05)), seed=cfg.get("seed", 42)
        )
        train.to_parquet(out_dir / f"{name}_train.parquet", index=False)
        splits[f"{name}_train"] = len(train)
        if not validation.empty:
            validation.to_parquet(out_dir / f"{name}_validation.parquet", index=False)
            splits[f"{name}_validation"] = len(validation)
        log.info("%s: %d train / %d validation", name, len(train), len(validation))

    generation_cfg = BK.GenerationConfig(**cfg.get("generation", {})).as_dict()
    (out_dir / "generation_report.json").write_text(
        json.dumps(
            {
                "teacher_model": cfg["teacher_model"],
                "generation": generation_cfg,
                "languages": languages,
                "protected_groups": list(cfg.get("protected_groups", [])),
                "conditions": list(cfg.get("conditions", [])),
                "rejected_source": cfg.get("rejected_source", "teacher_biased"),
                "splits": splits,
                "filters": reports,
                "seed": cfg.get("seed", 42),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    D.write_dataset_card(
        out_dir / "README.md",
        name=cfg.get("dataset_name", "hiring-bias-mitigation-semisynthetic"),
        teacher_model=cfg["teacher_model"],
        generation=generation_cfg,
        filter_report=reports,
        splits=splits,
        groups=list(cfg.get("protected_groups", [])),
        languages=languages,
        n_rows=sum(splits.values()),
    )
    log.info("artifacts written to %s", out_dir)


if __name__ == "__main__":
    main()
