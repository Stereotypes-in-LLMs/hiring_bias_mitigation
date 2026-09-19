"""Builds the SFT and preference datasets from whatever raw teacher dumps exist.

`generate_training_data.py` assembles these at the very end, after every language and every
pass. That is fine for an uninterrupted run and useless when one is cut short — the raw dumps
are all on disk, but nothing has been turned into a dataset.

This does the assembly step on its own, from whatever is present:

    python scripts/assemble_datasets.py --artifacts-dir artifacts/semisynthetic-v1

The two datasets have different prerequisites, which is what makes a partial run useful:

    SFT        needs `reference` + `invariant`         → complete as soon as both exist
    DPO/ORPO   needs `reference` + `invariant` + `biased` → per language

So a run stopped after the invariant pass yields a **complete SFT dataset** and a preference
dataset covering only the languages whose biased pass finished. Both are written, and the
report records which languages each covers, so a training run cannot silently use a dataset
that is missing half its data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.data.benchmark import assert_no_leakage  # noqa: E402
from hiring_bias_mitigation.generation import dataset as D  # noqa: E402
from hiring_bias_mitigation.generation import filters as F  # noqa: E402
from hiring_bias_mitigation.utils.config import load_config, resolve_output_path  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("assemble_datasets")

PASSES = ("reference", "invariant", "biased")


def available(raw_dir: Path) -> dict[str, set[str]]:
    """Which passes finished, per language."""
    out: dict[str, set[str]] = {}
    for path in sorted(raw_dir.glob("*.parquet")):
        lang, _, name = path.stem.partition("_")
        if name in PASSES:
            out.setdefault(lang, set()).add(name)
    return out


def _refilter(raw_dir: Path, lang: str, name: str, filter_fn):
    """Re-applies the pass's quality filter to its raw dump.

    Filtering here rather than trusting a cached filtered copy means a revised filter can be
    applied to an interrupted run without regenerating anything — which is the whole reason
    the raw dumps are kept.
    """
    frame = pd.read_parquet(raw_dir / f"{lang}_{name}.parquet")
    raw = frame.pop("raw_output").tolist()
    return filter_fn(raw, frame, lang)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/generation/teacher.yaml")
    parser.add_argument("--artifacts-dir", help="defaults to the config's artifacts_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(resolve_output_path(args.artifacts_dir or cfg["artifacts_dir"]))
    raw_dir = out_dir / "raw"
    if not raw_dir.exists():
        raise SystemExit(f"no raw dumps under {raw_dir} — run the generator first")

    present = available(raw_dir)
    log.info("raw dumps present: %s", {k: sorted(v) for k, v in present.items()})

    invariant_all, biased_all, reports = [], [], {}
    for lang in sorted(present):
        passes = present[lang]
        if not {"reference", "invariant"} <= passes:
            log.warning("%s: no invariant pass — skipped entirely", lang)
            continue

        _anchored, reports.setdefault(lang, {})["reference"] = _refilter(
            raw_dir, lang, "reference", F.filter_reference
        )
        invariant, reports[lang]["invariant"] = _refilter(
            raw_dir, lang, "invariant", F.filter_invariant
        )
        invariant_all.append(invariant)
        log.info("%s: %d invariant rows kept", lang, len(invariant))

        if "biased" in passes:
            biased, reports[lang]["biased"] = _refilter(
                raw_dir, lang, "biased", F.filter_biased
            )
            biased_all.append(biased)
            log.info("%s: %d biased rows kept", lang, len(biased))
        else:
            log.warning(
                "%s: biased pass missing — this language contributes to SFT but NOT to the "
                "preference dataset", lang
            )

    if not invariant_all:
        raise SystemExit("nothing to assemble")

    invariant = pd.concat(invariant_all, ignore_index=True)
    biased = pd.concat(biased_all, ignore_index=True) if biased_all else pd.DataFrame()

    sft = D.build_sft_dataset(invariant)
    if cfg.get("balance_decisions", True):
        sft, reports["balance"] = F.balance_decisions(
            sft, seed=cfg.get("seed", 42), column="decision"
        )
    dpo = D.build_dpo_dataset(invariant, biased)
    # Derived from the DPO frame, not generated separately, so KTO and DPO see identical
    # responses and any difference between them is the objective rather than the data.
    kto = D.build_kto_dataset(dpo)

    splits, coverage = {}, {}
    for name, frame in (("sft", sft), ("dpo", dpo), ("kto", kto)):
        if frame.empty:
            log.warning("%s dataset is empty — nothing written", name)
            continue
        assert_no_leakage(frame, f"assemble_datasets.{name}")
        coverage[name] = sorted(frame["lang"].unique())
        train, validation = D.train_val_split(
            frame, val_fraction=float(cfg.get("val_fraction", 0.05)), seed=cfg.get("seed", 42)
        )
        train.to_parquet(out_dir / f"{name}_train.parquet", index=False)
        splits[f"{name}_train"] = len(train)
        if not validation.empty:
            validation.to_parquet(out_dir / f"{name}_validation.parquet", index=False)
            splits[f"{name}_validation"] = len(validation)
        log.info("%s: %d train / %d validation, languages %s",
                 name, len(train), len(validation), coverage[name])

    (out_dir / "generation_report.json").write_text(
        json.dumps(
            {
                "teacher_model": cfg["teacher_model"],
                "assembled_from_raw": True,
                "passes_present": {k: sorted(v) for k, v in present.items()},
                "language_coverage": coverage,
                "splits": splits,
                "filters": reports,
                "seed": cfg.get("seed", 42),
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nassembled into {out_dir}")
    for key, value in splits.items():
        print(f"  {key}: {value:,} rows")
    for name, langs in coverage.items():
        expected = set(cfg.get("languages", ("en", "uk")))
        missing = expected - set(langs)
        marker = f"  ⚠ MISSING {sorted(missing)}" if missing else ""
        print(f"  {name} covers {langs}{marker}")


if __name__ == "__main__":
    main()
