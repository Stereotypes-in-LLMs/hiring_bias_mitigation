"""Builds the paper figure set, in English and Ukrainian.

    python scripts/make_figures.py                      # both languages, SVG + PDF
    python scripts/make_figures.py --formats svg png    # pick formats
    python scripts/make_figures.py --only tradeoff      # one figure, while iterating
    python scripts/make_figures.py --list               # what would be built

Needs the optional plotting extra, which is deliberately not part of the default install:

    uv pip install -e '.[viz]'

Reads scored runs only (`eval/results/*.json`) -- no model is loaded, so this is safe to run
while a sweep is in progress and cheap to re-run after it. Every figure is written beside its
own CSV: a reader who wants to re-plot a number should not have to run this repository, and a
figure whose data cannot be inspected is not evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402
from hiring_bias_mitigation.viz import charts as C  # noqa: E402
from hiring_bias_mitigation.viz import data as D  # noqa: E402
from hiring_bias_mitigation.viz import theme as T  # noqa: E402
from hiring_bias_mitigation.viz.labels import LANGUAGES  # noqa: E402

log = get_logger("make_figures")

DEFAULT_FORMATS = ("svg", "pdf")


def _require_altair() -> None:
    try:
        import altair  # noqa: F401
        import vl_convert  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            f"missing plotting dependency ({exc.name}). Install the extra:\n"
            "    uv pip install -e '.[viz]'"
        ) from exc


def _stability_frame():
    import pandas as pd

    path = Path(__file__).resolve().parents[1] / "reports" / "mitigation_decision_table.csv"
    return pd.read_csv(path) if path.exists() else None


def save(chart, path: Path, formats: tuple[str, ...]) -> list[Path]:
    written = []
    for fmt in formats:
        target = path.with_suffix(f".{fmt}")
        # ppi only affects the raster formats; Vega-Lite ignores it for svg/pdf.
        chart.save(str(target), ppi=300) if fmt == "png" else chart.save(str(target))
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="eval/results")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES), choices=LANGUAGES)
    parser.add_argument("--formats", nargs="+", default=list(DEFAULT_FORMATS),
                        choices=("svg", "pdf", "png", "html", "json"))
    parser.add_argument("--only", nargs="+", help="build only these figures (substring match)")
    parser.add_argument("--list", action="store_true", help="print the figure names and exit")
    parser.add_argument("--keep-going", action="store_true",
                        help="report a failing figure and continue with the rest")
    args = parser.parse_args()

    if args.list:
        for name in C.FIGURES:
            print(name)
        return

    _require_altair()
    T.register()

    family, available = T.check_font()
    if "uk" in args.languages and not family:
        log.warning(
            "no Cyrillic-capable font among %s found by vl-convert (%d families available). "
            "Ukrainian figures may render as empty boxes -- install DejaVu Sans or Noto Sans.",
            T.FONT_STACK, len(available),
        )
    elif family:
        log.info("rendering text with %s", family)

    records = D.load(args.results_dir)
    if not records:
        raise SystemExit(f"no scored runs in {args.results_dir}")
    frames = {
        "runs": D.run_level(records),
        "cells": D.cell_level(records),
        "attributes": D.attribute_level(records),
        # Produced by scripts/set_stability.py + the decision table; absent until it has run.
        "stability": _stability_frame(),
    }
    log.info(
        "%d run(s): %d cell rows, %d attribute rows",
        len(records), len(frames["cells"]), len(frames["attributes"]),
    )

    selected = {
        name: spec for name, spec in C.FIGURES.items()
        if not args.only or any(token in name for token in args.only)
    }
    if not selected:
        raise SystemExit(f"no figure matches {args.only}")

    manifest: dict[str, dict] = {}
    for lang in args.languages:
        out_dir = Path(args.out_dir) / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, (builder, frame_name) in selected.items():
            try:
                chart = builder(frames[frame_name], lang)
            except Exception as exc:
                if not args.keep_going:
                    raise
                log.error("%s [%s] failed: %s", name, lang, exc)
                continue
            if chart is None:
                log.info("%s [%s]: not enough data yet -- skipped", name, lang)
                continue
            written = save(chart, out_dir / name, tuple(args.formats))
            # The frame beside the figure, so a number in the paper can be traced without
            # re-running anything.
            csv_path = out_dir / f"{name}.csv"
            if frames[frame_name] is not None:
                frames[frame_name].to_csv(csv_path, index=False)
            manifest.setdefault(name, {})[lang] = [str(p) for p in written] + [str(csv_path)]
            log.info("%s [%s] -> %s", name, lang, ", ".join(p.name for p in written))

    manifest_path = Path(args.out_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s (%d figure(s))", manifest_path, len(manifest))


if __name__ == "__main__":
    main()
