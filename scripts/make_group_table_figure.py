# ruff: noqa: E501  (bilingual caption strings)
"""Renders one protected group's full comparison table as an image, in English and Ukrainian.

Every mitigation on one model and language, against the baseline on the same rows, with every
metric: counterfactual set stability (paired, matched variants, candidate-clustered CI), then
the report's disparity and utility measures recomputed over the same cells. The lexical
scrubber is starred and footnoted as the oracle upper bound it is.

    python scripts/make_group_table_figure.py --model Qwen3.5-9B --lang en --group military_status
    # -> figures/{en,uk}/table_<group>_<model>_<lang>.png (+ .pdf, + .csv)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from hiring_bias_mitigation.analysis import stability as S  # noqa: E402
from hiring_bias_mitigation.analysis.stability import _decision  # noqa: E402
from hiring_bias_mitigation.eval.report import (  # noqa: E402
    _restricted_metrics,
    load_records,
)
from hiring_bias_mitigation.utils.config import REPO_ROOT, resolve_output_path  # noqa: E402

TEXT = {
    "en": {
        "title": "{model} {lang}: {group} — every mitigation against the baseline on the same rows",
        "cols": ["Mitigation", "Condition", "Unstable sets %\n[95% CI]", "Δ pp", "Fixed :\nbroken",
                 "p", "MAD\npp ↓", "AR range\npp ↓", "Cohen's h\n↓", "Inconsist.\n% ↓",
                 "Hire\n%", "Utility\n% ↑", "Mean\nFS", "Attr.\nmentioned %"],
        "baseline": "baseline (no mitigation)",
        "legend": "Values: mitigated run, baseline in brackets, both on exactly the rows the run "
                  "evaluated. Unstable sets: share with its own 95% CI (bootstrap over candidates); "
                  "Δ pp: change against the baseline; bold coloured Δ = p < 0.05 (exact sign test on "
                  "fixed vs broken sets, uncorrected). A run with parse failures is compared on its "
                  "complete sets only; its own baseline on those sets is shown under its value. Italic ◆ = best value in the "
                  "column within its condition (real mitigations only; hire rate and FS have no "
                  "better direction).",
        "fixed": "Fixed : broken — each counterfactual set (one candidate–job pair, one condition, "
                 "all attribute variants) paired with the same set at baseline. Fixed: unstable at "
                 "baseline (the attribute alone changed the decision), stable after the "
                 "mitigation. Broken: stable at baseline, unstable after — the mitigation "
                 "introduced a new dependence on the attribute. Δ unstable sets = (broken − fixed) "
                 "/ sets; e.g. 58 : 57 means the method fixes about as many sets as it breaks.",
        "oracle": "* Lexical scrubbing is an oracle upper bound, not a comparable mitigation: it "
                  "removes the attribute with the study's own injection templates, so every "
                  "variant becomes character-for-character the attribute-free CV (100% of rows) "
                  "and the model sees one identical prompt — zero instability by construction. "
                  "Real CVs do not use our templates; read this row only as the ceiling. "
                  "LLM scrubbing, which must find the attribute itself, is the realistic version.",
        "sft": "SFT (LoRA, merged)",
        "own_base": "own baseline",
        "cond": {"explicit": "explicit", "implicit": "implicit"},
        "group": {"military_status": "military status", "gender": "gender", "religion": "religion"},
    },
    "uk": {
        "title": "{model} {lang}: {group} — усі мітигації проти базової моделі на тих самих рядках",
        "cols": ["Мітигація", "Умова", "Нестабільні\nнабори % [95% ДІ]", "Δ в.п.", "Виправл. :\nзламано",
                 "p", "MAD\nв.п. ↓", "Розмах AR\nв.п. ↓", "Cohen's h\n↓", "Неузгодж.\n% ↓",
                 "Найм\n%", "Корисність\n% ↑", "Серед.\nFS", "Згадки\nатрибута %"],
        "baseline": "базова модель (без мітигації)",
        "legend": "Значення: після мітигації, у дужках — базова модель, обидва на тих самих рядках. "
                  "Нестабільні набори: частка з власним 95% ДІ (бутстреп за кандидатами); Δ в.п.: "
                  "зміна відносно базової моделі; жирна кольорова Δ — p < 0,05 (точний знаковий тест "
                  "виправлених проти зламаних наборів, без корекції). Ран зі збоями парсингу "
                  "порівнюється лише на повністю розібраних наборах; його власна база на них — під значенням. Курсив ◆ — найкраще "
                  "значення в стовпці в межах умови (лише реальні мітигації; для частки найму "
                  "і FS «кращого» напряму немає).",
        "fixed": "Виправлено : зламано — кожен контрфактичний набір (одна пара кандидат–вакансія, "
                 "одна умова, усі варіанти атрибута) порівнюється з тим самим набором у базовій "
                 "моделі. Виправлено: у базовій моделі набір був нестабільним (сам атрибут змінював "
                 "рішення), після мітигації — стабільний. Зламано: був стабільним, став "
                 "нестабільним — мітигація внесла нову залежність від атрибута. Δ нестабільних "
                 "наборів = (зламано − виправлено) / кількість наборів; напр., 58 : 57 означає, що "
                 "метод виправляє приблизно стільки ж, скільки ламає.",
        "oracle": "* Лексичне очищення — оракульна верхня межа, а не порівнювана мітигація: воно "
                  "вирізає атрибут за власними шаблонами ін'єкцій дослідження, тож кожен варіант "
                  "посимвольно збігається з CV без атрибута (100% рядків), і модель бачить один і "
                  "той самий промпт — нульова нестабільність за побудовою. Реальні CV не пишуться "
                  "нашими шаблонами; цей рядок — лише стеля. Реалістична версія — очищення LLM, "
                  "яке мусить знайти атрибут саме.",
        "sft": "SFT (LoRA, злиті ваги)",
        "own_base": "власна база",
        "cond": {"explicit": "явна", "implicit": "неявна"},
        "group": {"military_status": "військовий статус", "gender": "стать", "religion": "релігія"},
    },
}


def arm_name(record: dict, lang: str) -> str:
    m = record["meta"].get("mitigation") or {}
    family = m.get("family")
    if family == "sft":
        return TEXT[lang]["sft"]
    if family == "prompt" or record["meta"].get("prompt_strategy", "baseline") != "baseline":
        return f"prompt: {record['meta']['prompt_strategy']}"
    if family == "scrub":
        mode = m.get("mode", "?")
        return f"scrub: {mode}*" if mode == "lexical" else f"scrub: {mode}"
    if family == "embedding":
        return f"LEACE (layer {','.join(map(str, m.get('layers', [])))})"
    return family or "?"


def build(model: str, lang: str, group: str) -> pd.DataFrame:
    raw_dir = Path(resolve_output_path("outputs/raw"))
    records = {r["run_name"]: r for r in load_records(REPO_ROOT / "eval" / "results")}
    base_name = f"{model}--{lang}--baseline"
    base_rec = records[base_name]
    base_rows = S.load_rows(str(raw_dir / f"{base_name}.parquet"))
    base_dec = pd.read_parquet(raw_dir / f"{base_name}.parquet",
                               columns=["protected_group", "condition", "decision"])

    def hire(frame, cond):
        d = frame[(frame.protected_group == group) & (frame.condition == cond)].decision.map(_decision)
        return 100 * (d == "hire").mean()

    out = []
    for name, rec in sorted(records.items()):
        if not name.startswith(f"{model}--{lang}--") or name == base_name or "--smoke" in name:
            continue
        for cond in ("implicit", "explicit"):
            cell = f"{group}::{cond}"
            if cell not in rec["groups"]:
                continue
            rows = S.load_rows(str(raw_dir / f"{name}.parquet"))
            rows = rows[(rows.protected_group == group) & (rows.condition.isin([cond, "attr_free"]))]
            st = S.compare(base_rows, rows)
            a, b = _restricted_metrics(rec, {cell}), _restricted_metrics(base_rec, {cell})
            dec = pd.read_parquet(raw_dir / f"{name}.parquet",
                                  columns=["protected_group", "condition", "decision"])
            out.append({
                "run": name, "record": rec, "condition": cond, "sets": st["sets"],
                "unstable": st["unstable_run_pct"], "unstable_base": st["unstable_base_pct"],
                "delta": st["unstable_run_pct"] - st["unstable_base_pct"],
                "ci_low": st["ci_low"], "ci_high": st["ci_high"],
                "run_ci_low": st["run_ci_low"], "run_ci_high": st["run_ci_high"],
                "base_ci_low": st["base_ci_low"], "base_ci_high": st["base_ci_high"],
                "fixed": st["fixed"],
                "broken": st["broken"], "p": st["p_sign"],
                **{f"{k}": a.get(k) for k in ("ar_mad", "ar_range", "mean_abs_cohens_h",
                                              "inconsistency_rate", "reference_agreement",
                                              "feedback_similarity", "attribute_mention_rate")},
                **{f"{k}_base": b.get(k) for k in ("ar_mad", "ar_range", "mean_abs_cohens_h",
                                                   "inconsistency_rate", "reference_agreement",
                                                   "feedback_similarity",
                                                   "attribute_mention_rate")},
                "hire": hire(dec, cond), "hire_base": hire(base_dec, cond),
            })
    return pd.DataFrame(out)


def render(frame: pd.DataFrame, model: str, lang_run: str, group: str, lang: str, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    txt = TEXT[lang]
    frame = frame.assign(arm=frame["record"].map(lambda r: arm_name(r, lang)))
    # The oracle goes last in its block: it is the ceiling, not a competitor.
    frame = frame.assign(oracle=frame["arm"].str.endswith("*")).sort_values(
        ["condition", "oracle", "delta"], ascending=[False, True, True])

    def pct(v, vb, digits=2, scale=100):
        return f"{v * scale:.{digits}f} ({vb * scale:.{digits}f})"

    cells, styles, blocks, significant, raw = [], [], [], [], []
    for cond, block in frame.groupby("condition", sort=False):
        # The baseline row comes from the arm measured on the most sets: an arm with parse
        # failures (LEACE on Ukrainian) is compared on its complete sets only, a subset whose
        # baseline differs, and it must not set the block's reference row.
        first = block.loc[block["sets"].idxmax()]
        cells.append([txt["baseline"], txt["cond"][cond],
                      f"{first.unstable_base:.2f}  [{first.base_ci_low:.2f}, {first.base_ci_high:.2f}]",
                      "", "", "", f"{first.ar_mad_base * 100:.2f}", f"{first.ar_range_base * 100:.2f}",
                      f"{first.mean_abs_cohens_h_base:.2f}",
                      f"{first.inconsistency_rate_base * 100:.2f}", f"{first.hire_base:.2f}",
                      f"{first.reference_agreement_base * 100:.2f}",
                      f"{first.feedback_similarity_base:.2f}",
                      f"{first.attribute_mention_rate_base * 100:.2f}"])
        styles.append("baseline")
        blocks.append(cond)
        significant.append(False)
        raw.append({})
        for r in block.itertuples():
            cells.append([
                r.arm, txt["cond"][cond],
                f"{r.unstable:.2f}  [{r.run_ci_low:.2f}, {r.run_ci_high:.2f}]"
                + (f"\n({txt['own_base']} {r.unstable_base:.2f}, n={r.sets})"
                   if abs(r.unstable_base - first.unstable_base) > 0.5 else ""),
                f"{r.delta:+.2f}",
                f"{r.fixed} : {r.broken}", f"{r.p:.0e}" if r.p < 0.01 else f"{r.p:.2f}",
                pct(r.ar_mad, r.ar_mad_base), pct(r.ar_range, r.ar_range_base),
                pct(r.mean_abs_cohens_h, r.mean_abs_cohens_h_base, 2, 1),
                pct(r.inconsistency_rate, r.inconsistency_rate_base),
                f"{r.hire:.2f} ({r.hire_base:.2f})",
                pct(r.reference_agreement, r.reference_agreement_base),
                pct(r.feedback_similarity, r.feedback_similarity_base, 2, 1),
                pct(r.attribute_mention_rate, r.attribute_mention_rate_base),
            ])
            blocks.append(cond)
            significant.append(r.p < 0.05)
            # unrounded values, so "best" is decided before rounding creates ties
            # an arm measured on a much smaller subset is not ranked against the others
            raw.append({} if r.sets < 0.9 * first.sets else {2: r.unstable, 3: r.delta, 6: r.ar_mad, 7: r.ar_range,
                        8: r.mean_abs_cohens_h, 9: r.inconsistency_rate,
                        11: r.reference_agreement, 13: r.attribute_mention_rate})
            styles.append("oracle" if r.arm.endswith("*") else
                          "sft" if r.arm.startswith("SFT") else
                          ("sig" if r.p < 0.05 else "plain"))

    # Best value per column within each condition block, among real mitigations only: the
    # baseline is the reference and the oracle scrubber is a ceiling, not a competitor.
    # Hire rate and FS have no "better" direction, so they are not marked.
    direction = {2: min, 3: min, 6: min, 7: min, 8: min, 9: min, 11: max, 13: min}
    best: set[tuple[int, int]] = set()
    for cond in dict.fromkeys(blocks):
        rows = [i for i, (b, st) in enumerate(zip(blocks, styles))
                if b == cond and st not in ("baseline", "oracle") and raw[i]]
        if len(rows) < 2:
            continue
        for col, pick in direction.items():
            values = {i: raw[i][col] for i in rows}
            target = pick(values.values())
            best |= {(i, col) for i, v in values.items() if v == target}

    row_h, notes_h = 0.34, 1.45  # inches
    fig_h = row_h * (len(cells) + 1.6) + notes_h + 0.6
    fig = plt.figure(figsize=(22, fig_h))
    ax = fig.add_axes([0.01, (notes_h + 0.1) / fig_h, 0.98, 1 - (notes_h + 0.65) / fig_h])
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=txt["cols"], bbox=[0, 0, 1, 1], cellLoc="center",
                     colLoc="center",
                     colWidths=[0.15, 0.045, 0.11, 0.05, 0.05, 0.04, 0.065, 0.07, 0.07, 0.065,
                                0.07, 0.07, 0.065, 0.065])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        if row == 0:
            cell.set_facecolor("#2f3e4e")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            continue
        style = styles[row - 1]
        if col == 0:
            cell.get_text().set_ha("left")
            cell._loc = "left"
        if (row - 1, col) in best:
            cell.get_text().set_fontstyle("italic")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_text(cell.get_text().get_text() + " ◆")
        if style == "baseline":
            cell.set_facecolor("#eeeeee")
        elif style == "sft":
            cell.set_facecolor("#dcecf9")
        elif style == "oracle":
            cell.set_facecolor("#fbf3e4")
            cell.get_text().set_color("#7a6a50")
        if col == 3 and significant[row - 1]:
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color("#1a7f37" if cells[row - 1][3].startswith("-") else "#b42318")
    group_name = txt["group"][group]
    ax.set_title(txt["title"].format(model=model, lang=lang_run.upper(), group=group_name),
                 loc="left", fontsize=13, fontweight="bold", pad=12)
    import textwrap

    notes = [(txt["legend"], "#444444"), (txt["fixed"], "#444444"), (txt["oracle"], "#7a6a50")]
    y = (notes_h - 0.05) / fig_h
    for note, color in notes:
        wrapped = textwrap.fill(note, 230)
        fig.text(0.012, y, wrapped, fontsize=9, color=color, va="top")
        y -= (0.2 + 0.17 * wrapped.count("\n")) / fig_h
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        # not with_suffix: the model name carries a dot ("Qwen3.5")
        fig.savefig(out.parent / f"{out.name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--group", default="military_status")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "figures"))
    args = parser.parse_args()
    if not os.environ.get("HBM_OUTPUT_ROOT"):
        raise SystemExit("HBM_OUTPUT_ROOT is not set -- load .env (the raw generations live there)")
    frame = build(args.model, args.lang, args.group)
    stem = f"table_{args.group}_{args.model}_{args.lang}"
    frame.drop(columns=["record"]).to_csv(Path(args.out_dir) / f"{stem}.csv", index=False)
    for lang in TEXT:
        render(frame, args.model, args.lang, args.group, lang, Path(args.out_dir) / lang / stem)
        print(Path(args.out_dir) / lang / f"{stem}.png")
    print(json.dumps({"rows": len(frame)}))


if __name__ == "__main__":
    main()
