"""The figures themselves.

Each builder takes a tidy frame plus a language and returns an Altair chart. No builder writes
a literal caption -- every string goes through `labels.t`, which is what keeps the English and
Ukrainian sets from drifting apart.

A builder returns `None` when the data it needs is not there yet. A sweep is days long, and
rebuilding the figure set halfway through should produce the figures that are ready rather
than an exception.
"""

from __future__ import annotations

import pandas as pd

from . import data as D
from .labels import family as family_label
from .labels import strategy, t


def _alt():
    import altair as alt

    return alt


#: Injection conditions, in the order they are always plotted.
CONDITIONS = ("explicit", "implicit", "attr_free")


def _label_variants(frame: pd.DataFrame, lang: str) -> pd.DataFrame:
    out = frame.copy()
    out["variant_label"] = out["variant"].map(lambda v: strategy(v, lang))
    out["group_label"] = out["group"].map(lambda g: t(g, lang)) if "group" in out else None
    if "condition" in out:
        out["condition_label"] = out["condition"].map(
            lambda c: t(c, lang) if c in CONDITIONS else c
        )
    return out


#: Canonical orders, so a series keeps its colour across the English and Ukrainian figures.
#: Without a domain, Vega builds the scale from the sorted label values, and the Ukrainian
#: labels sort differently from the English ones -- the same model came out blue in one figure
#: and orange in its twin.
GROUP_ORDER = ("military_status", "gender", "religion",
               "military_status_x_gender", "military_status_x_religion")
FAMILY_ORDER = ("none", "prompt", "scrub", "embedding", "sft", "dpo")
PALETTE = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#9d755d")


def _pinned_scale(keys: tuple[str, ...], label, frame: pd.DataFrame, column: str, lang: str):
    """A colour scale whose domain is the canonical order, restricted to what is plotted."""
    alt = _alt()
    present = set(frame[column].dropna())
    domain, colours = [], []
    for key, colour in zip(keys, PALETTE):
        value = label(key, lang)
        if value in present:
            domain.append(value)
            colours.append(colour)
    return alt.Scale(domain=domain, range=colours) if domain else alt.Undefined


def _family_scale(frame: pd.DataFrame, lang: str):
    return _pinned_scale(FAMILY_ORDER, family_label, frame, "family_label", lang)


def _condition_sort(lang: str) -> list[str]:
    """Explicit, implicit, attribute-free -- in the figure's language.

    Sorting on the translated label rather than the raw value keeps the panel order identical
    between the English and Ukrainian figures; alphabetical order would not.
    """
    return [t(c, lang) for c in CONDITIONS]


def _wrap(text: str, width: int = 22) -> list[str]:
    """An axis title as lines, for a faceted chart whose panels are shorter than the title.

    A rotated y-axis title is clipped by the panel height, and the Ukrainian strings run about
    a third longer than the English ones they were laid out for -- long enough that two rows of
    panels had their titles overlap in the middle of the figure. Altair accepts a list of
    lines, so the title wraps instead of colliding.
    """
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def baseline_disparity(cells: pd.DataFrame, lang: str):
    """Where the bias is: MAD-style disparity per model, language, group and condition.

    A heatmap rather than bars. The question this answers is "which cells are hot", and a
    reader scanning a 10x6 grid of bars answers it far more slowly than one scanning colour.
    """
    alt = _alt()
    frame = cells[cells["family"] == "none"]
    if frame.empty:
        return None
    frame = _label_variants(frame, lang)

    return (
        alt.Chart(frame)
        .mark_rect(stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("condition_label:N", title=None,
                    axis=alt.Axis(labelAngle=0), sort=_condition_sort(lang)),
            y=alt.Y("group_label:N", title=None),
            color=alt.Color(
                "ar_range_pp:Q",
                title=t("ar", lang),
                scale=alt.Scale(scheme="oranges"),
            ),
            tooltip=["model", "lang", "group_label", "condition_label",
                     alt.Tooltip("ar_range_pp:Q", format=".1f")],
        )
        .properties(width=150, height=alt.Step(26))
        .facet(column=alt.Column("model:N", title=None), row=alt.Row("lang:N", title=None))
        .properties(title=t("fig_baseline_title", lang))
    )


def fairness_utility_tradeoff(runs: pd.DataFrame, lang: str):
    """The figure that decides which mitigation is worth having.

    Change in disparity against change in utility, one point per run. Quadrants carry the whole
    argument: down-and-right is a free win, down-and-left bought fairness by damaging the
    model, and nothing else matters much. The rules at zero are what make the quadrants
    readable; without them a reader has to find the origin themselves.
    """
    alt = _alt()
    frame = D.with_baseline(runs, ("mad_pp", "utility_pct"))
    frame = frame[frame["family"] != "none"].dropna(subset=["mad_pp_delta", "utility_pct_delta"])
    if frame.empty:
        return None
    frame = _label_variants(frame, lang)

    base = alt.Chart(frame)
    points = base.mark_point(filled=True, opacity=0.85).encode(
        x=alt.X("mad_pp_delta:Q", title=t("mad_delta", lang)),
        y=alt.Y("utility_pct_delta:Q", title=t("utility_delta", lang)),
        color=alt.Color("family:N", title=t("strategy", lang)),
        shape=alt.Shape("lang:N", title=None),
        tooltip=["run", "model", "lang", "family", "variant_label",
                 alt.Tooltip("mad_pp_delta:Q", format="+.2f"),
                 alt.Tooltip("utility_pct_delta:Q", format="+.2f")],
    )
    zero_x = base.mark_rule(strokeDash=[4, 4]).encode(x=alt.datum(0))
    zero_y = base.mark_rule(strokeDash=[4, 4]).encode(y=alt.datum(0))
    return (
        (zero_x + zero_y + points)
        .properties(
            width=420, height=320,
            title=alt.TitleParams(
                t("fig_tradeoff_title", lang),
                subtitle=[t("n_runs", lang, n=len(frame)), t("tradeoff_hint", lang)],
            ),
        )
    )


def strategy_ranking(runs: pd.DataFrame, lang: str, family: str = "prompt"):
    """Disparity after each strategy, with the model's own baseline drawn as a rule.

    Sorted by effect within each panel, so the ordering itself is a result rather than
    alphabetical noise.

    The baseline arrives as a *column* on the same frame rather than as a second data source.
    Vega-Lite cannot facet a layered chart whose layers carry different data, so joining first
    is what lets the baseline rule appear inside every panel.
    """
    alt = _alt()
    frame = runs[runs["family"] == family].dropna(subset=["mad_pp"])
    if frame.empty:
        return None
    frame = D.with_baseline(runs, ("mad_pp",))
    frame = frame[frame["family"] == family].dropna(subset=["mad_pp"])
    if frame.empty:
        return None
    frame = _label_variants(frame, lang)

    base = alt.Chart(frame)
    bars = base.mark_bar().encode(
        x=alt.X("mad_pp:Q", title=t("mad", lang)),
        # The Ukrainian labels carry a gloss under the config name, so both the axis labels
        # and the legend title need more room than Vega's 180px default, which clips them.
        y=alt.Y("variant_label:N", title=None, sort="x",
                axis=alt.Axis(labelLimit=400, labelExpr="split(datum.label, '\\n')")),
        color=alt.Color("utility_pct:Q", title=t("utility", lang),
                        scale=alt.Scale(scheme="blues"),
                        legend=alt.Legend(titleLimit=400)),
        tooltip=["model", "lang", "variant_label",
                 alt.Tooltip("mad_pp:Q", format=".2f"),
                 alt.Tooltip("utility_pct:Q", format=".1f")],
    )
    rule = base.mark_rule(color="#D55E00", strokeDash=[4, 3], size=1.5).encode(
        x="mad_pp_baseline:Q",
        tooltip=[alt.Tooltip("mad_pp_baseline:Q", format=".2f")],
    )
    # The Ukrainian labels carry a gloss on a second line, which needs a taller row than the
    # one-line English ones -- at the English step the gloss overlaps the name below it.
    step = 34 if frame["variant_label"].str.contains("\n").any() else 20
    return (
        alt.layer(bars, rule)
        .properties(width=220, height=alt.Step(step))
        .facet(column=alt.Column("model:N", title=None), row=alt.Row("lang:N", title=None))
        .properties(
            title=alt.TitleParams(
                t("fig_strategy_title", lang),
                subtitle=f"{t('baseline', lang)}: - - -",
            )
        )
    )


def leak_versus_disparity(runs: pd.DataFrame, lang: str):
    """Residual attribute mentions against residual disparity.

    The scrub arms sit at one end and the untreated baselines at the other, which is what makes
    the relation visible: a mitigation that removes the attribute from the text removes the
    disparity with it, and whatever fraction leaks back brings some disparity back too.
    """
    alt = _alt()
    frame = runs.dropna(subset=["leak_pct", "mad_pp"])
    if frame.empty or frame["leak_pct"].nunique() < 2:
        return None
    frame = _label_variants(frame, lang)

    points = alt.Chart(frame).mark_point(filled=True, opacity=0.85).encode(
        x=alt.X("leak_pct:Q", title=t("leak", lang)),
        y=alt.Y("mad_pp:Q", title=t("mad", lang)),
        color=alt.Color("family:N", title=t("strategy", lang)),
        shape=alt.Shape("lang:N", title=None),
        tooltip=["run", "family", "variant_label",
                 alt.Tooltip("leak_pct:Q", format=".2f"),
                 alt.Tooltip("mad_pp:Q", format=".2f")],
    )
    trend = points.transform_regression("leak_pct", "mad_pp").mark_line(
        color="#555555", strokeDash=[5, 3]
    )
    return (points + trend).properties(
        width=420, height=300,
        title=alt.TitleParams(t("fig_leak_title", lang), subtitle=t("leak_hint", lang)),
    )


def attribute_gaps(attributes: pd.DataFrame, lang: str, run: str | None = None):
    """Per-attribute acceptance rate with bootstrap intervals -- a forest plot.

    This is the figure that shows *direction*: which attribute is advantaged, by how much, and
    whether the interval clears the reference level. The aggregate figures cannot show that,
    and a fairness claim about a specific group needs it.
    """
    alt = _alt()
    frame = attributes if run is None else attributes[attributes["run"] == run]
    frame = frame.dropna(subset=["ar_pct"])
    if frame.empty:
        return None
    frame = frame.copy()
    frame["marker"] = frame["significant"].map(
        {True: t("significant", lang), False: ""}
    )

    base = alt.Chart(frame)
    intervals = base.mark_rule(size=1.5, color="#888888").encode(
        y=alt.Y("attribute:N", title=None, sort="-x"),
        x=alt.X("ci_low_pct:Q", title=t("ar", lang)),
        x2="ci_high_pct:Q",
    )
    points = base.mark_point(filled=True).encode(
        y=alt.Y("attribute:N", title=None, sort="-x"),
        x="ar_pct:Q",
        color=alt.Color("significant:N", title=t("significant", lang)),
        tooltip=["attribute", alt.Tooltip("ar_pct:Q", format=".1f"),
                 alt.Tooltip("gap_pp:Q", format="+.1f"),
                 alt.Tooltip("paired_p:Q", format=".4f")],
    )
    reference = base.transform_filter(alt.datum.is_reference).mark_rule(
        color="#0072B2", strokeDash=[4, 3]
    ).encode(x="ar_pct:Q")
    return (
        (intervals + reference + points)
        .properties(width=300, height=alt.Step(18))
        .facet(column=alt.Column("condition:N", title=None))
        .properties(title=t("fig_attribute_title", lang))
    )


def language_transfer(runs: pd.DataFrame, lang: str, family: str = "prompt"):
    """The same strategy in both languages, paired by model.

    Built because the prompt results showed a strategy inverting between languages, and a
    claim that large needs a figure a reader can check at a glance rather than a sentence.
    """
    alt = _alt()
    frame = runs[runs["family"] == family].dropna(subset=["mad_pp"])
    wide = frame.pivot_table(
        index=["model", "variant"], columns="lang", values="mad_pp"
    ).reset_index()
    if not {"en", "uk"} <= set(wide.columns):
        return None
    wide = wide.dropna(subset=["en", "uk"])
    if wide.empty:
        return None
    wide["variant_label"] = wide["variant"].map(lambda v: strategy(v, lang))

    limit = float(max(wide["en"].max(), wide["uk"].max())) * 1.1
    base = alt.Chart(wide)
    diagonal = base.mark_line(color="#BBBBBB", strokeDash=[4, 4]).encode(
        x=alt.datum(0), y=alt.datum(0), x2=alt.datum(limit), y2=alt.datum(limit)
    )
    points = base.mark_point(filled=True, opacity=0.85).encode(
        x=alt.X("en:Q", title=f"{t('mad', lang)} — en",
                scale=alt.Scale(domain=[0, limit])),
        y=alt.Y("uk:Q", title=f"{t('mad', lang)} — uk",
                scale=alt.Scale(domain=[0, limit])),
        color=alt.Color("model:N", title=t("model", lang)),
        tooltip=["model", "variant_label",
                 alt.Tooltip("en:Q", format=".2f"), alt.Tooltip("uk:Q", format=".2f")],
    )
    return (diagonal + points).properties(
        width=340, height=340, title=t("fig_language_title", lang)
    )


def condition_contrast(cells: pd.DataFrame, lang: str):
    """Explicit against implicit injection, per model and group.

    The audit's central finding was that these two can disagree, so the paper needs the figure
    that shows the disagreement rather than two numbers in a table.
    """
    alt = _alt()
    frame = cells[cells["condition"].isin(["explicit", "implicit"])]
    frame = frame.dropna(subset=["ar_range_pp"])
    if frame.empty:
        return None
    frame = _label_variants(frame, lang)

    return (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("condition_label:N", title=None, sort=_condition_sort(lang)[:2],
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("ar_range_pp:Q", title=t("ar", lang)),
            color=alt.Color("group_label:N", title=None,
                            scale=_pinned_scale(GROUP_ORDER, lambda g, lg: t(g, lg),
                                                frame, "group_label", lang),
                            legend=alt.Legend(labelLimit=400)),
            detail="group_label:N",
            tooltip=["model", "lang", "group_label", "condition_label",
                     alt.Tooltip("ar_range_pp:Q", format=".1f")],
        )
        .properties(width=120, height=220)
        .facet(column=alt.Column("model:N", title=None), row=alt.Row("lang:N", title=None))
        .properties(title=t("fig_condition_title", lang))
    )


def stability_utility_tradeoff(table: pd.DataFrame, lang: str):
    """The paper's decision figure: consistency gained against utility lost, one point per run.

    Built on counterfactual set stability rather than MAD. Set stability is compared on matched
    variants and pairs each set with itself at baseline, so a run that fixes a few dozen sets is
    visible here while it is invisible in an attribute-averaged disparity. Hollow points did not
    survive FDR correction.
    """
    alt = _alt()
    if table is None or table.empty:
        return None
    frame = table.copy()
    frame["significant"] = frame["p_fdr"] < 0.05
    frame["variant_label"] = frame["variant"].fillna("").map(
        lambda v: strategy(v, lang) if v else ""
    )
    frame["label"] = (frame["model"].str.replace("Qwen3.5-", "") + " " + frame["lang"])
    frame["family_label"] = frame["family"].map(lambda f: family_label(f, lang))

    base = alt.Chart(frame)
    encoding = {
        "x": alt.X("delta_unstable_pp:Q", title=t("stability_delta", lang)),
        "y": alt.Y("delta_utility_pp:Q", title=t("utility_delta", lang)),
        "color": alt.Color("family_label:N", title=t("strategy", lang),
                           scale=_family_scale(frame, lang),
                           legend=alt.Legend(labelLimit=400)),
        "shape": alt.Shape("label:N", title=None),
        "tooltip": ["model", "lang", "family_label", "variant_label",
                    alt.Tooltip("delta_unstable_pp:Q", format="+.1f"),
                    alt.Tooltip("ci_low:Q", format="+.1f"),
                    alt.Tooltip("ci_high:Q", format="+.1f"),
                    alt.Tooltip("delta_utility_pp:Q", format="+.1f"),
                    alt.Tooltip("p_fdr:Q", format=".1e")],
    }
    # Two layers rather than a conditional fill: a white fill on a white page makes a
    # non-significant point disappear, and the non-significant cluster at the origin -- every
    # training adapter -- is one of the figure's two findings.
    significant = base.transform_filter("datum.significant").mark_point(
        filled=True, size=90, opacity=0.9
    ).encode(**encoding)
    not_significant = base.transform_filter("!datum.significant").mark_point(
        filled=False, size=90, strokeWidth=2
    ).encode(**encoding)
    errors = base.mark_rule(opacity=0.35).encode(
        x="ci_low:Q", x2="ci_high:Q", y="delta_utility_pp:Q",
        color=alt.Color("family_label:N", title=t("strategy", lang),
                        scale=_family_scale(frame, lang)),
    )
    zero_x = base.mark_rule(strokeDash=[4, 4]).encode(x=alt.datum(0))
    zero_y = base.mark_rule(strokeDash=[4, 4]).encode(y=alt.datum(0))
    points = significant + not_significant
    return (zero_x + zero_y + errors + points).properties(
        width=460, height=340,
        title=alt.TitleParams(t("fig_stability_title", lang), subtitle=t("stability_hint", lang)),
    )


def training_curves(curves: pd.DataFrame, lang: str):
    """SFT training curves exported from W&B, one line per run, the audited checkpoint marked."""
    alt = _alt()
    if curves is None or curves.empty:
        return None
    frame = curves.copy()
    frame["metric_label"] = frame["metric"].map(lambda m: t(m, lang))
    frame["run_label"] = (frame["run"].str.replace("qwen3.5-", "Qwen3.5-")
                          .str.replace("lapa-12b", "LAPA-12B")
                          .str.replace(r"-(\d+)b_", lambda m: f"-{m.group(1)}B_", regex=True)
                          .str.replace("_only", "").str.replace("_", " ").str.replace(" en", " EN")
                          .str.replace(" uk", " UK"))
    order = [t(m, lang) for m in ("train/loss", "eval/loss", "eval/mean_token_accuracy")]
    frame = frame[frame["metric_label"].isin(order)]
    color = alt.Color("run_label:N", title=t("model", lang))
    # Layers without their own data, so the facet can own it (Altair requires that).
    lines = alt.Chart().mark_line(point=alt.OverlayMarkDef(size=18)).encode(
        x=alt.X("step:Q", title=t("step", lang)),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
        color=color,
        tooltip=["run_label", "metric_label", "step", alt.Tooltip("value:Q", format=".4f")],
    )
    kept = alt.Chart().mark_rule(strokeDash=[4, 3], opacity=0.5).encode(
        x="kept_step:Q", color=color)
    return alt.layer(lines, kept, data=frame).properties(width=230, height=170).facet(
        column=alt.Column("metric_label:N", title=None, sort=order),
    ).resolve_scale(y="independent").properties(
        title=alt.TitleParams(t("fig_training_title", lang), subtitle=t("training_hint", lang)))


def operating_point(curves: pd.DataFrame, lang: str):
    """Unstable-set share against hire rate, base vs SFT, from the threshold sweep."""
    alt = _alt()
    if curves is None or curves.empty:
        return None
    frame = curves.copy()
    frame["hire_pct"] = 100 * frame["hire_rate"]
    frame["unstable_pct"] = 100 * frame["unstable"]
    order = [t("base_model", lang), t("sft_model", lang)]
    frame["model_label"] = frame["model"].map(dict(zip(("base", "adapter"), order)))
    # An explicit domain, so the base model is the same colour in both languages. Without it
    # the scale is built from the sorted label values, and "Базова модель" sorts against
    # "SFT (LoRA, ...)" the other way round than "Base model" does.
    color = alt.Color("model_label:N", title=None,
                      scale=alt.Scale(domain=order, range=["#4c78a8", "#f58518"]),
                      legend=alt.Legend(labelLimit=400))
    base = alt.Chart(frame)
    lines = base.transform_filter("datum.kind == 'sweep'").mark_line().encode(
        x=alt.X("hire_pct:Q", title=t("hire_rate", lang)),
        y=alt.Y("unstable_pct:Q", title=_wrap(t("unstable_sets", lang))),
        color=color,
        tooltip=["cell", "model_label", alt.Tooltip("hire_pct:Q", format=".1f"),
                 alt.Tooltip("unstable_pct:Q", format=".1f")],
    )
    dots = base.transform_filter("datum.kind == 'greedy'").mark_point(
        filled=True, size=90).encode(x="hire_pct:Q", y="unstable_pct:Q", color=color)
    return (lines + dots).properties(width=220, height=180).facet(
        facet=alt.Facet("cell:N", title=None), columns=3,
    ).properties(title=alt.TitleParams(t("fig_operating_title", lang),
                                       subtitle=t("operating_hint", lang)))


def matched_scope(table: pd.DataFrame, lang: str):
    """Every Qwen3.5-9B English arm on the identical 450 sets: consistency and utility."""
    alt = _alt()
    if table is None or table.empty:
        return None
    frame = table.copy()
    frame["family"] = frame["arm"].map(
        lambda a: "sft" if a.startswith("SFT") else a.split(":")[0].replace("LEACE", "embedding"))
    frame["family_label"] = frame["family"].map(lambda f: family_label(f, lang))
    frame["arm_label"] = frame["arm"].map(
        lambda a: "SFT" if a.startswith("SFT") else
        ("LEACE" if a == "LEACE" else strategy(a.split(": ", 1)[1], lang).split("\n")[0]
         if a.startswith("prompt") else a))
    frame["significant"] = frame["p_sign"] < 0.05
    sort = alt.EncodingSortField(field="delta_pp", order="ascending")
    y = alt.Y("arm_label:N", title=None, sort=sort)
    color = alt.Color("family_label:N", title=t("strategy", lang),
                      scale=_family_scale(frame, lang),
                      legend=alt.Legend(labelLimit=400))
    base = alt.Chart(frame)
    err = base.mark_rule(strokeWidth=2).encode(
        x=alt.X("ci_low:Q", title=t("stability_delta", lang)), x2="ci_high:Q", y=y, color=color)
    pts = base.mark_point(filled=True, size=80).encode(
        x="delta_pp:Q", y=y, color=color,
        tooltip=["arm", alt.Tooltip("delta_pp:Q", format="+.1f"),
                 alt.Tooltip("ci_low:Q", format="+.1f"), alt.Tooltip("ci_high:Q", format="+.1f"),
                 alt.Tooltip("d_utility_pp:Q", format="+.1f"), "fixed", "broken"])
    zero = base.mark_rule(strokeDash=[4, 4]).encode(x=alt.datum(0))
    left = (zero + err + pts).properties(width=300, height=300)
    util = base.mark_bar().encode(
        x=alt.X("d_utility_pp:Q", title=t("utility_delta", lang)),
        y=alt.Y("arm_label:N", sort=sort, axis=None), color=color).properties(width=150, height=300)
    return alt.hconcat(left, util).properties(
        title=alt.TitleParams(t("fig_scope_title", lang), subtitle=t("scope_hint", lang)))


#: name -> (builder, which frame it needs)
FIGURES = {
    "baseline_disparity": (baseline_disparity, "cells"),
    "stability_utility_tradeoff": (stability_utility_tradeoff, "stability"),
    "fairness_utility_tradeoff": (fairness_utility_tradeoff, "runs"),
    "strategy_ranking": (strategy_ranking, "runs"),
    "leak_versus_disparity": (leak_versus_disparity, "runs"),
    "attribute_gaps": (attribute_gaps, "attributes"),
    "language_transfer": (language_transfer, "runs"),
    "condition_contrast": (condition_contrast, "cells"),
    "training_curves": (training_curves, "training"),
    "operating_point": (operating_point, "operating"),
    "matched_scope": (matched_scope, "scope"),
}
