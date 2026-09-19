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
from .labels import strategy, t


def _alt():
    import altair as alt

    return alt


def _label_variants(frame: pd.DataFrame, lang: str) -> pd.DataFrame:
    out = frame.copy()
    out["variant_label"] = out["variant"].map(lambda v: strategy(v, lang))
    out["group_label"] = out["group"].map(lambda g: t(g, lang)) if "group" in out else None
    return out


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
            x=alt.X("condition:N", title=None,
                    axis=alt.Axis(labelAngle=0), sort=["explicit", "implicit", "attr_free"]),
            y=alt.Y("group_label:N", title=None),
            color=alt.Color(
                "ar_range_pp:Q",
                title=t("ar", lang),
                scale=alt.Scale(scheme="oranges"),
            ),
            tooltip=["model", "lang", "group_label", "condition",
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
        y=alt.Y("variant_label:N", title=None, sort="x"),
        color=alt.Color("utility_pct:Q", title=t("utility", lang),
                        scale=alt.Scale(scheme="blues")),
        tooltip=["model", "lang", "variant_label",
                 alt.Tooltip("mad_pp:Q", format=".2f"),
                 alt.Tooltip("utility_pct:Q", format=".1f")],
    )
    rule = base.mark_rule(color="#D55E00", strokeDash=[4, 3], size=1.5).encode(
        x="mad_pp_baseline:Q",
        tooltip=[alt.Tooltip("mad_pp_baseline:Q", format=".2f")],
    )
    return (
        alt.layer(bars, rule)
        .properties(width=220, height=alt.Step(20))
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
            x=alt.X("condition:N", title=None, sort=["explicit", "implicit"],
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("ar_range_pp:Q", title=t("ar", lang)),
            color=alt.Color("group_label:N", title=None),
            detail="group_label:N",
            tooltip=["model", "lang", "group_label", "condition",
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
    survive FDR correction; the training arms sit in a tight cluster at the origin.
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

    base = alt.Chart(frame)
    encoding = {
        "x": alt.X("delta_unstable_pp:Q", title=t("stability_delta", lang)),
        "y": alt.Y("delta_utility_pp:Q", title=t("utility_delta", lang)),
        "color": alt.Color("family:N", title=t("strategy", lang)),
        "shape": alt.Shape("label:N", title=None),
        "tooltip": ["model", "lang", "family", "variant_label",
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
        color=alt.Color("family:N", title=t("strategy", lang)),
    )
    zero_x = base.mark_rule(strokeDash=[4, 4]).encode(x=alt.datum(0))
    zero_y = base.mark_rule(strokeDash=[4, 4]).encode(y=alt.datum(0))
    points = significant + not_significant
    return (zero_x + zero_y + errors + points).properties(
        width=460, height=340,
        title=alt.TitleParams(t("fig_stability_title", lang), subtitle=t("stability_hint", lang)),
    )


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
}
