"""Figure-layer tests.

The data and label layers are pure and are tested unconditionally. The chart builders need
Altair, which is an optional extra, so those tests skip when it is absent rather than turning
a sweep machine's test run red for a plotting dependency it deliberately does not have.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hiring_bias_mitigation.viz import data as D
from hiring_bias_mitigation.viz import labels as L


def _record(run, family, variant, lang="uk", mad=0.05, utility=0.80, leak=0.0):
    return {
        "run_name": run,
        "meta": {
            "model": "org/M", "lang": lang, "n_prompts": 100,
            "mitigation": {"family": family, "strategy": variant} if family != "none"
            else {"family": "none"},
        },
        "summary": {
            "reference_agreement": utility, "refusal_rate": 0.0,
            "attribute_mention_rate": leak, "mean_feedback_similarity": 0.7,
            "disparity": {"overall": {
                "ar_mad": mad, "ar_range": 0.2, "mean_abs_cohens_h": 0.1,
                "inconsistency_rate": 0.05,
            }},
        },
        "groups": {
            "military_status::explicit": {
                "protected_group": "military_status", "condition": "explicit",
                "n_attributes": 2,
                "population": {"reference_agreement": utility, "inconsistency_rate": 0.05,
                               "n_decided": 100},
                "effect_sizes": {
                    "acceptance_rate_range": 0.2, "largest_gap": 0.1,
                    "largest_gap_attr": "War veteran", "largest_gap_cohens_h": 0.2,
                    "reference_attr": "Civilian",
                    "gap_vs_reference": {"Civilian": 0.0, "War veteran": 0.1},
                },
                "attributes": [
                    {"protected_attr": "Civilian", "acceptance_rate": 0.40,
                     "acceptance_rate__ci_low": 0.35, "acceptance_rate__ci_high": 0.45,
                     "acceptance_rate__significant": False, "acceptance_rate__paired_p": 0.9,
                     "inconsistency_rate": 0.04, "n_decided": 50},
                    {"protected_attr": "War veteran", "acceptance_rate": 0.50,
                     "acceptance_rate__ci_low": 0.45, "acceptance_rate__ci_high": 0.55,
                     "acceptance_rate__significant": True, "acceptance_rate__paired_p": 0.01,
                     "inconsistency_rate": 0.06, "n_decided": 50},
                ],
            }
        },
    }


# --------------------------------------------------------------------------- labels


def test_every_string_has_both_languages():
    """A half-translated figure in a paper is worse than an obviously missing one."""
    for key, table in {**L.STRINGS, **L.STRATEGIES}.items():
        assert set(table) >= set(L.LANGUAGES), f"{key} is missing {set(L.LANGUAGES) - set(table)}"
        for lang, text in table.items():
            assert text.strip(), f"{key}[{lang}] is empty"


def test_missing_key_raises_rather_than_falling_back():
    with pytest.raises(KeyError):
        L.t("no_such_key", "uk")
    with pytest.raises(ValueError):
        L.t("mad", "de")


def test_unknown_strategy_keeps_its_identifier():
    """A newly added mitigation should plot under its config name, not crash the run."""
    assert L.strategy("brand_new_method", "uk") == "brand_new_method"


# --------------------------------------------------------------------------- data


def test_run_level_converts_rates_to_points():
    frame = D.run_level([_record("r", "prompt", "s", mad=0.0123, utility=0.805)])
    row = frame.iloc[0]
    assert row["mad_pp"] == pytest.approx(1.23)
    assert row["utility_pct"] == pytest.approx(80.5)


def test_smoke_runs_never_reach_a_figure(tmp_path):
    """A 20-pair smoke run beside real points is twenty times the noise, silently."""
    import json

    for name in ("M--uk--baseline", "M--uk--baseline--smoke"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(_record(name, "none", "none")), encoding="utf-8"
        )
    assert len(D.load(tmp_path)) == 1
    assert len(D.load(tmp_path, drop_smoke=False)) == 2


def test_baseline_join_is_per_model_and_language():
    """A mitigation is read against its own model, never against another one."""
    records = [
        _record("M--uk--baseline", "none", "none", lang="uk", mad=0.05),
        _record("M--en--baseline", "none", "none", lang="en", mad=0.02),
        _record("M--uk--prompt--s", "prompt", "s", lang="uk", mad=0.03),
    ]
    merged = D.with_baseline(D.run_level(records), ("mad_pp",))
    row = merged[merged["run"] == "M--uk--prompt--s"].iloc[0]
    assert row["mad_pp_baseline"] == pytest.approx(5.0), "joined to the English baseline"
    assert row["mad_pp_delta"] == pytest.approx(-2.0)


def test_attribute_level_carries_intervals_and_reference_flag():
    frame = D.attribute_level([_record("r", "none", "none")])
    reference = frame[frame["is_reference"]].iloc[0]
    assert reference["attribute"] == "Civilian"
    assert frame["ci_low_pct"].notna().all()
    assert frame["significant"].sum() == 1


# --------------------------------------------------------------------------- charts


@pytest.fixture()
def frames():
    records = [
        _record("M--uk--baseline", "none", "none", lang="uk", mad=0.05, leak=0.02),
        _record("M--en--baseline", "none", "none", lang="en", mad=0.04, leak=0.02),
        _record("M--uk--prompt--a", "prompt", "a", lang="uk", mad=0.03, leak=0.01),
        _record("M--en--prompt--a", "prompt", "a", lang="en", mad=0.02, leak=0.01),
        _record("M--uk--scrub--lexical", "scrub", "lexical", lang="uk", mad=0.00, leak=0.0),
    ]
    return {
        "runs": D.run_level(records),
        "cells": D.cell_level(records),
        "attributes": D.attribute_level(records),
        "stability": pd.DataFrame([
            {"model": "M", "lang": "uk", "family": "prompt", "variant": "a",
             "delta_unstable_pp": -8.0, "ci_low": -10.0, "ci_high": -6.0, "p_fdr": 1e-5,
             "delta_utility_pp": 0.5},
            {"model": "M", "lang": "uk", "family": "sft", "variant": "uk_only",
             "delta_unstable_pp": 0.1, "ci_low": -0.2, "ci_high": 0.4, "p_fdr": 0.7,
             "delta_utility_pp": 0.0},
        ]),
    }


def test_every_figure_builds_in_both_languages(frames):
    pytest.importorskip("altair")
    from hiring_bias_mitigation.viz import charts as C
    from hiring_bias_mitigation.viz import theme as T

    T.register()
    for name, (builder, frame_name) in C.FIGURES.items():
        for lang in L.LANGUAGES:
            chart = builder(frames[frame_name], lang)
            assert chart is not None, f"{name} [{lang}] returned nothing"
            spec = chart.to_dict()          # raises on an invalid spec
            assert spec.get("$schema", "").startswith("https://vega.github.io/schema/vega-lite")


def test_a_figure_with_no_data_is_skipped_not_an_error():
    """A sweep is days long; rebuilding halfway must not raise."""
    pytest.importorskip("altair")
    from hiring_bias_mitigation.viz import charts as C

    empty = pd.DataFrame(
        columns=["run", "model", "lang", "family", "variant", "mad_pp", "utility_pct",
                 "leak_pct", "ar_range_pp", "condition", "group", "ar_pct", "significant"]
    )
    for name, (builder, _) in C.FIGURES.items():
        assert builder(empty, "uk") is None, f"{name} should skip an empty frame"


def test_ukrainian_and_english_specs_differ_only_in_text(frames):
    """The pair must be the same figure, not two figures that drifted."""
    pytest.importorskip("altair")
    import json
    import re

    from hiring_bias_mitigation.viz import charts as C

    builder, frame_name = C.FIGURES["fairness_utility_tradeoff"]
    specs = {
        lang: json.dumps(builder(frames[frame_name], lang).to_dict(), sort_keys=True)
        for lang in ("en", "uk")
    }
    # Strip every Cyrillic run and every quoted title/label, then the structure must match.
    def skeleton(text: str) -> str:
        return re.sub(r'"[^"]*"', '""', text)

    assert skeleton(specs["en"]) == skeleton(specs["uk"])
    assert specs["en"] != specs["uk"], "the Ukrainian figure carries no Ukrainian text"


def test_stability_figure_keeps_non_significant_points_visible():
    """A non-significant point is a finding here -- every training adapter is one -- so it must
    render as an outline, not as a white fill on a white page."""
    pytest.importorskip("altair")
    from hiring_bias_mitigation.viz import charts as C

    table = pd.DataFrame([
        {"model": "Qwen3.5-4B", "lang": "en", "family": "prompt", "variant": "a",
         "delta_unstable_pp": -10.0, "ci_low": -12.0, "ci_high": -8.0, "p_fdr": 1e-6,
         "delta_utility_pp": -1.0},
        {"model": "Qwen3.5-4B", "lang": "en", "family": "sft", "variant": "en_only",
         "delta_unstable_pp": 0.2, "ci_low": -0.3, "ci_high": 0.7, "p_fdr": 0.6,
         "delta_utility_pp": 0.0},
    ])
    spec = C.stability_utility_tradeoff(table, "uk").to_dict()
    marks = [layer.get("mark") for layer in spec["layer"]
             for layer in (layer.get("layer") or [layer])]
    hollow = [m for m in marks if isinstance(m, dict) and m.get("filled") is False]
    assert hollow, "non-significant points need an outlined layer"
    assert C.stability_utility_tradeoff(None, "en") is None
