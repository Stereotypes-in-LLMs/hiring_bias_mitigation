"""Scored runs to tidy frames.

Altair wants long-form data, and so does anyone who wants to re-plot these numbers without
this repository: every figure writes its frame beside it as CSV. The extraction lives here
rather than in the chart builders so the numbers in a figure and the numbers in a CSV cannot
come from two different code paths.

Nothing here touches a GPU or a model. It reads `eval/results/*.json` -- the same records the
Markdown report is built from -- so figures can be rebuilt at any point in a sweep.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..eval.report import load_records

#: Families whose runs carry a variant (a strategy, a method, an objective).
VARIANT_KEYS = ("strategy", "method", "objective", "variant")


def _model_short(record: dict) -> str:
    return str(record["meta"]["model"]).split("/")[-1]


def _family(record: dict) -> str:
    return (record["meta"].get("mitigation") or {}).get("family", "none")


def _variant(record: dict) -> str:
    mitigation = record["meta"].get("mitigation") or {}
    for key in VARIANT_KEYS:
        if mitigation.get(key):
            return str(mitigation[key])
    return "none"


def load(results_dir: str | Path = "eval/results", drop_smoke: bool = True) -> list[dict]:
    """Every scored record, smoke runs excluded by default.

    A smoke run is 20 pairs. It is proof the pipeline works, not a measurement, and letting one
    into a figure puts a point with twenty times the noise beside the real ones.
    """
    records = [r for r in load_records(results_dir) if "meta" in r and "summary" in r]
    if drop_smoke:
        records = [r for r in records if "smoke" not in r["run_name"]]
    return records


def run_level(records: list[dict]) -> pd.DataFrame:
    """One row per run: the aggregate fairness and utility numbers."""
    rows = []
    for record in records:
        overall = record["summary"].get("disparity", {}).get("overall", {})
        summary = record["summary"]
        rows.append(
            {
                "run": record["run_name"],
                "model": _model_short(record),
                "lang": record["meta"]["lang"],
                "family": _family(record),
                "variant": _variant(record),
                "n_prompts": record["meta"].get("n_prompts"),
                "mad_pp": 100 * overall.get("ar_mad", float("nan")),
                "ar_range_pp": 100 * overall.get("ar_range", float("nan")),
                "cohens_h": overall.get("mean_abs_cohens_h"),
                "inconsistency_pct": 100 * overall.get("inconsistency_rate", float("nan")),
                "utility_pct": 100 * summary.get("reference_agreement", float("nan")),
                "refusal_pct": 100 * summary.get("refusal_rate", float("nan")),
                "leak_pct": 100 * summary.get("attribute_mention_rate", float("nan")),
                "mean_fs": summary.get("mean_feedback_similarity"),
                "cells": len(record.get("groups", {})),
            }
        )
    return pd.DataFrame(rows)


def cell_level(records: list[dict]) -> pd.DataFrame:
    """One row per run x protected group x condition."""
    rows = []
    for record in records:
        for key, payload in record.get("groups", {}).items():
            effects = payload.get("effect_sizes", {})
            population = payload.get("population", {})
            rows.append(
                {
                    "run": record["run_name"],
                    "model": _model_short(record),
                    "lang": record["meta"]["lang"],
                    "family": _family(record),
                    "variant": _variant(record),
                    "group": payload.get("protected_group", key.split("::")[0]),
                    "condition": payload.get("condition", key.split("::")[-1]),
                    "n_attributes": payload.get("n_attributes"),
                    "ar_range_pp": 100 * effects.get("acceptance_rate_range", float("nan")),
                    "largest_gap_pp": 100 * effects.get("largest_gap", float("nan")),
                    "largest_gap_attr": effects.get("largest_gap_attr"),
                    "cohens_h": effects.get("largest_gap_cohens_h"),
                    "utility_pct": 100 * population.get("reference_agreement", float("nan")),
                    "inconsistency_pct": 100 * population.get("inconsistency_rate", float("nan")),
                }
            )
    return pd.DataFrame(rows)


def attribute_level(records: list[dict]) -> pd.DataFrame:
    """One row per attribute, with its gap, interval and significance flag."""
    rows = []
    for record in records:
        for key, payload in record.get("groups", {}).items():
            effects = payload.get("effect_sizes", {})
            reference = effects.get("reference_attr")
            gaps = effects.get("gap_vs_reference", {})
            for attribute in payload.get("attributes", []):
                name = attribute.get("protected_attr")
                rows.append(
                    {
                        "run": record["run_name"],
                        "model": _model_short(record),
                        "lang": record["meta"]["lang"],
                        "family": _family(record),
                        "variant": _variant(record),
                        "group": payload.get("protected_group", key.split("::")[0]),
                        "condition": payload.get("condition", key.split("::")[-1]),
                        "attribute": name,
                        "is_reference": name == reference,
                        "n_decided": attribute.get("n_decided"),
                        "ar_pct": 100 * attribute.get("acceptance_rate", float("nan")),
                        "ci_low_pct": 100 * attribute.get("acceptance_rate__ci_low", float("nan")),
                        "ci_high_pct": 100
                        * attribute.get("acceptance_rate__ci_high", float("nan")),
                        "gap_pp": 100 * gaps.get(name, float("nan")),
                        "paired_p": attribute.get("acceptance_rate__paired_p"),
                        "significant": bool(attribute.get("acceptance_rate__significant")),
                        "inconsistency_pct": 100
                        * attribute.get("inconsistency_rate", float("nan")),
                    }
                )
    return pd.DataFrame(rows)


def with_baseline(frame: pd.DataFrame, value_columns: tuple[str, ...]) -> pd.DataFrame:
    """Joins each mitigated row to its own baseline and adds `<column>_delta`.

    The join is on model and language, which is the comparison every figure in the paper
    makes: a mitigation is read against the same model on the same benchmark, never against
    another model.
    """
    baselines = frame[frame["family"] == "none"]
    keys = ["model", "lang"]
    merged = frame.merge(
        baselines[keys + list(value_columns)].rename(
            columns={c: f"{c}_baseline" for c in value_columns}
        ),
        on=keys,
        how="left",
    )
    for column in value_columns:
        merged[f"{column}_delta"] = merged[column] - merged[f"{column}_baseline"]
    return merged
