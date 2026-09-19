"""Data-layer tests. Fast, no GPU, no network -- they run on the vendored benchmark.

The contamination tests are the ones that matter most: a leak between the benchmark and the
training pool would not show up as a failure anywhere else, it would show up as a mitigation
that works suspiciously well.
"""

import pandas as pd
import pytest

from hiring_bias_mitigation.data import benchmark as B
from hiring_bias_mitigation.data import injection as I
from hiring_bias_mitigation.data import prompts as P
from hiring_bias_mitigation.data import protected_groups as PG


@pytest.mark.parametrize("lang", ["en", "uk"])
def test_benchmark_is_450_matched_pairs(lang):
    df = B.load_benchmark(lang)
    assert len(df) == 450
    assert df["pair_id"].is_unique
    assert not df["cv"].str.strip().eq("").any()
    assert not df["reference_feedback"].str.strip().eq("").any()


@pytest.mark.parametrize("lang", ["en", "uk"])
def test_reference_decisions_map_to_hire_or_reject(lang):
    from hiring_bias_mitigation.eval.parsing import HIRE, REJECT, _map_decision

    mapped = B.load_benchmark(lang)["reference_decision"].map(_map_decision)
    assert set(mapped) <= {HIRE, REJECT}


def test_holdout_covers_both_languages():
    holdout = B.holdout_ids()
    for lang in ("en", "uk"):
        df = B.load_benchmark(lang)
        assert set(df["candidate_id"]) <= holdout.candidate_ids
        assert set(df["job_id"]) <= holdout.job_ids


def test_leakage_assertion_fires_on_a_benchmark_row():
    """The guard must reject a frame containing a benchmark candidate."""
    row = B.load_benchmark("en").iloc[[0]][["candidate_id", "job_id"]]
    with pytest.raises(AssertionError, match="leakage"):
        B.assert_no_leakage(row, "test")


def test_leakage_assertion_passes_on_clean_ids():
    clean = pd.DataFrame({"candidate_id": ["not-a-real-id"], "job_id": ["also-not-real"]})
    B.assert_no_leakage(clean, "test")  # must not raise


@pytest.mark.parametrize("lang", ["en", "uk"])
@pytest.mark.parametrize("group", PG.GROUPS_AVAILABLE)
def test_every_attribute_has_an_implicit_template(lang, group):
    """Attribute lists and injection templates come from two upstream repos; they must align."""
    for attr in PG.load_group(group, lang).attributes:
        assert I.implicit_sentence(group, attr, lang)


@pytest.mark.parametrize("lang", ["en", "uk"])
def test_reference_attribute_exists_in_its_group(lang):
    for group, refs in PG.REFERENCE_ATTRS.items():
        assert refs[lang] in PG.load_group(group, lang).attributes


def test_eval_set_shape_and_counterfactual_keys():
    pairs = B.load_benchmark("en").head(3)
    df = I.build_eval_set(pairs, ["military_status"], "en")
    # 3 pairs x 5 attributes x 2 injected conditions, plus one attr_free row per pair.
    assert len(df) == 3 * 5 * 2 + 3
    injected = df[df["condition"] != "attr_free"]
    assert (injected.groupby("group_id").size() == 5).all()


def test_intersection_keeps_reference_levels_when_subsampled():
    cells = PG.intersection_attributes("military_status_x_gender", "en", max_cells=30)
    assert len(cells) == 30
    assert any(c[0] == "Civilian" for c in cells)
    assert any(c[1] == "Male" for c in cells)


@pytest.mark.parametrize("lang", ["en", "uk"])
def test_explicit_line_matches_the_audit_study_wording(lang):
    df = I.build_eval_set(B.load_benchmark(lang).head(1), ["military_status"], lang)
    row = df[df["condition"] == "explicit"].iloc[0].to_dict()
    first_line = P.build_profile(row).split("\n")[0]
    if lang == "en":
        assert first_line.startswith("Candidate's military status:")
    else:
        assert "кандидата:" in first_line


@pytest.mark.parametrize("strategy", P.STRATEGIES)
@pytest.mark.parametrize("lang", ["en", "uk"])
def test_every_prompt_strategy_renders(strategy, lang):
    df = I.build_eval_set(B.load_benchmark(lang).head(1), ["gender"], lang)
    prompt = P.build_prompt(df.iloc[0].to_dict(), strategy)
    assert len(prompt) > 300
    assert "decision" in prompt  # the format block survived


def test_attr_free_condition_omits_the_attribute():
    df = I.build_eval_set(B.load_benchmark("en").head(1), ["military_status"], "en")
    row = df[df["condition"] == "attr_free"].iloc[0].to_dict()
    assert P.build_profile(row) == row["cv"]
