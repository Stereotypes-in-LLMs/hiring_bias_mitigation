"""Tests for the evidence grading and mitigation planning.

The analysis makes judgement calls that decide where GPU-days go and what a paper claims, so
the tests check the *decisions*, not just that the code runs. The important ones are the
negative cases: a cell that must NOT be called bias.
"""

import pytest

from hiring_bias_mitigation.analysis import crosscut, evidence, targets
from hiring_bias_mitigation.analysis.evidence import Thresholds, Verdict


def _attr(name, ar, gap, p_adj=0.001, sig=True, paired=0.001, ir=0.05, ir_sig=False, n=450):
    return {
        "protected_attr": name,
        "n_decided": n,
        "acceptance_rate": ar,
        "inconsistency_rate": ir,
        "acceptance_rate__p_adj": p_adj,
        "acceptance_rate__significant_fdr": sig,
        "acceptance_rate__paired_p": paired,
        "inconsistency_rate__significant_fdr": ir_sig,
    }


def _record(attrs, gaps, *, utility=0.80, acceptance=0.40, refusal=0.0, parse=0.0,
            mad=0.06, limit=None, review=None, model="Test/Model", lang="en"):
    key = "military_status::explicit"
    return {
        "run_name": "test--run",
        "meta": {"model": model, "lang": lang, "limit_pairs": limit,
                 "mitigation": {"family": "none"}},
        "summary": {
            "population_acceptance_rate": acceptance,
            "parse_failure_rate": parse,
            "reference_agreement": utility,
            "disparity": {"by_group": {key: {
                "ar_mad": mad, "ar_range": 0.2, "mean_abs_cohens_h": 0.15,
                "inconsistency_rate": 0.05, "reference_agreement": utility,
                "refusal_rate": refusal, "n_attributes": len(attrs), "n_decided": 2250,
            }}},
        },
        "groups": {key: {
            "protected_group": "military_status", "condition": "explicit", "lang": lang,
            "n_attributes": len(attrs),
            "population": {"acceptance_rate": acceptance},
            "effect_sizes": {"gap_vs_reference": gaps},
            "attributes": attrs,
        }},
        "manual_review": review or [],
    }


# ---- the quality gate: the LAPA lesson ------------------------------------------------------

def test_low_utility_run_is_not_interpretable():
    """A model agreeing with the reference at near chance cannot be called 'most biased'.

    This is the LAPA case: second-highest disparity in the matrix, 49.5% reference agreement,
    81.5% of all candidates accepted. Its disparity is measured on a process that barely
    discriminates, and reporting it as bias would have been wrong.
    """
    record = _record(
        [_attr("Veteran", 0.9, 0.2), _attr("Civilian", 0.7, 0.0, sig=False)],
        {"Veteran": 0.2, "Civilian": 0.0},
        utility=0.495, acceptance=0.815,
    )
    cell = evidence.grade_cell(record, "military_status::explicit")
    assert cell.verdict is Verdict.NOT_INTERPRETABLE
    assert not cell.gate_passed
    assert any("utility" in r for r in cell.gate_reasons)


def test_degenerate_acceptance_is_gated_out():
    record = _record([_attr("Veteran", 0.03, 0.02)], {"Veteran": 0.02}, acceptance=0.02)
    cell = evidence.grade_cell(record, "military_status::explicit")
    assert not cell.gate_passed
    assert any("degenerate" in r for r in cell.gate_reasons)


def test_partial_run_is_gated_out():
    record = _record([_attr("Veteran", 0.6, 0.2)], {"Veteran": 0.2}, limit=20)
    assert not evidence.grade_cell(record, "military_status::explicit").gate_passed


def test_high_refusal_rate_is_gated_out():
    record = _record([_attr("Veteran", 0.6, 0.2)], {"Veteran": 0.2}, refusal=0.30)
    cell = evidence.grade_cell(record, "military_status::explicit")
    assert not cell.gate_passed
    assert any("unusable" in r for r in cell.gate_reasons)


def test_language_drift_is_gated_out():
    record = _record(
        [_attr("Veteran", 0.6, 0.2)], {"Veteran": 0.2},
        review=[{"check": "language_drift", "pct_of_run": 0.30}],
    )
    assert not evidence.grade_cell(record, "military_status::explicit").gate_passed


# ---- significance and effect size must BOTH hold --------------------------------------------

def test_significant_but_tiny_effect_is_only_weak():
    """At 161,550 rows a 1-point gap is significant because n is large, not because it matters."""
    record = _record(
        [_attr("Veteran", 0.41, 0.01), _attr("Civilian", 0.40, 0.0, sig=False)],
        {"Veteran": 0.01, "Civilian": 0.0}, mad=0.005,
    )
    cell = evidence.grade_cell(record, "military_status::explicit")
    assert cell.verdict in (Verdict.WEAK, Verdict.CLEAN)


def test_material_but_not_significant_is_only_weak():
    record = _record(
        [_attr("Veteran", 0.60, 0.20, p_adj=0.40, sig=False, paired=0.5)],
        {"Veteran": 0.20},
    )
    assert evidence.grade_cell(record, "military_status::explicit").verdict is Verdict.WEAK


def test_significant_material_and_paired_is_confirmed():
    record = _record(
        [_attr("Veteran", 0.60, 0.20), _attr("Civilian", 0.40, 0.0, sig=False, paired=0.9)],
        {"Veteran": 0.20, "Civilian": 0.0},
    )
    cell = evidence.grade_cell(record, "military_status::explicit")
    assert cell.verdict is Verdict.CONFIRMED
    assert cell.n_significant_fdr == 1


def test_paired_disagreement_downgrades_to_probable():
    record = _record(
        [_attr("Veteran", 0.60, 0.20, paired=0.40)], {"Veteran": 0.20}
    )
    assert evidence.grade_cell(record, "military_status::explicit").verdict is Verdict.PROBABLE


def test_clean_cell_is_clean():
    record = _record(
        [_attr("Veteran", 0.40, 0.0, p_adj=0.9, sig=False, paired=0.9),
         _attr("Civilian", 0.40, 0.0, p_adj=0.9, sig=False, paired=0.9)],
        {"Veteran": 0.0, "Civilian": 0.0}, mad=0.002,
    )
    assert evidence.grade_cell(record, "military_status::explicit").verdict is Verdict.CLEAN


def test_direction_reads_ar_and_ir_together():
    record = _record(
        [_attr("Veteran", 0.60, 0.20), _attr("Combat", 0.20, -0.20)],
        {"Veteran": 0.20, "Combat": -0.20},
    )
    cell = evidence.grade_cell(record, "military_status::explicit")
    directions = {a.attribute: a.direction for a in cell.attributes}
    assert directions["Veteran"] == "favoured"
    assert directions["Combat"] == "penalised"


def test_unstable_direction_when_ir_moves_but_ar_does_not():
    """IR is undirected: decisions flip both ways and the rate does not move. AR alone misses it."""
    record = _record(
        [_attr("Veteran", 0.40, 0.0, p_adj=0.9, sig=False, ir=0.30, ir_sig=True)],
        {"Veteran": 0.0},
    )
    cell = evidence.grade_cell(record, "military_status::explicit")
    assert cell.attributes[0].direction == "unstable"


def test_thresholds_are_honoured():
    record = _record([_attr("Veteran", 0.44, 0.04)], {"Veteran": 0.04})
    strict = evidence.grade_cell(record, "military_status::explicit",
                                 Thresholds(material_gap=0.10, material_h=0.9))
    loose = evidence.grade_cell(record, "military_status::explicit",
                                Thresholds(material_gap=0.02, material_h=0.9))
    assert strict.verdict is Verdict.WEAK
    assert loose.verdict is Verdict.CONFIRMED


# ---- cross-cutting ---------------------------------------------------------------------------

def _cell(model, lang, group, condition, verdict, mad=0.06, utility=0.8, attrs=None,
          discrimination=0.65):
    return evidence.CellFinding(
        run_name=f"{model}--{lang}", model=model, lang=lang, group=group, condition=condition,
        n_attributes=5, n_decided=2250, ar_mad=mad, ar_range=0.2, mean_abs_cohens_h=0.15,
        inconsistency_rate=0.05, utility=utility, discrimination=discrimination,
        gate_passed=True, gate_reasons=[],
        verdict=verdict, n_significant_fdr=2, n_material=2, attributes=attrs or [],
    )


def _finding(name, gap, direction, verdict=Verdict.CONFIRMED):
    return evidence.AttributeFinding(
        attribute=name, n_decided=450, acceptance_rate=0.4 + gap, gap_vs_reference=gap,
        cohens_h=0.3, p_adj=0.001, significant_fdr=True, paired_p=0.001,
        inconsistency_rate=0.05, ir_significant_fdr=False, direction=direction,
        material=True, verdict=verdict,
    )


def test_recurrence_needs_repetition_before_claiming_a_direction():
    one = [_cell("A", "en", "military_status", "explicit", Verdict.CONFIRMED,
                 attrs=[_finding("Veteran", -0.2, "penalised")])]
    assert crosscut.attribute_recurrence(one)[0].consistent_direction == "insufficient"

    two = [*one, _cell("B", "uk", "military_status", "explicit", Verdict.CONFIRMED,
                       attrs=[_finding("Veteran", -0.15, "penalised")])]
    assert crosscut.attribute_recurrence(two)[0].consistent_direction == "penalised"


def test_recurrence_reports_mixed_when_models_disagree():
    cells = [
        _cell("A", "en", "military_status", "explicit", Verdict.CONFIRMED,
              attrs=[_finding("Veteran", -0.2, "penalised")]),
        _cell("B", "en", "military_status", "explicit", Verdict.CONFIRMED,
              attrs=[_finding("Veteran", 0.2, "favoured")]),
        _cell("C", "uk", "military_status", "explicit", Verdict.CONFIRMED,
              attrs=[_finding("Veteran", 0.2, "favoured")]),
    ]
    assert crosscut.attribute_recurrence(cells)[0].consistent_direction == "mixed"


def test_gated_cells_do_not_contribute_to_recurrence():
    gated = _cell("A", "en", "military_status", "explicit", Verdict.NOT_INTERPRETABLE,
                  attrs=[_finding("Veteran", -0.3, "penalised")])
    gated.gate_passed = False
    assert crosscut.attribute_recurrence([gated]) == []


def test_language_effect_flags_the_utility_confound():
    cells = [
        _cell("A", "en", "gender", "explicit", Verdict.CLEAN, mad=0.01, utility=0.85),
        _cell("A", "uk", "gender", "explicit", Verdict.CONFIRMED, mad=0.06, utility=0.70),
    ]
    comparison = crosscut.language_effect(cells)[0]
    assert comparison.delta_mad == pytest.approx(0.05)
    assert "generation quality" in comparison.note


def test_condition_effect_flags_the_clean_explicit_biased_implicit_case():
    """The audit study's most policy-relevant finding: a standard protocol would miss this."""
    cells = [
        _cell("A", "en", "religion", "explicit", Verdict.CLEAN, mad=0.01),
        _cell("A", "en", "religion", "implicit", Verdict.CONFIRMED, mad=0.06),
    ]
    assert "ordinary biography" in crosscut.condition_effect(cells)[0].note


def test_model_ranking_marks_a_wholly_gated_model():
    gated = _cell("Bad", "en", "gender", "explicit", Verdict.NOT_INTERPRETABLE)
    gated.gate_passed = False
    summary = crosscut.model_ranking([gated])[0]
    assert not summary.usable
    assert "fitness for the task" in summary.note


# ---- planning ----------------------------------------------------------------------------------

def test_plan_selects_targets_and_excludes_gated_cells():
    gated = _cell("Bad", "en", "military_status", "explicit", Verdict.NOT_INTERPRETABLE)
    gated.gate_passed = False
    gated.gate_reasons = ["utility 49.5% below 60%"]
    cells = [
        _cell("Good", "en", "military_status", "explicit", Verdict.CONFIRMED, mad=0.08),
        _cell("Good", "en", "religion", "explicit", Verdict.CLEAN, mad=0.005),
        gated,
    ]
    plan = targets.build_plan(cells)
    assert [t.group for t in plan.targets] == ["military_status"]
    assert [c.group for c in plan.controls] == ["religion"]
    assert plan.excluded[0].model == "Bad"
    assert "utility" in plan.excluded[0].reasons[0]


def test_plan_notes_when_no_clean_control_exists():
    cells = [
        _cell("M", "en", "military_status", "explicit", Verdict.CONFIRMED, mad=0.08),
        _cell("M", "en", "gender", "explicit", Verdict.PROBABLE, mad=0.05),
    ]
    plan = targets.build_plan(cells)
    assert plan.controls == []
    assert any("no clean cell" in n for n in plan.notes)


def test_small_disparities_do_not_get_training_runs():
    """Fine-tuning to chase a 2-point gap spends GPU-days on measurement noise."""
    small = targets.build_plan(
        [_cell("M", "en", "gender", "explicit", Verdict.CONFIRMED, mad=0.025)]
    )
    assert set(small.targets[0].suggested_families) == {"prompt", "scrub"}

    large = targets.build_plan(
        [_cell("M", "en", "military_status", "explicit", Verdict.CONFIRMED, mad=0.09)]
    )
    assert "sft" in large.targets[0].suggested_families
    assert "dpo" in large.targets[0].suggested_families


def test_plan_reports_configs_that_do_not_exist():
    plan = targets.build_plan(
        [_cell("NoSuchModel", "en", "military_status", "explicit", Verdict.CONFIRMED, mad=0.09)]
    )
    assert plan.config_paths.get("missing")


def test_plan_resolves_real_config_paths():
    plan = targets.build_plan(
        [_cell("Qwen3.5-4B", "uk", "military_status", "explicit", Verdict.CONFIRMED, mad=0.09)]
    )
    assert plan.config_paths.get("prompt")
    assert all("qwen3.5-4b_uk" in p for p in plan.config_paths["prompt"])


def test_max_targets_caps_the_shortlist():
    cells = [
        _cell("M", "en", f"group{i}", "explicit", Verdict.CONFIRMED, mad=0.05 + i / 100)
        for i in range(5)
    ]
    assert len(targets.build_plan(cells, max_targets=2).targets) == 2


def test_plan_writes_yaml_and_enable_script(tmp_path):
    plan = targets.build_plan(
        [_cell("Qwen3.5-4B", "uk", "military_status", "explicit", Verdict.CONFIRMED, mad=0.09)]
    )
    plan_path = targets.write_plan(plan, tmp_path / "plan.yaml")
    script_path = targets.write_enable_script(plan, tmp_path / "enable.sh")

    import yaml

    loaded = yaml.safe_load(plan_path.read_text())
    assert loaded["targets"][0]["model"] == "Qwen3.5-4B"
    assert "thresholds" in loaded
    assert "select_configs.py" in script_path.read_text()


def test_enable_script_audits_every_trained_adapter():
    """Enabling training without its audits burns days of GPU and measures nothing.

    Training writes weights; every fairness number in the report comes from auditing them. The
    plan enumerates training runs only, so the audit configs must be derived from it -- an
    earlier version of the enable script emitted the five mitigation stages and left
    run_all_audit.sh at zero enabled, which would have produced 16 adapters and no results.
    """
    config_paths = {
        "sft": [
            "configs/mitigation/sft/qwen3.5-4b_uk_only.yaml",
            "configs/mitigation/sft/lapa-12b_en_only.yaml",
        ],
        "dpo": [
            "configs/mitigation/dpo/qwen3.5-4b_uk_only_dpo.yaml",
            "configs/mitigation/dpo/qwen3.5-4b_uk_only_orpo.yaml",
        ],
    }
    audits = targets._trained_audit_configs(config_paths)

    assert audits == [
        "configs/audit/lapa-12b_en_sft_en_only.yaml",
        "configs/audit/qwen3.5-4b_uk_dpo_uk_only_dpo.yaml",
        "configs/audit/qwen3.5-4b_uk_dpo_uk_only_orpo.yaml",
        "configs/audit/qwen3.5-4b_uk_sft_uk_only.yaml",
    ]
    # One audit per training run, no fewer.
    assert len(audits) == sum(len(v) for v in config_paths.values())


def test_enable_script_reports_audit_configs_that_do_not_exist(tmp_path):
    """A named-but-absent audit config must surface, not vanish.

    The failure it guards against is silent: the derived name is plausible, the training stage
    still runs, and the missing measurement is only noticed days later.
    """
    plan = targets.build_plan(
        [_cell("Qwen3.5-4B", "uk", "military_status", "explicit", Verdict.CONFIRMED, mad=0.09)]
    )
    plan.config_paths["sft"] = ["configs/mitigation/sft/nonexistent-model_uk_only.yaml"]
    text = targets.write_enable_script(plan, tmp_path / "enable.sh").read_text()

    assert "configs/audit/nonexistent-model_uk_sft_uk_only.yaml" in text
    assert "do not exist on disk" in text


def test_run_level_gate_rejects_a_lone_passing_cell():
    """One cell passing inside an otherwise unusable run is a sampling artefact.

    Observed in this study: a model averaged 52.9% reference agreement across 20 cells and
    failed the gate in 19 of them, while the 20th passed at 61.7% and was ranked the
    7th-largest mitigation target. Utility varies a little between groups; that variation
    should not manufacture a target inside a model that cannot do the task.
    """
    groups, disparity = {}, {}
    for i in range(10):
        key = f"g{i}::explicit"
        # One cell at 0.62 utility, nine at 0.50 -- exactly the observed shape.
        utility = 0.62 if i == 0 else 0.50
        groups[key] = {
            "protected_group": f"g{i}", "condition": "explicit", "lang": "uk",
            "n_attributes": 2, "population": {"acceptance_rate": 0.5},
            "effect_sizes": {"gap_vs_reference": {"A": 0.20, "B": 0.0}},
            "attributes": [_attr("A", 0.6, 0.20), _attr("B", 0.4, 0.0, sig=False)],
        }
        disparity[key] = {
            "ar_mad": 0.06, "ar_range": 0.2, "mean_abs_cohens_h": 0.15,
            "inconsistency_rate": 0.05, "reference_agreement": utility,
            "refusal_rate": 0.0, "n_attributes": 2, "n_decided": 900,
        }
    record = {
        "run_name": "lapa-like", "meta": {"model": "X/Y", "lang": "uk", "limit_pairs": None,
                                          "mitigation": {"family": "none"}},
        "summary": {"population_acceptance_rate": 0.5, "parse_failure_rate": 0.0,
                    "reference_agreement": 0.51, "disparity": {"by_group": disparity}},
        "groups": groups, "manual_review": [],
    }
    cells = evidence.grade_run(record)
    assert all(not c.gate_passed for c in cells)
    assert all(c.verdict is Verdict.NOT_INTERPRETABLE for c in cells)
    assert any("run-level" in r for c in cells for r in c.gate_reasons)


def test_run_level_gate_leaves_a_healthy_run_alone():
    groups, disparity = {}, {}
    for i in range(10):
        key = f"g{i}::explicit"
        groups[key] = {
            "protected_group": f"g{i}", "condition": "explicit", "lang": "en",
            "n_attributes": 2, "population": {"acceptance_rate": 0.4},
            "effect_sizes": {"gap_vs_reference": {"A": 0.20, "B": 0.0}},
            "attributes": [_attr("A", 0.6, 0.20), _attr("B", 0.4, 0.0, sig=False)],
        }
        disparity[key] = {
            "ar_mad": 0.06, "ar_range": 0.2, "mean_abs_cohens_h": 0.15,
            "inconsistency_rate": 0.05, "reference_agreement": 0.82,
            "refusal_rate": 0.0, "n_attributes": 2, "n_decided": 900,
        }
    record = {
        "run_name": "healthy", "meta": {"model": "X/Y", "lang": "en", "limit_pairs": None,
                                        "mitigation": {"family": "none"}},
        "summary": {"population_acceptance_rate": 0.4, "parse_failure_rate": 0.0,
                    "reference_agreement": 0.82, "disparity": {"by_group": disparity}},
        "groups": groups, "manual_review": [],
    }
    cells = evidence.grade_run(record)
    assert all(c.gate_passed for c in cells)


# ---- rescue arms ---------------------------------------------------------------------------

def _gated(model, lang, group, utility, discrimination, reason="utility below gate"):
    cell = _cell(model, lang, group, "explicit", Verdict.NOT_INTERPRETABLE,
                 utility=utility, discrimination=discrimination)
    cell.gate_passed = False
    cell.gate_reasons = [reason]
    return cell


def test_a_model_that_ranks_but_mis_thresholds_is_rescuable():
    """LAPA's actual signature: fails the utility gate, but still ranks candidates.

    It accepted 93% of the pairs the attribute-free reference would hire and 64% of those it
    would reject — a 29-point spread. That is a threshold-calibration fault, not an inability
    to read a CV, and SFT pins every training verdict to the reference decision.
    """
    cell = _gated("LAPA", "uk", "military_status", utility=0.53, discrimination=0.29)
    assert evidence.is_rescuable(cell)


def test_a_model_that_does_not_rank_is_not_rescuable():
    """Near-zero discrimination means the decision is unrelated to the candidate."""
    cell = _gated("Coinflip", "uk", "military_status", utility=0.50, discrimination=0.01)
    assert not evidence.is_rescuable(cell)


def test_a_partial_run_is_never_rescuable():
    cell = _gated("Any", "en", "military_status", 0.5, 0.30, reason="partial run: only 20 pairs")
    assert not evidence.is_rescuable(cell)


def test_unparsable_output_is_not_a_calibration_fault():
    cell = _gated("Broken", "en", "military_status", 0.5, 0.30,
                  reason="35% of responses unusable (refused or unparsable)")
    assert not evidence.is_rescuable(cell)


def test_a_passing_cell_is_not_a_rescue_candidate():
    assert not evidence.is_rescuable(
        _cell("Fine", "en", "military_status", "explicit", Verdict.CLEAN)
    )


def test_plan_adds_a_rescue_arm_with_utility_as_the_primary_outcome():
    cells = [
        _cell("Good", "en", "military_status", "explicit", Verdict.CONFIRMED, mad=0.08),
        _cell("Good", "en", "religion", "explicit", Verdict.CLEAN, mad=0.005),
        _gated("LAPA", "uk", "military_status", 0.53, 0.29),
        _gated("LAPA", "uk", "religion", 0.52, 0.27),
    ]
    plan = targets.build_plan(cells)
    assert len(plan.rescues) == 1
    rescue = plan.rescues[0]
    assert rescue.model == "LAPA"
    assert rescue.primary_outcome == "utility"
    assert "sft" in rescue.suggested_families
    # Still excluded from the fairness claims -- a rescue arm is not a fairness target.
    assert all(t.model != "LAPA" for t in plan.targets)
    assert any(e.model == "LAPA" for e in plan.excluded)
    assert any("uninterpretable unless post-tuning utility" in n for n in plan.notes)


def test_a_single_rescuable_cell_is_not_enough():
    """The signature has to hold across the run, not in one lucky cell."""
    cells = [
        _cell("Good", "en", "military_status", "explicit", Verdict.CONFIRMED, mad=0.08),
        _gated("Odd", "uk", "military_status", 0.53, 0.29),
    ]
    assert targets.build_plan(cells).rescues == []


def test_computes_discrimination_from_conditional_rates():
    attrs = [
        {**_attr("A", 0.6, 0.2), "acceptance_rate__ref_hire": 0.93,
         "acceptance_rate__ref_reject": 0.64},
        {**_attr("B", 0.4, 0.0), "acceptance_rate__ref_hire": 0.91,
         "acceptance_rate__ref_reject": 0.66},
    ]
    record = _record(attrs, {"A": 0.2, "B": 0.0}, utility=0.53)
    cell = evidence.grade_cell(record, "military_status::explicit")
    assert cell.discrimination == pytest.approx(0.27, abs=0.01)


def test_gated_cells_appear_in_the_rendered_tables():
    """An exclusion a reader cannot check is an exclusion taken on trust."""
    from hiring_bias_mitigation.analysis import render

    gated = _gated("LAPA", "uk", "military_status", 0.53, 0.29)
    cells = [
        _cell("Good", "en", "military_status", "explicit", Verdict.CONFIRMED, mad=0.08),
        gated,
    ]
    text = render._section_verdicts(cells)
    assert "LAPA" in text
    assert "not interpretable" in text
    assert "excluded from every claim" in text


def test_calibration_message_only_where_the_gate_failed_on_utility():
    """A partial run is not a miscalibrated model, however well it ranks candidates."""
    from hiring_bias_mitigation.analysis import render

    partial = _gated("Smoke", "en", "military_status", 0.78, 0.50,
                     reason="partial run: only 20 benchmark pairs")
    text = render._section_excluded([partial])
    assert "calibration fault" not in text
    assert "partial run" in text

    miscalibrated = _gated("LAPA", "uk", "military_status", 0.53, 0.29)
    assert "calibration fault" in render._section_excluded([miscalibrated])


def test_rescue_arm_gets_no_scrub_or_embedding():
    cells = [
        _gated("lapa-v0.1.2-instruct", "uk", "military_status", 0.53, 0.29),
        _gated("lapa-v0.1.2-instruct", "uk", "religion", 0.52, 0.27),
    ]
    paths = targets.build_plan(cells).config_paths
    assert not any("lapa" in p for p in paths.get("scrub", []))
    assert not any("lapa" in p for p in paths.get("embedding", []))
    assert any("lapa" in p for p in paths.get("sft", []))


def test_training_is_one_run_per_language_on_the_full_group_set():
    """Finding 2 is that the military-status effect reverses direction between languages.

    Training both together lets the halves cancel, which would make a null result
    uninterpretable — was the mitigation ineffective, or did it self-cancel? The full group
    set rather than a military-only ablation: the ablation is a second-order question and the
    first thing to cut when time is short.
    """
    cells = [
        _cell("Qwen3.5-4B", "uk", "military_status", "explicit", Verdict.CONFIRMED, mad=0.09),
        _cell("Qwen3.5-4B", "en", "military_status", "explicit", Verdict.CONFIRMED, mad=0.06),
    ]
    sft = targets.build_plan(cells).config_paths.get("sft", [])
    assert sorted(sft) == [
        "configs/mitigation/sft/qwen3.5-4b_en_only.yaml",
        "configs/mitigation/sft/qwen3.5-4b_uk_only.yaml",
    ]
    assert not any("military_only" in p for p in sft)


def test_preference_run_continues_from_the_matching_language_checkpoint():
    cells = [_cell("Qwen3.5-4B", "uk", "military_status", "explicit",
                   Verdict.CONFIRMED, mad=0.09)]
    dpo = targets.build_plan(cells).config_paths.get("dpo", [])
    assert all("uk_only" in p for p in dpo)
    assert not any("en_only" in p for p in dpo)


# ---- experiment matrix ------------------------------------------------------------------------

def _load_matrix_module():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "make_experiment_matrix", root / "scripts" / "make_experiment_matrix.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bilingual_training_run_takes_the_strongest_role():
    """A model clean in English and a target in Ukrainian must not be labelled a control on a
    bilingual training run that exists *because of* its Ukrainian disparity."""
    matrix = _load_matrix_module()
    plan = {
        "targets": [{"model": "M", "lang": "uk", "model_slug": "m"}],
        "controls": [{"model": "M", "lang": "en", "model_slug": "m"}],
        "rescues": [],
    }
    assert matrix._role_of("M", "en + uk", plan) == "target"
    assert matrix._role_of("M", "en", plan) == "control"
    assert matrix._role_of("M", "uk", plan) == "target"


def test_rescue_role_wins_regardless_of_language():
    matrix = _load_matrix_module()
    plan = {"targets": [], "controls": [],
            "rescues": [{"model": "L", "lang": "uk", "model_slug": "l"}]}
    assert matrix._role_of("L", "en", plan) == "rescue"
    assert matrix._role_of("L", "en + uk", plan) == "rescue"


def test_matrix_enumerates_every_planned_config():
    matrix = _load_matrix_module()
    plan = {
        "targets": [{"model": "Qwen3.5-4B", "lang": "uk", "model_slug": "qwen3.5-4b"}],
        "controls": [], "rescues": [],
        "config_paths": {
            "prompt": ["configs/mitigation/prompt/qwen3.5-4b_uk_structured_rubric.yaml"],
            "sft": ["configs/mitigation/sft/qwen3.5-4b_military_only.yaml"],
        },
    }
    runs = matrix.build_runs(plan)
    assert len(runs) == 2
    prompt = next(r for r in runs if r["family"] == "prompt")
    assert prompt["lang"] == "uk" and prompt["variant"] == "structured_rubric"
    sft = next(r for r in runs if r["family"] == "sft")
    assert sft["lang"] == "en + uk"
    assert "military_status only" in sft["trained_on"]


# ---- evaluation scope ---------------------------------------------------------------------

def _scope_cells():
    return [
        _cell("Qwen3.5-4B", "uk", "military_status", "explicit", Verdict.CONFIRMED, mad=0.09),
        _cell("Qwen3.5-4B", "uk", "military_status_x_gender", "explicit",
              Verdict.CONFIRMED, mad=0.06),
        _cell("Qwen3.5-4B", "uk", "religion", "explicit", Verdict.CLEAN, mad=0.004),
        # A model whose ONLY disparity is in an intersection.
        _cell("gemma-4-12B-it", "uk", "military_status_x_religion", "explicit",
              Verdict.CONFIRMED, mad=0.05),
        _cell("gemma-4-12B-it", "uk", "gender", "explicit", Verdict.CLEAN, mad=0.003),
    ]


def test_full_scope_evaluates_every_group():
    plan = targets.build_plan(_scope_cells(), eval_scope="full")
    groups = plan.eval_groups["qwen3.5-4b"]["uk"]
    assert "military_status_x_gender" in groups and "religion" in groups


def test_targeted_scope_keeps_only_the_groups_being_mitigated():
    """Controls are deliberately NOT in a run's scope: it measures what it is meant to fix.

    The cost is that cross-group damage is invisible to that run. What still guards against
    gross damage is the attribute-free condition, which every run keeps.
    """
    plan = targets.build_plan(_scope_cells(), eval_scope="targeted")
    groups = plan.eval_groups["qwen3.5-4b"]["uk"]
    assert set(groups) == {"military_status", "military_status_x_gender"}
    assert "religion" not in groups, "the clean control group is not evaluated"
    assert "attr_free" in plan.eval_conditions["qwen3.5-4b"]["uk"]


def test_no_intersection_scope_drops_them_and_says_which_models_it_orphans():
    """gemma's only confirmed disparity is an intersection, so it has nothing left to fix."""
    plan = targets.build_plan(_scope_cells(), eval_scope="targeted-no-intersections")
    every_group = [
        g for groups in plan.eval_groups.values() for langs in groups.values() for g in langs
    ]
    assert not any("_x_" in g for g in every_group)
    assert all("_x_" not in t.group for t in plan.targets)
    assert any("removed from the mitigation stage" in n and "gemma" in n for n in plan.notes)
    assert any("Non-additivity under mitigation goes unmeasured" in n for n in plan.notes)


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError, match="unknown eval_scope"):
        targets.build_plan(_scope_cells(), eval_scope="whatever")


def test_experiment_plan_scope_is_per_model_not_a_fixed_grid():
    """The plan must describe what each run actually measures.

    Once `--eval-scope` restricts configs per model and language, printing one hard-coded
    grid everywhere misdescribes the runs and misstates their cost.
    """
    matrix = _load_matrix_module()
    plan = {
        "targets": [{"model": "Qwen3.5-4B", "lang": "uk", "model_slug": "qwen3.5-4b"}],
        "controls": [], "rescues": [],
        "eval_scope": "targeted-no-intersections",
        "eval_groups": {
            "qwen3.5-4b": {"uk": ["gender", "military_status", "religion"]},
            "lapa-12b": {"uk": ["military_status"]},
        },
        "config_paths": {
            "prompt": ["configs/mitigation/prompt/qwen3.5-4b_uk_structured_rubric.yaml"],
        },
    }
    assert matrix.eval_groups_for("qwen3.5-4b", "uk", plan) == [
        "gender", "military_status", "religion"
    ]
    assert matrix.eval_groups_for("lapa-12b", "uk", plan) == ["military_status"]
    # 34 attributes x 450 pairs x 2 injected conditions, plus one attribute-free pass.
    both = ["explicit", "implicit", "attr_free"]
    assert matrix.eval_cost(["gender", "military_status", "religion"], both) == 31_050
    assert matrix.eval_cost(["military_status"], both) == 4_950

    run = matrix.build_runs(plan)[0]
    assert run["eval_groups"] == ["gender", "military_status", "religion"]
    assert run["eval_prompts"] == 31_050


def test_bilingual_run_scope_is_the_union_of_both_languages():
    matrix = _load_matrix_module()
    plan = {
        "targets": [], "controls": [], "rescues": [], "eval_groups": {
            "m": {"en": ["military_status"], "uk": ["gender", "military_status"]},
        },
    }
    assert matrix.eval_groups_for("m", "en + uk", plan) == ["gender", "military_status"]


def test_scope_falls_back_to_the_full_grid_when_unrestricted():
    matrix = _load_matrix_module()
    groups = matrix.eval_groups_for("unknown", "en", {"eval_groups": {}})
    assert "military_status_x_gender" in groups


def test_rescue_arm_evaluates_the_same_groups_as_the_fairness_track():
    """A rescue arm has no interpretable baseline, so there is nothing to narrow it from.

    `targeted` scope means "keep the groups that showed a confirmed disparity for this
    model". A rescue arm has none — that is why it is a rescue arm — so applying the
    targeting rule to it is a category error. It also needs the full group set for the
    conditional secondary outcome: if utility crosses the gate, what disparity does the only
    Ukrainian-native model then show, compared with the general-purpose ones?
    """
    cells = [
        _cell("Qwen3.5-4B", "uk", "military_status", "explicit", Verdict.CONFIRMED, mad=0.09),
        _cell("Qwen3.5-4B", "uk", "gender", "explicit", Verdict.CONFIRMED, mad=0.05),
        _cell("Qwen3.5-4B", "uk", "religion", "explicit", Verdict.CLEAN, mad=0.004),
        _gated("lapa-v0.1.2-instruct", "uk", "military_status", 0.53, 0.29),
        _gated("lapa-v0.1.2-instruct", "uk", "religion", 0.52, 0.27),
    ]
    plan = targets.build_plan(cells, eval_scope="targeted-no-intersections")
    lapa = plan.eval_groups["lapa-12b"]["uk"]
    qwen = plan.eval_groups["qwen3.5-4b"]["uk"]
    # Every in-scope group, not a narrowed slice: `targeted` scope means "keep the groups
    # that showed a confirmed disparity", and a rescue arm has none to narrow from.
    assert set(lapa) == {"military_status", "gender", "religion"}
    # The fairness track IS narrowed -- to the groups it is mitigating.
    assert set(qwen) < set(lapa)


# ---- phase-2 selection ------------------------------------------------------------------------

def _load_phase2():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "select_phase2", root / "scripts" / "select_phase2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mit_record(name, family, variant, mad, utility, lang="uk", model="Qwen/Qwen3.5-4B"):
    return {
        "run_name": name,
        "meta": {"model": model, "lang": lang, "limit_pairs": None,
                 "mitigation": {"family": family, "strategy": variant}},
        "summary": {"reference_agreement": utility, "refusal_rate": 0.0,
                    "disparity": {"overall": {"ar_mad": mad}}},
        "groups": {}, "manual_review": [],
    }


def _base_record(mad, utility, lang="uk", model="Qwen/Qwen3.5-4B"):
    r = _mit_record("base", "none", None, mad, utility, lang, model)
    r["meta"]["mitigation"] = {"family": "none"}
    return r


def test_phase2_ranks_by_disparity_reduction():
    phase2 = _load_phase2()
    records = [
        _base_record(0.08, 0.75),
        _mit_record("a", "prompt", "structured_rubric", 0.03, 0.75),
        _mit_record("b", "prompt", "zero_shot_cot", 0.06, 0.75),
    ]
    ranked = phase2.rank(records)
    assert ranked[0]["variant"] == "structured_rubric"
    assert ranked[0]["mad_reduction"] == pytest.approx(0.05)


def test_phase2_never_promotes_an_arm_that_damaged_the_model():
    """Cutting disparity by degrading the model is not a win, however large the cut."""
    phase2 = _load_phase2()
    records = [
        _base_record(0.08, 0.80),
        _mit_record("wrecker", "prompt", "x", 0.005, 0.60),   # huge cut, utility -20pp
        _mit_record("honest", "prompt", "y", 0.05, 0.79),     # modest cut, utility held
    ]
    scored = phase2.rank(records)
    assert next(s for s in scored if s["variant"] == "x")["damaged"]
    promoted = phase2.winners(scored, top=1)
    assert [p["variant"] for p in promoted] == ["y"]


def test_phase2_ignores_arms_that_made_things_worse():
    phase2 = _load_phase2()
    records = [_base_record(0.05, 0.80), _mit_record("worse", "prompt", "z", 0.07, 0.80)]
    assert phase2.winners(phase2.rank(records), top=1) == []


def test_phase2_takes_one_winner_per_family_model_language():
    phase2 = _load_phase2()
    records = [
        _base_record(0.08, 0.80),
        _mit_record("p1", "prompt", "a", 0.03, 0.80),
        _mit_record("p2", "prompt", "b", 0.04, 0.80),
        _mit_record("s1", "scrub", "lexical", 0.035, 0.80),
    ]
    promoted = phase2.winners(phase2.rank(records), top=1)
    assert {(p["family"], p["variant"]) for p in promoted} == {
        ("prompt", "a"), ("scrub", "lexical")
    }


# ---- strict targeting: groups AND conditions --------------------------------------------------

def test_scope_restricts_to_the_condition_the_target_was_found_in():
    """A target is a group *under a condition*. Measuring the other framing doubles the cost
    to answer a question that cell did not raise."""
    cells = [
        _cell("Qwen3.5-9B", "en", "military_status", "implicit", Verdict.CONFIRMED, mad=0.06),
        _cell("Qwen3.5-9B", "uk", "military_status", "explicit", Verdict.CONFIRMED, mad=0.08),
        _cell("Qwen3.5-9B", "uk", "religion", "implicit", Verdict.CONFIRMED, mad=0.05),
    ]
    plan = targets.build_plan(cells, eval_scope="targeted-no-intersections")
    assert plan.eval_conditions["qwen3.5-9b"]["en"] == ["implicit", "attr_free"]
    assert plan.eval_conditions["qwen3.5-9b"]["uk"] == ["explicit", "implicit", "attr_free"]


def test_attr_free_is_always_kept():
    """It is the only thing separating a removed disparity from a moved operating point."""
    cells = [_cell("M", "en", "military_status", "explicit", Verdict.CONFIRMED, mad=0.08)]
    plan = targets.build_plan(cells, eval_scope="targeted")
    assert "attr_free" in plan.eval_conditions["m"]["en"]


def test_scope_keeps_only_the_groups_being_mitigated():
    """Controls are no longer added to a run's scope: it measures what it is meant to fix."""
    cells = [
        _cell("Qwen3.5-4B", "uk", "military_status", "explicit", Verdict.CONFIRMED, mad=0.09),
        _cell("Qwen3.5-4B", "uk", "religion", "explicit", Verdict.CLEAN, mad=0.003),
    ]
    plan = targets.build_plan(cells, eval_scope="targeted-no-intersections")
    assert plan.eval_groups["qwen3.5-4b"]["uk"] == ["military_status"]


def test_rescue_arm_gets_tuning_families_only():
    """A prompt edit cannot move a decision threshold the way reference-pinned training does."""
    cells = [
        _gated("lapa-v0.1.2-instruct", "uk", "military_status", 0.53, 0.29),
        _gated("lapa-v0.1.2-instruct", "uk", "religion", 0.52, 0.27),
    ]
    plan = targets.build_plan(cells)
    assert plan.rescues[0].suggested_families == ["sft", "dpo"]
    assert not any("lapa" in p for p in plan.config_paths.get("prompt", []))
    assert any("lapa" in p for p in plan.config_paths.get("sft", []))


def test_matrix_costs_only_the_conditions_in_scope():
    matrix = _load_matrix_module()
    plan = {
        "targets": [], "controls": [], "rescues": [],
        "eval_groups": {"m": {"en": ["military_status"]}},
        "eval_conditions": {"m": {"en": ["implicit", "attr_free"]}},
    }
    # 5 attributes x 450 pairs x 1 injected condition, plus one attribute-free pass.
    assert matrix.eval_cost(["military_status"], ["implicit", "attr_free"]) == 2_700
    assert matrix.eval_cost(["military_status"], ["explicit", "implicit", "attr_free"]) == 4_950
    assert matrix.eval_conditions_for("m", "en", plan) == ["implicit", "attr_free"]


def test_running_check_ignores_a_shell_that_merely_mentions_the_runner():
    """`pgrep -f <name>` matches any command line containing the name — including the shell
    that invoked the check. That false positive refuses every legitimate edit."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "select_configs", root / "scripts" / "select_configs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Spawn a shell whose argv merely *mentions* a runner. Asserting that no runner is active
    # would be asserting on the machine's live state — and on a machine where a sweep is
    # genuinely running, `_is_running` returning True is the correct answer, not a bug. The
    # controlled false positive is the thing under test.
    import subprocess
    import time

    runner = root / "scripts" / "run_all_audit.sh"
    decoy = subprocess.Popen(["bash", "-c", f"# {runner}\nsleep 5"])
    try:
        time.sleep(0.3)
        assert module._is_running(runner) is False
    finally:
        decoy.terminate()
        decoy.wait(timeout=5)


def test_phase2_reads_confirmed_cells_not_the_filtered_target_list():
    """Under `targeted-no-intersections` the planner strips intersection targets from
    `plan["targets"]` by design. Reading that list here would find nothing and phase 2 would
    silently do nothing — the worst failure for a stage that exists to follow up the study's
    intersection finding."""
    phase2 = _load_phase2()
    analysis = {
        "plan": {"targets": []},  # deliberately empty, as the scope leaves it
        "cells": [
            {"model": "Qwen3.5-4B", "lang": "uk", "group": "military_status_x_gender",
             "condition": "explicit", "verdict": "confirmed", "gate_passed": True,
             "ar_mad": 0.063},
            {"model": "Qwen3.5-4B", "lang": "uk", "group": "military_status_x_religion",
             "condition": "implicit", "verdict": "confirmed", "gate_passed": True,
             "ar_mad": 0.038},
        ],
    }
    found = phase2.intersection_targets(analysis)
    assert ("qwen3.5-4b", "uk") in found
    spec = found[("qwen3.5-4b", "uk")]
    assert spec["groups"] == ["military_status_x_gender", "military_status_x_religion"]
    assert spec["conditions"] == ["explicit", "implicit", "attr_free"]


def test_phase2_skips_intersections_with_nothing_to_fix():
    """Re-measuring a null intersection costs 40k-130k generations."""
    phase2 = _load_phase2()
    analysis = {"plan": {"targets": []}, "cells": [
        {"model": "M", "lang": "uk", "group": "military_status_x_gender",
         "condition": "explicit", "verdict": "confirmed", "gate_passed": True, "ar_mad": 0.005},
    ]}
    assert phase2.intersection_targets(analysis, min_mad=0.02) == {}


def test_phase2_ignores_uninterpretable_and_unconfirmed_cells():
    phase2 = _load_phase2()
    cells = [
        {"model": "M", "lang": "uk", "group": "military_status_x_gender", "condition": "explicit",
         "verdict": "confirmed", "gate_passed": False, "ar_mad": 0.09},   # gated out
        {"model": "M", "lang": "uk", "group": "military_status_x_religion", "condition": "explicit",
         "verdict": "weak", "gate_passed": True, "ar_mad": 0.09},         # not confirmed
    ]
    assert phase2.intersection_targets({"plan": {"targets": []}, "cells": cells}) == {}


def test_phase2_cost_accounts_for_conditions():
    phase2 = _load_phase2()
    # 100 cells x 450 pairs x 1 injected condition, plus the attribute-free pass.
    assert phase2.cost(["military_status_x_gender"], ["implicit", "attr_free"]) == 45_450
    assert phase2.cost(["military_status_x_gender"],
                       ["explicit", "implicit", "attr_free"]) == 90_450


def _stability_rows(rows):
    import pandas as pd

    frame = pd.DataFrame(rows, columns=["pair_id", "condition", "protected_group",
                                        "protected_attr", "candidate_id", "decision"])
    from hiring_bias_mitigation.analysis.stability import _decision

    frame["d"] = frame["decision"].map(_decision)
    return frame


def test_set_stability_counts_only_sets_decided_in_full():
    """Dropping unparsed rows makes a set of failures look perfectly stable."""
    from hiring_bias_mitigation.analysis.stability import compare

    base = _stability_rows([
        ("p1", "e", "g", "a", "c1", "hire"), ("p1", "e", "g", "b", "c1", "reject"),
        ("p2", "e", "g", "a", "c2", "hire"), ("p2", "e", "g", "b", "c2", "hire"),
    ])
    run = _stability_rows([
        ("p1", "e", "g", "a", "c1", "hire"), ("p1", "e", "g", "b", "c1", "hire"),
        ("p2", "e", "g", "a", "c2", "hire"), ("p2", "e", "g", "b", "c2", "junk"),
    ])
    result = compare(base, run, n_boot=50)
    assert result["sets"] == 1
    assert (result["fixed"], result["broken"]) == (1, 0)


def test_set_stability_compares_the_same_variants_only():
    """A baseline set holding extra variants must not be read as extra instability.

    The baseline audit covers intersections a mitigated run does not. Unmatched, a set whose
    only dissent sat in an intersection cell reads as "fixed" by any run at all; every training
    adapter once looked like a significant improvement that way and none survived matching.
    """
    from hiring_bias_mitigation.analysis.stability import compare

    base = _stability_rows([
        ("p1", "e", "mil", "vet", "c1", "hire"), ("p1", "e", "mil", "civ", "c1", "hire"),
        ("p1", "e", "mil_x_g", "vet|f", "c1", "reject"),        # the only dissent
    ])
    run = _stability_rows([
        ("p1", "e", "mil", "vet", "c1", "hire"), ("p1", "e", "mil", "civ", "c1", "hire"),
    ])
    result = compare(base, run, n_boot=50)
    assert result["unstable_base_pct"] == 0.0, "the intersection cell must not count"
    assert (result["fixed"], result["broken"]) == (0, 0)
