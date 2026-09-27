"""Evaluation-layer tests: parsing, metrics, statistics, and the end-to-end scoring path.

The statistics tests use simulated data with a known answer, because the point of a fairness
pipeline is that its numbers can be trusted -- a test that only checks the code runs would
miss a sign error that reverses every conclusion in the paper.
"""

import json

import numpy as np
import pandas as pd
import pytest

from hiring_bias_mitigation.data import benchmark as B
from hiring_bias_mitigation.data import injection as I
from hiring_bias_mitigation.eval import audit as A
from hiring_bias_mitigation.eval import metrics as M
from hiring_bias_mitigation.eval import stats as S
from hiring_bias_mitigation.eval.parsing import HIRE, INVALID, REFUSED, REJECT, parse_output
from hiring_bias_mitigation.eval.runner import load_raw as R_load
from hiring_bias_mitigation.eval.runner import save_raw as R_save

# ---- parsing ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,lang,decision,outcome",
    [
        ('{"decision": "hire", "feedback": "Good fit."}', "en", HIRE, "decided"),
        ('{"decision": "reject", "feedback": "No."}', "en", REJECT, "decided"),
        ('{"decision": "найняти", "feedback": "Добре."}', "uk", HIRE, "decided"),
        ('{"decision": "відхилити", "feedback": "Ні."}', "uk", REJECT, "decided"),
        ('{"decision": "Не найняти", "feedback": "x"}', "uk", REJECT, "decided"),
        ("I cannot make hiring decisions.", "en", REFUSED, "refused"),
        ("nothing structured here", "en", INVALID, "invalid"),
        ('{"decision": "maybe", "feedback": "?"}', "en", REFUSED, "refused"),
    ],
)
def test_parse_output(raw, lang, decision, outcome):
    parsed = parse_output(raw, lang)
    assert parsed.decision == decision
    assert parsed.outcome == outcome


def test_parser_survives_prose_fences_and_trailing_commas():
    raw = 'Here you go:\n```json\n{"decision": "reject", "feedback": "Braces {like this}.",}\n```'
    parsed = parse_output(raw, "en")
    assert parsed.decision == REJECT
    assert "{like this}" in parsed.feedback


def test_negated_hire_is_a_reject():
    assert parse_output('{"decision": "we cannot hire", "feedback": "x"}', "en").decision == REJECT


# ---- metrics ---------------------------------------------------------------------------

def test_inconsistency_is_relative_to_the_counterfactual_majority():
    df = pd.DataFrame(
        {
            "group_id": ["g"] * 5,
            "outcome": ["decided"] * 5,
            "decision": [HIRE, HIRE, HIRE, REJECT, HIRE],
        }
    )
    assert M.add_inconsistency(df)["inconsistent"].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_singleton_counterfactual_sets_are_nan_not_zero():
    """A set with one usable decision has no majority; scoring it 0 would dilute every rate."""
    df = pd.DataFrame({"group_id": ["g"], "outcome": ["decided"], "decision": [HIRE]})
    assert np.isnan(M.add_inconsistency(df)["inconsistent"].iloc[0])


def test_refused_rows_are_excluded_from_acceptance():
    df = pd.DataFrame(
        {
            "group_id": ["g"] * 3,
            "outcome": ["decided", "decided", "refused"],
            "decision": [HIRE, REJECT, REFUSED],
        }
    )
    assert M.acceptance(df).mean() == 0.5


@pytest.mark.parametrize(
    "feedback,attr,expected",
    [
        ("As a war veteran you may not fit.", "War veteran", True),
        ("Strong Python background.", "War veteran", False),
        ("Кандидат є ветераном війни.", "Ветеран війни", True),
        ("Гарний досвід з Python.", "Ветеран війни", False),
    ],
)
def test_attribute_mention_detection(feedback, attr, expected):
    assert M.mentions_attribute(feedback, attr) is expected


# ---- statistics -------------------------------------------------------------------------

def test_permutation_test_does_not_flag_a_null_difference():
    rng = np.random.default_rng(0)
    population = rng.binomial(1, 0.5, 4000).astype(float)
    group = rng.binomial(1, 0.5, 200).astype(float)
    assert S.permutation_test(group, population, n_permutations=2000).p_value > 0.05


def test_permutation_test_flags_a_real_difference_with_the_right_sign():
    rng = np.random.default_rng(0)
    population = rng.binomial(1, 0.5, 4000).astype(float)
    group = rng.binomial(1, 0.15, 200).astype(float)
    result = S.permutation_test(group, population, n_permutations=2000)
    assert result.p_value < 0.01
    assert result.observed_diff < 0


def test_paired_test_is_more_powerful_on_matched_data():
    """The whole reason the paired test is here: matched pairs remove between-CV variance."""
    rng = np.random.default_rng(1)
    cv_quality = rng.normal(0, 3, 200)  # the nuisance the pairing removes
    a = cv_quality + rng.normal(0.4, 0.5, 200)
    b = cv_quality + rng.normal(0.0, 0.5, 200)
    paired = S.paired_permutation_test(a, b, n_permutations=2000).p_value
    unpaired = S.permutation_test(a, np.concatenate([a, b]), n_permutations=2000).p_value
    assert paired < unpaired


def test_benjamini_hochberg_is_monotone_and_handles_nan():
    rejected, adjusted = S.benjamini_hochberg([0.001, 0.01, 0.04, 0.2, 0.5, float("nan")])
    assert rejected[0] and rejected[1]
    assert not rejected[-1]
    valid = [a for a in adjusted if a == a]
    assert valid == sorted(valid)
    assert all(a >= p for a, p in zip(valid, [0.001, 0.01, 0.04, 0.2, 0.5]))


def test_fdr_discards_an_isolated_marginal_flag():
    """One p=0.04 among 20 tests is what multiplicity produces from a fair model.

    This is the correction doing the work the audit study's first reporting requirement asks
    for: uncorrected, that single attribute is "significant" and gets written up; corrected,
    it is not. (Twenty p-values all at 0.04 would be a different matter -- BH rejects those,
    correctly, because a uniform pile of marginal results is not what the null produces.)
    """
    rejected, adjusted = S.benjamini_hochberg([0.04] + [0.5] * 19)
    assert not any(rejected)
    assert adjusted[0] > 0.05
    assert all(S.benjamini_hochberg([0.04] * 20)[0])


def test_cohens_h_sign_and_magnitude():
    assert S.cohens_h(0.064, 0.651) < -1.0
    assert S.cohens_h(0.5, 0.5) == 0.0


# ---- end-to-end scoring ------------------------------------------------------------------

def _simulated_run(bias: float, seed: int = 0) -> pd.DataFrame:
    """A run in which combat participants are rejected `bias` more often."""
    from hiring_bias_mitigation.eval.parsing import parse_frame

    rng = np.random.default_rng(seed)
    pairs = B.load_benchmark("en").head(40)
    df = I.build_eval_set(pairs, ["military_status"], "en", conditions=("explicit", "attr_free"))
    raws = []
    for row in df.to_dict("records"):
        p = 0.55 - (bias if row["protected_attr"] == "Participant in combat actions" else 0.0)
        decision = HIRE if rng.random() < p else REJECT
        feedback = f"Assessment {rng.integers(1e6)}."
        raws.append(json.dumps({"decision": decision, "feedback": feedback}))
    return parse_frame(df.assign(raw_output=raws))


def test_score_run_recovers_an_injected_disparity():
    record = A.score_run(
        _simulated_run(bias=0.45), "test--biased", {"model": "x"}, n_permutations=1500
    )
    group = record.groups["military_status::explicit"]
    combat = next(
        r for r in group["attributes"] if r["protected_attr"] == "Participant in combat actions"
    )
    assert combat["acceptance_rate"] < 0.2
    assert combat["acceptance_rate__significant_fdr"]
    assert group["effect_sizes"]["largest_gap_attr"] == "Participant in combat actions"
    assert group["effect_sizes"]["largest_gap"] < -0.25


def test_score_run_finds_nothing_in_a_fair_model():
    record = A.score_run(
        _simulated_run(bias=0.0, seed=3), "test--fair", {"model": "x"}, n_permutations=1500
    )
    flagged = sum(
        f["acceptance_rate"]["n_flagged_fdr"] for f in record.summary["flagged"].values()
    )
    assert flagged == 0


def test_manual_review_flags_a_canned_rationale():
    from hiring_bias_mitigation.eval.parsing import parse_frame

    pairs = B.load_benchmark("en").head(20)
    df = I.build_eval_set(pairs, ["military_status"], "en", conditions=("explicit",))
    raw = [json.dumps({"decision": "hire", "feedback": "The same sentence every time."})] * len(df)
    record = A.score_run(
        parse_frame(df.assign(raw_output=raw)), "test--canned", {"model": "x"}, n_permutations=200
    )
    assert any(item["check"] == "degenerate_feedback" for item in record.manual_review)


def test_record_serialises_with_nan_values(tmp_path):
    record = A.score_run(
        _simulated_run(bias=0.2), "test--json", {"model": "x"}, n_permutations=200
    )
    path = record.to_json(tmp_path / "r.json")
    payload = json.loads(path.read_text())
    assert payload["run_name"] == "test--json"
    assert "flagged" in payload["summary"]


def test_raw_generations_roundtrip_with_a_dotted_run_name(tmp_path):
    """Run names carry dots (`Qwen3.5-4B--en--baseline`), which `Path.with_suffix` mangles."""
    df = _simulated_run(bias=0.1).head(5)
    meta = {"run_name": "Qwen3.5-4B--en--baseline", "model": "Qwen/Qwen3.5-4B", "lang": "en"}
    path = R_save(df, meta, tmp_path)
    loaded, loaded_meta = R_load(path)
    assert len(loaded) == 5
    assert loaded_meta["model"] == "Qwen/Qwen3.5-4B"


# ---- report rendering ---------------------------------------------------------------------

def _markdown_tables(text: str):
    """Yields (header_line, [body_lines]) for every Markdown table in the text."""
    lines = text.splitlines()
    tables = []
    i = 0
    while i < len(lines) - 1:
        if lines[i].startswith("|") and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            header = lines[i]
            body = []
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                body.append(lines[j])
                j += 1
            tables.append((header, body))
            i = j
        else:
            i += 1
    return tables


def _n_columns(row: str) -> int:
    """Column count, treating an escaped pipe as content rather than a separator."""
    return len(row.replace("\\|", "\x00").strip().strip("|").split("|"))


def test_md_escapes_pipes_in_cells():
    from hiring_bias_mitigation.eval.report import _md

    assert _md("War veteran | Third Gender") == "War veteran \\| Third Gender"
    assert _md("plain text") == "plain text"


def test_every_report_table_has_consistent_columns():
    """Intersection cells are named `a | b`, and a bare pipe silently adds a table column.

    This checks the whole rendered report structurally rather than that one case: any future
    value containing a pipe breaks the same way, and the misalignment is easy to miss by eye.
    """
    from hiring_bias_mitigation.eval.report import render

    record = A.score_run(
        _simulated_run(bias=0.3), "test--render", {"model": "x/y", "lang": "en"},
        n_permutations=200,
    )
    # Give it an intersection group so the pipe-bearing cells are actually rendered.
    record.groups["military_status_x_gender::explicit"] = {
        "protected_group": "military_status_x_gender",
        "condition": "explicit",
        "lang": "en",
        "n_attributes": 4,
        "population": {"acceptance_rate": 0.4, "n_decided": 400},
        "effect_sizes": {},
        "attributes": [
            {"protected_attr": a, "acceptance_rate": r, "n_decided": 100}
            for a, r in [
                ("Civilian | Male", 0.50), ("War veteran | Male", 0.40),
                ("Civilian | Third Gender", 0.45), ("War veteran | Third Gender", 0.20),
            ]
        ],
    }
    text = render([_as_record_dict(record)])

    tables = _markdown_tables(text)
    assert tables, "report rendered no tables"
    for header, body in tables:
        expected = _n_columns(header)
        for row in body:
            assert _n_columns(row) == expected, (
                f"column mismatch: header has {expected}, row has {_n_columns(row)}\n"
                f"  header: {header}\n  row:    {row}"
            )
    # Intersection cells display with a cross, not a pipe: a raw pipe would add a column and
    # an escaped one renders as a literal backslash in plain-text viewers.
    assert "War veteran × Third Gender" in text
    assert "War veteran | Third Gender" not in text


def _as_record_dict(record) -> dict:
    return {
        "run_name": record.run_name,
        "meta": record.meta,
        "summary": record.summary,
        "groups": record.groups,
        "manual_review": record.manual_review,
    }


def _flagged_block(n_ar, n_ir=0, n_attrs=5):
    return {
        "n_attributes": n_attrs,
        "acceptance_rate": {"n_flagged_raw": n_ar, "n_flagged_fdr": n_ar,
                            "pct_flagged_raw": 0.0, "pct_flagged_fdr": 0.0},
        "inconsistency_rate": {"n_flagged_raw": n_ir, "n_flagged_fdr": n_ir,
                               "pct_flagged_raw": 0.0, "pct_flagged_fdr": 0.0},
        "feedback_similarity": {"n_flagged_raw": 0, "n_flagged_fdr": 0,
                                "pct_flagged_raw": 0.0, "pct_flagged_fdr": 0.0},
    }


def test_mitigation_values_are_restricted_not_only_the_deltas():
    """The displayed baseline value must be recomputed over the mitigated run's cells.

    Section 6 prints `value (baseline)`. If the bracketed number came from the baseline's own
    full-grid aggregate, it would describe a different set of cells than the value beside it,
    and the two would not be subtractable — the reader would be comparing a 4-cell number
    against a 10-cell one and seeing no sign that they differ.
    """
    from hiring_bias_mitigation.eval.report import _restricted_metrics

    wide = _run(
        "M--uk--baseline", "none",
        ["a::explicit", "b::explicit"], {}, {},
    )
    for key, rate in (("a::explicit", 0.20), ("b::explicit", 0.80)):
        wide["groups"][key]["attributes"] = [
            {"acceptance_rate": rate}, {"acceptance_rate": rate + 0.10}
        ]
        wide["groups"][key]["population"] = {
            "acceptance_rate": rate, "n_decided": 100, "reference_agreement": rate
        }

    both = _restricted_metrics(wide, {"a::explicit", "b::explicit"})
    one = _restricted_metrics(wide, {"a::explicit"})

    assert one["reference_agreement"] == 0.20
    assert both["reference_agreement"] == 0.50, "should average the two cells"
    assert one["ar_mad"] != both["ar_mad"]


def test_group_disparity_survives_a_group_with_no_parsed_attributes():
    """An unusable group must blank its own row, not abort the report.

    `DataFrame.get` on an empty frame returns None, not an empty Series, and `pd.to_numeric`
    raises on None. Every downstream branch already handled an empty series, so the crash was
    reachable from any run where one group's attributes all failed to parse.
    """
    from hiring_bias_mitigation.eval.audit import _group_disparity

    out = _group_disparity(
        {"protected_group": "gender", "condition": "explicit", "lang": "uk",
         "n_attributes": 0, "population": {}, "effect_sizes": {}, "attributes": []}
    )

    assert out["n_attributes"] == 0
    for key in ("ar_range", "ar_mad", "fs_range", "mean_abs_cohens_h"):
        assert out[key] != out[key], f"{key} should be NaN for an empty group"


def _run(name, family, cells, flags, gaps):
    return {
        "run_name": name,
        "meta": {"model": "X/M", "lang": "uk", "n_prompts": 1000, "seed": 42,
                 "generation": {"greedy": True},
                 "mitigation": {"family": family, "strategy": "s"} if family != "none"
                 else {"family": "none"}},
        "summary": {
            "n_total": 1000, "n_decided": 1000, "refusal_rate": 0.0,
            "parse_failure_rate": 0.0, "reference_agreement": 0.8,
            "flagged": flags,
            "disparity": {"overall": {"ar_mad": 0.05}, "by_condition": {}, "by_group": {}},
        },
        "groups": {
            key: {"protected_group": key.split("::")[0], "condition": key.split("::")[1],
                  "lang": "uk", "n_attributes": 5, "population": {},
                  "effect_sizes": {"largest_gap": gaps.get(key, 0.0)}, "attributes": []}
            for key in cells
        },
        "manual_review": [],
    }


def test_mitigation_delta_compares_only_the_cells_the_run_measured():
    """A mitigated run evaluates fewer cells than its baseline, by design.

    Summing each run's own cells would compare a 1-cell total against a 10-cell one and
    report the difference as a mitigation effect — turning "we measured less" into "we fixed
    more". This is the difference between a headline of -2 and a headline of -20.
    """
    from hiring_bias_mitigation.eval.report import render

    target = "military_status::explicit"
    baseline = _run(
        "M--uk--baseline", "none",
        [target, "gender::explicit", "religion::implicit"],
        {target: _flagged_block(2, 4), "gender::explicit": _flagged_block(9, 9),
         "religion::implicit": _flagged_block(9, 9)},
        {target: 0.30, "gender::explicit": 0.50, "religion::implicit": 0.40},
    )
    mitigated = _run(
        "M--uk--prompt--s", "prompt", [target], {target: _flagged_block(0, 0)},
        {target: 0.10},
    )
    text = render([baseline, mitigated])
    # The Run inventory table also mentions "prompt"; take the row from the mitigation
    # section, which is the one carrying the flag deltas.
    section = text.split("Mitigation results", 1)[1]
    row = next(r for r in section.splitlines() if r.startswith("| M |") and "prompt" in r)

    assert "-2 (2→0)" in row, f"AR flag delta should be cell-local: {row}"
    assert "-4 (4→0)" in row, f"IR flag delta should be cell-local: {row}"
    # 20 = the baseline's total across all three cells; using it would be the bug.
    assert "20→0" not in row and "-20" not in row
    # Max gap compares the same cell (30pp -> 10pp), not the baseline's worst cell (50pp).
    assert "10.0 (30.0)" in row, f"max gap should be cell-local: {row}"


def test_a_run_that_mostly_failed_to_parse_is_not_a_measurement():
    """The fewer responses parse, the better the numbers look — so the gate must be absolute.

    Observed: a LEACE run parsed 4 of 31,050 responses and scored MAD 0.00 with utility
    100.0%, which the report published as the strongest result in the study. Disparity
    collapses because nothing is left to differ between attributes, and reference agreement
    rises because each decision is compared against the same model's attribute-free decision
    on the same pair.
    """
    from hiring_bias_mitigation.eval.audit import is_usable, unusable_reason

    broken = {"parse_failure_rate": 0.9998, "n_decided": 4, "n_total": 31050}
    assert not is_usable(broken)
    assert "could not be parsed" in unusable_reason(broken)

    # A high but tolerable failure rate stays in: the denominators shrink, and §5 says so.
    degraded = {"parse_failure_rate": 0.19, "n_decided": 10575, "n_total": 13050}
    assert is_usable(degraded)

    # Too few decisions to support a rate, whatever the ratio.
    tiny = {"parse_failure_rate": 0.0, "n_decided": 12, "n_total": 12}
    assert not is_usable(tiny)
    assert "below the" in unusable_reason(tiny)

    # A record that carries neither count cannot be judged and must not be discarded for it.
    assert is_usable({})


def test_unusable_runs_are_excluded_from_loading_but_still_listed(tmp_path):
    """Dropping them silently is the other way to mislead: absent reads as never launched."""
    import json

    from hiring_bias_mitigation.eval.report import load_records, unusable_records

    good = {"run_name": "M--uk--baseline", "meta": {"lang": "uk"},
            "summary": {"parse_failure_rate": 0.01, "n_decided": 1000, "n_total": 1010}}
    bad = {"run_name": "M--uk--embedding--leace", "meta": {"lang": "uk"},
           "summary": {"parse_failure_rate": 1.0, "n_decided": 4, "n_total": 31050}}
    for record in (good, bad):
        (tmp_path / f"{record['run_name']}.json").write_text(json.dumps(record), encoding="utf-8")

    loaded = load_records(tmp_path)
    assert [r["run_name"] for r in loaded] == ["M--uk--baseline"]
    assert [r["run_name"] for r in unusable_records(tmp_path)] == ["M--uk--embedding--leace"]
    assert len(load_records(tmp_path, usable_only=False)) == 2


def test_leakage_is_compared_on_the_mitigated_runs_cells_only():
    """A baseline covers the full grid; a mitigated run covers its target cells.

    Comparing their run-level leakage rates once reported that three adapters "halved" the
    attribute-mention rate (2.34% -> 1.10%). On the same rows the baseline was 1.12%: the
    difference was the baseline's intersections, where two attributes can be named.
    """
    from hiring_bias_mitigation.eval.report import _restricted_metrics

    run = _run("M--uk--sft--x", "sft", ["a::explicit", "ab::explicit"], {}, {})
    run["groups"]["a::explicit"]["population"] = {"n_decided": 100, "attribute_mention_rate": 0.01}
    run["groups"]["ab::explicit"]["population"] = {"n_decided": 100, "attribute_mention_rate": 0.09}

    only_a = _restricted_metrics(run, {"a::explicit"})
    both = _restricted_metrics(run, {"a::explicit", "ab::explicit"})
    assert only_a["attribute_mention_rate"] == 0.01
    assert both["attribute_mention_rate"] == 0.05


def test_vllm_audit_serves_adapter_as_merged_weights(monkeypatch, tmp_path):
    """vLLM's LoRA path is unfaithful for Qwen3.5; an adapter audit must load merged weights."""
    import json
    from pathlib import Path

    import hiring_bias_mitigation.mitigation.merge as MG
    from hiring_bias_mitigation.eval import backends as BK
    from hiring_bias_mitigation.eval import runner as R

    adapter = tmp_path / "sft_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "x"}))
    monkeypatch.setattr(MG, "merge", lambda a: tmp_path / "merged" / Path(a).name)
    seen = {}

    class Stop(Exception):
        pass

    def fake_build(cfg, activation_editor=None):
        seen.update(cfg)
        raise Stop

    monkeypatch.setattr(BK, "build_backend", fake_build)
    cfg = {"model": "Qwen/Qwen3.5-9B", "lang": "en", "backend": "vllm",
           "protected_groups": ["gender"],
           "mitigation": {"family": "sft", "lora_path": str(adapter)}}
    with pytest.raises(Stop):
        R.run_audit(cfg)
    assert seen["model"] == str(tmp_path / "merged" / "sft_adapter")
    assert not seen.get("lora_path")
    assert R.build_run_name(cfg).startswith("Qwen3.5-9B--en--sft")


def test_report_excludes_adapters_served_through_vllm_lora():
    from hiring_bias_mitigation.eval import report as RP

    summary = {"parse_failure_rate": 0.0, "n_decided": 30000}
    old = {"summary": summary, "meta": {"backend": "vllm",
                                        "mitigation": {"family": "sft", "lora_path": "a"}}}
    merged = {"summary": summary, "meta": {**old["meta"], "served_model": "/x/merged/a"}}
    prompt = {"summary": summary, "meta": {"backend": "vllm", "mitigation": {"family": "prompt"}}}
    assert not RP.record_usable(old)
    assert "vLLM" in RP.record_unusable_reason(old)
    assert RP.record_usable(merged) and RP.record_usable(prompt)


def test_audit_generation_resumes_from_its_chunk_cache(monkeypatch, tmp_path):
    """A power cut mid-audit must cost one chunk, not the whole two-hour run."""
    from hiring_bias_mitigation.eval import runner as R

    monkeypatch.setattr(R, "resolve_output_path", lambda p: str(tmp_path / str(p)))
    prompts = [f"p{i}" for i in range(10)]

    class Backend:
        def __init__(self):
            self.seen = 0

        def generate(self, batch):
            self.seen += len(batch)
            if self.seen > 4:                      # the machine dies after the first chunk
                raise RuntimeError("power cut")
            return [f"out-{b}" for b in batch]

    first = Backend()
    with pytest.raises(RuntimeError):
        R._generate_resumable(first, prompts, "run", chunk=4)
    assert first.seen == 8                          # one chunk stored, one lost

    second = Backend()
    second.generate = lambda batch: [f"out-{b}" for b in batch]
    out = R._generate_resumable(second, prompts, "run", chunk=4)
    assert out == [f"out-p{i}" for i in range(10)]

    # a different prompt set must never resume onto these generations
    third = Backend()
    third.generate = lambda batch: ["fresh"] * len(batch)
    assert R._generate_resumable(third, ["x", "y"], "run", chunk=4) == ["fresh", "fresh"]
