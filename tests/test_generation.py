"""Generation-pipeline tests: filters, dataset assembly, and the contamination guards.

No network and no GPU -- the Djinni pool builder is exercised through its filter functions on
synthetic frames, and the teacher is replaced by canned strings.
"""

import json

import pandas as pd
import pytest

from hiring_bias_mitigation.data import benchmark as B
from hiring_bias_mitigation.generation import dataset as D
from hiring_bias_mitigation.generation import filters as F
from hiring_bias_mitigation.generation import pool as PL
from hiring_bias_mitigation.generation import synth as SY


def _pairs(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": [f"c{i}__j{i}" for i in range(n)],
            "candidate_id": [f"c{i}" for i in range(n)],
            "job_id": [f"j{i}" for i in range(n)],
            "cv": ["Eight years of Python, Django and PostgreSQL. Led a team of six."] * n,
            "job_description": ["Senior Python engineer. Django, PostgreSQL, team lead."] * n,
            "job_position": ["Python Developer"] * n,
            "lang": ["en"] * n,
        }
    )


def _response(decision: str, feedback: str) -> str:
    return json.dumps({"decision": decision, "feedback": feedback})


# ---- pool filters --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,lang,expected",
    [
        ("I am a married father of two.", "en", True),
        ("I served in the armed forces.", "en", True),
        ("Eight years of Go and Kubernetes.", "en", False),
        ("Я одружений, маю двох дітей.", "uk", True),
        ("Досвід з Kubernetes та Go.", "uk", False),
    ],
)
def test_protected_marker_detection(text, lang, expected):
    assert PL.has_protected_marker(text, lang) is expected


def test_candidate_filter_drops_held_out_candidates():
    """A CV in the benchmark must never reach the training pool."""
    held_out = next(iter(B.holdout_ids().candidate_ids))
    candidates = pd.DataFrame(
        {
            "id": [held_out, "fresh-id"],
            "CV": ["Eight years of Python and Django, led a team of six engineers. " * 12] * 2,
            "Primary Keyword": ["Python", "Python"],
            "Experience Years": [8.0, 8.0],
        }
    )
    kept = PL.filter_candidates(candidates, PL.PoolConfig(lang="en"))
    assert list(kept["id"]) == ["fresh-id"]


def test_matcher_rejects_leaked_pairs():
    held_out_job = next(iter(B.holdout_ids().job_ids))
    candidates = pd.DataFrame(
        {"id": ["fresh"], "CV": ["x" * 500], "Primary Keyword": ["Python"],
         "Experience Years": [5.0]}
    )
    jobs = pd.DataFrame(
        {"id": [held_out_job], "Long Description": ["y" * 500], "Position": ["Dev"],
         "Primary Keyword": ["Python"], "Exp Years": ["3y"]}
    )
    with pytest.raises(AssertionError, match="leakage"):
        PL.match_pairs(candidates, jobs, PL.PoolConfig(lang="en", n_pairs=1))


# ---- teacher-output filters ------------------------------------------------------------------

def test_reference_filter_keeps_only_usable_generations():
    pairs = _pairs(4)
    raw = [
        _response("hire", "Strong Django and PostgreSQL experience matches the requirements."),
        "unparseable garbage",
        _response("hire", "Good."),  # too short
        _response("reject", "Досвід не відповідає вимогам вакансії взагалі нічим."),  # wrong lang
    ]
    kept, report = F.filter_reference(raw, pairs, "en")
    assert len(kept) == 1
    assert report["n_input"] == 4
    assert set(report["dropped_by_reason"]) == {
        "reference_invalid", "reference_feedback_too_short", "reference_wrong_language"
    }


def _variants() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": ["c0__j0"] * 4,
            "candidate_id": ["c0"] * 4,
            "job_id": ["j0"] * 4,
            "lang": ["en"] * 4,
            "protected_group": ["military_status"] * 4,
            "protected_attr": ["War veteran"] * 4,
            "condition": ["explicit"] * 4,
            "reference_decision": ["hire"] * 4,
            "reference_feedback": ["Strong match on the stated requirements."] * 4,
        }
    )


def test_invariant_filter_enforces_the_properties_training_depends_on():
    raw = [
        _response("hire", "Eight years of Django and PostgreSQL meet every listed requirement."),
        _response("reject", "Eight years of Django meet the requirements."),  # verdict drift
        _response("hire", "As a war veteran, the candidate brings discipline to the team."),  # leak
        # meta-commentary: announces neutrality instead of simply being neutral
        _response("hire", "Regardless of personal characteristics, the experience is strong."),
    ]
    kept, report = F.filter_invariant(raw, _variants(), "en")
    assert len(kept) == 1
    assert report["dropped_by_reason"]["invariant_decision_drift"] == 1
    assert report["dropped_by_reason"]["invariant_attribute_leak"] == 1
    assert report["dropped_by_reason"]["invariant_meta_commentary"] == 1


def test_biased_filter_rejects_negatives_that_are_not_negative():
    raw = [
        _response("reject", "Service background suggests a poor cultural fit here."),  # flipped
        _response("hire", "As a war veteran, discipline is assured for this role."),  # names attr
        _response("hire", "Eight years of Django meet every listed requirement."),  # not biased
        _response("hire", "Solid PostgreSQL and Django experience throughout."),  # not biased
    ]
    kept, report = F.filter_biased(raw, _variants(), "en")
    assert len(kept) == 2
    assert report["dropped_by_reason"]["biased_not_actually_biased"] == 2


def test_decision_balancing_evens_out_the_verdict_skew():
    df = pd.DataFrame({"chosen_decision": ["reject"] * 90 + ["hire"] * 10, "x": range(100)})
    balanced, info = F.balance_decisions(df)
    assert info["balanced"]
    assert balanced["chosen_decision"].value_counts().nunique() == 1


# ---- dataset assembly -------------------------------------------------------------------------

def _invariant_frame() -> pd.DataFrame:
    return _variants().assign(
        cv=["Eight years of Python and Django."] * 4,
        job_description=["Senior Python engineer."] * 4,
        group_label=["military status"] * 4,
        chosen_decision=["hire"] * 4,
        chosen_feedback=["Django and PostgreSQL experience matches the requirements."] * 4,
        protected_attr=["War veteran", "Civilian", "Reservist", "Military retiree"],
    )


def test_sft_rows_use_the_evaluation_time_prompt():
    """Training on a different prompt format than the audit uses would be misread as the
    mitigation failing to work."""
    sft = D.build_sft_dataset(_invariant_frame())
    assert len(sft) == 4
    assert all("smart AI hiring system" in p for p in sft["prompt"])
    assert all("Candidate's military status:" in p for p in sft["prompt"])
    assert json.loads(sft["completion"].iloc[0])["decision"] == "hire"


def test_dpo_pairs_join_on_the_exact_variant():
    invariant = _invariant_frame()
    biased = invariant.head(2).assign(
        rejected_decision=["reject"] * 2,
        rejected_feedback=["Service background is a poor cultural fit."] * 2,
        rejected_flipped_decision=[True] * 2,
        rejected_mentions_attribute=[False] * 2,
    )
    dpo = D.build_dpo_dataset(invariant, biased)
    assert len(dpo) == 2  # only the variants present on both sides
    assert json.loads(dpo["chosen"].iloc[0])["decision"] == "hire"
    assert json.loads(dpo["rejected"].iloc[0])["decision"] == "reject"


def test_split_never_puts_one_candidate_on_both_sides():
    frame = pd.DataFrame(
        {
            "candidate_id": [f"c{i // 4}" for i in range(80)],
            "prompt": ["p"] * 80,
            "completion": ["c"] * 80,
        }
    )
    train, val = D.train_val_split(frame, val_fraction=0.25)
    assert set(train["candidate_id"]).isdisjoint(set(val["candidate_id"]))
    assert len(val) > 0


def test_variant_frame_covers_every_requested_group_and_condition():
    anchored = _pairs(3).assign(
        reference_decision="hire", reference_feedback="Strong match on the requirements."
    )
    variants = SY.build_variant_frame(
        anchored,
        SY.SynthConfig(
            lang="en",
            protected_groups=("military_status", "religion"),
            conditions=("explicit", "implicit"),
            attributes_per_pair=2,
        ),
    )
    assert len(variants) == 3 * 2 * 2 * 2  # pairs x groups x attributes x conditions
    assert set(variants["protected_group"]) == {"military_status", "religion"}
    explicit = variants[variants["condition"] == "explicit"].iloc[0]
    assert explicit["profile"].startswith("Candidate's military status:") or explicit[
        "profile"
    ].startswith("Candidate's religion:")


def test_kto_split_unpairs_the_dpo_triples():
    """KTO and DPO must see identical responses, so a difference is the objective not the data.

    Deriving the KTO rows from the DPO frame is what guarantees that. Generating them
    separately would let the two arms diverge on sampling noise and make the comparison
    meaningless.
    """
    import pandas as pd

    from hiring_bias_mitigation.generation.dataset import build_kto_dataset

    dpo = pd.DataFrame([
        {"prompt": "p1", "chosen": "good1", "rejected": "bad1", "lang": "uk",
         "protected_group": "military_status", "candidate_id": "c1", "job_id": "j1"},
        {"prompt": "p2", "chosen": "good2", "rejected": "bad2", "lang": "en",
         "protected_group": "gender", "candidate_id": "c2", "job_id": "j2"},
    ])
    kto = build_kto_dataset(dpo)

    assert len(kto) == 2 * len(dpo)
    assert set(kto.columns) >= {"prompt", "completion", "label", "lang"}
    # Balanced by construction: one desirable and one undesirable per pair.
    assert kto["label"].sum() == len(dpo)
    assert (~kto["label"]).sum() == len(dpo)

    desirable = set(kto[kto["label"]]["completion"])
    undesirable = set(kto[~kto["label"]]["completion"])
    assert desirable == {"good1", "good2"}
    assert undesirable == {"bad1", "bad2"}
    # Provenance survives, or contamination cannot be asserted on the split.
    assert set(kto["candidate_id"]) == {"c1", "c2"}


def test_kto_split_of_an_empty_frame_is_empty_not_an_error():
    import pandas as pd

    from hiring_bias_mitigation.generation.dataset import build_kto_dataset

    out = build_kto_dataset(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == ["prompt", "completion", "label", "lang"]
