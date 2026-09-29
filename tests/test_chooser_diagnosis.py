"""RR05: the classification chooser, diagnosed from its own retained records.

The finding this answers said the checkpoint used as a chooser "abstained on 28 of 28 corpora and ranked the same
option first on all 28 distinct state digests", and that was read as the model ignoring the state. The retained
records say something narrower: all 28 probability vectors DIFFER, the argmax never changes, the top probability sits
at 0.3730 on average over four options where chance is 0.25, and the cited threshold is 0.80. Those are four separate
facts and the diagnosis keeps them apart, because a diagnosis that overstates a defect is as unusable as one that
hides it. `docs/evidence/RR05_2026_09_26/CHOOSER_DIAGNOSIS.json` is the run of this code over those records.
"""

from m5phet.decide import (ARGMAX_INVARIANT_AT_THRESHOLD, ARGMAX_INVARIANT_BELOW_THRESHOLD, DISCRIMINATING,
                           NOT_DIAGNOSABLE, STATE_INDEPENDENT, diagnose_chooser)


def record(state, probabilities, chosen=None, threshold=0.8):
    return {"schema": "m5phet.decision.v1", "kind": "regime_method", "backend": "laya", "state_sha256": state,
            "chosen": chosen, "probabilities": probabilities, "probability_decimals": 4,
            "abstention": {"refusal": "LOW_CONFIDENCE_ABSTAINED", "top_option": max(probabilities, key=probabilities.get),
                           "top_probability": max(probabilities.values()),
                           "threshold": {"min_confidence": threshold}}}


FOUR = ("agglomerative", "kmeans", "dbscan", "gaussian_mixture")


def spread(n, top=0.38, step=0.002):
    """`n` records whose vectors all differ and whose argmax never changes -- the retained shape."""
    out = []
    for i in range(n):
        first = top + i * step
        rest = (1.0 - first) / 3
        out.append(record(f"{i:064x}", {FOUR[0]: first, FOUR[1]: rest, FOUR[2]: rest, FOUR[3]: rest}))
    return out


def test_the_retained_shape_is_named_argmax_invariant_below_threshold_and_not_state_independent():
    out = diagnose_chooser(spread(28))
    assert out["verdict"] == ARGMAX_INVARIANT_BELOW_THRESHOLD
    assert out["distinct_states"] == 28 and out["distinct_probability_vectors"] == 28
    assert out["argmax_invariant"] is True and out["abstained"] == 28 and out["chose"] == 0
    assert out["reached_threshold"] == 0 and out["thresholds"] == [0.8]
    assert out["chance_level"] == 0.25 and out["options"] == 4
    assert "the state IS read" in out["why"] and "chance level" in out["why"]


def test_identical_vectors_are_the_only_thing_called_state_independent():
    same = [record(f"{i:064x}", {FOUR[0]: 0.4, FOUR[1]: 0.2, FOUR[2]: 0.2, FOUR[3]: 0.2}) for i in range(5)]
    out = diagnose_chooser(same)
    assert out["verdict"] == STATE_INDEPENDENT and out["distinct_probability_vectors"] == 1
    assert out["distinct_states"] == 5


def test_a_chooser_whose_first_option_changes_is_discriminating():
    out = diagnose_chooser([record("a" * 64, {FOUR[0]: 0.4, FOUR[1]: 0.3, FOUR[2]: 0.2, FOUR[3]: 0.1}),
                            record("b" * 64, {FOUR[0]: 0.1, FOUR[1]: 0.5, FOUR[2]: 0.2, FOUR[3]: 0.2})])
    assert out["verdict"] == DISCRIMINATING and out["argmax_invariant"] is False


def test_an_invariant_argmax_that_did_reach_the_threshold_is_a_different_verdict():
    high = [record(f"{i:064x}", {FOUR[0]: 0.9 - i * 0.01, FOUR[1]: 0.04, FOUR[2]: 0.03, FOUR[3]: 0.03},
                   chosen=FOUR[0]) for i in range(4)]
    out = diagnose_chooser(high)
    assert out["verdict"] == ARGMAX_INVARIANT_AT_THRESHOLD and out["reached_threshold"] == 4 and out["chose"] == 4


def test_one_record_diagnoses_nothing_and_says_so():
    out = diagnose_chooser(spread(1))
    assert out["verdict"] == NOT_DIAGNOSABLE and out["distinct_probability_vectors"] is None


def test_the_diagnosis_carries_no_accuracy_because_the_records_carry_no_correct_answer():
    out = diagnose_chooser(spread(28))
    assert not any(key in out for key in ("accuracy", "macro_f1", "skill", "correct"))
