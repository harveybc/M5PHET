"""WP20: the declared abstention threshold, and the measurement it has to be cited from.

WP09 measured this checkpoint on 450 independently labelled rows and found the relation between its confidence and
how often it is right: below 0.8 the argmax is at chance (0.27-0.36 across four bins), in the 0.8-0.9 bin it is right
17 of 24 times, and in the 0.9-1.0 bin it is right in all 31 rows measured. These tests prove that `m5phet.decide`
turns that measurement into a rule and that the rule cannot be loosened by anybody who does not have a measurement:
a threshold with no citation, or one the cited report never resolved, refuses every question and asks nothing.

They prove nothing about whether abstaining is wise. That is the measurement's claim, not this file's; here the
question is only whether the gate does what the report says and records what it did.
"""

import json
from pathlib import Path

import pytest

from m5phet import decide
from m5phet.interpret import Interpreter
from m5phet.questions import refusal
from m5phet.runtime import Registry

#: the report WP09 wrote. Present on the machine that measured it; the fixture below has the same shape, so the
#: suite is complete without it and the one test that reads it proves the real file still resolves the real number.
WP09_REPORT = Path("~/.local/state/m5phet/wp09-quality-20260925/report_laya_zero_shot.json").expanduser()

PROFILE = {"column": "treatment", "distinct_values": 2, "binary": True, "missing_fraction": 0.0}
OPTIONS = [["treatment", "the intervention"], ["outcome", "what the effect is on"],
           ["confounder", "a common cause"], ["modifier", "the effect varies with it"], ["exclude", "not used"]]
QUESTION = {"treatment": {"options": OPTIONS, "instructions": "Which role does this column play?"}}


def report_with_bins(bins, **overrides):
    """An `m5phet-evaluation-report/1` document carrying exactly these reliability bins and nothing else that matters."""
    return {"version": "m5phet-evaluation-report/1", "stage": "laya_zero_shot", "family": "classification",
            "protocol_digest": "b9aefb3c" + "0" * 56, "corpus_seal": "31257d47" + "0" * 56,
            "metric_sets": [{"name": "classification", "values": {"accuracy": 0.3933}},
                            {"name": "calibration", "values": {"reliability": {"bin_count": len(bins),
                                                                               "rows": sum(b[2] for b in bins),
                                                                               "bins": [{"bin": [low, high],
                                                                                         "count": count,
                                                                                         "accuracy": accuracy}
                                                                                        for low, high, count, accuracy
                                                                                        in bins]}}}],
            **overrides}


#: the real report's own numbers, so the fixture and the machine's file resolve the same threshold
MEASURED_BINS = [(0.0, 0.1, 0, None), (0.1, 0.2, 0, None), (0.2, 0.3, 0, None),
                 (0.3, 0.4, 134, 0.3208955223880597), (0.4, 0.5, 152, 0.35526315789473684),
                 (0.5, 0.6, 58, 0.29310344827586204), (0.6, 0.7, 22, 0.2727272727272727),
                 (0.7, 0.8, 29, 0.3103448275862069), (0.8, 0.9, 24, 0.7083333333333334),
                 (0.9, 1.0, 31, 1.0)]


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "report_laya_zero_shot.json"
    path.write_text(json.dumps(report_with_bins(MEASURED_BINS), indent=2), encoding="utf-8")
    return path


class FakeLaya:
    """Laya's answer shape with the probabilities a test declares. Nothing here is a model; `backend` says `laya`
    only so the record path is exercised -- a real fixture backend is refused two checks earlier."""

    name, area = "laya_news", "classification"

    def __init__(self, probabilities, label):
        self.probabilities, self.label, self.seen = dict(probabilities), label, []

    def capabilities(self):
        return {"provider": self.name, "operations": ["infer"], "families": ["classification"],
                "output_kinds": ["typed_questions"], "uncertainty_methods": ["UNCALIBRATED_CLASS_PROBABILITIES"],
                "supported": [{"operation": "infer", "family": "classification", "output_kind": "typed_questions"}],
                "known_states": ["laya-checkpoint:abc"], "backend": "laya"}

    def question_types(self):
        return {"choice": {"required": ["options"], "optional": ["instructions"]}}

    def answer_questions(self, state, questions, data, as_of):
        self.seen.append({"state": state, "questions": questions})
        answers = {name: {"type": "choice", "status": "OK", "label": self.label, "backend": "laya",
                          "instructions": question.get("instructions"),
                          "options": [list(pair) for pair in question["options"]],
                          "uncalibrated_probabilities": dict(self.probabilities), "probability_decimals": 4,
                          "calibration": "UNCALIBRATED", "execution_authorized": False}
                   for name, question in questions.items()}
        answers["__state_ref__"] = "laya-checkpoint:abc"
        return answers


class Mute(Interpreter):
    def __init__(self):
        super().__init__(command="fixture", model="fixture-v1", environ={})

    @property
    def available(self):
        return False


def registry_with(provider):
    registry = Registry()
    registry.register(provider)
    return registry


def spread(top_key, top_value):
    """Probabilities over the declared roles with `top_key` at `top_value` and the rest sharing what is left."""
    rest = [key for key, _label in OPTIONS if key != top_key]
    share = round((1.0 - top_value) / len(rest), 6)
    return {top_key: top_value, **{key: share for key in rest}}


def ask(provider, source_path, threshold, record_dir=None):
    state = decide.decision_state("causal_study", PROFILE)
    return decide.ask(registry_with(provider), state, QUESTION, kind="causal_study", record_dir=record_dir,
                      min_confidence=threshold, abstention_source=source_path), state


# --- the threshold has to come from a measurement -----------------------------------------------------------------------

def test_a_threshold_with_no_cited_measurement_refuses_every_question_and_asks_nothing():
    provider = FakeLaya(spread("confounder", 0.41), "confounder")
    out, _state = ask(provider, None, 0.8)
    assert out["treatment"]["status"] == "REFUSED"
    assert out["treatment"]["refusal"] == decide.UNCITED_THRESHOLD
    assert "0.8" in out["treatment"]["why"]
    # nothing was asked: a gate nobody measured must not even spend a model call
    assert provider.seen == []


def test_a_threshold_the_cited_report_never_resolved_is_refused_by_name(source):
    provider = FakeLaya(spread("confounder", 0.41), "confounder")
    out, _state = ask(provider, source, 0.85)
    assert out["treatment"]["refusal"] == decide.THRESHOLD_NOT_MEASURED
    assert "0.85" in out["treatment"]["why"]
    assert provider.seen == []


def test_a_threshold_above_every_row_the_report_measured_is_refused(tmp_path):
    path = tmp_path / "thin.json"
    path.write_text(json.dumps(report_with_bins([(0.0, 0.5, 10, 0.3), (0.5, 1.0, 0, None)])), encoding="utf-8")
    provider = FakeLaya(spread("confounder", 0.41), "confounder")
    out, _state = ask(provider, path, 0.5)
    assert out["treatment"]["refusal"] == decide.THRESHOLD_NOT_MEASURED
    assert provider.seen == []


def test_a_source_that_is_not_an_evaluation_report_is_refused_by_name(tmp_path):
    path = tmp_path / "not-a-report.json"
    path.write_text(json.dumps({"version": "something-else/1", "threshold": 0.8}), encoding="utf-8")
    provider = FakeLaya(spread("confounder", 0.41), "confounder")
    out, _state = ask(provider, path, 0.8)
    assert out["treatment"]["refusal"] == decide.ABSTENTION_SOURCE_UNREADABLE
    out, _state = ask(provider, tmp_path / "absent.json", 0.8)
    assert out["treatment"]["refusal"] == decide.ABSTENTION_SOURCE_UNREADABLE
    assert provider.seen == []


def test_a_report_with_no_reliability_bins_measured_no_relation_to_cite(tmp_path):
    path = tmp_path / "no-calibration.json"
    body = report_with_bins(MEASURED_BINS)
    body["metric_sets"] = [body["metric_sets"][0]]
    path.write_text(json.dumps(body), encoding="utf-8")
    out, _state = ask(FakeLaya(spread("confounder", 0.41), "confounder"), path, 0.8)
    assert out["treatment"]["refusal"] == decide.ABSTENTION_SOURCE_UNREADABLE


def test_an_abstention_source_without_a_threshold_gates_nothing_and_is_a_caller_error(source):
    state = decide.decision_state("causal_study", PROFILE)
    with pytest.raises(decide.DecisionError, match="gates nothing"):
        decide.ask(registry_with(FakeLaya(spread("confounder", 0.41), "confounder")), state, QUESTION,
                   kind="causal_study", abstention_source=source)


def test_the_citation_carries_the_reports_digest_and_what_it_measured_above_the_threshold(source):
    citation, unresolved = decide.abstention_threshold(0.8, source)
    assert unresolved is None
    assert citation["min_confidence"] == 0.8
    assert citation["sha256"] == __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    assert citation["stage"] == "laya_zero_shot"
    # 24 rows at 0.8-0.9 with 17 correct, 31 rows at 0.9-1.0 with 31 correct
    assert citation["measured_rows_at_or_above"] == 55
    assert citation["measured_correct_at_or_above"] == 48.0
    assert citation["measured_accuracy_at_or_above"] == round(48 / 55, 6)
    assert [entry["bin"] for entry in citation["bins_at_or_above"]] == [[0.8, 0.9], [0.9, 1.0]]


@pytest.mark.skipif(not WP09_REPORT.exists(), reason="the WP09 report is written by the measurement, not committed")
def test_the_real_wp09_report_resolves_the_threshold_this_framework_uses():
    citation, unresolved = decide.abstention_threshold(0.8, WP09_REPORT)
    assert unresolved is None and citation["stage"] == "laya_zero_shot"
    assert citation["measured_rows_at_or_above"] == 55
    assert citation["measured_accuracy_at_or_above"] == round(48 / 55, 6)
    # and the number nobody measured is still refused against the real file
    assert decide.abstention_threshold(0.75, WP09_REPORT)[1][0] == decide.THRESHOLD_NOT_MEASURED


# --- what the gate does ---------------------------------------------------------------------------------------------

def test_an_answer_below_the_threshold_abstains_and_records_the_abstention_with_no_choice(source, tmp_path):
    provider = FakeLaya(spread("confounder", 0.41), "confounder")
    out, state = ask(provider, source, 0.8, record_dir=tmp_path / "decisions")
    entry = out["treatment"]
    assert entry["status"] == "REFUSED" and entry["refusal"] == decide.LOW_CONFIDENCE_ABSTAINED
    assert "0.41" in entry["why"] and "0.8" in entry["why"] and str(source) in entry["why"]
    decision = entry["decision"]
    assert decision["chosen"] is None
    assert decision["abstention"]["top_option"] == "confounder"
    assert decision["abstention"]["top_probability"] == 0.41
    assert decision["abstention"]["answer_label"] == "confounder"
    assert decision["abstention"]["threshold"]["path"] == str(source)
    assert decision["abstention"]["threshold"]["min_confidence"] == 0.8
    # the answer is kept whole: the probabilities are the head's own, unrounded and unrenormalised
    assert decision["probabilities"] == spread("confounder", 0.41)
    assert decision["state_sha256"] == decide.state_sha256(state)


def test_an_answer_at_or_above_the_threshold_is_a_choice_like_any_other(source, tmp_path):
    provider = FakeLaya(spread("treatment", 0.93), "treatment")
    out, _state = ask(provider, source, 0.8, record_dir=tmp_path / "decisions")
    entry = out["treatment"]
    assert entry["status"] == "OK"
    assert entry["decision"]["chosen"] == "treatment"
    assert "abstention" not in entry["decision"]
    # exactly at the threshold is above it: the report's bin is closed on the left
    at_edge = FakeLaya(spread("treatment", 0.8), "treatment")
    out, _state = ask(at_edge, source, 0.8)
    assert out["treatment"]["status"] == "OK"


def test_a_tie_at_the_top_abstains_because_the_head_separated_nothing(source):
    provider = FakeLaya({"treatment": 0.45, "outcome": 0.45, "confounder": 0.04, "modifier": 0.03, "exclude": 0.03},
                        "treatment")
    out, _state = ask(provider, source, 0.8)
    assert out["treatment"]["refusal"] == decide.LOW_CONFIDENCE_ABSTAINED
    assert out["treatment"]["decision"]["abstention"]["top_option"] == "outcome"     # first in key order, no choice


def test_with_no_threshold_declared_nothing_abstains_and_the_old_behaviour_stands():
    provider = FakeLaya(spread("confounder", 0.41), "confounder")
    state = decide.decision_state("causal_study", PROFILE)
    out = decide.ask(registry_with(provider), state, QUESTION, kind="causal_study")
    assert out["treatment"]["status"] == "OK" and out["treatment"]["decision"]["chosen"] == "confounder"


# --- the record -------------------------------------------------------------------------------------------------------

def test_an_abstention_record_round_trips_and_its_digest_names_the_file(source, tmp_path):
    folder = tmp_path / "decisions"
    out, _state = ask(FakeLaya(spread("confounder", 0.41), "confounder"), source, 0.8, record_dir=folder)
    path = Path(out["treatment"]["record_path"])
    assert path.parent == folder and path.stem == decide.decision_sha256(out["treatment"]["decision"])
    assert decide.load(path) == out["treatment"]["decision"]


def test_an_abstention_that_also_claims_a_choice_is_never_written(source, tmp_path):
    out, _state = ask(FakeLaya(spread("confounder", 0.41), "confounder"), source, 0.8)
    forged = dict(out["treatment"]["decision"], chosen="treatment")
    folder = tmp_path / "decisions"
    with pytest.raises(decide.DecisionError, match=decide.LOW_CONFIDENCE_ABSTAINED):
        decide.record(forged, folder)
    assert not folder.exists() or list(folder.iterdir()) == []


def test_an_abstention_whose_threshold_cites_nothing_is_never_written(source, tmp_path):
    out, _state = ask(FakeLaya(spread("confounder", 0.41), "confounder"), source, 0.8)
    decision = out["treatment"]["decision"]
    forged = dict(decision, abstention=dict(decision["abstention"], threshold={"min_confidence": 0.8}))
    with pytest.raises(decide.DecisionError, match=decide.UNCITED_THRESHOLD):
        decide.record(forged, tmp_path)


def test_an_abstention_is_never_given_an_outcome(source, tmp_path):
    out, _state = ask(FakeLaya(spread("confounder", 0.41), "confounder"), source, 0.8,
                      record_dir=tmp_path / "decisions")
    row = {"stage": "laya_chosen", "status": "OK", "comparability": decide.COMPARABLE, "rank": 1}
    linked = decide.outcome(out["treatment"]["record_path"], row, out_dir=tmp_path / "outcomes")
    assert linked == refusal(decide.ABSTENTION_HAS_NO_OUTCOME, linked["why"])
    assert not (tmp_path / "outcomes").exists()


# --- WP23: a person's choice is recorded too, and says so -----------------------------------------------------------------

def test_a_persons_choice_is_recorded_under_chosen_by_human_with_no_probabilities(tmp_path):
    state = decide.decision_state("causal_study", PROFILE)
    entry = decide.human_choice(kind="causal_study", state_text=state, question="treatment", options=OPTIONS,
                                chosen="treatment",
                                why="the column named treatment is the intervention this synthetic generator applies",
                                as_of="2026-09-25T00:00:00+00:00", record_dir=tmp_path)
    assert entry["status"] == "OK"
    decision = entry["decision"]
    # the merged API records a person under `chosen_by: HUMAN` with NO backend at all: a person is not one, and a
    # record that named one could be read as a model's later
    assert decision["chosen_by"] == decide.CHOSEN_BY_HUMAN and decision["backend"] is None
    assert decision["probabilities"] == {} and decision["probability_decimals"] is None
    assert decision["checkpoint"] is None and decision["execution_authorized"] is False
    assert decide.load(Path(entry["record_path"])) == decision


def test_a_persons_choice_carrying_probabilities_or_no_reason_is_never_written(tmp_path):
    state = decide.decision_state("causal_study", PROFILE)
    entry = decide.human_choice(kind="causal_study", state_text=state, question="treatment", options=OPTIONS,
                                chosen="treatment", why="because it is one")
    with pytest.raises(decide.DecisionError, match="no probabilities"):
        decide.record(dict(entry["decision"], probabilities={"treatment": 1.0}), tmp_path)
    with pytest.raises(decide.DecisionError, match="carries exactly"):
        decide.record({key: value for key, value in entry["decision"].items() if key != "why"}, tmp_path)
    with pytest.raises(decide.DecisionError, match=decide.CHOOSER_FIELDS):
        decide.record(dict(entry["decision"], backend="laya"), tmp_path)


def test_a_models_decision_never_carries_a_reason_a_person_wrote(source, tmp_path):
    out, _state = ask(FakeLaya(spread("treatment", 0.93), "treatment"), source, 0.8)
    forged = dict(out["treatment"]["decision"], why="it looked right to me")
    with pytest.raises(decide.DecisionError, match="carries exactly"):
        decide.record(forged, tmp_path)


def test_a_persons_choice_can_be_linked_to_a_row_and_the_outcome_says_who_chose(tmp_path):
    state = decide.decision_state("causal_study", PROFILE)
    entry = decide.human_choice(kind="causal_study", state_text=state, question="treatment", options=OPTIONS,
                                chosen="treatment", why="it is the intervention",
                                record_dir=tmp_path / "decisions")
    row = {"stage": "hand_spec", "status": "OK", "comparability": decide.COMPARABLE, "rank": 1}
    linked = decide.outcome(entry["record_path"], row, out_dir=tmp_path / "outcomes")
    assert linked["status"] == "OK"
    assert linked["outcome"]["chosen_by"] == decide.CHOSEN_BY_HUMAN
    assert linked["outcome"]["probabilities"] == {}
    assert decide.load_outcome(Path(linked["record_path"])) == linked["outcome"]
