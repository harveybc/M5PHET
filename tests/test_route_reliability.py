"""WP30: the router measured, and the gate stated no wider than it is.

Three things are pinned here. The scorer's verdicts on fixture proposals -- six outcomes, kept apart, each one a
different fact about the model, so that a wrong area is never counted as a wrong value and a refusal to propose is
never counted as a misreading. The catalog's shape for a measured router and for an unmeasured one, including the
refusal of a report measured on a different model. And the sentence the catalog owes a reader who sees an abstention
threshold and assumes it is applied wherever a language model decides anything: `route` reports no confidence with
any shipped plugin, and the catalog says so for that path by name.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from m5phet import orchestrate                                                          # noqa: E402
from m5phet.interpret import Interpreter                                                # noqa: E402
from m5phet.orchestrate import ROUTE_RELIABILITY_SCHEMA, route                          # noqa: E402
from m5phet.runtime import Registry                                                     # noqa: E402

import measure_route                                                                    # noqa: E402
from measure_route import (CORRECT, INVALID_PROPOSAL, REFUSED, WRONG_AREA, WRONG_TYPE, WRONG_VALUE,  # noqa: E402
                           verdict_of)


CASE = {"provider": "predictor_forecast", "fragment": "household-power", "area": "forecasting",
        "prompt": "predict household power one hour ahead", "types": ["point_forecast"],
        "values": {"target": "Global_active_power", "horizon": 60}}


def outcome(area="forecasting", questions=None, state=None, status="OK", problems=(), why=None, validated=True):
    """One `orchestrate.route` report, as the proposal endpoint returns it."""
    envelope = None
    if area is not None:
        envelope = {"area": area, "state": dict(state or {}),
                    "questions": questions if questions is not None else
                    {"p": {"type": "point_forecast", "horizon": 60, "target": "Global_active_power"}}}
    return {"status": status, "proposal": envelope, "task": envelope if validated else None,
            "problems": list(problems), "why": why}


# --- the six verdicts ------------------------------------------------------------------------------------------------

def test_the_expected_envelope_is_correct():
    verdict, detail = verdict_of(CASE, outcome())
    assert verdict == CORRECT
    assert detail["area"] == "forecasting" and detail["types"] == ["point_forecast"]
    assert detail["values"] == {"target": "Global_active_power", "horizon": 60}


def test_a_governed_value_in_the_state_counts_as_named_under_its_owner_s_spelling():
    verdict, _ = verdict_of(CASE, outcome(state={"target_variable": "Global_active_power"},
                                          questions={"p": {"type": "point_forecast", "horizon": 60}}))
    assert verdict == CORRECT, "state.target_variable is the owner's spelling of the forecaster's target"


def test_another_engine_is_a_wrong_area_even_when_the_envelope_is_valid():
    verdict, detail = verdict_of(CASE, outcome(area="causal", questions={"e": {"type": "ate"}}))
    assert verdict == WRONG_AREA and detail["wanted_area"] == "forecasting"


def test_a_proposal_the_checker_refused_is_counted_by_the_kind_of_problem():
    verdict, detail = verdict_of(CASE, outcome(
        status="INVALID_PROPOSAL", validated=False,
        problems=["question 'p': type 'nowcast' is not one this area answers ['point_forecast']",
                  "column 'potencia' is not in the attached data ['Global_active_power']"]))
    assert verdict == INVALID_PROPOSAL
    assert detail["problem_kinds"] == ["COLUMN_NOT_IN_DATA", "UNDECLARED_QUESTION_TYPE"]


def test_a_valid_envelope_asking_other_question_types_is_a_wrong_type():
    verdict, detail = verdict_of(CASE, outcome(questions={"r": {"type": "anomaly_risk", "threshold": "< 1"}}))
    assert verdict == WRONG_TYPE and detail["wanted_types"] == ["point_forecast"]


def test_a_declared_value_that_is_the_wrong_one_is_a_wrong_value():
    verdict, detail = verdict_of(CASE, outcome(
        questions={"p": {"type": "point_forecast", "horizon": 1, "target": "Global_active_power"}}))
    assert verdict == WRONG_VALUE and detail["value_problems"]["horizon"] == {"got": 1, "wanted": 60}


def test_a_governed_value_the_sentence_names_and_the_envelope_omits_is_absent_not_correct():
    verdict, detail = verdict_of(CASE, outcome(questions={"p": {"type": "point_forecast", "horizon": 60}}))
    assert verdict == WRONG_VALUE and detail["value_problems"] == {"target": "ABSENT"}


def test_a_horizon_written_as_text_still_matches_the_integer_the_engine_has():
    verdict, _ = verdict_of(CASE, outcome(
        questions={"p": {"type": "point_forecast", "horizon": "60", "target": "Global_active_power"}}))
    assert verdict == CORRECT


def test_nothing_proposed_at_all_is_a_refusal_and_not_a_misreading():
    verdict, detail = verdict_of(CASE, outcome(area=None, status="REFUSED",
                                               why="the interpreter returned no JSON object"))
    assert verdict == REFUSED and detail["why"] == "the interpreter returned no JSON object"


def test_a_field_the_sentence_does_not_name_is_not_scored():
    case = dict(CASE, values={"target": "Global_active_power"})
    verdict, _ = verdict_of(case, outcome(questions={"p": {"type": "point_forecast", "horizon": 1,
                                                           "target": "Global_active_power"}}))
    assert verdict == CORRECT, "an unnamed horizon is the engine's to settle, not the model's to get wrong"


def test_every_declared_case_names_an_area_types_and_a_reason_for_what_it_scores():
    for case in measure_route.CASES:
        assert case["area"] in ("classification", "forecasting", "unsupervised", "rl", "causal")
        assert case["types"], "a case with no expected question types scores nothing"
        for field in case["values"]:
            assert field in measure_route.FIELD_SPELLINGS, f"{field} has no declared spelling in an envelope"


# --- the measurement survives being killed ---------------------------------------------------------------------------

def test_a_sentence_is_resumed_only_under_the_same_case_instance_and_n(tmp_path):
    entry = {"provider": "predictor_forecast", "prompt": CASE["prompt"], "kind": "ROUTED", "runs": 5, "correct": 4,
             "verdicts": {CORRECT: 4, WRONG_VALUE: 1}, "results": [], "stable": False}
    measure_route.save_checkpoint(tmp_path, CASE, "http://127.0.0.1:8784", 5, entry)
    assert measure_route.load_checkpoint(tmp_path, CASE, "http://127.0.0.1:8784", 5)["correct"] == 4
    assert measure_route.load_checkpoint(tmp_path, CASE, "http://127.0.0.1:8784", 3) is None, \
        "N is part of the protocol: five routings and three routings are two measurements"
    assert measure_route.load_checkpoint(tmp_path, CASE, "http://127.0.0.1:8799", 5) is None, \
        "a result measured against another instance is not a partial result of this one"
    other = dict(CASE, values={"target": "direction_long"})
    assert measure_route.load_checkpoint(tmp_path, other, "http://127.0.0.1:8784", 5) is None, \
        "a result scored against another expectation is not a partial result of this one"


def test_a_checkpoint_directory_with_nothing_in_it_resumes_nothing(tmp_path):
    assert measure_route.load_checkpoint(tmp_path, CASE, "http://127.0.0.1:8784", 5) is None
    assert measure_route.load_checkpoint(None, CASE, "http://127.0.0.1:8784", 5) is None


# --- what the catalog publishes ----------------------------------------------------------------------------------------

class Silent(Interpreter):
    def __init__(self, model="fixture-v1"):
        super().__init__(command="fixture", model=model, environ={})

    @property
    def available(self):
        return True


def report(tmp_path, *, plugin="command", model="fixture-v1", reliability=0.75, name="route.json"):
    path = tmp_path / name
    path.write_text(json.dumps({
        "schema": ROUTE_RELIABILITY_SCHEMA, "runs_per_sentence": 5, "measured_at": "2026-09-25T00:00:00+00:00",
        "interpreter": {"plugin": plugin, "model": model}, "protocol": "the declared protocol",
        "corpus": "the declared corpus",
        "summary": {"sentences": 4, "runs": 20, "correct": 15, "reliability": reliability,
                    "reliability_when_it_proposed": 0.8,
                    "verdicts": {"CORRECT": 15, "WRONG_VALUE": 5},
                    "invalid_proposal_problems": {}}}), encoding="utf-8")
    return path


def test_nothing_declared_is_not_measured():
    assert orchestrate.declared_route_reliability(None, {}) == orchestrate.NOT_MEASURED


def test_a_declared_report_publishes_the_rate_its_counts_and_its_digest(tmp_path):
    path = report(tmp_path)
    published = orchestrate.declared_route_reliability(
        None, {orchestrate.ROUTE_RELIABILITY_VARIABLE: str(path)}, interpreter=Silent())
    assert published["reliability"] == 0.75
    assert published["n"] == {"sentences": 4, "runs": 20, "runs_per_sentence": 5}
    assert published["confidence"] == orchestrate.ROUTE_CONFIDENCE_NOT_REPORTED
    assert len(published["report_sha256"]) == 64
    assert published["protocol"] and published["corpus"]


def test_a_report_measured_on_another_model_is_refused_not_published_beside_this_one(tmp_path):
    path = report(tmp_path, model="another-model")
    published = orchestrate.declared_route_reliability(
        None, {orchestrate.ROUTE_RELIABILITY_VARIABLE: str(path)}, interpreter=Silent())
    assert published["refusal"] == orchestrate.ROUTE_MEASURED_ON_ANOTHER_INTERPRETER
    assert "another-model" in published["why"]


def test_a_file_that_is_not_one_of_our_reports_is_refused_by_name(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"schema": "something_else.v1", "summary": {"reliability": 1.0}}), encoding="utf-8")
    published = orchestrate.declared_route_reliability(None, {orchestrate.ROUTE_RELIABILITY_VARIABLE: str(path)})
    assert published["refusal"] == orchestrate.ROUTE_REPORT_UNREADABLE


def test_a_report_with_no_measured_rate_is_refused(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"schema": ROUTE_RELIABILITY_SCHEMA, "summary": {}}), encoding="utf-8")
    published = orchestrate.declared_route_reliability(None, {orchestrate.ROUTE_RELIABILITY_VARIABLE: str(path)})
    assert published["refusal"] == orchestrate.ROUTE_REPORT_UNREADABLE


# --- the gate, stated no wider than it is --------------------------------------------------------------------------------

def test_the_route_report_says_that_no_confidence_is_reported_for_this_path():
    out = route("forecast something", None, Registry(), interpreter=Interpreter(command="", environ={}))
    assert out["confidence"] == orchestrate.ROUTE_CONFIDENCE_NOT_REPORTED
    assert "check_proposal" in out["gate"] or "validated against the catalog" in out["gate"]


def test_the_catalog_says_which_paths_the_abstention_rule_covers_and_which_it_does_not():
    engine = pytest.importorskip("m5phet.web.engine")
    instance = engine.Engine(registry=Registry(), configuration=_empty_configuration(), environ={})
    paths = instance.abstention()["paths"]
    assert set(paths) == {"decide", "interpret", "route"}
    assert paths["decide"]["covered"] is True
    assert paths["route"]["covered"] is False
    assert paths["route"]["confidence"] == orchestrate.ROUTE_CONFIDENCE_NOT_REPORTED
    assert "_ask" in paths["route"]["why"], "the reason is structural and is stated, not asserted"
    assert instance.route_view()["reliability"] == orchestrate.NOT_MEASURED


def _empty_configuration():
    from m5phet.config import Configuration
    return Configuration(data=None, path=None, environ={})
