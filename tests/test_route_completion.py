"""RR05: the router's twelve measured WRONG_TYPE runs were under-answers, and what is done about them.

`docs/evidence/ROUTE_RELIABILITY_2026_09_25/route_reliability_n5.json` is the measurement these tests are written
against: 95 routings, `WRONG_AREA` 0, `WRONG_VALUE` 0, `WRONG_TYPE` 12 -- and all twelve on four sentences that ask
two things, where the envelope asked a strict SUBSET of the types asked for. Every test below is deterministic: the
interpreter is a fixture, so what is checked is the CONTRACT of the completion pass, never a model's mood.
"""

import json

import pytest

from m5phet.interpret import Interpreter
from m5phet.orchestrate import (COMPLETION_ADDED, COMPLETION_NONE, COMPLETION_NOTHING_FURTHER,
                                COMPLETION_NOTHING_TO_OFFER, COMPLETION_NOT_CONSULTED, COMPLETION_NOT_OFFERED,
                                COMPLETION_REJECTED, complete_under_answer, completable_types, route)
from m5phet.runtime import Registry


class Scripted(Interpreter):
    """Replies in order: the first for the envelope, the second for the narrow completion question."""

    def __init__(self, *replies):
        super().__init__(command="fixture", model="fixture-v1", environ={})
        self.replies, self.asked = list(replies), []

    @property
    def available(self):
        return True

    def _ask(self, text):
        self.asked.append(text)
        return self.replies[min(len(self.asked) - 1, len(self.replies) - 1)]


class Exploding(Scripted):
    """Unreachable from the call this test makes onward; `after` says how many replies come first."""

    def __init__(self, *replies, after=0):
        super().__init__(*replies)
        self.after = after

    def _ask(self, text):
        self.asked.append(text)
        if len(self.asked) <= self.after:
            return self.replies[len(self.asked) - 1]
        raise OSError("the interpreter is not reachable")


class Regimes:
    """The unsupervised provider's real shape: `clustering` needs nothing, `cluster_description` needs a metric."""

    name, area = "fake_regimes", "unsupervised"

    def capabilities(self):
        return {"provider": self.name, "operations": ["infer"], "families": ["clustering"],
                "output_kinds": ["clustering"], "uncertainty_methods": ["none"],
                "supported": [{"operation": "infer", "family": "clustering", "output_kind": "clustering"}],
                "known_states": ["r1"]}

    def question_types(self):
        return {"clustering": {"required": [], "optional": ["level"]},
                "cluster_description": {"required": ["target_metric"], "optional": ["level"]}}

    def data_requirement(self):
        return {"required": True, "why": "the reference assigns the caller's rows"}

    def answer_questions(self, state, questions, data, as_of):
        return {n: {"type": q["type"]} for n, q in questions.items()}


class Forecaster:
    """`interval` requires a `confidence_level` nobody supplied: rule 2 must refuse to offer it."""

    name, area = "fake_forecaster", "forecasting"

    def capabilities(self):
        return {"provider": self.name, "operations": ["infer"], "families": ["regression_forecasting"],
                "output_kinds": ["point_forecast"], "uncertainty_methods": ["none"],
                "supported": [{"operation": "infer", "family": "regression_forecasting",
                               "output_kind": "point_forecast"}],
                "known_states": ["s1"]}

    def question_types(self):
        return {"point_forecast": {"required": ["horizon"], "optional": ["target"]},
                "interval": {"required": ["horizon", "confidence_level"], "optional": ["target"]}}

    def answer_questions(self, state, questions, data, as_of):
        return {n: {"type": q["type"]} for n, q in questions.items()}


ROWS = [{"date": "2026-01-01", "open": "1.1", "close": "1.2"}, {"date": "2026-01-02", "open": "1.2", "close": "1.3"}]


def registry(provider):
    r = Registry()
    r.register(provider)
    return r


def catalog_of(provider):
    from m5phet.questions import catalog
    return catalog(registry(provider))


def profile_of(rows):
    from m5phet.orchestrate import dataset_profile
    return dataset_profile(rows)


# --- rule 2, with no model at all: what may even be offered -----------------------------------------------------------

def test_a_type_whose_required_field_is_present_is_offerable():
    task = {"area": "unsupervised", "state": {}, "questions": {"d": {"type": "cluster_description",
                                                                     "target_metric": "highest body"}}}
    declared = Regimes().question_types()
    assert completable_types(task, declared) == ["clustering"]


def test_a_type_whose_required_field_is_missing_is_never_offered():
    task = {"area": "forecasting", "state": {"target_variable": "close"},
            "questions": {"p": {"type": "point_forecast", "horizon": 1}}}
    # `interval` needs a confidence_level nobody named. Offering it would mean this layer choosing 0.95.
    assert completable_types(task, Forecaster().question_types()) == []


def test_a_required_field_may_come_from_the_state():
    task = {"area": "forecasting", "state": {"horizon": 1, "confidence_level": 0.9},
            "questions": {"p": {"type": "point_forecast", "horizon": 1}}}
    assert completable_types(task, Forecaster().question_types()) == ["interval"]


def test_a_type_already_asked_is_not_offered_again():
    task = {"area": "unsupervised", "state": {},
            "questions": {"c": {"type": "clustering"},
                          "d": {"type": "cluster_description", "target_metric": "highest body"}}}
    assert completable_types(task, Regimes().question_types()) == []


# --- the pass itself --------------------------------------------------------------------------------------------------

def test_the_under_answered_second_ask_is_recovered_and_validates():
    """The measured case: the model wrote cluster_description alone for a sentence that asks for the assignment too."""
    task = {"area": "unsupervised", "state": {},
            "questions": {"d": {"type": "cluster_description", "target_metric": "highest body"}}}
    scripted = Scripted(json.dumps({"also_asked_question_type": "clustering"}))
    out, record = complete_under_answer("describe el grupo y asigna los regimenes", task,
                                        catalog_of(Regimes()), profile_of(ROWS), scripted)
    assert record["offered"] == ["clustering"]
    assert record["outcome"] == COMPLETION_ADDED and record["added"] == "clustering"
    assert {q["type"] for q in out["questions"].values()} == {"cluster_description", "clustering"}
    # the second question carries ONLY fields its own type declares, and no value this layer chose
    added = next(q for q in out["questions"].values() if q["type"] == "clustering")
    assert set(added) == {"type"}


def test_the_model_is_shown_the_sentence_and_the_declared_types_and_never_a_row():
    task = {"area": "unsupervised", "state": {},
            "questions": {"d": {"type": "cluster_description", "target_metric": "highest body"}}}
    scripted = Scripted(json.dumps({"also_asked_question_type": COMPLETION_NONE}))
    complete_under_answer("describe el grupo", task, catalog_of(Regimes()), profile_of(ROWS), scripted)
    asked = scripted.asked[0]
    assert "clustering" in asked and COMPLETION_NONE in asked
    assert "1.1" not in asked and "1.2" not in asked and "close" not in asked


def test_nothing_further_asked_leaves_the_envelope_exactly_as_the_model_wrote_it():
    task = {"area": "unsupervised", "state": {},
            "questions": {"d": {"type": "cluster_description", "target_metric": "highest body"}}}
    before = json.dumps(task, sort_keys=True)
    out, record = complete_under_answer("describe el grupo con cuerpo alto", task, catalog_of(Regimes()),
                                        profile_of(ROWS), Scripted(json.dumps({"also_asked_question_type": "none"})))
    assert record["outcome"] == COMPLETION_NOTHING_FURTHER
    assert out is task and json.dumps(out, sort_keys=True) == before


def test_a_null_choice_is_read_as_nothing_further_and_not_as_a_type():
    task = {"area": "unsupervised", "state": {},
            "questions": {"d": {"type": "cluster_description", "target_metric": "highest body"}}}
    out, record = complete_under_answer("describe el grupo", task, catalog_of(Regimes()), profile_of(ROWS),
                                        Scripted(json.dumps({"also_asked_question_type": None})))
    assert record["outcome"] == COMPLETION_NOTHING_FURTHER and out is task


def test_a_type_the_pass_did_not_offer_is_discarded_by_name():
    task = {"area": "unsupervised", "state": {},
            "questions": {"d": {"type": "cluster_description", "target_metric": "highest body"}}}
    out, record = complete_under_answer("describe el grupo", task, catalog_of(Regimes()), profile_of(ROWS),
                                        Scripted(json.dumps({"also_asked_question_type": "anomaly_risk"})))
    assert record["outcome"] == COMPLETION_NOT_OFFERED and record["choice"] == "anomaly_risk"
    assert out is task and {q["type"] for q in out["questions"].values()} == {"cluster_description"}


def test_nothing_to_offer_costs_no_model_call_at_all():
    task = {"area": "unsupervised", "state": {},
            "questions": {"c": {"type": "clustering"},
                          "d": {"type": "cluster_description", "target_metric": "highest body"}}}
    scripted = Scripted(json.dumps({"also_asked_question_type": "clustering"}))
    out, record = complete_under_answer("q", task, catalog_of(Regimes()), profile_of(ROWS), scripted)
    assert record["outcome"] == COMPLETION_NOTHING_TO_OFFER and scripted.asked == [] and out is task


def test_an_unreachable_interpreter_leaves_the_first_envelope_standing():
    task = {"area": "unsupervised", "state": {},
            "questions": {"d": {"type": "cluster_description", "target_metric": "highest body"}}}
    out, record = complete_under_answer("describe el grupo", task, catalog_of(Regimes()), profile_of(ROWS),
                                        Exploding("{}"))
    assert record["outcome"] == COMPLETION_NOT_CONSULTED and out is task


def test_an_addition_that_does_not_validate_is_dropped_and_recorded():
    """The safety net: a declared type whose fields pass rule 2 and that `check_proposal` still refuses."""
    task = {"area": "unsupervised", "state": {},
            "questions": {"d": {"type": "cluster_description", "target_metric": "highest body"}}}
    # An empty profile makes the area's own data requirement bite, so the completed envelope cannot validate.
    out, record = complete_under_answer("describe y asigna", task, catalog_of(Regimes()), profile_of([]),
                                        Scripted(json.dumps({"also_asked_question_type": "clustering"})))
    assert record["outcome"] == COMPLETION_REJECTED and record["problems"]
    assert out is task and {q["type"] for q in out["questions"].values()} == {"cluster_description"}


# --- through `route`, which is what the product calls -----------------------------------------------------------------

def test_route_reports_the_completion_and_recovers_the_second_ask():
    envelope = json.dumps({"area": "unsupervised", "state": {},
                           "questions": {"d": {"type": "cluster_description", "target_metric": "highest body"}}})
    scripted = Scripted(envelope, json.dumps({"also_asked_question_type": "clustering"}))
    out = route("describe el grupo con cuerpo alto y asigna los regimenes", ROWS, registry(Regimes()),
                interpreter=scripted)
    assert out["status"] == "OK"
    assert out["completion"]["outcome"] == COMPLETION_ADDED
    assert {q["type"] for q in out["task"]["questions"].values()} == {"cluster_description", "clustering"}


def test_route_with_completion_off_is_the_old_behaviour_and_asks_the_model_once():
    envelope = json.dumps({"area": "unsupervised", "state": {},
                           "questions": {"d": {"type": "cluster_description", "target_metric": "highest body"}}})
    scripted = Scripted(envelope, json.dumps({"also_asked_question_type": "clustering"}))
    out = route("describe el grupo y asigna", ROWS, registry(Regimes()), interpreter=scripted, complete=False)
    assert out["status"] == "OK" and out["completion"] is None and len(scripted.asked) == 1
    assert {q["type"] for q in out["task"]["questions"].values()} == {"cluster_description"}


def test_a_refused_route_never_reaches_the_completion_pass():
    scripted = Scripted(json.dumps({"area": None, "why": "nothing here serves recipes"}))
    out = route("give me a recipe for paella", ROWS, registry(Regimes()), interpreter=scripted)
    assert out["status"] == "REFUSED" and out.get("completion") is None and len(scripted.asked) == 1
