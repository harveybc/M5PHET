"""The model proposes, the contract disposes; the model narrates, the answers check it."""

import json

from m5phet.interpret import Interpreter
from m5phet.orchestrate import (check_proposal, dataset_profile, narrate, narration_is_faithful, render, route)
from m5phet.questions import refusal
from m5phet.runtime import Registry


class Fixed(Interpreter):
    def __init__(self, reply):
        super().__init__(command="fixture", model="fixture-v1", environ={})
        self.reply, self.asked = reply, []

    @property
    def available(self):
        return True

    def _ask(self, text):
        self.asked.append(text)
        return self.reply


class Forecaster:
    name, area = "fake_forecaster", "forecasting"

    def capabilities(self):
        return {"provider": self.name, "operations": ["infer"], "families": ["regression_forecasting"],
                "output_kinds": ["point_forecast"], "uncertainty_methods": ["none"],
                "supported": [{"operation": "infer", "family": "regression_forecasting", "output_kind": "point_forecast"}],
                "known_states": ["s1"]}

    def question_types(self):
        return {"point_forecast": {"required": ["horizon"], "optional": ["target"]}}

    def answer_questions(self, state, questions, data, as_of):
        return {n: {"type": "point_forecast", "values": [0.5412], "unit": "kW"} for n in questions}


ROWS = [{"date": "2026-01-01", "ventas": "120.5", "festivo": "0"}, {"date": "2026-01-02", "ventas": "121.0", "festivo": "1"}]


def registry():
    r = Registry()
    r.register(Forecaster())
    return r


# --- the data profile shows shape, never rows ------------------------------------------------------------------------------

def test_the_profile_carries_columns_types_and_count_and_no_values():
    profile = dataset_profile(ROWS)
    assert profile["columns"] == ["date", "ventas", "festivo"]
    assert profile["types"] == {"date": "text", "ventas": "number", "festivo": "number"}
    assert profile["rows"] == 2
    assert "120.5" not in json.dumps(profile)


def test_csv_text_is_profiled_as_a_table():
    profile = dataset_profile("a,b\n1,2\n3,4\n")
    assert profile["kind"] == "table" and profile["columns"] == ["a", "b"] and profile["rows"] == 2


# --- the model proposes, the contract disposes -----------------------------------------------------------------------------

def test_a_valid_proposal_becomes_a_task_and_the_model_never_saw_the_rows():
    fixed = Fixed(json.dumps({"area": "forecasting", "state": {"target_variable": "ventas"},
                              "questions": {"p": {"type": "point_forecast", "horizon": 3}}}))
    out = route("forecast ventas three days ahead", ROWS, registry(), interpreter=fixed)
    assert out["status"] == "OK" and out["task"]["area"] == "forecasting"
    assert "120.5" not in fixed.asked[0], "the rows are not shipped to the model"
    assert "ventas" in fixed.asked[0], "the column names are"


def test_a_column_the_data_does_not_have_is_refused_by_name():
    fixed = Fixed(json.dumps({"area": "forecasting", "state": {"target_variable": "ingresos"},
                              "questions": {"p": {"type": "point_forecast", "horizon": 3}}}))
    out = route("forecast ingresos", ROWS, registry(), interpreter=fixed)
    assert out["status"] == "INVALID_PROPOSAL" and out["task"] is None
    assert any("ingresos" in p for p in out["problems"])


def test_a_question_type_the_area_does_not_answer_is_refused_by_name():
    fixed = Fixed(json.dumps({"area": "forecasting", "state": {}, "questions": {"i": {"type": "interval", "horizon": 3}}}))
    out = route("give me an interval", ROWS, registry(), interpreter=fixed)
    assert out["status"] == "INVALID_PROPOSAL" and any("interval" in p for p in out["problems"])


def test_an_area_nobody_serves_is_refused():
    fixed = Fixed(json.dumps({"area": "rl", "state": {}, "questions": {"a": {"type": "next_action"}}}))
    out = route("what should I do", ROWS, registry(), interpreter=fixed)
    assert out["status"] == "INVALID_PROPOSAL" and any("rl" in p for p in out["problems"])


def test_the_model_may_decline_and_that_is_reported_not_forced():
    fixed = Fixed(json.dumps({"area": None, "why": "this asks for a recipe"}))
    out = route("give me a recipe for paella", ROWS, registry(), interpreter=fixed)
    assert out["status"] == "REFUSED" and "recipe" in out["why"]


def test_without_an_interpreter_the_person_is_told_to_write_the_envelope():
    out = route("forecast ventas", ROWS, registry(), interpreter=Interpreter(command="", environ={}))
    assert out["status"] == "REFUSED" and "envelope" in out["why"]


def test_malformed_model_output_is_refused_not_guessed():
    out = route("x", ROWS, registry(), interpreter=Fixed("I think forecasting {maybe"))
    assert out["status"] == "REFUSED"


# --- the narration may omit figures; it may not add them --------------------------------------------------------------------

ANSWERS = {"p": {"type": "point_forecast", "status": "OK", "values": [0.5412], "unit": "kW"},
           "i": refusal("NOT_ESTIMABLE", "no predictive distribution", "interval")}
RESPONSE = {"answers": ANSWERS, "answered": 1, "refused": 1}


def test_a_faithful_narration_is_kept():
    out = narrate("q", RESPONSE, interpreter=Fixed("La previsión es 0.5412 kW; el intervalo no se pudo calcular."))
    assert out["source"] == "INTERPRETER" and "0.5412" in out["text"]


def test_a_narration_that_invents_a_number_is_discarded():
    out = narrate("q", RESPONSE, interpreter=Fixed("La previsión es 0.5412 kW con un 95% de confianza y error del 3%."))
    assert out["source"] == "DETERMINISTIC" and "introduced a figure" in out["why"]
    assert "0.5412" in out["text"] and "95" not in out["text"]


def test_percent_renderings_of_a_probability_are_accepted():
    answers = {"c": {"type": "choice", "status": "OK", "label": "euro_area",
                     "uncalibrated_probabilities": {"euro_area": 0.9666, "other": 0.0334}}}
    assert narration_is_faithful("euro_area con 96.7% (0.9666)", answers)
    assert not narration_is_faithful("euro_area con 99% de certeza", answers)


def test_the_deterministic_rendering_names_refusals():
    text = render(RESPONSE)
    assert "not answered" in text and "no predictive distribution" in text and "0.5412" in text
