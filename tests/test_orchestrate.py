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


# --- a data column is not a fitted target ---------------------------------------------------------------------------------

class Bundled(Forecaster):
    """A provider that declares its fitted vocabulary, as the real forecaster does through chat_slots()."""

    def chat_slots(self):
        return [{"name": "target", "allowed": ["Global_active_power"]}, {"name": "horizon", "allowed": [60], "type": "integer"}]


def bundled_registry():
    r = Registry()
    r.register(Bundled())
    return r


HISTORY = {"values": [1.0, 2.0], "scale": "original", "scaler_digest": "abc"}


def test_the_router_is_shown_the_fitted_vocabulary_not_only_the_datas_keys():
    fixed = Fixed(json.dumps({"area": "forecasting", "state": {},
                              "questions": {"p": {"type": "point_forecast", "horizon": 60, "target": "Global_active_power"}}}))
    out = route("pronostica la potencia a una hora", HISTORY, bundled_registry(), interpreter=fixed)
    assert out["status"] == "OK"
    assert "Global_active_power" in fixed.asked[0] and "allowed_values" in fixed.asked[0]


def test_a_data_key_proposed_as_the_target_is_refused_naming_the_fitted_targets():
    """Seen in the browser: the model picked `values` -- a key of the attached JSON -- as the target, the proposal passed
    the column check because `values` IS a key, and the engine refused. The refusal was honest; the experience was broken."""
    fixed = Fixed(json.dumps({"area": "forecasting", "state": {},
                              "questions": {"p": {"type": "point_forecast", "horizon": 60, "target": "values"}}}))
    out = route("pronostica la potencia", HISTORY, bundled_registry(), interpreter=fixed)
    assert out["status"] == "INVALID_PROPOSAL"
    assert any("'values'" in p and "Global_active_power" in p for p in out["problems"])


def test_an_unfitted_horizon_in_a_proposal_is_refused_before_running():
    fixed = Fixed(json.dumps({"area": "forecasting", "state": {},
                              "questions": {"p": {"type": "point_forecast", "horizon": 90, "target": "Global_active_power"}}}))
    out = route("pronostica 90 pasos", HISTORY, bundled_registry(), interpreter=fixed)
    assert out["status"] == "INVALID_PROPOSAL" and any("horizon 90" in p for p in out["problems"])


def test_a_governed_value_is_not_mistaken_for_a_missing_data_column():
    """`Global_active_power` is not a column of the attached history object; it is the fitted target, and must pass."""
    fixed = Fixed(json.dumps({"area": "forecasting", "state": {"target_variable": "Global_active_power"},
                              "questions": {"p": {"type": "point_forecast", "horizon": 60, "target": "Global_active_power"}}}))
    out = route("pronostica", HISTORY, bundled_registry(), interpreter=fixed)
    assert out["status"] == "OK", out["problems"]


# --- two bundles: the pair that no bundle has is refused before the person presses run ------------------------------------

class TwoBundles(Forecaster):
    def chat_slots(self):
        return [{"name": "target", "allowed": ["Global_active_power", "direction_long"]},
                {"name": "horizon", "allowed": [60, 1], "type": "integer",
                 "aliases": {"60": ["one hour", "una hora", "60 minutes"], "1": ["next bar", "one step"]}}]

    def chat_combinations(self):
        return [{"target": "Global_active_power", "horizon": 60}, {"target": "direction_long", "horizon": 1}]


def two_bundle_registry():
    r = Registry()
    r.register(TwoBundles())
    return r


def test_a_target_paired_with_the_other_bundles_horizon_is_refused_as_not_fitted():
    """Seen in the browser: `Global_active_power` at horizon 1. Each value is admissible; the pair is nobody's."""
    fixed = Fixed(json.dumps({"area": "forecasting", "state": {},
                              "questions": {"p": {"type": "point_forecast", "horizon": 1, "target": "Global_active_power"}}}))
    out = route("pronostica la potencia a una hora", HISTORY, two_bundle_registry(), interpreter=fixed)
    assert out["status"] == "INVALID_PROPOSAL" and any("not a fitted combination" in p for p in out["problems"])


def test_the_router_is_told_what_each_value_means_and_which_pairs_are_fitted():
    fixed = Fixed(json.dumps({"area": "forecasting", "state": {},
                              "questions": {"p": {"type": "point_forecast", "horizon": 60, "target": "Global_active_power"}}}))
    out = route("pronostica la potencia a una hora", HISTORY, two_bundle_registry(), interpreter=fixed)
    assert out["status"] == "OK"
    shown = fixed.asked[0]
    assert "una hora" in shown and "fitted_combinations" in shown and "direction_long" in shown


# --- Retsu's counterexamples (RETSU_TO_SATOSHI_AUDIT_M5PHET_ENVELOPE_2026_09_24, §5): a quantity said in words, and a
# --- percent alias applied to a unit that is not a probability, both walked through the digit-only guard ----------------

KW = {"p": {"type": "point_forecast", "status": "OK", "values": [0.5412255525588989], "unit": "kW"}}


def test_a_quantity_said_in_words_is_not_a_faithful_narration():
    for text in ("La potencia será casi el doble de la anterior: 0.5412 kW.",
                 "La mitad de lo habitual, 0.5412 kW.",
                 "Roughly twice the usual load, 0.5412 kW.",
                 "Half of yesterday's 0.5412 kW."):
        assert not narration_is_faithful(text, KW), text


def test_a_percent_alias_needs_a_probability_to_stand_on():
    # 0.5412 kW is power, not a share of anything; "54%" is a sentence about nothing the answers carry
    assert not narration_is_faithful("La previsión equivale a un 54% (0.5412 kW).", KW)
    assert not narration_is_faithful("Consumption at 54.12% of capacity, 0.5412 kW.", KW)
    # the same alias over a declared probability is still accepted
    probs = {"c": {"type": "choice", "status": "OK", "label": "euro_area",
                   "uncalibrated_probabilities": {"euro_area": 0.9666, "other": 0.0334}}}
    assert narration_is_faithful("euro_area con 96.66%", probs)
    assert narration_is_faithful("euro_area at 97% (0.9666)", probs)


def test_a_claim_of_profit_or_an_order_is_not_a_faithful_narration_of_a_critic_value():
    rl = {"v": {"type": "value_estimation", "status": "OK", "expected_return": 3.7995848655700684,
                "uncertainty_bounds": [3.7995848655700684, 4.006697177886963]},
          "a": {"type": "next_action", "status": "OK", "action": [0.05912280082702637],
                "unit": "target position fraction of the policy's own action scale"}}
    # every digit here is in the answers; the damage is in the verb (Retsu §5)
    assert not narration_is_faithful("Es ganancia realizada de 3.7995848655700684.", rl)
    assert not narration_is_faithful("Place a buy order for 0.05912280082702637 lots.", rl)
    assert not narration_is_faithful("Compra: la acción 0.0591 es una orden de compra.", rl)
    # the RL narration Retsu left standing: a negation of those words is not a claim of them
    assert narration_is_faithful("La acción 0.05912280082702637 es una fracción de posición, no una orden; el retorno "
                                 "3.7995848655700684 es bajo la recompensa de entrenamiento, no una ganancia.", rl)
    assert narration_is_faithful("Not an order and not a profit: the critic value is 3.7995848655700684.", rl)


def test_the_guard_says_why_it_discarded():
    from m5phet.orchestrate import narration_problems
    problems = narration_problems("Casi el doble, un 54%, ganancia de 0.5412 kW.", KW)
    assert any("doble" in p for p in problems)
    assert any("54%" in p or "percent" in p for p in problems)
    assert any("ganancia" in p for p in problems)
    assert narration_problems("La previsión es 0.5412 kW; el intervalo no se pudo calcular.", KW) == []
