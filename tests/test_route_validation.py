"""RB04: the three things the envelope did not validate, and the interval it could not ask for.

Written top-down, before the implementation, from five holes PROBED in the integrated tree on 2026-09-28. Each one
was confirmed to pass validation before any of this existed, so every counterexample below is a defect that was
really there and not a hypothetical:

  B  `horizon: True` was admitted whenever `1` was a fitted horizon, because `True == 1` in Python and membership
     was tested with `in`. A boolean reached an engine as a horizon. Musashi's `44130ae` repaired exactly this in the
     MEASUREMENT scorer; the product validator still had it.
  E  a question carrying the governed alias `target_variable` was not validated at all: any value passed.
  F  a question naming `target` and `target_variable` with DIFFERENT values passed, and which one the engine would
     read was left to the engine.
  G  `interval` with `confidence_level: 0.99` passed although no bundle fitted that pair.
  H  `confidence_level: 95` passed, which is not even a probability.
  I  `completable_types` never offered `interval`, because `confidence_level` was in no slot -- which is why the four
     two-ask sentences of the router measurement stayed under-answered.

The positive paths are asserted beside every counterexample, because a validator that refuses more than it should is
a worse product than one that refuses less: it refuses the person's real sentence.
"""

import json
import math

import pytest

from m5phet import outputs
from m5phet.orchestrate import (GOVERNED_ALIASES, admits, check_proposal, completable_types, complete_under_answer,
                                narrate, route)
from m5phet.questions import catalog
from m5phet.runtime import Registry

from test_orchestrate import Fixed, Forecaster

HISTORY = {"values": [1.0, 2.0], "scale": "original", "scaler_digest": "abc"}


class Quantile(Forecaster):
    """The real forecaster's shape where an interval is concerned: `interval` is a DECLARED type needing a horizon and
    a confidence level, and the capabilities declare which levels a bundle actually fitted -- 0.9 from the symmetric
    pair (0.05, 0.95). `fitted_confidence_levels` is the provider's own field; nothing here invents it."""

    levels = [0.9]

    def capabilities(self):
        return {**super().capabilities(),
                "bundles": [{"state_ref": "s1", "quantiles": [0.05, 0.5, 0.95],
                             "fitted_confidence_levels": list(self.levels)}]}

    def question_types(self):
        return {"point_forecast": {"required": ["horizon"], "optional": ["target"]},
                "interval": {"required": ["horizon", "confidence_level"], "optional": ["target"]}}

    def chat_slots(self):
        return [{"name": "target", "allowed": ["Global_active_power"]},
                {"name": "horizon", "allowed": [60], "type": "integer"}]


class PointOnly(Quantile):
    """A point bundle: `interval` is still DECLARED -- so the refusal can say the true thing -- and no level is fitted."""

    levels = []

    def capabilities(self):
        return {**Forecaster.capabilities(self),
                "bundles": [{"state_ref": "s1", "quantiles": [], "fitted_confidence_levels": []}]}


class TwoLevels(Quantile):
    levels = [0.5, 0.9]

    def capabilities(self):
        return {**Forecaster.capabilities(self),
                "bundles": [{"state_ref": "s1", "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
                             "fitted_confidence_levels": [0.5, 0.9]}]}


def registry_of(provider):
    r = Registry()
    r.register(provider)
    return r


def checked(envelope, provider=None):
    """(task, problems) from `check_proposal` against a real catalog. No model is consulted."""
    registry = registry_of(provider or Quantile())
    return check_proposal(envelope, catalog(registry), {"kind": "none", "columns": [], "rows": 0})


def envelope(**questions):
    return {"area": "forecasting", "state": {}, "questions": questions}


def point(**fields):
    return {"type": "point_forecast", **fields}


def interval(**fields):
    return {"type": "interval", **fields}


# --- A. numeric validation: the validator now refuses what the repaired scorer refuses ---------------------------------

def test_a_boolean_is_never_a_governed_number_even_when_one_is_a_fitted_value():
    """Hole B. `True == 1` in Python, so a slot whose fitted horizon is 1 admitted `True` by plain membership."""
    class Horizon1(Quantile):
        def chat_slots(self):
            return [{"name": "target", "allowed": ["Global_active_power"]},
                    {"name": "horizon", "allowed": [1], "type": "integer"}]

    task, problems = checked(envelope(p=point(horizon=True, target="Global_active_power")), Horizon1())
    assert task is None
    assert any("horizon" in p and "True" in p for p in problems), problems


def test_a_boolean_false_is_not_the_horizon_zero_either():
    class Horizon0(Quantile):
        def chat_slots(self):
            return [{"name": "target", "allowed": ["Global_active_power"]},
                    {"name": "horizon", "allowed": [0], "type": "integer"}]

    task, problems = checked(envelope(p=point(horizon=False, target="Global_active_power")), Horizon0())
    assert task is None and any("horizon" in p for p in problems)


def test_a_fractional_horizon_is_refused_and_never_truncated_to_the_fitted_one():
    task, problems = checked(envelope(p=point(horizon=60.5, target="Global_active_power")))
    assert task is None and any("60.5" in p for p in problems)


def test_a_non_finite_horizon_is_refused_by_name_and_does_not_raise():
    for value in (float("inf"), float("-inf"), float("nan")):
        task, problems = checked(envelope(p=point(horizon=value, target="Global_active_power")))
        assert task is None and problems, value


def test_the_same_number_written_as_a_float_is_still_the_fitted_horizon():
    """The positive path of the same rule: 60.0 IS 60, and refusing it would refuse a real sentence."""
    task, problems = checked(envelope(p=point(horizon=60.0, target="Global_active_power")))
    assert task is not None, problems


def test_an_integer_written_as_text_is_refused_by_the_validator_as_it_always_was():
    """Kept deliberately: the SCORER accepts an exact integer string in order to score what a model wrote, and the
    VALIDATOR refuses it so a string never reaches an engine expecting a number. The two are not in conflict --
    `verdict_of` reads `INVALID_PROPOSAL` before it reads any value, so a refused envelope is never scored CORRECT."""
    task, problems = checked(envelope(p=point(horizon="60", target="Global_active_power")))
    assert task is None and any("'60'" in p for p in problems)


def test_the_fitted_horizon_and_target_still_validate():
    task, problems = checked(envelope(p=point(horizon=60, target="Global_active_power")))
    assert task is not None and problems == []


@pytest.mark.parametrize("value,allowed,expected", [
    (True, [1], False), (False, [0], False), (1, [1], True), (60.0, [60], True), (60.5, [60], False),
    (float("nan"), [60], False), (float("inf"), [60], False), ("60", [60], False),
    ("a", ["a"], True), ("a", ["b"], False), (True, [True], True), (1, [True], False),
])
def test_type_aware_membership_is_the_whole_numeric_rule(value, allowed, expected):
    assert admits(value, allowed) is expected


# --- B. alias validation: a governed field is governed under every spelling it has -------------------------------------

def test_the_alias_of_a_governed_field_inside_a_question_is_validated_too():
    """Hole E. `target_variable` is the owner's spelling of the forecaster's `target`; an unfitted value under the
    alias passed untouched and was refused later, by the engine, after the person pressed run."""
    task, problems = checked(envelope(p=point(horizon=60, target_variable="NOT_A_FITTED_TARGET")))
    assert task is None
    assert any("target_variable" in p and "NOT_A_FITTED_TARGET" in p for p in problems), problems


def test_two_spellings_of_one_governed_field_disagreeing_is_refused_as_a_contradiction():
    """Hole F. Both values may be admissible; naming two of them for one field is not a question."""
    class Two(Quantile):
        def chat_slots(self):
            return [{"name": "target", "allowed": ["Global_active_power", "Voltage"]},
                    {"name": "horizon", "allowed": [60], "type": "integer"}]

    task, problems = checked(envelope(p=point(horizon=60, target="Global_active_power", target_variable="Voltage")),
                             Two())
    assert task is None
    assert any("target" in p and "Voltage" in p and "Global_active_power" in p for p in problems), problems


def test_the_same_value_under_both_spellings_is_not_a_contradiction():
    task, problems = checked(envelope(p=point(horizon=60, target="Global_active_power",
                                              target_variable="Global_active_power")))
    assert task is not None, problems


def test_a_governed_alias_in_the_state_is_validated_as_it_always_was():
    good = checked({"area": "forecasting", "state": {"target_variable": "Global_active_power"},
                    "questions": {"p": point(horizon=60)}})
    assert good[0] is not None, good[1]
    bad = checked({"area": "forecasting", "state": {"target_variable": "nope"},
                   "questions": {"p": point(horizon=60)}})
    assert bad[0] is None and any("state.target_variable" in p for p in bad[1])


def test_the_alias_map_the_validator_uses_is_the_one_the_measurement_scores_against():
    """The scorer and the validator must not drift: a spelling one of them governs and the other does not is a hole
    that reopens silently. `tools/measure_route.py` declares FIELD_SPELLINGS; the validator declares
    GOVERNED_ALIASES; this asserts they say the same thing about aliases."""
    import importlib.util
    import pathlib
    tool = pathlib.Path(__file__).resolve().parent.parent / "tools" / "measure_route.py"
    spec = importlib.util.spec_from_file_location("_measure_route_for_drift", tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from_tool = {spelling: field for field, spellings in module.FIELD_SPELLINGS.items()
                 for spelling in spellings if spelling != field}
    assert from_tool == GOVERNED_ALIASES


# --- C. interval requests: a level nobody fitted is refused, and the ones fitted are askable ---------------------------

def test_a_confidence_level_no_bundle_fitted_is_refused_naming_the_ones_that_are():
    """Hole G. A 0.95 interval is not obtained by widening a 0.90 one; its two bounds are two quantiles somebody
    fitted or they are two numbers somebody invented."""
    task, problems = checked(envelope(r=interval(horizon=60, confidence_level=0.99, target="Global_active_power")))
    assert task is None
    assert any("confidence_level" in p and "0.99" in p and "0.9" in p for p in problems), problems


def test_a_confidence_level_that_is_not_a_probability_is_refused():
    """Hole H. 95 is a percentage somebody typed; the field is a two-sided level strictly between 0 and 1."""
    task, problems = checked(envelope(r=interval(horizon=60, confidence_level=95, target="Global_active_power")))
    assert task is None and any("confidence_level" in p for p in problems)


def test_a_boolean_or_a_string_confidence_level_is_refused():
    for value in (True, "0.9", None):
        task, problems = checked(envelope(r=interval(horizon=60, confidence_level=value,
                                                     target="Global_active_power")))
        assert task is None and problems, value


def test_a_fitted_confidence_level_validates():
    task, problems = checked(envelope(r=interval(horizon=60, confidence_level=0.9, target="Global_active_power")))
    assert task is not None, problems


def test_each_of_several_fitted_levels_validates_and_an_unfitted_one_between_them_does_not():
    ok = checked(envelope(r=interval(horizon=60, confidence_level=0.5, target="Global_active_power")), TwoLevels())
    assert ok[0] is not None, ok[1]
    mid = checked(envelope(r=interval(horizon=60, confidence_level=0.7, target="Global_active_power")), TwoLevels())
    assert mid[0] is None and any("0.7" in p for p in mid[1])


def test_an_area_that_fitted_no_level_leaves_the_refusal_to_the_engine_and_answers_the_other_half():
    """The correction this test records. An earlier draft of the validator refused the WHOLE envelope when the area
    declared no fitted level -- and `m5phet.questions` is explicit that every question is answered separately: "a
    request asking for a point forecast and an interval gets the point forecast from an engine that has one and an
    explicit refusal for the interval from an engine that does not". Refusing both halves because one is
    unserviceable hides which half the engine can actually do, so a well-formed level an area did not fit passes
    validation and the PROVIDER refuses that one question by name."""
    from m5phet.questions import run_task

    class PointAnswers(PointOnly):
        def answer_questions(self, state, questions, data, as_of):
            from m5phet.questions import refusal
            out = {}
            for name, question in questions.items():
                out[name] = ({"type": "point_forecast", "values": [0.5412], "unit": "kW"}
                             if question["type"] == "point_forecast"
                             else refusal("NOT_ESTIMABLE", "it emits a point estimate and no predictive distribution",
                                          question["type"]))
            return out

    task, problems = checked(envelope(p=point(horizon=60, target="Global_active_power"),
                                      r=interval(horizon=60, confidence_level=0.9, target="Global_active_power")),
                             PointAnswers())
    assert task is not None, problems
    out = run_task(task, registry_of(PointAnswers()), data=HISTORY)
    assert out["answers"]["p"]["values"] == [0.5412]
    assert out["answers"]["r"]["refusal"] == "NOT_ESTIMABLE" and out["answered"] == 1 and out["refused"] == 1


def test_a_malformed_level_is_an_envelope_problem_even_where_no_level_is_fitted():
    """The other side of the same split: 95 is not a level any engine could mean, so the router misread the sentence
    and the person is told that instead of being handed a refusal that blames the model for lacking a distribution."""
    task, problems = checked(envelope(r=interval(horizon=60, confidence_level=95, target="Global_active_power")),
                             PointOnly())
    assert task is None and any("confidence_level" in p and "95" in p for p in problems), problems


def test_the_catalog_publishes_the_fitted_levels_so_a_caller_sees_them_before_asking():
    entry = catalog(registry_of(TwoLevels()))["forecasting"]
    assert entry["confidence_levels"] == [0.5, 0.9]
    assert catalog(registry_of(PointOnly()))["forecasting"]["confidence_levels"] == []


def test_a_point_forecast_is_unaffected_by_any_of_this():
    task, problems = checked(envelope(p=point(horizon=60, target="Global_active_power")), PointOnly())
    assert task is not None, problems


# --- D. the under-answered interval, recovered from declared levels only -----------------------------------------------

class Scripted(Fixed):
    def __init__(self, *replies):
        super().__init__(replies[0])
        self.replies = list(replies)

    def _ask(self, text):
        self.asked.append(text)
        return self.replies[min(len(self.asked) - 1, len(self.replies) - 1)]


def test_interval_is_offerable_once_the_levels_it_needs_are_declared_values():
    """Hole I. `confidence_level` is now a governed value like a horizon, so rule 2 of the completion pass is
    satisfied from the area's OWN declaration and not by this layer choosing a number."""
    task = envelope(p=point(horizon=60, target="Global_active_power"))
    entry = catalog(registry_of(Quantile()))["forecasting"]
    assert "interval" in completable_types(task, entry["question_types"], area=entry)


def test_interval_is_not_offerable_when_no_level_was_fitted():
    """The counterexample that keeps the repair honest: with nothing fitted there is no level to choose among, so the
    type is not offered at all rather than offered and then filled with a guess."""
    task = envelope(p=point(horizon=60, target="Global_active_power"))
    entry = catalog(registry_of(PointOnly()))["forecasting"]
    assert "interval" not in completable_types(task, entry["question_types"], area=entry)


def test_the_second_ask_of_a_two_part_sentence_is_recovered_as_a_validating_interval():
    """The measured failure mode: "pronostica la potencia y dame un rango" produced `interval` alone or
    `point_forecast` alone. The completion pass asks once, among declared types, and then once among declared
    levels."""
    registry = registry_of(Quantile())
    cat = catalog(registry)
    task = envelope(p=point(horizon=60, target="Global_active_power"))
    interpreter = Scripted(json.dumps({"also_asked_question_type": "interval"}),
                           json.dumps({"confidence_level": 0.9}))
    completed, record = complete_under_answer("pronostica la potencia y dame un rango", task, cat,
                                              {"kind": "none", "columns": [], "rows": 0}, interpreter)
    assert record["outcome"] == "ADDED" and record["added"] == "interval"
    assert sorted(q["type"] for q in completed["questions"].values()) == ["interval", "point_forecast"]
    added = next(q for q in completed["questions"].values() if q["type"] == "interval")
    assert added["confidence_level"] == 0.9 and added["horizon"] == 60
    assert check_proposal(completed, cat, {"kind": "none", "columns": [], "rows": 0})[0] is not None


def test_the_model_choosing_an_unfitted_level_for_the_completion_drops_the_addition():
    registry = registry_of(Quantile())
    cat = catalog(registry)
    task = envelope(p=point(horizon=60, target="Global_active_power"))
    interpreter = Scripted(json.dumps({"also_asked_question_type": "interval"}),
                           json.dumps({"confidence_level": 0.99}))
    completed, record = complete_under_answer("pronostica y dame un rango", task, cat,
                                              {"kind": "none", "columns": [], "rows": 0}, interpreter)
    assert record["outcome"] in ("ADDITION_DID_NOT_VALIDATE", "CHOICE_WAS_NOT_OFFERED")
    assert sorted(q["type"] for q in completed["questions"].values()) == ["point_forecast"]


def test_the_completion_is_shown_only_the_declared_levels_and_never_a_row():
    registry = registry_of(TwoLevels())
    cat = catalog(registry)
    interpreter = Scripted(json.dumps({"also_asked_question_type": "interval"}),
                           json.dumps({"confidence_level": 0.5}))
    complete_under_answer("dame un rango", envelope(p=point(horizon=60, target="Global_active_power")), cat,
                          {"kind": "none", "columns": [], "rows": 0}, interpreter)
    shown = interpreter.asked[-1]
    assert "0.5" in shown and "0.9" in shown and "0.99" not in shown
    assert "1.0" not in shown and "2.0" not in shown


def test_route_recovers_the_interval_end_to_end():
    interpreter = Scripted(json.dumps(envelope(p=point(horizon=60, target="Global_active_power"))),
                           json.dumps({"also_asked_question_type": "interval"}),
                           json.dumps({"confidence_level": 0.9}))
    out = route("pronostica la potencia a una hora y dame un rango", HISTORY, registry_of(Quantile()),
                interpreter=interpreter)
    assert out["status"] == "OK", out["problems"]
    assert sorted(q["type"] for q in out["task"]["questions"].values()) == ["interval", "point_forecast"]
    assert out["completion"]["outcome"] == "ADDED"


# --- E. output selection: the request may name the procedure, and an unknown one is never substituted -----------------

def test_an_envelope_may_select_an_installed_output_procedure():
    task, problems = checked({**envelope(p=point(horizon=60, target="Global_active_power")),
                              "output": {"plugin": "telegram"}})
    assert task is not None, problems
    assert task["output"]["plugin"] == "telegram"


def test_an_output_procedure_that_is_not_installed_is_refused_by_name_and_not_silently_defaulted():
    """The rule the whole output component rests on: nothing is substituted. A request that asks for a surface this
    installation does not have must be told so, not quietly answered by `default` as if it had been honoured."""
    task, problems = checked({**envelope(p=point(horizon=60, target="Global_active_power")),
                              "output": {"plugin": "carrier_pigeon"}})
    assert task is None
    assert any("carrier_pigeon" in p and "default" in p for p in problems), problems


def test_an_envelope_with_no_output_selection_is_exactly_todays_behaviour():
    task, problems = checked(envelope(p=point(horizon=60, target="Global_active_power")))
    assert task is not None and "output" not in task, problems


@pytest.mark.parametrize("bad", [{"plugin": 5}, {"plugin": ""}, {"plugin": None}, "telegram", [], {"surface": "x"}])
def test_a_malformed_output_selection_is_refused(bad):
    task, problems = checked({**envelope(p=point(horizon=60, target="Global_active_power")), "output": bad})
    assert task is None and problems, bad


def test_the_selected_procedure_is_the_one_that_renders_the_answer():
    response = {"area": "forecasting", "answered": 1, "refused": 0,
                "answers": {"p": {"type": "point_forecast", "values": [0.5412], "unit": "kW", "status": "OK"}}}
    task = {"area": "forecasting", "output": {"plugin": "telegram"}}
    assert narrate("pronostica", response, task=task)["output_plugin"] == "telegram"
    assert narrate("pronostica", response)["output_plugin"] == "default"


def test_selecting_a_surface_does_not_change_what_the_area_returns():
    """The header is a property of the area and its provider, never of the screen the answer lands on."""
    assert outputs.header("forecasting") == outputs.header("forecasting")
    plain = outputs.header("forecasting")
    assert "telegram" not in json.dumps(plain) and plain["output_kind"] == "point_forecast"


def test_every_name_the_selection_accepts_is_one_this_installation_can_load():
    for name in outputs.names():
        assert outputs.load(name) is not None
