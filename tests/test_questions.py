"""The envelope: every question answered on its own, every refusal typed, and no path from "cannot" to a number."""

import pytest

from m5phet.questions import (MALFORMED_QUESTION, MISSING_FIELD, NO_PROVIDER, NOT_ESTIMABLE, PROVIDER_ERROR,
                              TASK_SCHEMA, UNSUPPORTED_QUESTION_TYPE, TaskError, catalog, refusal, run_task,
                              validate_task)
from m5phet.runtime import Registry


class Forecaster:
    """A provider that answers point forecasts and honestly refuses intervals."""

    name = "fake_forecaster"
    area = "forecasting"

    def capabilities(self):
        return {"provider": self.name, "operations": ["infer"], "families": ["regression_forecasting"],
                "output_kinds": ["point_forecast"], "uncertainty_methods": ["none"],
                "supported": [{"operation": "infer", "family": "regression_forecasting", "output_kind": "point_forecast"}],
                "known_states": ["s1"]}

    def question_types(self):
        return {"point_forecast": {"required": ["horizon"], "optional": ["target"]},
                "interval": {"required": ["horizon", "confidence_level"]}}

    def answer_questions(self, state, questions, data, as_of):
        out = {"__state_ref__": "s1"}
        for name, q in questions.items():
            if q["type"] == "point_forecast":
                out[name] = {"type": "point_forecast", "values": [1.0] * q["horizon"], "unit": "kW",
                             "execution_authorized": False}
            else:
                out[name] = refusal(NOT_ESTIMABLE, "this model emits a point estimate and no predictive distribution",
                                    q["type"])
        return out


def task(**questions):
    return {"schema": TASK_SCHEMA, "area": "forecasting", "state": {"dataset_id": "ds"}, "questions": questions}


@pytest.fixture
def registry():
    r = Registry()
    r.register(Forecaster())
    return r


def test_each_question_is_answered_on_its_own(registry):
    out = run_task(task(p={"type": "point_forecast", "horizon": 3},
                        i={"type": "interval", "horizon": 3, "confidence_level": 0.95}), registry)
    assert out["answers"]["p"]["status"] == "OK" and out["answers"]["p"]["values"] == [1.0, 1.0, 1.0]
    assert out["answers"]["i"]["status"] == "REFUSED" and out["answers"]["i"]["refusal"] == NOT_ESTIMABLE
    assert out["answered"] == 1 and out["refused"] == 1
    assert out["execution_authorized"] is False and out["provider"] == "fake_forecaster"


def test_an_undeclared_type_is_refused_by_name_before_the_provider(registry):
    out = run_task(task(a={"type": "anomaly_risk", "threshold": "< 100"}), registry)
    assert out["answers"]["a"]["refusal"] == UNSUPPORTED_QUESTION_TYPE
    assert "point_forecast" in out["answers"]["a"]["why"]


def test_a_missing_required_field_is_refused_by_name(registry):
    out = run_task(task(p={"type": "point_forecast"}), registry)
    assert out["answers"]["p"]["refusal"] == MISSING_FIELD and "horizon" in out["answers"]["p"]["why"]


def test_an_unknown_field_is_refused_rather_than_ignored(registry):
    out = run_task(task(p={"type": "point_forecast", "horizon": 2, "seasonality": "weekly"}), registry)
    assert out["answers"]["p"]["refusal"] == MALFORMED_QUESTION


def test_answers_keep_the_callers_order(registry):
    out = run_task(task(z={"type": "point_forecast", "horizon": 1}, a={"type": "point_forecast", "horizon": 1}), registry)
    assert list(out["answers"]) == ["z", "a"]


def test_an_area_with_no_provider_refuses_every_question():
    out = run_task({"area": "rl", "state": {}, "questions": {"n": {"type": "next_action"}}}, Registry())
    assert out["answers"]["n"]["refusal"] == NO_PROVIDER and out["provider"] is None


@pytest.mark.parametrize("bad", [
    {"area": "astrology", "state": {}, "questions": {"q": {"type": "x"}}},
    {"area": "forecasting", "questions": {"q": {"type": "x"}}},
    {"area": "forecasting", "state": {}, "questions": {}},
    {"area": "forecasting", "state": {}, "questions": {"q": "not a mapping"}},
    {"area": "forecasting", "state": {}, "questions": {"q": {}}},
    {"schema": "other", "area": "forecasting", "state": {}, "questions": {"q": {"type": "x"}}},
])
def test_a_malformed_envelope_is_refused_before_any_provider(bad):
    with pytest.raises(TaskError):
        validate_task(bad)


def test_a_provider_that_answers_the_wrong_type_or_claims_authority_is_caught():
    class Wrong(Forecaster):
        name = "wrong"

        def answer_questions(self, state, questions, data, as_of):
            return {"p": {"type": "interval", "values": [1]},
                    "q": {"type": "point_forecast", "values": [1], "execution_authorized": True}}

    r = Registry()
    r.register(Wrong())
    out = run_task(task(p={"type": "point_forecast", "horizon": 1}, q={"type": "point_forecast", "horizon": 1}), r)
    assert out["answers"]["p"]["refusal"] == PROVIDER_ERROR and out["answers"]["q"]["refusal"] == PROVIDER_ERROR


def test_a_provider_that_raises_refuses_its_questions_instead_of_the_envelope():
    class Broken(Forecaster):
        name = "broken"

        def answer_questions(self, *a):
            raise RuntimeError("engine fell over")

    r = Registry()
    r.register(Broken())
    out = run_task(task(p={"type": "point_forecast", "horizon": 1}), r)
    assert out["answers"]["p"]["refusal"] == PROVIDER_ERROR and "fell over" in out["answers"]["p"]["why"]


def test_the_catalog_lists_what_may_be_asked(registry):
    cat = catalog(registry)
    assert cat["forecasting"]["provider"] == "fake_forecaster"
    assert set(cat["forecasting"]["question_types"]) == {"point_forecast", "interval"}
    assert cat["rl"]["provider"] is None and cat["rl"]["question_types"] == {}
