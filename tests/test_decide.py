"""WP17: Laya as a configuration chooser. A state, a declared option set, a recorded choice -- and nothing else.

These tests are written against a FAKE classification provider, in the style of `test_web_tasks.py`. They prove the
plumbing and the refusals; they establish nothing about whether a choice is any good. That judgement belongs to the
closure table, not here.
"""

import json
from pathlib import Path

import pytest

from m5phet import decide
from m5phet.interpret import Interpreter
from m5phet.questions import MALFORMED_QUESTION, PROVIDER_ERROR, STATE_REQUIRED, refusal
from m5phet.runtime import Registry

PROFILE = {"feature": "Global_active_power", "stationarity": "STATIONARY", "acf_peaks": [74, 1443],
           "missing_fraction": 0.00002, "step_seconds": 60, "regular": True, "note": None}

OPTIONS = [["level", "keep the level"], ["diff", "first difference"], ["log_return", "log return"]]
QUESTION = {"transform": {"options": OPTIONS,
                          "instructions": "Which transform suits this series for one-hour-ahead forecasting?"}}


class FakeLaya:
    """A classification provider with Laya's answer shape. `backend` is what decides whether a decision may exist."""

    name, area = "laya_news", "classification"

    def __init__(self, backend="laya", probabilities=None, label=None, refuse=None, options_echo=True):
        self.backend, self.refuse = backend, refuse
        self.probabilities = probabilities or {"level": 0.1041, "diff": 0.8123, "log_return": 0.0836}
        self.label = label or "diff"
        self.options_echo = options_echo
        self.seen = []

    def capabilities(self):
        return {"provider": self.name, "operations": ["infer"], "families": ["classification"],
                "output_kinds": ["typed_questions"], "uncertainty_methods": ["UNCALIBRATED_CLASS_PROBABILITIES"],
                "supported": [{"operation": "infer", "family": "classification", "output_kind": "typed_questions"}],
                "known_states": ["laya-checkpoint:abc"], "backend": self.backend}

    def question_types(self):
        return {"choice": {"required": ["options"], "optional": ["instructions"]}}

    def answer_questions(self, state, questions, data, as_of):
        self.seen.append({"state": state, "questions": questions, "data": data, "as_of": as_of})
        answers = {}
        for name, question in questions.items():
            if self.refuse is not None:
                answers[name] = refusal(self.refuse[0], self.refuse[1], question["type"])
                continue
            answers[name] = {"type": "choice", "status": "OK", "label": self.label, "backend": self.backend,
                             "instructions": question.get("instructions"),
                             **({"options": [list(o) for o in question["options"]]} if self.options_echo else {}),
                             "uncalibrated_probabilities": dict(self.probabilities),
                             "probability_decimals": 4, "calibration": "UNCALIBRATED",
                             "execution_authorized": False,
                             **({"non_model_fixture": True, "warning": "NON_MODEL_FIXTURE: a declared fixture"}
                                if self.backend == "fixture" else {})}
        answers["__state_ref__"] = "laya-checkpoint:abc"
        return answers


class Mute(Interpreter):
    """An interpreter that is not available, so a narration never leaves the deterministic rendering."""

    def __init__(self):
        super().__init__(command="fixture", model="fixture-v1", environ={})

    @property
    def available(self):
        return False


def registry_with(provider):
    registry = Registry()
    registry.register(provider)
    return registry


def engine_with(provider):
    from m5phet.web.engine import Engine
    engine = Engine(registry=registry_with(provider), environ={})
    engine.interpreter = Mute()
    return engine


# --- the state text ---------------------------------------------------------------------------------------------------

def test_the_same_profile_yields_the_same_text_and_the_same_digest_whatever_the_key_order():
    one = decide.decision_state("feature_profile", PROFILE)
    shuffled = {k: PROFILE[k] for k in reversed(list(PROFILE))}
    two = decide.decision_state("feature_profile", shuffled)
    assert one == two
    assert decide.state_sha256(one) == decide.state_sha256(two)
    # the keys come out sorted, the declared decimals are used, and a list keeps the caller's order
    assert one.splitlines()[0] == "decision_kind: feature_profile"
    assert one.splitlines()[1:4] == ["acf_peaks:", "  - 74", "  - 1443"]
    assert "missing_fraction: 0.000020" in one
    assert "step_seconds: 60" in one and "regular: true" in one and "note: null" in one


def test_a_nested_payload_is_rendered_depth_first_with_sorted_keys_at_every_level():
    text = decide.decision_state("k", {"b": {"z": 1.5, "a": [1, {"y": 2, "x": 3}]}, "a": "first"})
    assert text.splitlines() == ["decision_kind: k", "a: first", "b:", "  a:", "    - 1", "    -",
                                 "      x: 3", "      y: 2", "  z: 1.500000"]


def test_a_state_is_a_description_and_refuses_to_carry_rows():
    with pytest.raises(decide.DecisionError, match=decide.ROWS_IN_STATE):
        decide.decision_state("k", {"rows": list(range(decide.MAX_LIST_ITEMS + 1))})


# --- what reaches the provider ------------------------------------------------------------------------------------------

def test_the_envelope_reaching_the_provider_carries_exactly_the_declared_options_and_instructions():
    provider = FakeLaya()
    state = decide.decision_state("feature_profile", PROFILE)
    out = decide.ask(engine_with(provider), state, QUESTION, kind="feature_profile")
    assert out["transform"]["status"] == "OK"
    seen = provider.seen[0]
    assert seen["state"] == {"news": state, "asset": "M5PHET", "language": "en"}
    assert seen["questions"] == {"transform": {"type": "choice", "options": OPTIONS,
                                               "instructions": QUESTION["transform"]["instructions"]}}
    # the clock stamps the record; it is not sent, because a state text carries no clocks of its own to replay
    assert seen["as_of"] is None
    assert out["transform"]["decision"]["as_of"] == out["transform"]["decision"]["as_of"]


def test_a_bare_registry_takes_the_run_task_route_and_reaches_the_same_provider():
    provider = FakeLaya()
    state = decide.decision_state("feature_profile", PROFILE)
    out = decide.ask(registry_with(provider), state, QUESTION, kind="feature_profile", as_of="2026-09-25T00:00:00Z")
    assert out["transform"]["decision"]["chosen"] == "diff"
    assert out["transform"]["decision"]["as_of"] == "2026-09-25T00:00:00Z"
    assert provider.seen[0]["state"]["asset"] == "M5PHET"


# --- what is refused --------------------------------------------------------------------------------------------------

def test_a_fixture_backend_is_refused_and_nothing_is_written(tmp_path):
    provider = FakeLaya(backend="fixture")
    state = decide.decision_state("feature_profile", PROFILE)
    out = decide.ask(engine_with(provider), state, QUESTION, kind="feature_profile", record_dir=tmp_path)
    assert out["transform"]["status"] == "REFUSED"
    assert out["transform"]["refusal"] == decide.NON_MODEL_FIXTURE
    assert "decision" not in out["transform"]
    assert list(tmp_path.iterdir()) == []


def test_a_backend_that_is_not_laya_is_refused_even_without_the_fixture_marker(tmp_path):
    provider = FakeLaya(backend="some_other_model")
    state = decide.decision_state("feature_profile", PROFILE)
    out = decide.ask(registry_with(provider), state, QUESTION, kind="feature_profile", record_dir=tmp_path)
    assert out["transform"]["refusal"] == decide.NON_MODEL_FIXTURE
    assert list(tmp_path.iterdir()) == []


def test_fewer_than_two_options_is_refused_before_anything_is_asked(tmp_path):
    provider = FakeLaya()
    state = decide.decision_state("feature_profile", PROFILE)
    out = decide.ask(engine_with(provider), state, {"transform": {"options": [["level", "keep the level"]],
                                                                  "instructions": "Which transform?"}},
                     kind="feature_profile", record_dir=tmp_path)
    assert out["transform"]["refusal"] == decide.MALFORMED_OPTIONS
    assert "at least two" in out["transform"]["why"]
    assert provider.seen == [], "a question with nothing to choose between never reaches the model"
    assert list(tmp_path.iterdir()) == []


def test_a_duplicate_option_key_is_refused_before_anything_is_asked():
    provider = FakeLaya()
    state = decide.decision_state("feature_profile", PROFILE)
    out = decide.ask(engine_with(provider), state,
                     {"transform": {"options": [["diff", "first difference"], ["diff", "the difference"]],
                                    "instructions": "Which transform?"}}, kind="feature_profile")
    assert out["transform"]["refusal"] == decide.MALFORMED_OPTIONS and "diff" in out["transform"]["why"]
    assert provider.seen == []


def test_an_empty_state_text_is_refused_before_anything_is_asked():
    provider = FakeLaya()
    out = decide.ask(engine_with(provider), "   ", QUESTION, kind="feature_profile")
    assert out["transform"]["refusal"] == STATE_REQUIRED
    assert provider.seen == []


def test_a_question_without_instructions_is_refused_before_anything_is_asked():
    provider = FakeLaya()
    state = decide.decision_state("feature_profile", PROFILE)
    out = decide.ask(engine_with(provider), state, {"transform": {"options": OPTIONS}}, kind="feature_profile")
    assert out["transform"]["refusal"] == decide.INSTRUCTIONS_REQUIRED
    assert provider.seen == []


def test_one_malformed_question_does_not_stop_the_others():
    provider = FakeLaya()
    state = decide.decision_state("feature_profile", PROFILE)
    out = decide.ask(engine_with(provider), state,
                     {"bad": {"options": [["a", "A"]], "instructions": "?"},
                      "transform": QUESTION["transform"]}, kind="feature_profile")
    assert out["bad"]["refusal"] == decide.MALFORMED_OPTIONS
    assert out["transform"]["status"] == "OK"
    assert list(provider.seen[0]["questions"]) == ["transform"]


def test_the_providers_own_token_budget_refusal_passes_through_by_name(tmp_path):
    provider = FakeLaya(refuse=(MALFORMED_QUESTION, "TOKEN_BUDGET_EXCEEDED: 'transform' would be truncated by the "
                                                    "pinned SDK; shorten the instructions or the options"))
    state = decide.decision_state("feature_profile", PROFILE)
    out = decide.ask(engine_with(provider), state, QUESTION, kind="feature_profile", record_dir=tmp_path)
    assert out["transform"]["status"] == "REFUSED"
    assert out["transform"]["refusal"] == MALFORMED_QUESTION
    assert "TOKEN_BUDGET_EXCEEDED" in out["transform"]["why"], "the provider's own name survives this layer"
    assert list(tmp_path.iterdir()) == []


def test_a_choice_outside_the_declared_options_is_refused(tmp_path):
    provider = FakeLaya(label="wavelet")
    state = decide.decision_state("feature_profile", PROFILE)
    out = decide.ask(engine_with(provider), state, QUESTION, kind="feature_profile", record_dir=tmp_path)
    assert out["transform"]["refusal"] == decide.CHOICE_OUTSIDE_OPTIONS and "wavelet" in out["transform"]["why"]
    assert list(tmp_path.iterdir()) == []


def test_a_probability_over_an_option_that_was_never_declared_is_refused():
    provider = FakeLaya(probabilities={"level": 0.2, "diff": 0.5, "wavelet": 0.3})
    state = decide.decision_state("feature_profile", PROFILE)
    out = decide.ask(engine_with(provider), state, QUESTION, kind="feature_profile")
    assert out["transform"]["refusal"] == decide.CHOICE_OUTSIDE_OPTIONS


def test_a_provider_that_cannot_be_reached_refuses_and_records_nothing(tmp_path):
    class Unreachable(FakeLaya):
        def answer_questions(self, state, questions, data, as_of):
            raise OSError("the worker did not answer")

    state = decide.decision_state("feature_profile", PROFILE)
    out = decide.ask(registry_with(Unreachable()), state, QUESTION, kind="feature_profile", record_dir=tmp_path)
    assert out["transform"]["status"] == "REFUSED" and out["transform"]["refusal"] == PROVIDER_ERROR
    assert list(tmp_path.iterdir()) == []


# --- the record -------------------------------------------------------------------------------------------------------

def test_the_decision_carries_the_answers_own_numbers_and_no_other():
    exact = {"level": 0.10413456789, "diff": 0.81230987654, "log_return": 0.08355555557}
    provider = FakeLaya(probabilities=exact)
    state = decide.decision_state("feature_profile", PROFILE)
    out = decide.ask(engine_with(provider), state, QUESTION, kind="feature_profile")
    decision = out["transform"]["decision"]
    assert decision["probabilities"] == exact, "verbatim: no rounding, no rescaling, no renormalisation"
    assert decision["probability_decimals"] == 4, "the decimals the ANSWER declared, not a number invented here"
    assert decision["checkpoint"] == "laya-checkpoint:abc"
    assert decision["backend"] == "laya" and decision["execution_authorized"] is False
    assert decision["state_sha256"] == decide.state_sha256(state)
    assert decision["kind"] == "feature_profile" and decision["question"] == "transform"
    assert decision["options"] == OPTIONS and decision["chosen"] == "diff"
    assert set(decision) == {"schema", "kind", "state_sha256", "question", "options", "chosen", "probabilities",
                             "probability_decimals", "checkpoint", "backend", "as_of", "execution_authorized"}


def test_a_record_round_trips_and_its_digest_is_verified(tmp_path):
    provider = FakeLaya()
    state = decide.decision_state("feature_profile", PROFILE)
    out = decide.ask(engine_with(provider), state, QUESTION, kind="feature_profile", record_dir=tmp_path)
    path = Path(out["transform"]["record_path"])
    assert path.parent == tmp_path and path.name == f"{decide.decision_sha256(out['transform']['decision'])}.json"
    assert decide.load(path) == out["transform"]["decision"]
    # the same decision written twice is the same file: the record is content-addressed
    again = decide.record(out["transform"]["decision"], tmp_path)
    assert again == path and len(list(tmp_path.iterdir())) == 1


def test_a_tampered_record_does_not_load(tmp_path):
    provider = FakeLaya()
    state = decide.decision_state("feature_profile", PROFILE)
    out = decide.ask(registry_with(provider), state, QUESTION, kind="feature_profile", record_dir=tmp_path)
    path = Path(out["transform"]["record_path"])
    body = json.loads(path.read_text())
    body["probabilities"]["diff"] = 0.99
    path.write_text(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    with pytest.raises(decide.DecisionError, match="digest"):
        decide.load(path)


def test_a_decision_on_a_fixture_is_never_written_even_if_a_caller_asks_for_it(tmp_path):
    provider = FakeLaya()
    state = decide.decision_state("feature_profile", PROFILE)
    out = decide.ask(registry_with(provider), state, QUESTION, kind="feature_profile")
    forged = dict(out["transform"]["decision"], backend="fixture")
    with pytest.raises(decide.DecisionError, match=decide.NON_MODEL_FIXTURE):
        decide.record(forged, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_decision_claiming_execution_authority_is_never_written(tmp_path):
    provider = FakeLaya()
    state = decide.decision_state("feature_profile", PROFILE)
    out = decide.ask(registry_with(provider), state, QUESTION, kind="feature_profile")
    forged = dict(out["transform"]["decision"], execution_authorized=True)
    with pytest.raises(decide.DecisionError, match="execution"):
        decide.record(forged, tmp_path)
    assert list(tmp_path.iterdir()) == []
