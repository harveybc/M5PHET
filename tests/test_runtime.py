"""Behavioral acceptance tests for the M5PHET runtime (P02 provider lifecycle and the request/result envelope).

Written BEFORE the implementation, from docs/INTERFACES.md and docs/IMPLEMENTATION_PLAN.md. Every test states a behaviour a
caller can observe, not a structure: a refusal must happen before the model is loaded, a fitted state must not move during
inference, and a refused output must carry no invented number.
"""

import copy

import pytest

from m5phet import ContractError
from m5phet.runtime import (
    REQUEST_SCHEMA,
    Registry,
    RequestError,
    Status,
    run,
    validate_request,
)


# --- a provider double that RECORDS what the runtime did to it -----------------------------------------------------------

class RecordingProvider:
    """Declares narrow capabilities and records every call, so a test can prove the runtime refused before doing work."""

    name = "recording"

    def __init__(self, *, families=("classification",), output_kinds=("typed_questions",),
                 operations=("infer", "fit", "calibrate", "evaluate"), states=("state-1",)):
        self.calls = []
        self._caps = {
            "provider": self.name,
            "operations": list(operations),
            "families": list(families),
            "output_kinds": list(output_kinds),
            "uncertainty_methods": ["UNCALIBRATED_CLASS_PROBABILITIES"],
            "fit_required": True,
            "known_states": list(states),
            "resource_limits": {"max_batch": 8},
        }

    def capabilities(self):
        self.calls.append(("capabilities", None))
        return copy.deepcopy(self._caps)

    def load(self, state_ref):
        self.calls.append(("load", state_ref))
        if state_ref not in self._caps["known_states"]:
            raise AssertionError("load must never be reached for an unknown state")
        return {"state_ref": state_ref, "digest": "d" * 64}

    def infer(self, request, state):
        self.calls.append(("infer", request["request_id"]))
        state["mutated_by_infer"] = True          # the runtime must not hand back a mutated state
        return {"outputs": {"tone": {"status": "OK", "payload": {"label": "neutral"},
                                     "uncertainty": "UNCALIBRATED_CLASS_PROBABILITIES"}}}

    def fit(self, request):
        self.calls.append(("fit", request["request_id"]))
        return {"state_ref": "state-2", "digest": "e" * 64,
                "train_population": {"rows": 10}, "clocks": {"train_end": "2026-01-01T00:00:00Z"}}

    def calibrate(self, request, state):
        self.calls.append(("calibrate", request["request_id"]))
        return {"calibration_ref": "cal-1", "population": {"rows": 5},
                "clocks": {"calibration_end": "2026-02-01T00:00:00Z"}}

    def evaluate(self, request, state):
        self.calls.append(("evaluate", request["request_id"]))
        return {"metrics": {"mae": 0.5}, "population": {"rows": 7}}

    def called(self, what):
        return [c for c in self.calls if c[0] == what]


def base_request(**overrides):
    request = {
        "schema_version": REQUEST_SCHEMA,
        "request_id": "req-1",
        "task_id": "task-1",
        "operation": "infer",
        "family": "classification",
        "output_kind": "typed_questions",
        "state": {"asset": "EURUSD", "text": "a released statement"},
        "input_schema": {"fields": ["asset", "text"]},
        "output_schema": {"questions": ["tone"]},
        "as_of": "2026-09-24T00:00:00Z",
        "temporal_contract": {"receipt": "first_receipt", "processing_delay_seconds": 0},
        "provider_ref": "recording",
        "fitted_state_ref": "state-1",
        "execution_constraints": {"mode": "SHADOW", "partial_results": False},
        "data_refs": ["delivery-1"],
    }
    request.update(overrides)
    return request


@pytest.fixture()
def registry():
    return Registry()


# --- request validation ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("missing", ["schema_version", "request_id", "task_id", "operation", "family",
                                     "output_kind", "as_of", "provider_ref"])
def test_a_request_missing_a_required_field_is_refused(missing):
    request = base_request()
    request.pop(missing)
    with pytest.raises(RequestError) as exc:
        validate_request(request)
    assert missing in str(exc.value)


def test_an_unknown_schema_version_is_refused():
    with pytest.raises(RequestError):
        validate_request(base_request(schema_version="m5phet.task.from_the_future"))


def test_an_operation_outside_the_declared_set_is_refused():
    with pytest.raises(RequestError) as exc:
        validate_request(base_request(operation="train_a_bit"))
    assert "operation" in str(exc.value)


def test_validation_does_not_mutate_the_callers_request():
    request = base_request()
    before = copy.deepcopy(request)
    validate_request(request)
    assert request == before


# --- provider registration ------------------------------------------------------------------------------------------------

def test_a_duplicate_provider_name_is_rejected(registry):
    registry.register(RecordingProvider())
    with pytest.raises(ContractError):
        registry.register(RecordingProvider())


def test_capabilities_must_declare_the_required_facts(registry):
    class Vague:
        name = "vague"

        def capabilities(self):
            return {"provider": "vague"}

    with pytest.raises(ContractError) as exc:
        registry.register(Vague())
    assert "operations" in str(exc.value) or "families" in str(exc.value)


# --- refusal BEFORE the model is loaded -------------------------------------------------------------------------------------

def test_an_unsupported_family_refuses_before_the_model_is_loaded(registry):
    provider = RecordingProvider(families=("regression_forecasting",))
    registry.register(provider)
    result = run(base_request(), registry)
    assert result["status"] == Status.UNSUPPORTED_TASK
    assert provider.called("load") == [] and provider.called("infer") == []


def test_an_unsupported_output_kind_refuses_before_the_model_is_loaded(registry):
    provider = RecordingProvider(output_kinds=("marginal_quantiles",))
    registry.register(provider)
    result = run(base_request(), registry)
    assert result["status"] == Status.UNSUPPORTED_TASK
    assert provider.called("load") == []


def test_an_unknown_fitted_state_refuses_before_the_model_is_loaded(registry):
    provider = RecordingProvider()
    registry.register(provider)
    result = run(base_request(fitted_state_ref="state-from-another-run"), registry)
    assert result["status"] == Status.MODEL_NOT_FITTED
    assert provider.called("load") == []


def test_infer_without_a_fitted_state_is_not_implicit_training(registry):
    provider = RecordingProvider()
    registry.register(provider)
    request = base_request()
    request.pop("fitted_state_ref")
    result = run(request, registry)
    assert result["status"] == Status.MODEL_NOT_FITTED
    assert provider.called("fit") == [] and provider.called("load") == []


def test_an_operation_the_provider_does_not_offer_refuses_before_loading(registry):
    provider = RecordingProvider(operations=("infer",))
    registry.register(provider)
    result = run(base_request(operation="fit"), registry)
    assert result["status"] == Status.UNSUPPORTED_TASK
    assert provider.called("fit") == [] and provider.called("load") == []


def test_an_unregistered_provider_ref_refuses(registry):
    result = run(base_request(provider_ref="nobody"), registry)
    assert result["status"] == Status.UNSUPPORTED_TASK
    assert "nobody" in result["why"]


# --- the result envelope ----------------------------------------------------------------------------------------------------

def test_a_successful_result_binds_request_provider_and_state(registry):
    registry.register(RecordingProvider())
    result = run(base_request(), registry)
    assert result["status"] == Status.OK
    assert result["request_sha256"] and len(result["request_sha256"]) == 64
    assert result["binding"]["provider"] == "recording"
    assert result["binding"]["fitted_state_ref"] == "state-1"
    assert result["binding"]["state_digest"] == "d" * 64
    assert result["binding"]["task_id"] == "task-1" and result["binding"]["as_of"] == "2026-09-24T00:00:00Z"


def test_the_envelope_never_collapses_separate_facts_into_one_verified_flag(registry):
    registry.register(RecordingProvider())
    result = run(base_request(), registry)
    assert "verified" not in result
    facts = result["facts"]
    for key in ("schema_valid", "capability_checked", "replayed", "calibration_bound",
                "governance_accepted", "application_eligible"):
        assert key in facts
    assert facts["schema_valid"] is True
    assert facts["calibration_bound"] is False and facts["application_eligible"] is False
    assert result["execution_authorized"] is False


def test_each_output_carries_its_own_status_and_declared_uncertainty(registry):
    registry.register(RecordingProvider())
    result = run(base_request(), registry)
    tone = result["outputs"]["tone"]
    assert tone["status"] == Status.OK
    assert tone["uncertainty"] == "UNCALIBRATED_CLASS_PROBABILITIES"
    assert tone["payload"] == {"label": "neutral"}


def test_a_refused_output_carries_no_invented_score(registry):
    class Refusing(RecordingProvider):
        name = "refusing"

        def infer(self, request, state):
            self.calls.append(("infer", request["request_id"]))
            return {"outputs": {"tone": {"status": "ABSTAINED", "why": "the input was truncated"}}}

    registry.register(Refusing())
    result = run(base_request(provider_ref="refusing"), registry)
    tone = result["outputs"]["tone"]
    assert tone["status"] == Status.ABSTAINED
    assert "payload" not in tone or tone["payload"] is None
    assert result["status"] == Status.ABSTAINED


def test_an_omitted_question_never_counts_as_success(registry):
    class Partial(RecordingProvider):
        name = "partial"

        def infer(self, request, state):
            self.calls.append(("infer", request["request_id"]))
            return {"outputs": {"tone": {"status": "OK", "payload": {"label": "neutral"},
                                         "uncertainty": "UNCALIBRATED_CLASS_PROBABILITIES"}}}

    registry.register(Partial())
    request = base_request(provider_ref="partial", output_schema={"questions": ["tone", "event"]})
    result = run(request, registry)
    assert result["outputs"]["event"]["status"] == Status.INVALID_INPUT
    assert result["status"] != Status.OK
    # and when partial responses are declared allowed, the omission is still not an OK output
    allowed = run(base_request(provider_ref="partial", output_schema={"questions": ["tone", "event"]},
                               execution_constraints={"mode": "SHADOW", "partial_results": True}), registry)
    assert allowed["status"] == Status.PARTIAL
    assert allowed["outputs"]["event"]["status"] == Status.INVALID_INPUT


def test_a_provider_that_answers_a_question_nobody_asked_is_invalid(registry):
    class Extra(RecordingProvider):
        name = "extra"

        def infer(self, request, state):
            self.calls.append(("infer", request["request_id"]))
            return {"outputs": {"tone": {"status": "OK", "payload": {"label": "neutral"},
                                         "uncertainty": "UNCALIBRATED_CLASS_PROBABILITIES"},
                                "smuggled": {"status": "OK", "payload": {"label": "x"},
                                             "uncertainty": "UNCALIBRATED_CLASS_PROBABILITIES"}}}

    registry.register(Extra())
    result = run(base_request(provider_ref="extra"), registry)
    assert result["status"] == Status.INVALID_INPUT
    assert "smuggled" in result["why"]


def test_an_output_declaring_an_undeclared_uncertainty_method_is_invalid(registry):
    class Fantasy(RecordingProvider):
        name = "fantasy"

        def infer(self, request, state):
            self.calls.append(("infer", request["request_id"]))
            return {"outputs": {"tone": {"status": "OK", "payload": {"label": "neutral"},
                                         "uncertainty": "CALIBRATED_POSTERIOR"}}}

    registry.register(Fantasy())
    result = run(base_request(provider_ref="fantasy"), registry)
    assert result["status"] == Status.INVALID_INPUT
    assert "CALIBRATED_POSTERIOR" in result["why"]


# --- lifecycle boundaries ----------------------------------------------------------------------------------------------------

def test_inference_cannot_modify_the_fitted_state(registry):
    provider = RecordingProvider()
    registry.register(provider)
    first = run(base_request(), registry)
    second = run(base_request(request_id="req-2"), registry)
    assert first["binding"]["state_digest"] == second["binding"]["state_digest"]
    assert "mutated_by_infer" not in second["binding"]


def test_fit_returns_an_immutable_state_reference_with_population_and_clocks(registry):
    provider = RecordingProvider()
    registry.register(provider)
    result = run(base_request(operation="fit"), registry)
    assert result["status"] == Status.OK
    assert result["fitted_state"]["state_ref"] == "state-2"
    assert result["fitted_state"]["train_population"] == {"rows": 10}
    assert result["fitted_state"]["clocks"]["train_end"] == "2026-01-01T00:00:00Z"
    assert provider.called("infer") == []


def test_calibrate_binds_its_population_and_clocks(registry):
    registry.register(RecordingProvider())
    result = run(base_request(operation="calibrate"), registry)
    assert result["status"] == Status.OK
    assert result["calibration"]["calibration_ref"] == "cal-1"
    assert result["calibration"]["population"] == {"rows": 5}
    assert result["facts"]["calibration_bound"] is True


def test_evaluate_reports_a_population_and_does_not_authorize_execution(registry):
    registry.register(RecordingProvider())
    result = run(base_request(operation="evaluate"), registry)
    assert result["status"] == Status.OK
    assert result["evaluation"]["population"] == {"rows": 7}
    assert result["execution_authorized"] is False


def test_a_provider_that_raises_is_reported_not_swallowed(registry):
    class Broken(RecordingProvider):
        name = "broken"

        def infer(self, request, state):
            raise RuntimeError("the engine died")

    registry.register(Broken())
    result = run(base_request(provider_ref="broken"), registry)
    assert result["status"] == Status.RESOURCE_EXCEEDED or result["status"] == Status.INVALID_INPUT
    assert "the engine died" in result["why"]
    assert "outputs" not in result or not result.get("outputs")


def test_the_same_request_binds_to_the_same_digest_and_a_changed_one_does_not(registry):
    registry.register(RecordingProvider())
    a = run(base_request(), registry)
    b = run(base_request(), registry)
    c = run(base_request(as_of="2026-09-25T00:00:00Z"), registry)
    assert a["request_sha256"] == b["request_sha256"]
    assert c["request_sha256"] != a["request_sha256"]


# --- external registration through the designed entry-point group -------------------------------------------------------------

def test_entry_point_discovery_reports_a_broken_provider_and_keeps_the_others(registry, monkeypatch):
    """An external distribution owns its own dependencies, so one that cannot import must be reported by name rather than
    taking the process down or silently disappearing."""
    from m5phet import runtime as R

    class EP:
        def __init__(self, name, loader):
            self.name, self._loader = name, loader

        def load(self):
            return self._loader()

    def good():
        return RecordingProvider()

    def broken():
        raise ImportError("torch is not installed in this environment")

    monkeypatch.setattr(R, "entry_points", None, raising=False)
    monkeypatch.setattr("importlib.metadata.entry_points",
                        lambda group=None: [EP("good", good), EP("broken", broken)])
    report = registry.load_entry_points()
    assert report["registered"] == ["good"]
    assert "torch is not installed" in report["refused"]["broken"]
    assert registry.names() == ["recording"]


def test_a_second_provider_claiming_a_registered_name_is_refused_at_discovery(registry, monkeypatch):
    class EP:
        def __init__(self, name):
            self.name = name

        def load(self):
            return RecordingProvider

    registry.register(RecordingProvider())
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None: [EP("impostor")])
    report = registry.load_entry_points()
    assert report["registered"] == []
    assert "already registered" in report["refused"]["impostor"]
    assert registry.names() == ["recording"]
