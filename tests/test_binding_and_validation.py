"""The 7385937 review: calibration binding, task and population identity, typed payloads, manifest integrity, task-kind gates.

Each test is one of Musashi's one-factor counterexamples expressed as the behaviour the API must have instead. The accepted
dispatch behaviour from that review is not retested here; it stays in test_specialized_providers.py.
"""

import copy
import json

import pytest

from m5phet import ContractError
from m5phet.evidence import open_run
from m5phet.runtime import REQUEST_SCHEMA, Registry, Status, run


class Provider:
    name = "probe"

    def __init__(self):
        self.calls = []
        self.loaded_task = "task-1"
        self.state_digest = "d" * 64
        self.rows = ["row-1", "row-2"]
        self.payload = {"label": "neutral", "score": 0.9}
        self.calibration = {"calibration_ref": "cal", "population": {"rows": 5, "row_ids": ["row-1", "row-2"]},
                            "clocks": {"calibration_end": "2026-01-01T00:00:00Z"},
                            "task_id": "task-1", "state_ref": "state-1", "state_digest": "d" * 64}

    def capabilities(self):
        triples = [("infer", "classification", "typed_questions"),
                   ("calibrate", "classification", "typed_questions"),
                   ("infer", "regression_forecasting", "marginal_quantiles")]
        return {"operations": sorted({t[0] for t in triples}), "families": sorted({t[1] for t in triples}),
                "output_kinds": sorted({t[2] for t in triples}),
                "uncertainty_methods": ["UNCALIBRATED_CLASS_PROBABILITIES"],
                "supported": [{"operation": o, "family": f, "output_kind": k} for o, f, k in triples],
                "known_states": ["state-1"]}

    def load(self, state_ref):
        self.calls.append(("load", state_ref))
        return {"state_ref": state_ref, "digest": self.state_digest, "task_id": self.loaded_task,
                "model_sha256": "m" * 64}

    def infer(self, request, state):
        self.calls.append(("infer", request["request_id"]))
        if request["output_kind"] == "marginal_quantiles":
            return {"outputs": {"eurusd": {"status": "OK", "uncertainty": "UNCALIBRATED_CLASS_PROBABILITIES",
                                           "payload": {"quantiles": {"0.1": 1.0, "0.5": 1.1, "0.9": 1.2},
                                                       "horizon": 6, "target": "close"}}},
                    "population": {"row_ids": list(self.rows)}}
        return {"outputs": {"tone": {"status": "OK", "uncertainty": "UNCALIBRATED_CLASS_PROBABILITIES",
                                     "payload": copy.deepcopy(self.payload)}},
                "population": {"row_ids": list(self.rows)}}

    def calibrate(self, request, state):
        self.calls.append(("calibrate", request["request_id"]))
        return copy.deepcopy(self.calibration)


def request(**over):
    base = {"schema_version": REQUEST_SCHEMA, "request_id": "r1", "task_id": "task-1", "operation": "infer",
            "family": "classification", "output_kind": "typed_questions", "as_of": "2026-09-24T00:00:00Z",
            "provider_ref": "probe", "fitted_state_ref": "state-1",
            "output_schema": {"questions": ["tone"]},
            "population": {"row_ids": ["row-1", "row-2"]},
            "execution_constraints": {"partial_results": False}}
    base.update(over)
    return base


@pytest.fixture()
def registry():
    r = Registry()
    r.provider = Provider()
    r.register(r.provider)
    return r


def test_the_positive_baseline_still_passes(registry):
    result = run(request(), registry)
    assert result["status"] == Status.OK
    assert result["binding"]["task_id"] == "task-1"
    assert result["binding"]["population_sha256"]


# --- finding 1: calibration binds actual evidence -----------------------------------------------------------------------------

@pytest.mark.parametrize("drop", ["task_id", "state_ref", "state_digest"])
def test_a_calibration_missing_a_binding_field_does_not_bind(registry, drop):
    registry.provider.calibration.pop(drop)
    result = run(request(operation="calibrate"), registry)
    assert result["facts"]["calibration_bound"] is False
    assert result["status"] == Status.CALIBRATION_UNAVAILABLE
    assert drop in (result["why"] or "")


def test_a_calibration_with_another_state_digest_does_not_bind(registry):
    registry.provider.calibration["state_digest"] = "f" * 64
    result = run(request(operation="calibrate"), registry)
    assert result["facts"]["calibration_bound"] is False
    assert "digest" in (result["why"] or "").lower()


def test_a_future_calibration_in_another_timezone_does_not_bind(registry):
    """2026-09-23T23:30:00-05:00 is 04:30Z on the 24th: four and a half hours AFTER the decision clock."""
    registry.provider.calibration["clocks"]["calibration_end"] = "2026-09-23T23:30:00-05:00"
    result = run(request(operation="calibrate", as_of="2026-09-24T00:00:00Z"), registry)
    assert result["facts"]["calibration_bound"] is False
    assert "after" in (result["why"] or "").lower()


def test_a_past_calibration_in_another_timezone_still_binds(registry):
    registry.provider.calibration["clocks"]["calibration_end"] = "2026-09-23T18:00:00-05:00"   # 23:00Z, before as_of
    result = run(request(operation="calibrate", as_of="2026-09-24T00:00:00Z"), registry)
    assert result["facts"]["calibration_bound"] is True and result["status"] == Status.OK


def test_an_unparseable_clock_does_not_bind(registry):
    registry.provider.calibration["clocks"]["calibration_end"] = "whenever"
    result = run(request(operation="calibrate"), registry)
    assert result["facts"]["calibration_bound"] is False


# --- finding 2: task and population identity are separate from provider identity ----------------------------------------------

def test_a_state_declaring_another_task_is_a_contradiction(registry):
    registry.provider.loaded_task = "different-task"
    result = run(request(), registry)
    assert result["status"] == Status.INVALID_INPUT
    assert "task" in (result["why"] or "").lower()
    assert result["binding"].get("task_id") == "task-1", "the requested task is never overwritten by the state's"


def test_a_state_that_declares_cross_task_compatibility_is_accepted_explicitly(registry):
    class Foundation(Provider):
        name = "foundation"

        def load(self, state_ref):
            self.calls.append(("load", state_ref))
            return {"state_ref": state_ref, "digest": self.state_digest, "task_id": "pretraining-corpus",
                    "compatible_task_ids": ["task-1"], "model_sha256": "m" * 64}

    registry.register(Foundation())
    result = run(request(provider_ref="foundation"), registry)
    assert result["status"] == Status.OK
    assert result["binding"]["task_compatibility"] == "DECLARED_BY_STATE"


def test_a_returned_population_that_is_not_the_requested_one_refuses(registry):
    registry.provider.rows = ["row-9"]
    result = run(request(), registry)
    assert result["status"] == Status.INVALID_INPUT
    assert "population" in (result["why"] or "").lower()


def test_the_population_identity_is_carried_and_moves_with_the_rows(registry):
    first = run(request(), registry)
    second = run(request(population={"row_ids": ["row-1"]}, request_id="r2"), registry)
    registry.provider.rows = ["row-1"]
    third = run(request(population={"row_ids": ["row-1"]}, request_id="r3"), registry)
    assert first["binding"]["population_sha256"] != second["binding"]["population_sha256"]
    assert third["status"] == Status.OK
    assert third["binding"]["population_sha256"] == second["binding"]["population_sha256"]


# --- finding 3: typed payloads, per output kind -------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_number_anywhere_in_a_payload_is_invalid(registry, bad):
    registry.provider.payload = {"label": "neutral", "score": bad}
    result = run(request(), registry)
    assert result["status"] == Status.INVALID_INPUT
    assert result["outputs"]["tone"]["status"] == Status.INVALID_INPUT
    assert "finite" in (result["why"] or "").lower()


def test_a_typed_question_payload_must_carry_a_label(registry):
    registry.provider.payload = {"score": 0.9}
    result = run(request(), registry)
    assert result["status"] == Status.INVALID_INPUT


def test_quantiles_must_be_keyed_and_monotonic(registry):
    forecast = request(family="regression_forecasting", output_kind="marginal_quantiles",
                       output_schema={"targets": ["eurusd"], "quantiles": [0.1, 0.5, 0.9], "horizons": [6]})
    assert run(forecast, registry)["status"] == Status.OK

    class Crossing(Provider):
        name = "crossing"

        def infer(self, req, state):
            return {"outputs": {"eurusd": {"status": "OK", "uncertainty": "UNCALIBRATED_CLASS_PROBABILITIES",
                                           "payload": {"quantiles": {"0.1": 1.5, "0.5": 1.1, "0.9": 1.2},
                                                       "horizon": 6, "target": "close"}}},
                    "population": {"row_ids": list(self.rows)}}

    registry.register(Crossing())
    crossed = run(dict(forecast, provider_ref="crossing", request_id="r-cross"), registry)
    assert crossed["status"] == Status.INVALID_INPUT
    assert "monotonic" in (crossed["why"] or "").lower()


# --- finding 5: the questions gate depends on the output kind -----------------------------------------------------------------

def test_a_declared_forecast_without_questions_is_not_rejected_for_lacking_them(registry):
    forecast = request(family="regression_forecasting", output_kind="marginal_quantiles",
                       output_schema={"targets": ["eurusd"], "quantiles": [0.1, 0.5, 0.9], "horizons": [6]})
    result = run(forecast, registry)
    assert result["status"] == Status.OK
    assert "questions" not in (result["why"] or "")


def test_a_classification_request_still_needs_its_questions(registry):
    result = run(request(output_schema={}), registry)
    assert result["status"] == Status.INVALID_INPUT
    assert "questions" in (result["why"] or "")
    assert registry.provider.calls == []


def test_a_forecast_without_its_own_declared_schema_refuses_before_loading(registry):
    result = run(request(family="regression_forecasting", output_kind="marginal_quantiles",
                         output_schema={}, request_id="r-bare"), registry)
    assert result["status"] == Status.INVALID_INPUT
    assert registry.provider.calls == []


# --- finding 4: manifest integrity and event-log validation on the normal path -------------------------------------------------

def test_a_modified_manifest_is_detected_even_when_the_old_digest_is_kept(tmp_path):
    run_ = open_run(tmp_path, run_id="x1", task={"task_id": "A"}, code_identity={"rev": "A"})
    run_.close()
    path = tmp_path / "runs" / "x1" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["identity"]["task"] = {"task_id": "B"}
    manifest["task"] = {"task_id": "B"}
    manifest["authority"] = "GOVERNED"
    path.write_text(json.dumps(manifest))                  # the stored digest is deliberately left untouched
    with pytest.raises(ContractError) as exc:
        open_run(tmp_path, run_id="x1", task={"task_id": "B"}, code_identity={"rev": "A"}, resume=True)
    assert "integrity" in str(exc.value).lower() or "recomputed" in str(exc.value).lower()


def test_a_manifest_whose_authority_contradicts_its_profile_is_refused(tmp_path):
    run_ = open_run(tmp_path, run_id="x2", task={"task_id": "A"}, code_identity={"rev": "A"})
    run_.close()
    path = tmp_path / "runs" / "x2" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["authority"] = "GOVERNED"                     # governed stays false: a contradiction
    path.write_text(json.dumps(manifest))
    with pytest.raises(ContractError):
        open_run(tmp_path, run_id="x2", task={"task_id": "A"}, code_identity={"rev": "A"}, resume=True)


def test_interior_corruption_is_refused_on_the_normal_path_without_calling_recover(tmp_path):
    run_ = open_run(tmp_path, run_id="x3", task={"task_id": "A"}, code_identity={"rev": "A"})
    first = run_.start_attempt(candidate="c")
    run_.finish_attempt(first, status="OK")
    path = tmp_path / "runs" / "x3" / "attempts.jsonl"
    lines = path.read_text().splitlines()
    lines.insert(1, '{"event": "attempt_started", "attempt_id": "corr')
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ContractError) as exc:
        open_run(tmp_path, run_id="x3", task={"task_id": "A"}, code_identity={"rev": "A"}, resume=True)
    assert "interior" in str(exc.value).lower()
