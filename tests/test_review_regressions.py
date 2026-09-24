"""Musashi's integration-review counterexamples, each turned into a rejection or recovery regression.

Every test below is one scenario from `docs/audits/evidence/M5PHET_INTEGRATION_REVIEW_2026_09_24/probe.py`, expressed as the
behaviour the API must have instead. An exception raised after a correct rejection is not a reason to weaken the expectation.
"""

import copy
import json

import pytest

from m5phet import ContractError
from m5phet.evidence import AUTHORITY_GOVERNED, AUTHORITY_LOCAL, DeliveryDenied, import_historical, open_run
from m5phet.runtime import REQUEST_SCHEMA, ProviderError, Registry, Status, run


def task_a():
    return {"task_id": "A", "family": "classification"}


def code():
    return {"tool_sha256": "a" * 64}


def metric(**over):
    record = {"metric": "mae", "definition": "mean absolute error", "definition_version": "v1", "unit": "z",
              "scale": "normalized", "aggregation": "mean", "value": 0.25, "task": "A", "split": "validation",
              "horizon": 96, "target": "all", "population": 100, "status": "OK"}
    record.update(over)
    return record


# --- finding 1: resume must not misattribute work or lose an unfinished attempt ---------------------------------------------

def test_resume_with_a_changed_identity_is_refused(tmp_path):
    run_ = open_run(tmp_path, run_id="r1", task=task_a(), code_identity=code())
    run_.close()
    with pytest.raises(ContractError) as exc:
        open_run(tmp_path, run_id="r1", task={"task_id": "B", "family": "classification"},
                 code_identity=code(), resume=True)
    assert "identity" in str(exc.value).lower()


def test_resume_with_the_same_identity_is_allowed_and_keeps_its_digest(tmp_path):
    first = open_run(tmp_path, run_id="r2", task=task_a(), code_identity=code())
    first.close()
    again = open_run(tmp_path, run_id="r2", task=task_a(), code_identity=code(), resume=True)
    assert again.identity_sha256 == first.identity_sha256


def test_resume_rebuilds_the_attempts_that_were_left_open(tmp_path):
    run_ = open_run(tmp_path, run_id="r3", task=task_a(), code_identity=code())
    pending = run_.start_attempt(candidate="c1")
    finished = run_.start_attempt(candidate="c2")
    run_.finish_attempt(finished, status="OK")
    reopened = open_run(tmp_path, run_id="r3", task=task_a(), code_identity=code(), resume=True)
    report = reopened.recover()
    assert report["open_attempts"] == [pending]
    reopened.close()
    manifest = json.loads((tmp_path / "runs" / "r3" / "manifest.json").read_text())
    assert manifest["open_attempts_at_close"] == [pending], "an unfinished attempt must survive a restart"


def test_interior_corruption_is_rejected_and_preserved_not_rewritten(tmp_path):
    run_ = open_run(tmp_path, run_id="r4", task=task_a(), code_identity=code())
    first = run_.start_attempt(candidate="c1")
    run_.finish_attempt(first, status="OK")
    path = tmp_path / "runs" / "r4" / "attempts.jsonl"
    lines = path.read_text().splitlines()
    lines.insert(1, '{"event": "attempt_started", "attempt_id": "corr')      # an INTERIOR broken record
    path.write_text("\n".join(lines) + "\n")
    reopened = open_run(tmp_path, run_id="r4", task=task_a(), code_identity=code(), resume=True)
    with pytest.raises(ContractError) as exc:
        reopened.recover()
    assert "interior" in str(exc.value).lower()
    quarantine = tmp_path / "runs" / "r4" / "attempts.jsonl.corrupt"
    assert quarantine.is_file() and "corr" in quarantine.read_text()
    assert path.read_text().splitlines() == lines, "the original bytes are preserved, not silently rewritten"


def test_only_an_identified_torn_trailing_append_is_recovered(tmp_path):
    run_ = open_run(tmp_path, run_id="r5", task=task_a(), code_identity=code())
    first = run_.start_attempt(candidate="c1")
    run_.finish_attempt(first, status="OK")
    path = tmp_path / "runs" / "r5" / "attempts.jsonl"
    with path.open("a") as handle:
        handle.write('{"event": "attempt_started", "attempt_id": "torn')
    reopened = open_run(tmp_path, run_id="r5", task=task_a(), code_identity=code(), resume=True)
    report = reopened.recover()
    assert report["truncated_trailing_records_dropped"] == 1 and report["interior_corruption"] == 0


# --- finding 2: governed authority is a validated contract, not truthiness ---------------------------------------------------

def test_a_denied_receipt_is_not_an_accepted_delivery(tmp_path):
    with pytest.raises(DeliveryDenied):
        open_run(tmp_path, run_id="g1", task=task_a(), code_identity=code(), governed=True,
                 deliveries={"input": "lake://r"}, delivery_resolver=lambda r: {"status": "DENIED"})
    assert not (tmp_path / "runs" / "g1").exists()


@pytest.mark.parametrize("receipt", [{"delivery_id": "d"}, {"sha256": "b" * 64}, {"delivery_id": "d", "sha256": "short"},
                                     {"delivery_id": "d", "sha256": "b" * 64, "status": "PENDING"}, "a string", 7])
def test_an_incomplete_receipt_is_refused(tmp_path, receipt):
    with pytest.raises(DeliveryDenied):
        open_run(tmp_path, run_id="g2", task=task_a(), code_identity=code(), governed=True,
                 deliveries={"input": "lake://r"}, delivery_resolver=lambda r: receipt)


def test_a_governed_run_cannot_be_resumed_as_a_local_one(tmp_path):
    good = {"delivery_id": "d1", "sha256": "b" * 64, "status": "ACCEPTED"}
    run_ = open_run(tmp_path, run_id="g3", task=task_a(), code_identity=code(), governed=True,
                    deliveries={"input": "lake://r"}, delivery_resolver=lambda r: good)
    assert run_.authority == AUTHORITY_GOVERNED
    run_.close()
    with pytest.raises(ContractError) as exc:
        open_run(tmp_path, run_id="g3", task=task_a(), code_identity=code(), resume=True)
    assert "profile" in str(exc.value).lower() or "governed" in str(exc.value).lower()


def test_a_local_run_cannot_be_resumed_into_governed_authority(tmp_path):
    run_ = open_run(tmp_path, run_id="g4", task=task_a(), code_identity=code())
    run_.close()
    with pytest.raises(ContractError):
        open_run(tmp_path, run_id="g4", task=task_a(), code_identity=code(), governed=True,
                 deliveries={"input": "lake://r"},
                 delivery_resolver=lambda r: {"delivery_id": "d", "sha256": "b" * 64, "status": "ACCEPTED"},
                 resume=True)
    assert json.loads((tmp_path / "runs" / "g4" / "manifest.json").read_text())["authority"] == AUTHORITY_LOCAL


# --- finding 4: typed, identified, idempotent records -----------------------------------------------------------------------

def test_finishing_an_unknown_attempt_is_refused(tmp_path):
    run_ = open_run(tmp_path, run_id="m1", task=task_a(), code_identity=code())
    with pytest.raises(ContractError):
        run_.finish_attempt("never-started", status="OK")


def test_an_attempt_cannot_be_finished_twice_with_a_different_status(tmp_path):
    run_ = open_run(tmp_path, run_id="m2", task=task_a(), code_identity=code())
    attempt = run_.start_attempt(candidate="c1")
    run_.finish_attempt(attempt, status="OK")
    run_.finish_attempt(attempt, status="OK")                    # the same event retransmitted: one effect
    with pytest.raises(ContractError):
        run_.finish_attempt(attempt, status="FAILED")            # a contradictory finish is refused
    events = [json.loads(l) for l in (tmp_path / "runs" / "m2" / "attempts.jsonl").read_text().splitlines() if l.strip()]
    finished = [e for e in events if e["event"] == "attempt_finished"]
    assert len(finished) == 1, "an idempotent retransmission must not double the record"


@pytest.mark.parametrize("bad", [{"value": float("nan")}, {"value": float("inf")}, {"value": True},
                                 {"population": True}, {"population": -1}, {"population": 1.5},
                                 {"value": None, "status": "OK"}, {"unit": ""}, {"definition_version": None}])
def test_an_invalid_metric_value_or_population_is_refused(tmp_path, bad):
    run_ = open_run(tmp_path, run_id=f"m3-{abs(hash(json.dumps(bad, default=str)))}", task=task_a(), code_identity=code())
    attempt = run_.start_attempt(candidate="c1")
    with pytest.raises(ContractError):
        run_.record_metric(attempt, metric(**bad))


def test_a_metric_event_has_a_stable_identity_and_is_idempotent(tmp_path):
    run_ = open_run(tmp_path, run_id="m4", task=task_a(), code_identity=code())
    attempt = run_.start_attempt(candidate="c1")
    first = run_.record_metric(attempt, metric(), event_id="e1")
    again = run_.record_metric(attempt, metric(), event_id="e1")      # retransmission of the same event
    assert first == again
    rows = [json.loads(l) for l in (tmp_path / "runs" / "m4" / "metrics.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == 1, "one event id is one record"
    with pytest.raises(ContractError):
        run_.record_metric(attempt, metric(value=0.9), event_id="e1")  # same id, different content
    other = run_.record_metric(attempt, metric(value=0.9))
    assert other != first, "a genuinely different observation is its own event"


def test_a_metric_for_a_foreign_attempt_is_refused(tmp_path):
    run_ = open_run(tmp_path, run_id="m5", task=task_a(), code_identity=code())
    with pytest.raises(ContractError):
        run_.record_metric("not-an-attempt", metric())


# --- finding 3: the runtime certifies only what it checked -------------------------------------------------------------------

class Recording:
    name = "recording"

    def __init__(self, **over):
        self.calls = []
        self._caps = {"provider": "recording", "operations": ["infer", "fit", "calibrate", "evaluate"],
                      "families": ["classification"], "output_kinds": ["typed_questions"],
                      "uncertainty_methods": ["UNCALIBRATED_CLASS_PROBABILITIES"], "known_states": ["state-1"],
                      "supported": [{"operation": o, "family": "classification", "output_kind": "typed_questions"}
                                    for o in ("infer", "fit", "calibrate", "evaluate")]}
        self._caps.update(over)
        self._infer = {"outputs": {"tone": {"status": "OK", "payload": {"label": "n"},
                                            "uncertainty": "UNCALIBRATED_CLASS_PROBABILITIES"}}}

    def capabilities(self):
        return copy.deepcopy(self._caps)

    def load(self, state_ref):
        self.calls.append("load")
        return {"state_ref": state_ref, "digest": "d" * 64, "task_id": "task-1"}

    def infer(self, request, state):
        self.calls.append("infer")
        return copy.deepcopy(self._infer)

    def calibrate(self, request, state):
        self.calls.append("calibrate")
        return {"calibration_ref": "cal-1", "population": {"rows": 5}, "clocks": {"calibration_end": "2026-02-01T00:00:00Z"},
                "task_id": "another-task", "state_ref": "state-from-elsewhere"}

    def fit(self, request):
        return {"state_ref": "s2", "digest": "e" * 64, "train_population": {"rows": 1}, "clocks": {"train_end": "x"}}

    def evaluate(self, request, state):
        return {"metrics": {}, "population": {"rows": 1}}


def request(**over):
    base = {"schema_version": REQUEST_SCHEMA, "request_id": "r", "task_id": "task-1", "operation": "infer",
            "family": "classification", "output_kind": "typed_questions", "as_of": "2026-09-24T00:00:00Z",
            "provider_ref": "recording", "fitted_state_ref": "state-1",
            "output_schema": {"questions": ["tone"]}, "execution_constraints": {"partial_results": False}}
    base.update(over)
    return base


def test_an_ok_answer_without_a_payload_is_invalid():
    registry, provider = Registry(), Recording()
    provider._infer = {"outputs": {"tone": {"status": "OK", "payload": None,
                                            "uncertainty": "UNCALIBRATED_CLASS_PROBABILITIES"}}}
    registry.register(provider)
    result = run(request(), registry)
    assert result["status"] == Status.INVALID_INPUT
    assert result["outputs"]["tone"]["status"] == Status.INVALID_INPUT


def test_a_request_without_questions_refuses_before_load_and_infer():
    registry, provider = Registry(), Recording()
    registry.register(provider)
    result = run(request(output_schema={}), registry)
    assert result["status"] == Status.INVALID_INPUT
    assert provider.calls == [], "no work may happen before an operation-specific refusal"


def test_a_foreign_or_future_calibration_does_not_bind():
    registry, provider = Registry(), Recording()
    registry.register(provider)
    result = run(request(operation="calibrate", as_of="2026-01-01T00:00:00Z"), registry)
    assert result["facts"]["calibration_bound"] is False
    assert result["status"] in (Status.CALIBRATION_UNAVAILABLE, Status.INVALID_INPUT)
    assert "task" in (result["why"] or "").lower() or "state" in (result["why"] or "").lower()


def test_a_provider_exception_keeps_its_class():
    class Boom(Recording):
        name = "boom"

        def infer(self, req, state):
            raise ValueError("a malformed tensor")

    registry = Registry(); registry.register(Boom())
    result = run(request(provider_ref="boom"), registry)
    assert result["status"] == Status.PROVIDER_ERROR
    assert result["provider_exception"]["type"] == "ValueError"
    assert "a malformed tensor" in result["provider_exception"]["message"]
    assert result["status"] != Status.RESOURCE_EXCEEDED


# --- finding 6: a class entry point loads -------------------------------------------------------------------------------------

def test_a_class_entry_point_is_instantiated(monkeypatch):
    class EP:
        name = "probe"

        def load(self):
            return Recording

    registry = Registry()
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None: [EP()])
    report = registry.load_entry_points()
    assert report["registered"] == ["probe"], report["refused"]
    assert registry.names() == ["recording"]
