"""INT01, INT03, INT05, INT11 and INT12: local-first durable evidence and the optional embedded projection.

Written before the implementation, from docs/INTEGRATION_AND_OPTIMIZATION.md. Each test states a behaviour: a local run works
with no services and says it is ungoverned, a denied governed delivery refuses before any work and never falls back to a local
file, an identity moves when the task moves, an interrupted write is recovered without losing or duplicating an attempt, the
analytical projection is rebuildable and agrees with the records exactly, and nothing imported later gains authority it never
had.
"""

import json
import sys
from pathlib import Path

import pytest

from m5phet import ContractError
from m5phet.evidence import (
    AUTHORITY_GOVERNED,
    AUTHORITY_LOCAL,
    DeliveryDenied,
    EvidenceRun,
    import_historical,
    open_run,
)


def metric(**over):
    record = {"metric": "mae", "definition": "mean absolute error", "definition_version": "v1",
              "unit": "z", "scale": "normalized", "aggregation": "mean", "value": 0.25,
              "task": "task-1", "split": "validation", "horizon": 96, "target": "all_channels",
              "population": 5165, "status": "OK"}
    record.update(over)
    return record


# --- INT01: a local run needs no service and says what it is ---------------------------------------------------------------

def test_INT01_a_local_run_needs_no_service_and_is_labelled_ungoverned(tmp_path):
    run = open_run(tmp_path, run_id="local-1", task={"task_id": "task-1", "family": "classification"},
                   code_identity={"tool_sha256": "a" * 64})
    assert isinstance(run, EvidenceRun)
    assert run.authority == AUTHORITY_LOCAL
    attempt = run.start_attempt(candidate="c1")
    run.record_metric(attempt, metric())
    run.finish_attempt(attempt, status="OK")
    run.close()
    manifest = json.loads((tmp_path / "runs" / "local-1" / "manifest.json").read_text())
    assert manifest["authority"] == "LOCAL_UNGOVERNED"
    assert manifest["governed"] is False and manifest.get("delivery_receipts") in (None, [])
    assert manifest["task"]["task_id"] == "task-1"
    for name in ("manifest.json", "attempts.jsonl", "metrics.jsonl", "artifacts.json"):
        assert (tmp_path / "runs" / "local-1" / name).is_file()


def test_INT01_a_metric_record_carries_its_own_meaning(tmp_path):
    run = open_run(tmp_path, run_id="local-2", task={"task_id": "t", "family": "classification"},
                   code_identity={"tool_sha256": "a" * 64})
    attempt = run.start_attempt(candidate="c1")
    with pytest.raises(ContractError) as exc:
        run.record_metric(attempt, {"metric": "mae", "value": 0.2})
    assert "unit" in str(exc.value) or "population" in str(exc.value)
    run.record_metric(attempt, metric(value=None, status="UNDEFINED"))
    run.close()
    rows = [json.loads(line) for line in (tmp_path / "runs" / "local-2" / "metrics.jsonl").read_text().splitlines()]
    assert rows[0]["value"] is None and rows[0]["status"] == "UNDEFINED", "a missing metric is explicit, never a zero"


# --- INT03: a denied governed delivery refuses before work, with no local fallback -------------------------------------------

def test_INT03_a_denied_delivery_refuses_before_any_work_and_never_falls_back(tmp_path):
    def deny(resource):
        raise DeliveryDenied(f"no authorization for {resource}")

    with pytest.raises(DeliveryDenied):
        open_run(tmp_path, run_id="gov-1", task={"task_id": "t", "family": "classification"},
                 code_identity={"tool_sha256": "a" * 64}, governed=True,
                 deliveries={"input": "lake://resource"}, delivery_resolver=deny)
    assert not (tmp_path / "runs" / "gov-1").exists(), "a refused run must not leave a started local run behind"


def test_INT03_a_governed_run_records_its_receipts_and_a_distinct_authority(tmp_path):
    run = open_run(tmp_path, run_id="gov-2", task={"task_id": "t", "family": "classification"},
                   code_identity={"tool_sha256": "a" * 64}, governed=True,
                   deliveries={"input": "lake://resource"},
                   delivery_resolver=lambda r: {"delivery_id": "d1", "sha256": "b" * 64})
    assert run.authority == AUTHORITY_GOVERNED
    run.close()
    manifest = json.loads((tmp_path / "runs" / "gov-2" / "manifest.json").read_text())
    assert manifest["authority"] == "GOVERNED" and manifest["governed"] is True
    assert manifest["delivery_receipts"]["input"]["delivery_id"] == "d1"


# --- INT05: identity moves with the task, the model, the data or the calibration ---------------------------------------------

def test_INT05_a_changed_task_model_or_data_moves_the_run_identity(tmp_path):
    counter = {"n": 0}

    def identity(**over):
        # each call is its OWN run: a rerun is a new run id, never an overwrite, which the runtime enforces separately
        task = {"task_id": "t", "family": "classification"}
        args = {"task": task, "code_identity": {"tool_sha256": "a" * 64},
                "model_identity": {"model_sha256": "m" * 64}, "data_identity": {"data_sha256": "d" * 64},
                "calibration_identity": None}
        args.update(over)
        counter["n"] += 1
        run = open_run(tmp_path, run_id=f"id-{counter['n']}", **args)
        run.close()
        return run.identity_sha256

    base = identity()
    assert identity() == base, "the same task, code, model and data give one identity under two run ids"
    assert identity(model_identity={"model_sha256": "n" * 64}) != base
    assert identity(data_identity={"data_sha256": "e" * 64}) != base
    assert identity(calibration_identity={"calibration_ref": "cal-1"}) != base
    assert identity(task={"task_id": "other", "family": "classification"}) != base


# --- INT11: interrupted writes, retries, and an analytical projection that agrees exactly -------------------------------------

def test_INT11_an_interrupted_write_is_recovered_without_losing_or_duplicating_an_attempt(tmp_path):
    run = open_run(tmp_path, run_id="crash-1", task={"task_id": "t", "family": "classification"},
                   code_identity={"tool_sha256": "a" * 64})
    first = run.start_attempt(candidate="c1")
    run.record_metric(first, metric(value=0.3))
    run.finish_attempt(first, status="OK")
    # a torn line, as an interrupted process leaves behind
    path = tmp_path / "runs" / "crash-1" / "attempts.jsonl"
    with path.open("a") as handle:
        handle.write('{"event": "attempt_started", "attempt_id": "torn')
    reopened = open_run(tmp_path, run_id="crash-1", task={"task_id": "t", "family": "classification"},
                        code_identity={"tool_sha256": "a" * 64}, resume=True)
    report = reopened.recover()
    assert report["truncated_trailing_records_dropped"] == 1 and report["interior_corruption"] == 0
    assert report["attempts_recovered"] == 1
    second = reopened.start_attempt(candidate="c1")
    assert second != first, "a retry is a new attempt, not the same one"
    reopened.finish_attempt(second, status="OK", retry_of=first)
    reopened.close()
    events = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ids = [e["attempt_id"] for e in events if e["event"] == "attempt_started"]
    assert len(ids) == len(set(ids)), "event identities must be unique"
    assert any(e.get("retry_of") == first for e in events)


def test_INT11_the_embedded_projection_is_rebuildable_and_agrees_with_the_records(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    run = open_run(tmp_path, run_id="olap-1", task={"task_id": "t", "family": "classification"},
                   code_identity={"tool_sha256": "a" * 64})
    attempt = run.start_attempt(candidate="c1")
    for value, horizon in ((0.25, 96), (0.31, 192)):
        run.record_metric(attempt, metric(value=value, horizon=horizon, population=100 + horizon))
    run.finish_attempt(attempt, status="OK")
    run.close()
    from m5phet.analytics import build_projection, read_metrics

    first = build_projection(tmp_path / "runs" / "olap-1")
    rows = read_metrics(first)
    assert {r["horizon"] for r in rows} == {96, 192}
    assert {r["unit"] for r in rows} == {"z"}
    assert {r["population"] for r in rows} == {196, 292}
    second = build_projection(tmp_path / "runs" / "olap-1", rebuild=True)
    assert read_metrics(second) == rows, "a rebuilt projection must give identical values, units and populations"


def test_INT12_the_core_path_does_not_import_the_optional_analytics_package(tmp_path, monkeypatch):
    """Core inference must not require DuckDB. Importing it is blocked and a local run still works."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] == "duckdb":
            raise ImportError("duckdb is not installed in this environment")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for module in [m for m in list(sys.modules) if m.startswith("m5phet")]:
        sys.modules.pop(module, None)
    from m5phet.evidence import open_run as fresh_open

    run = fresh_open(tmp_path, run_id="offline-1", task={"task_id": "t", "family": "classification"},
                     code_identity={"tool_sha256": "a" * 64})
    attempt = run.start_attempt(candidate="c1")
    run.record_metric(attempt, metric())
    run.finish_attempt(attempt, status="OK")
    run.close()
    assert json.loads((tmp_path / "runs" / "offline-1" / "manifest.json").read_text())["authority"] == "LOCAL_UNGOVERNED"


def test_INT12_a_historical_local_import_never_gains_retrospective_authority(tmp_path):
    run = open_run(tmp_path, run_id="hist-1", task={"task_id": "t", "family": "classification"},
                   code_identity={"tool_sha256": "a" * 64})
    run.close()
    imported = import_historical(tmp_path / "runs" / "hist-1", campaign="campaign-1")
    assert imported["authority"] == AUTHORITY_LOCAL
    assert imported["campaign_authority"] == "NOT_CREATED_BY_IMPORT"
    assert imported["delivery_authorization"] == "NONE"
    assert "cannot retroactively" in imported["reading"]
    with pytest.raises(ContractError):
        import_historical(tmp_path / "runs" / "hist-1", campaign="campaign-1", claim_governed=True)
