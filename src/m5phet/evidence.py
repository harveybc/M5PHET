"""Local-first durable evidence for a run, with an optional governed counterpart (INT01, INT03, INT05, INT11, INT12).

The layout is the designed one: a manifest written atomically, append-only attempt and metric records, and an artifact index.
Two things this module refuses to blur. A local run is labelled LOCAL_UNGOVERNED and never manufactures a delivery receipt; a
governed run resolves its deliveries FIRST and, if one is denied, refuses before any work and leaves nothing behind. And a
metric record carries its own meaning, so a missing value is explicit rather than a zero.

Nothing here imports the optional analytics package: the embedded projection lives in `m5phet.analytics` and is a rebuildable
view over these records, never a second authoritative accounting.
"""

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from .classification import ContractError

AUTHORITY_LOCAL = "LOCAL_UNGOVERNED"
AUTHORITY_GOVERNED = "GOVERNED"

#: a metric record says what it measured, on what, in what unit, over how many rows, and whether it is defined at all
REQUIRED_METRIC_FIELDS = ("metric", "definition", "definition_version", "unit", "scale", "aggregation",
                          "task", "split", "population", "status")


class DeliveryDenied(ContractError):
    """A governed delivery that was refused. There is no local fallback and no partial run."""


def _atomic_write(path: Path, text: str) -> None:
    """A manifest is never half-written: the replacement is atomic within the same directory."""
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


class EvidenceRun:
    """One run's durable records. Every append is flushed, so an interrupted process loses at most its last line."""

    def __init__(self, directory: Path, manifest: dict):
        self.directory = Path(directory)
        self.manifest = manifest
        self.authority = manifest["authority"]
        self.identity_sha256 = manifest["identity_sha256"]
        self._attempts = self.directory / "attempts.jsonl"
        self._metrics = self.directory / "metrics.jsonl"
        self._artifacts = self.directory / "artifacts.json"
        self._open_attempts = set()

    # --- records -----------------------------------------------------------------------------------------------------
    def _append(self, path: Path, record: dict) -> None:
        with path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def start_attempt(self, *, candidate: str, parameters: dict | None = None) -> str:
        attempt_id = uuid.uuid4().hex
        self._open_attempts.add(attempt_id)
        self._append(self._attempts, {"event": "attempt_started", "attempt_id": attempt_id, "candidate": candidate,
                                      "parameters": parameters or {}, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                      "run_id": self.manifest["run_id"]})
        return attempt_id

    def finish_attempt(self, attempt_id: str, *, status: str, retry_of: str | None = None, cost: dict | None = None) -> None:
        record = {"event": "attempt_finished", "attempt_id": attempt_id, "status": status,
                  "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_id": self.manifest["run_id"]}
        if retry_of:
            record["retry_of"] = retry_of                # a retry is a NEW attempt that names the one it repeats
        if cost:
            record["cost"] = cost
        self._append(self._attempts, record)
        self._open_attempts.discard(attempt_id)

    def record_metric(self, attempt_id: str, record: dict) -> None:
        missing = [f for f in REQUIRED_METRIC_FIELDS if f not in record]
        if missing:
            raise ContractError(f"a metric record must carry {', '.join(missing)}")
        if record.get("value") is None and record.get("status") in (None, "OK"):
            raise ContractError("a metric with no value must declare a status other than OK; it is never a zero")
        self._append(self._metrics, {**record, "attempt_id": attempt_id, "run_id": self.manifest["run_id"],
                                     "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    def record_artifact(self, name: str, *, reference: str, sha256: str, bytes_: int | None = None,
                        retention: str = "UNDECLARED", available: bool = True) -> None:
        index = json.loads(self._artifacts.read_text()) if self._artifacts.is_file() else {"artifacts": {}}
        index["artifacts"][name] = {"reference": reference, "sha256": sha256, "bytes": bytes_,
                                    "retention": retention, "available": available}
        _atomic_write(self._artifacts, json.dumps(index, indent=1, sort_keys=True))

    # --- recovery ----------------------------------------------------------------------------------------------------
    def recover(self) -> dict:
        """Drop a torn trailing record and report what survived. An interrupted write never silently changes a count."""
        report = {"truncated_records_dropped": 0, "attempts_recovered": 0, "metrics_recovered": 0}
        for path, key in ((self._attempts, "attempts_recovered"), (self._metrics, "metrics_recovered")):
            if not path.is_file():
                continue
            kept, dropped = [], 0
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except ValueError:
                    dropped += 1
                    continue
                kept.append(line)
            if dropped:
                _atomic_write(path, "\n".join(kept) + ("\n" if kept else ""))
            report["truncated_records_dropped"] += dropped
            if key == "attempts_recovered":
                report[key] = len({json.loads(l)["attempt_id"] for l in kept
                                   if json.loads(l).get("event") == "attempt_started"})
            else:
                report[key] = len(kept)
        return report

    def close(self) -> dict:
        self.manifest["closed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.manifest["open_attempts_at_close"] = sorted(self._open_attempts)
        _atomic_write(self.directory / "manifest.json", json.dumps(self.manifest, indent=1, sort_keys=True, default=str))
        return self.manifest


def open_run(root: Path, *, run_id: str, task: dict, code_identity: dict, model_identity: dict | None = None,
             data_identity: dict | None = None, calibration_identity: dict | None = None,
             governed: bool = False, deliveries: dict | None = None, delivery_resolver=None,
             resume: bool = False) -> EvidenceRun:
    """Open a run directory. A governed run resolves every delivery BEFORE the directory exists, so a denial leaves nothing."""
    root = Path(root)
    directory = root / "runs" / run_id
    receipts = None
    if governed:
        if not deliveries or delivery_resolver is None:
            raise ContractError("a governed run needs its deliveries and a resolver; it may not fall back to local access")
        receipts = {}
        for name, resource in deliveries.items():
            receipts[name] = delivery_resolver(resource)          # DeliveryDenied propagates: nothing is created
            if not receipts[name]:
                raise DeliveryDenied(f"the delivery for {name} returned no receipt")
    if directory.exists() and not resume:
        raise ContractError(f"run {run_id!r} already exists; a rerun is a new run id, not an overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"task": task, "code_identity": code_identity, "model_identity": model_identity,
                "data_identity": data_identity, "calibration_identity": calibration_identity}
    manifest = {
        "schema": "m5phet.evidence.manifest.v1",
        "run_id": run_id,
        "opened_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": AUTHORITY_GOVERNED if governed else AUTHORITY_LOCAL,
        "governed": bool(governed),
        "delivery_receipts": receipts,
        "task": task,
        "identity": identity,
        "identity_sha256": _digest(identity),
        "reading": ("local records are traceability, not authenticated governance; a LOCAL_UNGOVERNED run carries no campaign "
                    "or delivery authority and cannot acquire one later"),
    }
    if resume and (directory / "manifest.json").is_file():
        manifest = json.loads((directory / "manifest.json").read_text())
    else:
        _atomic_write(directory / "manifest.json", json.dumps(manifest, indent=1, sort_keys=True, default=str))
    for name in ("attempts.jsonl", "metrics.jsonl"):
        (directory / name).touch()
    if not (directory / "artifacts.json").is_file():
        _atomic_write(directory / "artifacts.json", json.dumps({"artifacts": {}}, indent=1))
    return EvidenceRun(directory, manifest)


def import_historical(run_directory: Path, *, campaign: str, claim_governed: bool = False) -> dict:
    """INT12: a local run may be imported for later analysis, and that import creates no authority it never had."""
    if claim_governed:
        raise ContractError("a historical local import cannot be declared governed: it has no delivery authorization and no "
                            "campaign-before-work record")
    directory = Path(run_directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    return {
        "run_id": manifest["run_id"],
        "campaign": campaign,
        "authority": manifest["authority"],
        "campaign_authority": "NOT_CREATED_BY_IMPORT",
        "delivery_authorization": "NONE",
        "imported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reading": ("importing historical local results preserves their scope: it cannot retroactively create campaign-"
                    "before-work or delivery authorization, and it does not relax any sealed governed experiment"),
    }
