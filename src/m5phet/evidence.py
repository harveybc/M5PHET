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
import math
import os
import re
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


def _validate_receipt(name: str, receipt) -> dict:
    """An accepted delivery, not anything truthy. A denied or pending receipt is a denial, and a malformed one is not evidence."""
    if not isinstance(receipt, dict):
        raise DeliveryDenied(f"the delivery for {name} returned {type(receipt).__name__}, not a receipt")
    status = receipt.get("status", "ACCEPTED")
    if status != "ACCEPTED":
        raise DeliveryDenied(f"the delivery for {name} is {status!r}, not ACCEPTED")
    if not isinstance(receipt.get("delivery_id"), str) or not receipt["delivery_id"].strip():
        raise DeliveryDenied(f"the delivery for {name} carries no delivery_id")
    if not isinstance(receipt.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"]) is None:
        raise DeliveryDenied(f"the delivery for {name} carries no lowercase sha256 digest")
    return dict(receipt)


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
        self._started, self._finished, self._metric_events = {}, {}, {}
        self._rebuild_state()

    def _rebuild_state(self) -> None:
        """Durable events are the state. Anything the process forgot is read back from them, so a restart neither loses an
        unfinished attempt nor forgets which event ids have already been recorded."""
        for line in (self._attempts.read_text().splitlines() if self._attempts.is_file() else []):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue                                        # recover() decides what a broken record means
            if event.get("event") == "attempt_started":
                self._started[event["attempt_id"]] = event
            elif event.get("event") == "attempt_finished":
                self._finished[event["attempt_id"]] = event
        for line in (self._metrics.read_text().splitlines() if self._metrics.is_file() else []):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("event_id"):
                self._metric_events[record["event_id"]] = record.get("content_sha256")

    @property
    def _open_attempts(self):
        return {a for a in self._started if a not in self._finished}

    # --- records -----------------------------------------------------------------------------------------------------
    def _append(self, path: Path, record: dict) -> None:
        with path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def start_attempt(self, *, candidate: str, parameters: dict | None = None) -> str:
        attempt_id = uuid.uuid4().hex
        record = {"event": "attempt_started", "attempt_id": attempt_id, "candidate": candidate,
                  "parameters": parameters or {}, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "run_id": self.manifest["run_id"]}
        self._append(self._attempts, record)
        self._started[attempt_id] = record          # open attempts are derived from the durable events, never tracked apart
        return attempt_id

    def finish_attempt(self, attempt_id: str, *, status: str, retry_of: str | None = None, cost: dict | None = None) -> None:
        """A transition, not an append. An unknown attempt is refused, the same finish retransmitted has one effect, and a
        contradictory finish is refused rather than recorded twice."""
        if attempt_id not in self._started:
            raise ContractError(f"attempt {attempt_id!r} was never started in this run")
        previous = self._finished.get(attempt_id)
        if previous is not None:
            if previous.get("status") == status and previous.get("retry_of") == retry_of:
                return                                          # idempotent retransmission
            raise ContractError(f"attempt {attempt_id!r} is already finished as {previous.get('status')!r}; "
                                f"a contradictory finish is refused")
        if retry_of is not None and retry_of not in self._started:
            raise ContractError(f"retry_of {retry_of!r} is not an attempt of this run")
        record = {"event": "attempt_finished", "attempt_id": attempt_id, "status": status,
                  "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_id": self.manifest["run_id"]}
        if retry_of:
            record["retry_of"] = retry_of                # a retry is a NEW attempt that names the one it repeats
        if cost:
            record["cost"] = cost
        self._append(self._attempts, record)
        self._finished[attempt_id] = record

    def record_metric(self, attempt_id: str, record: dict, *, event_id: str | None = None) -> str:
        """A typed observation with a stable identity. Returns the event id; the same id with the same content is idempotent,
        and the same id with different content is a contradiction."""
        if attempt_id not in self._started:
            raise ContractError(f"attempt {attempt_id!r} was never started in this run")
        missing = [f for f in REQUIRED_METRIC_FIELDS if f not in record]
        if missing:
            raise ContractError(f"a metric record must carry {', '.join(missing)}")
        for field in ("metric", "definition", "definition_version", "unit", "scale", "aggregation", "task", "split", "status"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ContractError(f"metric field {field!r} must be a non-empty string")
        population = record.get("population")
        if isinstance(population, bool) or not isinstance(population, int) or population < 0:
            raise ContractError("population must be a non-negative integer count, and a boolean is not a count")
        value = record.get("value")
        if value is None:
            if record["status"] == "OK":
                raise ContractError("a metric with no value must declare a status other than OK; it is never a zero")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ContractError(f"metric value {value!r} must be a finite number and never a boolean")
            if record["status"] != "OK":
                raise ContractError(f"a metric with a value must not declare status {record['status']!r}")
        content = {**record, "attempt_id": attempt_id, "run_id": self.manifest["run_id"]}
        content_sha = _digest(content)
        identity = event_id or content_sha
        seen = self._metric_events.get(identity)
        if seen is not None:
            if seen == content_sha:
                return identity                                 # the same event retransmitted
            raise ContractError(f"metric event {identity!r} already recorded with other content")
        self._append(self._metrics, {**content, "event_id": identity, "content_sha256": content_sha,
                                     "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self._metric_events[identity] = content_sha
        return identity

    def record_artifact(self, name: str, *, reference: str, sha256: str, bytes_: int | None = None,
                        retention: str = "UNDECLARED", available: bool = True) -> None:
        index = json.loads(self._artifacts.read_text()) if self._artifacts.is_file() else {"artifacts": {}}
        index["artifacts"][name] = {"reference": reference, "sha256": sha256, "bytes": bytes_,
                                    "retention": retention, "available": available}
        _atomic_write(self._artifacts, json.dumps(index, indent=1, sort_keys=True))

    # --- recovery ----------------------------------------------------------------------------------------------------
    def recover(self) -> dict:
        """Only an identified TORN TRAILING append is recoverable. Interior corruption is quarantined and refused: rewriting it
        away would silently change a count. Open attempts are rebuilt from the events that survived."""
        report = {"truncated_trailing_records_dropped": 0, "interior_corruption": 0, "attempts_recovered": 0,
                  "metrics_recovered": 0, "open_attempts": []}
        problems = []
        for path, key in ((self._attempts, "attempts_recovered"), (self._metrics, "metrics_recovered")):
            if not path.is_file():
                continue
            lines = [l for l in path.read_text().splitlines() if l.strip()]
            bad = []
            for index, line in enumerate(lines):
                try:
                    json.loads(line)
                except ValueError:
                    bad.append(index)
            interior = [i for i in bad if i != len(lines) - 1]
            if interior:
                quarantine = path.with_suffix(path.suffix + ".corrupt")
                quarantine.write_text("\n".join(lines[i] for i in interior) + "\n")
                report["interior_corruption"] += len(interior)
                problems.append(f"{path.name}: {len(interior)} interior record(s) are corrupt and were quarantined to "
                                f"{quarantine.name}; the file is left exactly as found")
                continue
            if bad:
                _atomic_write(path, "\n".join(lines[:-1]) + ("\n" if len(lines) > 1 else ""))
                report["truncated_trailing_records_dropped"] += 1
                lines = lines[:-1]
            report[key] = len(lines)
        if problems:
            raise ContractError("; ".join(problems))
        self._started, self._finished, self._metric_events = {}, {}, {}
        self._rebuild_state()
        report["attempts_recovered"] = len(self._started)
        report["metrics_recovered"] = len(self._metric_events)
        report["open_attempts"] = sorted(self._open_attempts)
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
            receipts[name] = _validate_receipt(name, delivery_resolver(resource))   # denial propagates: nothing is created
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
        stored = json.loads((directory / "manifest.json").read_text())
        # RP154 (finding 4): the stored digest is RECOMPUTED from the stored identity. Comparing two stored strings cannot see
        # a manifest whose task was rewritten while its old digest was left in place.
        recomputed = _digest(stored.get("identity"))
        if recomputed != stored.get("identity_sha256"):
            raise ContractError(
                f"run {run_id!r} fails its own integrity check: the manifest's identity hashes to {recomputed[:12]} and it "
                f"stores {str(stored.get('identity_sha256'))[:12]}; the record was altered after it was written")
        expected_authority = AUTHORITY_GOVERNED if stored.get("governed") else AUTHORITY_LOCAL
        if stored.get("authority") != expected_authority:
            raise ContractError(
                f"run {run_id!r} declares authority {stored.get('authority')!r} with governed={stored.get('governed')!r}: "
                f"a profile and its authority label cannot contradict each other")
        if stored.get("identity_sha256") != manifest["identity_sha256"]:
            raise ContractError(
                f"run {run_id!r} was opened under identity {str(stored.get('identity_sha256'))[:12]} and the request asks for "
                f"{manifest['identity_sha256'][:12]}: a changed task, code, model, data or calibration identity requires a NEW run")
        if bool(stored.get("governed")) != bool(governed):
            raise ContractError(
                f"run {run_id!r} was opened under the {stored.get('authority')} profile and is being resumed as "
                f"{'GOVERNED' if governed else AUTHORITY_LOCAL}: a profile is not changed by resuming, and local history never "
                f"gains governed authority")
        manifest = stored
    else:
        _atomic_write(directory / "manifest.json", json.dumps(manifest, indent=1, sort_keys=True, default=str))
    for name in ("attempts.jsonl", "metrics.jsonl"):
        (directory / name).touch()
    _refuse_interior_corruption(directory)
    if not (directory / "artifacts.json").is_file():
        _atomic_write(directory / "artifacts.json", json.dumps({"artifacts": {}}, indent=1))
    return EvidenceRun(directory, manifest)


def _refuse_interior_corruption(directory: Path) -> None:
    """RP154 (finding 4): an unreadable record that is not the final append is refused whenever the run is opened, not only
    when a caller happens to ask for recovery. A torn tail is left for recover() to decide."""
    for name in ("attempts.jsonl", "metrics.jsonl"):
        path = directory / name
        if not path.is_file():
            continue
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        for index, line in enumerate(lines):
            try:
                json.loads(line)
            except ValueError:
                if index != len(lines) - 1:
                    raise ContractError(
                        f"{name} holds an interior record this run cannot read (line {index + 1} of {len(lines)}); it is not a "
                        f"torn final append, so the log is refused rather than silently skipped")


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
