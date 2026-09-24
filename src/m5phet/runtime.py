"""P02: the provider lifecycle and the request/result envelope.

The runtime owns three responsibilities and no more. It validates a versioned request. It checks a provider's DECLARED
capabilities and refuses an unsupported task, an operation the provider does not offer or an unknown fitted state BEFORE any
model is loaded or any work is done. And it binds what came back to the request, the provider and the fitted state, keeping
the separate facts separate.

What it deliberately does not do: train implicitly, invent a number for a refused output, authorize execution, or let one
boolean stand for schema validity, replay, calibration, governance and application eligibility at once.
"""

import copy
import hashlib
import json

from .classification import ContractError

REQUEST_SCHEMA = "m5phet.task.draft2"

REQUIRED_REQUEST_FIELDS = ("schema_version", "request_id", "task_id", "operation", "family",
                           "output_kind", "as_of", "provider_ref")

OPERATIONS = ("fit", "calibrate", "infer", "evaluate", "identify")

REQUIRED_CAPABILITY_FIELDS = ("operations", "families", "output_kinds", "uncertainty_methods")

#: the keys that identify one DECLARED, TESTED combination. Support is a list of these, never the Cartesian product of the
#: independent lists above: a provider that classifies typed questions and forecasts quantiles has not thereby declared that it
#: emits quantiles for a classification task.
COMBINATION_KEYS = ("operation", "family", "output_kind")

#: the designed entry-point group. An external distribution owns its own backend dependencies; this package requires none of
#: them, so discovery must survive a provider whose import fails and must never load a second provider under a taken name.
ENTRY_POINT_GROUP = "m5phet.providers"

#: operations that consume a fitted model and therefore require a state reference the provider recognises
STATE_CONSUMING = ("infer", "calibrate", "evaluate")


class Status:
    """Distinct outcomes. A refusal names which one it is; nothing collapses into a generic failure."""

    OK = "OK"
    PARTIAL = "PARTIAL"
    ABSTAINED = "ABSTAINED"
    UNSUPPORTED_TASK = "UNSUPPORTED_TASK"
    INPUT_UNAVAILABLE = "INPUT_UNAVAILABLE"
    INVALID_INPUT = "INVALID_INPUT"
    MODEL_NOT_FITTED = "MODEL_NOT_FITTED"
    CALIBRATION_UNAVAILABLE = "CALIBRATION_UNAVAILABLE"
    NOT_IDENTIFIED = "NOT_IDENTIFIED"
    RESOURCE_EXCEEDED = "RESOURCE_EXCEEDED"
    PROVIDER_ERROR = "PROVIDER_ERROR"


class ProviderError(ContractError):
    """A provider that failed. Its real exception class and message are carried through, never renamed to exhaustion."""


class RequestError(ContractError):
    """A request that cannot be read as a task at all. Raised before a provider is consulted."""


def validate_request(request) -> dict:
    """Return a validated deep copy. The caller's object is never touched, so a validator cannot quietly fill a field in."""
    if not isinstance(request, dict):
        raise RequestError("a request must be a mapping")
    checked = copy.deepcopy(request)
    missing = [f for f in REQUIRED_REQUEST_FIELDS if f not in checked or checked[f] in (None, "")]
    if missing:
        raise RequestError(f"missing required request field(s): {', '.join(missing)}")
    if checked["schema_version"] != REQUEST_SCHEMA:
        raise RequestError(f"schema_version {checked['schema_version']!r} is not {REQUEST_SCHEMA!r}")
    if checked["operation"] not in OPERATIONS:
        raise RequestError(f"operation {checked['operation']!r} is not one of {OPERATIONS}")
    for field in ("request_id", "task_id", "family", "output_kind", "as_of", "provider_ref"):
        if not isinstance(checked[field], str) or not checked[field].strip():
            raise RequestError(f"{field} must be a non-empty string")
    constraints = checked.get("execution_constraints") or {}
    if not isinstance(constraints, dict):
        raise RequestError("execution_constraints must be a mapping when present")
    if "partial_results" in constraints and not isinstance(constraints["partial_results"], bool):
        raise RequestError("execution_constraints.partial_results must be a boolean")
    return checked


def request_digest(request: dict) -> str:
    """The identity a result binds to: the whole request, canonically serialised."""
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class Registry:
    """The providers this process may use. Registration is the only way in, and a name is claimed once."""

    def __init__(self):
        self._providers = {}
        self._capabilities = {}

    def register(self, provider) -> dict:
        name = getattr(provider, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ContractError("a provider must declare a non-empty name")
        if name in self._providers:
            raise ContractError(f"provider {name!r} is already registered; duplicate registrations are rejected")
        if not callable(getattr(provider, "capabilities", None)):
            raise ContractError(f"provider {name!r} declares no capabilities()")
        caps = provider.capabilities()
        if not isinstance(caps, dict):
            raise ContractError(f"provider {name!r}: capabilities() must return a mapping")
        absent = [f for f in REQUIRED_CAPABILITY_FIELDS if not caps.get(f)]
        if absent:
            raise ContractError(f"provider {name!r}: capabilities() must declare {', '.join(absent)}")
        for field in REQUIRED_CAPABILITY_FIELDS:
            if not isinstance(caps[field], (list, tuple)) or not all(isinstance(v, str) for v in caps[field]):
                raise ContractError(f"provider {name!r}: capabilities()[{field!r}] must be a list of names")
        supported = caps.get("supported")
        if not isinstance(supported, (list, tuple)) or not supported:
            raise ContractError(f"provider {name!r}: capabilities() must declare `supported`, the list of tested "
                                f"operation/family/output_kind combinations; support is never inferred from a product")
        for entry in supported:
            if not isinstance(entry, dict) or any(not isinstance(entry.get(k), str) or not entry.get(k) for k in COMBINATION_KEYS):
                raise ContractError(f"provider {name!r}: each `supported` entry must name {', '.join(COMBINATION_KEYS)}")
        self._providers[name] = provider
        self._capabilities[name] = copy.deepcopy(caps)
        return copy.deepcopy(caps)

    def get(self, name):
        return self._providers.get(name)

    def capabilities(self, name):
        return copy.deepcopy(self._capabilities.get(name)) if name in self._capabilities else None

    def names(self):
        return sorted(self._providers)

    def load_entry_points(self, group: str = ENTRY_POINT_GROUP) -> dict:
        """Discover external providers through the designed group. A provider whose import or registration fails is REPORTED
        with its reason and does not stop the others; a duplicate name is refused rather than silently replacing the incumbent."""
        from importlib.metadata import entry_points
        report = {"group": group, "registered": [], "refused": {}}
        try:
            found = list(entry_points(group=group))
        except Exception as exc:                                    # noqa: BLE001
            report["refused"]["<discovery>"] = f"entry point discovery failed: {exc}"
            return report
        for ep in found:
            try:
                loaded = ep.load()
                # a class exposes capabilities as an UNBOUND function, so `hasattr` cannot tell a class from an instance
                provider = loaded() if isinstance(loaded, type) else (loaded() if callable(loaded) and not hasattr(loaded, "capabilities") else loaded)
                self.register(provider)
                report["registered"].append(ep.name)
            except Exception as exc:                                # noqa: BLE001
                report["refused"][ep.name] = str(exc)
        return report


def _envelope(request, status, why=None, **extra) -> dict:
    """Every result carries the same separate facts; none of them is a single VERIFIED boolean."""
    out = {
        "schema": "m5phet.result.draft2",
        "status": status,
        "why": why,
        "request_sha256": request_digest(request) if isinstance(request, dict) else None,
        "execution_authorized": False,
        "facts": {
            "schema_valid": extra.pop("schema_valid", False),
            "capability_checked": extra.pop("capability_checked", False),
            "replayed": False,
            "calibration_bound": extra.pop("calibration_bound", False),
            "governance_accepted": False,
            "application_eligible": False,
        },
    }
    out.update(extra)
    return out


def _check_outputs(requested, returned, caps):
    """Per-output status against what was asked for. An omitted question is never a success and an unasked answer is invalid."""
    outputs, problems = {}, []
    returned = returned if isinstance(returned, dict) else {}
    extra = sorted(set(returned) - set(requested))
    if extra:
        problems.append(f"the provider answered question(s) nobody asked: {', '.join(extra)}")
    declared_uncertainty = set(caps.get("uncertainty_methods") or ())
    for question in requested:
        answer = returned.get(question)
        if not isinstance(answer, dict):
            outputs[question] = {"status": Status.INVALID_INPUT,
                                 "why": "the provider returned no answer for this question"}
            continue
        status = answer.get("status")
        if status not in vars(Status).values():
            outputs[question] = {"status": Status.INVALID_INPUT, "why": f"unknown output status {status!r}"}
            problems.append(f"{question}: unknown output status {status!r}")
            continue
        entry = {"status": status, "why": answer.get("why")}
        if status == Status.OK:
            uncertainty = answer.get("uncertainty")
            if uncertainty not in declared_uncertainty:
                entry = {"status": Status.INVALID_INPUT,
                         "why": f"uncertainty {uncertainty!r} is not declared by this provider"}
                problems.append(f"{question}: uncertainty {uncertainty!r} is not declared by this provider")
            elif answer.get("payload") is None:
                entry = {"status": Status.INVALID_INPUT,
                         "why": "an OK answer must carry a payload; a null payload certifies nothing"}
                problems.append(f"{question}: an OK answer with a null payload")
            else:
                entry["uncertainty"] = uncertainty
                entry["payload"] = answer.get("payload")
        else:
            # a refused output carries no number: whatever the provider put in `payload` is dropped here
            entry["payload"] = None
        outputs[question] = entry
    return outputs, problems


def _overall(outputs, problems, partial_allowed):
    if problems:
        return Status.INVALID_INPUT
    statuses = {q: o["status"] for q, o in outputs.items()}
    if not statuses:
        return Status.INVALID_INPUT
    if all(s == Status.OK for s in statuses.values()):
        return Status.OK
    if all(s == Status.ABSTAINED for s in statuses.values()):
        return Status.ABSTAINED
    if partial_allowed and any(s == Status.OK for s in statuses.values()):
        return Status.PARTIAL
    return Status.INVALID_INPUT


def run(request, registry: Registry) -> dict:
    """Validate, check capabilities, refuse early, then dispatch. Nothing below loads a model before the checks above pass."""
    try:
        checked = validate_request(request)
    except RequestError as exc:
        return _envelope(request if isinstance(request, dict) else {}, Status.INVALID_INPUT, str(exc))
    provider = registry.get(checked["provider_ref"])
    if provider is None:
        return _envelope(checked, Status.UNSUPPORTED_TASK,
                         f"provider {checked['provider_ref']!r} is not registered", schema_valid=True)
    caps = registry.capabilities(checked["provider_ref"]) or {}
    operation = checked["operation"]
    for field, declared in (("operation", "operations"), ("family", "families"), ("output_kind", "output_kinds")):
        if checked[field] not in (caps.get(declared) or ()):
            return _envelope(checked, Status.UNSUPPORTED_TASK,
                             f"provider {checked['provider_ref']!r} does not declare {field} {checked[field]!r}",
                             schema_valid=True, capability_checked=True)
    wanted = {k: checked[k] for k in COMBINATION_KEYS}
    if not any(all(entry.get(k) == wanted[k] for k in COMBINATION_KEYS) for entry in (caps.get("supported") or ())):
        return _envelope(checked, Status.UNSUPPORTED_TASK,
                         f"provider {checked['provider_ref']!r} declares each of these separately but not the combination "
                         f"{wanted}; support is a tested combination, not a product of capability lists",
                         schema_valid=True, capability_checked=True)
    if operation == "infer" and not ((checked.get("output_schema") or {}).get("questions")):
        return _envelope(checked, Status.INVALID_INPUT,
                         "an infer request must declare the questions to answer; nothing is loaded for an unanswerable request",
                         schema_valid=True, capability_checked=True)
    state_ref = checked.get("fitted_state_ref")
    if operation in STATE_CONSUMING:
        if not state_ref:
            return _envelope(checked, Status.MODEL_NOT_FITTED,
                             f"{operation} needs a fitted_state_ref; this runtime never trains implicitly",
                             schema_valid=True, capability_checked=True)
        known = caps.get("known_states")
        if known is not None and state_ref not in known:
            return _envelope(checked, Status.MODEL_NOT_FITTED,
                             f"fitted_state_ref {state_ref!r} is not a state this provider holds",
                             schema_valid=True, capability_checked=True)
    # every refusal above happened without loading anything
    state, binding = None, {"provider": checked["provider_ref"], "task_id": checked["task_id"],
                            "as_of": checked["as_of"], "fitted_state_ref": state_ref}
    try:
        if operation in STATE_CONSUMING:
            state = provider.load(state_ref)
            if not isinstance(state, dict) or not state.get("digest"):
                return _envelope(checked, Status.INVALID_INPUT, "the provider's loaded state declares no digest",
                                 schema_valid=True, capability_checked=True)
            binding["state_digest"] = state["digest"]
            if state.get("model_sha256"):
                binding["model_sha256"] = state["model_sha256"]     # a provider swap must show a changed model identity
        if operation == "infer":
            returned = provider.infer(copy.deepcopy(checked), copy.deepcopy(state))
            requested = list((checked.get("output_schema") or {}).get("questions") or [])
            outputs, problems = _check_outputs(requested, (returned or {}).get("outputs"), caps)
            partial_allowed = bool((checked.get("execution_constraints") or {}).get("partial_results"))
            status = _overall(outputs, problems, partial_allowed)
            return _envelope(checked, status, "; ".join(problems) or None, schema_valid=True, capability_checked=True,
                             binding=binding, outputs=outputs)
        if operation == "fit":
            fitted = provider.fit(copy.deepcopy(checked))
            for field in ("state_ref", "train_population", "clocks"):
                if not (isinstance(fitted, dict) and fitted.get(field)):
                    return _envelope(checked, Status.INVALID_INPUT, f"fit returned no {field}",
                                     schema_valid=True, capability_checked=True, binding=binding)
            return _envelope(checked, Status.OK, None, schema_valid=True, capability_checked=True,
                             binding=binding, fitted_state=copy.deepcopy(fitted))
        if operation == "calibrate":
            calibration = provider.calibrate(copy.deepcopy(checked), copy.deepcopy(state))
            for field in ("calibration_ref", "population", "clocks"):
                if not (isinstance(calibration, dict) and calibration.get(field)):
                    return _envelope(checked, Status.CALIBRATION_UNAVAILABLE, f"calibrate returned no {field}",
                                     schema_valid=True, capability_checked=True, binding=binding)
            # a calibration binds THIS task, THIS fitted state and a clock at or before the decision time
            mismatch = []
            if calibration.get("task_id") not in (None, checked["task_id"]):
                mismatch.append(f"it was calibrated for task {calibration['task_id']!r}, not {checked['task_id']!r}")
            if calibration.get("state_ref") not in (None, state_ref):
                mismatch.append(f"it binds state {calibration['state_ref']!r}, not {state_ref!r}")
            end = (calibration.get("clocks") or {}).get("calibration_end")
            if isinstance(end, str) and end > checked["as_of"]:
                mismatch.append(f"its calibration_end {end} is after the decision clock {checked['as_of']}")
            if mismatch:
                return _envelope(checked, Status.CALIBRATION_UNAVAILABLE, "; ".join(mismatch),
                                 schema_valid=True, capability_checked=True, binding=binding,
                                 calibration=copy.deepcopy(calibration))
            return _envelope(checked, Status.OK, None, schema_valid=True, capability_checked=True,
                             calibration_bound=True, binding=binding, calibration=copy.deepcopy(calibration))
        if operation == "evaluate":
            evaluation = provider.evaluate(copy.deepcopy(checked), copy.deepcopy(state))
            if not (isinstance(evaluation, dict) and evaluation.get("population")):
                return _envelope(checked, Status.INVALID_INPUT, "evaluate returned no population",
                                 schema_valid=True, capability_checked=True, binding=binding)
            return _envelope(checked, Status.OK, None, schema_valid=True, capability_checked=True,
                             binding=binding, evaluation=copy.deepcopy(evaluation))
        return _envelope(checked, Status.NOT_IDENTIFIED,
                         f"operation {operation!r} has no implemented path in this runtime",
                         schema_valid=True, capability_checked=True, binding=binding)
    except Exception as exc:                                        # noqa: BLE001
        # a provider that fails is reported with what it said AND with its real class; nothing is renamed to exhaustion
        return _envelope(checked, Status.PROVIDER_ERROR, f"the provider raised: {exc}",
                         schema_valid=True, capability_checked=True, binding=binding,
                         provider_exception={"type": type(exc).__name__, "message": str(exc)})
