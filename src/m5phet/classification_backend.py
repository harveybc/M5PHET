"""CB05: which classification backend is in force, decided ONCE, refused when the configuration disagrees with itself.

The incident this module exists because of. On 2026-09-28 the operator's `chat.env` set
`NEWS_SIGNAL_BACKEND="fixture"` -- read by the in-process `laya_news` provider -- and also bound
`M5PHET_CHAT_LAYA_WORKER`, which made `web.engine.Engine` send every classification to a private worker whose own
environment sets `NEWS_SIGNAL_BACKEND=laya`. Two settings, in two processes, saying opposite things about whether a
model or a declared non-model answers; and the binding won, silently, because the engine simply checked the worker
variable first. Real weights answered while the configuration said fixture. That day the silent winner happened to be
the useful one. Nothing in the code made that the direction it had to resolve in, and a reader of the configuration
could not have told which of the two would answer -- which means the receipt could not be trusted either, because it
was read from the configuration rather than from what answered.

The rule here, in one sentence: an explicit disagreement is REFUSED BY NAME, an ambiguity requires ONE explicit mode,
and nothing is ever overridden silently.

    selections                              -> outcome
    NEWS_SIGNAL_BACKEND=fixture, no worker  -> mode fixture
    NEWS_SIGNAL_BACKEND=laya, no worker     -> mode local_weights
    worker bound, backend unset             -> mode remote_worker
    NEWS_SIGNAL_BACKEND=fixture + worker    -> REFUSED  CLASSIFICATION_BACKEND_CONTRADICTION
    NEWS_SIGNAL_BACKEND=laya + worker       -> REFUSED  CLASSIFICATION_MODE_REQUIRED, until M5PHET_CLASSIFICATION_MODE
                                               names one of them

`M5PHET_CLASSIFICATION_MODE` may settle an AMBIGUITY -- two selections that both mean real weights, in two places --
and it may not settle a CONTRADICTION. A mode that overrides an explicit `fixture` is the same silent override with a
label on it, so a declared mode is checked against the selections and never replaces them.

Validation is the second half, and it is the half that makes a receipt worth reading. A resolution says what the
configuration DECLARED; `validate()` compares it with the capabilities of the path that actually answered -- the
worker's own `describe`, or the installed provider's -- and refuses a fixture serving a declared real mode, real
weights serving a declared fixture, absent weights, a checkpoint other than a pinned one, and an answer that came
from the other path entirely. What the receipt then carries is measured from the answering path, not inferred from
the environment.

Nothing in this module may carry the worker's host. The resolution publishes `worker_bound: true`; the name of the
machine stays in the operator's environment, exactly as `m5phet.config` requires of the JSON file.
"""

SCHEMA = "m5phet.classification_backend.v1"

#: the three modes. One of them is in force at any moment, and it is always the operator's declaration or an
#: unambiguous reading of it -- never this module's preference.
FIXTURE = "fixture"
LOCAL_WEIGHTS = "local_weights"
REMOTE_WORKER = "remote_worker"
MODES = (FIXTURE, LOCAL_WEIGHTS, REMOTE_WORKER)

#: the variables that select a backend. They belong to two different components -- the provider reads the first group,
#: the chat engine binds the worker -- which is exactly why nothing but this module may decide between them.
BACKEND_VARIABLE = "NEWS_SIGNAL_BACKEND"
WORKER_VARIABLE = "M5PHET_CHAT_LAYA_WORKER"
COMMAND_VARIABLE = "M5PHET_CHAT_LAYA_COMMAND"
MODE_VARIABLE = "M5PHET_CLASSIFICATION_MODE"
#: optional. The checkpoint the operator says must be serving; a different one is refused instead of answered.
CHECKPOINT_PIN_VARIABLE = "M5PHET_CLASSIFICATION_EXPECT_CHECKPOINT"

#: the backends the installed provider accepts. Kept here as well so a misspelling is refused at start-up rather than
#: at the first question, and asserted against `news_signal.provider` by the test suite where that package is present.
BACKENDS = ("laya", FIXTURE)

# --- refusal codes. Each one names a different disagreement; none of them is a repair -----------------------------------
CONTRADICTION = "CLASSIFICATION_BACKEND_CONTRADICTION"
MODE_REQUIRED = "CLASSIFICATION_MODE_REQUIRED"
MODE_UNKNOWN = "CLASSIFICATION_MODE_UNKNOWN"
MODE_CONTRADICTS = "CLASSIFICATION_MODE_CONTRADICTS_SELECTION"
BACKEND_UNKNOWN = "CLASSIFICATION_BACKEND_UNKNOWN"
WORKER_NOT_BOUND = "CLASSIFICATION_WORKER_NOT_BOUND"
WORKER_COMMAND_NOT_BOUND = "CLASSIFICATION_WORKER_COMMAND_NOT_BOUND"
WORKER_UNREACHABLE = "CLASSIFICATION_WORKER_UNREACHABLE"
BACKEND_MISMATCH = "CLASSIFICATION_BACKEND_MISMATCH"
WEIGHTS_ABSENT = "CLASSIFICATION_WEIGHTS_ABSENT"
CHECKPOINT_MISMATCH = "CLASSIFICATION_CHECKPOINT_MISMATCH"
PATH_MISMATCH = "CLASSIFICATION_PATH_MISMATCH"

#: where an answer came from. `in_process` is the provider installed in this interpreter; `remote_worker` is the
#: private worker reached over the administrator's one-shot transport.
IN_PROCESS = "in_process"

VALIDATED = "VALIDATED"


class BackendRefusal(ValueError):
    """A named refusal. `code` is the machine-readable name; the message carries the repair."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _value(environ, name):
    value = environ.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def resolve(environ, *, remote_permitted=True):
    """The mode in force, or a `BackendRefusal`.

    `remote_permitted` is False when the caller supplied its own registry -- an embedded or test engine, which never
    reaches a worker. A worker binding is then not in force, and the resolution SAYS so (`worker_binding_ignored`)
    rather than quietly dropping it: the whole point of this module is that nothing about the backend is quiet.
    """
    backend = _value(environ, BACKEND_VARIABLE)
    worker = _value(environ, WORKER_VARIABLE) if remote_permitted else None
    ignored = bool(_value(environ, WORKER_VARIABLE)) and not remote_permitted
    command = _value(environ, COMMAND_VARIABLE)
    declared = _value(environ, MODE_VARIABLE)
    pin = _value(environ, CHECKPOINT_PIN_VARIABLE)

    if backend is not None and backend not in BACKENDS:
        raise BackendRefusal(BACKEND_UNKNOWN,
                             f"{BACKEND_VARIABLE}={backend!r} is not one of {BACKENDS}. It is refused here, at "
                             f"start-up, rather than at the first question the provider is asked.")
    if declared is not None and declared not in MODES:
        raise BackendRefusal(MODE_UNKNOWN, f"{MODE_VARIABLE}={declared!r} is not one of {MODES}.")

    # 1. the contradiction. A declared non-model AND a bound real worker: no mode may settle this, because settling it
    #    means overriding something the operator wrote explicitly, which is the defect and not the fix.
    if backend == FIXTURE and worker:
        raise BackendRefusal(
            CONTRADICTION,
            f"{BACKEND_VARIABLE}=fixture declares a non-model backend while {WORKER_VARIABLE} binds a worker that "
            f"answers from real weights. These cannot both be in force and neither silently wins. Repair it in one "
            f"of two ways: to serve the fixture, unset {WORKER_VARIABLE} (and {COMMAND_VARIABLE}); to serve the "
            f"worker's weights, unset {BACKEND_VARIABLE} or set it to 'laya', and declare "
            f"{MODE_VARIABLE}=remote_worker.")

    # 2. the ambiguity. Two selections that both mean real weights, in two places. This is what one explicit mode may
    #    settle, and the only thing it may settle.
    if backend == "laya" and worker and declared is None:
        raise BackendRefusal(
            MODE_REQUIRED,
            f"{BACKEND_VARIABLE}=laya selects real weights in this process and {WORKER_VARIABLE} selects a worker's. "
            f"Which set of weights answers is not something this code may choose: declare "
            f"{MODE_VARIABLE}={REMOTE_WORKER} or {MODE_VARIABLE}={LOCAL_WEIGHTS}.")

    if declared is not None:
        mode, source = declared, "DECLARED"
        if mode == REMOTE_WORKER and not worker:
            raise BackendRefusal(WORKER_NOT_BOUND,
                                 f"{MODE_VARIABLE}={REMOTE_WORKER} declares that a private worker answers, and "
                                 f"{WORKER_VARIABLE} binds none. Nothing local stands in for it.")
        if mode in (FIXTURE, LOCAL_WEIGHTS) and worker:
            raise BackendRefusal(MODE_CONTRADICTS,
                                 f"{MODE_VARIABLE}={mode} declares that this process answers, and {WORKER_VARIABLE} "
                                 f"still binds a worker. Unset it, or declare {MODE_VARIABLE}={REMOTE_WORKER}.")
        if mode == FIXTURE and backend == "laya":
            raise BackendRefusal(MODE_CONTRADICTS,
                                 f"{MODE_VARIABLE}={FIXTURE} and {BACKEND_VARIABLE}=laya disagree about whether a "
                                 f"model answers.")
        if mode == LOCAL_WEIGHTS and backend == FIXTURE:
            raise BackendRefusal(MODE_CONTRADICTS,
                                 f"{MODE_VARIABLE}={LOCAL_WEIGHTS} and {BACKEND_VARIABLE}=fixture disagree about "
                                 f"whether a model answers.")
    elif worker:
        mode, source = REMOTE_WORKER, "WORKER_BOUND"
    elif backend == FIXTURE:
        mode, source = FIXTURE, "BACKEND_SELECTED"
    elif backend == "laya":
        mode, source = LOCAL_WEIGHTS, "BACKEND_SELECTED"
    else:
        # nothing declared at all. `news_signal.provider.configuration` defaults to 'laya', and that default is what
        # is in force -- said out loud, because a default nobody wrote is still a decision somebody will be held to.
        mode, source = LOCAL_WEIGHTS, "PROVIDER_DEFAULT"

    if mode == REMOTE_WORKER and not command:
        raise BackendRefusal(WORKER_COMMAND_NOT_BOUND,
                             f"{WORKER_VARIABLE} binds a worker and {COMMAND_VARIABLE} names no command to run on "
                             f"it. Until 2026-09-28 this was discovered at the first question, after the catalog had "
                             f"already published the locally installed provider in the worker's place.")

    return {
        "schema": SCHEMA,
        "mode": mode,
        "mode_source": source,
        "mode_declared": source == "DECLARED",
        "worker_bound": bool(worker),
        "worker_command_bound": bool(command) if worker else False,
        "worker_binding_ignored": ignored,
        "local_backend_selection": backend,
        "expects_backend": FIXTURE if mode == FIXTURE else "laya",
        "expects_weights": mode != FIXTURE,
        "expects_path": REMOTE_WORKER if mode == REMOTE_WORKER else IN_PROCESS,
        "checkpoint_pin": pin,
        "variables": {"backend": BACKEND_VARIABLE, "worker": WORKER_VARIABLE, "command": COMMAND_VARIABLE,
                      "mode": MODE_VARIABLE, "checkpoint_pin": CHECKPOINT_PIN_VARIABLE},
        "reading": ("which backend answers classification, decided from the operator's declarations alone. A "
                    "contradiction refuses by name; an ambiguity requires one explicit mode; nothing is overridden "
                    "silently. The host itself is never published here."),
    }


def validate(resolution, capabilities, *, answered_by):
    """What ACTUALLY answered, checked against what was declared. Returns the effective block, or refuses by name.

    `capabilities` is the answering path's own declaration -- the worker's `describe` response, or the installed
    provider's `capabilities()`. Nothing here reads the environment a second time: a receipt built from the
    configuration would say what the operator intended, which is the thing that went wrong.
    """
    if answered_by != resolution["expects_path"]:
        raise BackendRefusal(PATH_MISMATCH,
                             f"the declared mode {resolution['mode']} is answered by {resolution['expects_path']} "
                             f"and this answer came from {answered_by}.")
    caps = capabilities if isinstance(capabilities, dict) else {}
    backend = caps.get("backend")
    expected = resolution["expects_backend"]
    if backend != expected:
        raise BackendRefusal(BACKEND_MISMATCH,
                             f"the configuration declares backend {expected!r} and the {answered_by} path declares "
                             f"{backend!r}. A declared non-model that answers from a checkpoint, and a declared model "
                             f"that answers from a fixture, are the same defect in two directions.")
    weights = bool(caps.get("weights_present"))
    if resolution["expects_weights"] and not weights:
        raise BackendRefusal(WEIGHTS_ABSENT,
                             f"the {answered_by} path declares backend 'laya' with no weights present; a real mode is "
                             f"not served by a provider that holds no checkpoint.")
    states = caps.get("known_states") or []
    checkpoint = states[0] if states else None
    pin = resolution["checkpoint_pin"]
    if pin and checkpoint != pin:
        raise BackendRefusal(CHECKPOINT_MISMATCH,
                             f"{CHECKPOINT_PIN_VARIABLE} pins {pin!r} and the {answered_by} path serves "
                             f"{checkpoint!r}.")
    return {
        "schema": SCHEMA + ".effective",
        "mode": resolution["mode"],
        "mode_source": resolution["mode_source"],
        "answered_by": answered_by,
        "backend": backend,
        "weights_present": weights,
        "device": caps.get("device"),
        "checkpoint": checkpoint,
        "checkpoint_pinned": bool(pin),
        "status": VALIDATED,
        "validated": True,
        "reading": ("the backend and checkpoint that ANSWERED, read from the declaration of the path that answered "
                    "and checked against the configured mode; not inferred from the configuration."),
    }


def declared(capabilities, *, answered_by=IN_PROCESS):
    """The effective block when the caller SUPPLIED the provider instead of configuring it.

    `Engine(registry=...)` is an embedded or test engine: the provider was handed to it already built, from an
    environment this process never read, so there is no configuration contract to check it against. What the receipt
    can still say -- and does -- is what that provider declares it is. The block says which of the two situations it
    is in, so a reader never mistakes an unchecked declaration for a validated one.
    """
    caps = capabilities if isinstance(capabilities, dict) else {}
    states = caps.get("known_states") or []
    return {
        "schema": SCHEMA + ".effective",
        "mode": "PROVIDER_SUPPLIED",
        "mode_source": "REGISTRY_INJECTED",
        "answered_by": answered_by,
        "backend": caps.get("backend"),
        "weights_present": bool(caps.get("weights_present")),
        "device": caps.get("device"),
        "checkpoint": states[0] if states else None,
        "checkpoint_pinned": False,
        "status": "PROVIDER_SUPPLIED",
        "validated": True,
        "reading": ("the registry was supplied by the caller, so no operator configuration was resolved; the backend "
                    "and checkpoint here are the supplied provider's own declaration, unchecked against a mode."),
    }


def unavailable(resolution, code, detail):
    """The effective block when nothing may answer: the mode still stated, the refusal named, no backend invented."""
    return {
        "schema": SCHEMA + ".effective",
        "mode": resolution["mode"],
        "mode_source": resolution["mode_source"],
        "answered_by": None,
        "backend": None,
        "weights_present": False,
        "device": None,
        "checkpoint": None,
        "checkpoint_pinned": bool(resolution["checkpoint_pin"]),
        "status": code,
        "validated": False,
        "why": detail,
        "reading": ("the declared mode cannot be served. No question of this area is answered from anywhere else: "
                    "there is no path from this refusal to a local reading of the same question."),
    }
