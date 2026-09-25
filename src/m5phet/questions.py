"""One envelope for every area: a state, named typed questions in; named typed answers out.

This is the shape the owner asked for, and it is Laya's shape generalised. A caller does not learn five request formats. It
names an AREA, describes the STATE it is asking about -- a dataset, a fitted model, an observation -- and asks any number of
QUESTIONS, each with a name and a type. Each question comes back under its own name with its own type, answered or refused.

Two rules make the envelope worth trusting.

Every question is answered separately. A request asking for a point forecast and an interval gets the point forecast from an
engine that has one and an explicit refusal for the interval from an engine that does not. The alternative -- refusing the
whole request, or worse, inventing the interval -- would hide which half the engine can actually do.

A refusal is typed and carries its reason. There is no path from "the engine cannot compute this" to a number. `NOT_ESTIMABLE`
for a counterfactual the study cannot condition on, `UNSUPPORTED_QUESTION_TYPE` for a type the area does not declare, and so
on. The names are fixed here so a caller can match on them, and every one of them says why.

Providers take part by declaring `question_types()` -- the types they answer and the fields each needs -- and implementing
`answer_questions(state, questions, data, as_of)`. Unknown types are refused before the provider is reached; declared types
reach it, and it may still refuse any one of them by name.
"""

import copy
import hashlib
import json
import time

TASK_SCHEMA = "m5phet.task.questions.v1"
ANSWERS_SCHEMA = "m5phet.answers.v1"

AREAS = ("classification", "forecasting", "causal", "rl", "unsupervised")

#: refusal codes a caller may match on. Each is a statement about the engine or the request, never about the answer.
UNSUPPORTED_AREA = "UNSUPPORTED_AREA"
NO_PROVIDER = "NO_PROVIDER_FOR_AREA"
UNSUPPORTED_QUESTION_TYPE = "UNSUPPORTED_QUESTION_TYPE"
MALFORMED_QUESTION = "MALFORMED_QUESTION"
MISSING_FIELD = "MISSING_QUESTION_FIELD"
NOT_ESTIMABLE = "NOT_ESTIMABLE"
STATE_REQUIRED = "STATE_REQUIRED"
PROVIDER_ERROR = "PROVIDER_ERROR"


class TaskError(ValueError):
    """The envelope itself cannot be read as a task. Raised before any provider is consulted."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def refusal(kind, why, question_type=None):
    """A typed refusal. It carries no number and cannot be mistaken for an answer that carries one."""
    out = {"status": "REFUSED", "refusal": kind, "why": why}
    if question_type is not None:
        out["type"] = question_type
    return out


def validate_task(payload):
    """Return a validated deep copy of the envelope, or raise TaskError naming what is wrong with it."""
    if not isinstance(payload, dict):
        raise TaskError("a task must be a mapping")
    task = copy.deepcopy(payload)
    if task.get("schema", TASK_SCHEMA) != TASK_SCHEMA:
        raise TaskError(f"schema {task.get('schema')!r} is not {TASK_SCHEMA!r}")
    task["schema"] = TASK_SCHEMA
    area = task.get("area")
    if area not in AREAS:
        raise TaskError(f"{UNSUPPORTED_AREA}: {area!r} is not one of {list(AREAS)}")
    state = task.get("state")
    if not isinstance(state, dict):
        raise TaskError(f"{STATE_REQUIRED}: `state` must be a mapping describing what is being asked about")
    questions = task.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise TaskError("`questions` must be a non-empty mapping of name -> question")
    for name, question in questions.items():
        if not isinstance(name, str) or not name.strip():
            raise TaskError("every question needs a non-empty name")
        if not isinstance(question, dict) or not isinstance(question.get("type"), str) or not question["type"].strip():
            raise TaskError(f"{MALFORMED_QUESTION}: question {name!r} must be a mapping with a string `type`")
    if "as_of" in task and task["as_of"] is not None and not isinstance(task["as_of"], str):
        raise TaskError("`as_of` must be an ISO-8601 string when present")
    return task


def provider_for(registry, area):
    """The registered provider that declares this area, or None. A provider declares one area; two claiming the same one is
    a registration mistake and is reported rather than resolved by picking the first."""
    found = []
    for name in registry.names():
        provider = registry.get(name)
        declared = getattr(provider, "area", None)
        if declared == area:
            found.append(provider)
    if len(found) > 1:
        raise TaskError(f"{len(found)} providers declare area {area!r}: {[p.name for p in found]}; one area, one provider")
    return found[0] if found else None


def declared_types(provider):
    """The question types a provider answers, with the fields each requires. Missing declaration means none."""
    method = getattr(provider, "question_types", None)
    if not callable(method):
        return {}
    declared = method()
    if not isinstance(declared, dict):
        raise TaskError(f"provider {provider.name!r}: question_types() must return a mapping")
    return declared


def _check_fields(question, spec):
    required = spec.get("required") or []
    missing = [f for f in required if f not in question]
    if missing:
        return f"{MISSING_FIELD}: question type {question['type']!r} needs {missing}"
    allowed = set(required) | set(spec.get("optional") or []) | {"type", "instructions"}
    extra = sorted(set(question) - allowed)
    if extra:
        return f"{MALFORMED_QUESTION}: question type {question['type']!r} does not take {extra}"
    return None


def area_quality(area, provider, configuration=None, environ=None):
    """WP31: what is known about how well this area answers, for the answer to carry.

    Read, never computed: the provider's own `capabilities()['quality']` for classification, the evaluation report an
    operator declared for forecasting and unsupervised, and the refusal `m5phet_evaluation` owns for causal and rl.
    A provider whose capabilities cannot be read is not a reason to publish nothing: the area still says
    NOT_MEASURED, which is the same statement with a different cause and is recorded as such."""
    from . import quality as quality_module
    capabilities = None
    if provider is not None and callable(getattr(provider, "capabilities", None)):
        try:
            capabilities = provider.capabilities()
        except Exception:                                               # noqa: BLE001
            capabilities = None
    return quality_module.for_area(area, capabilities, configuration, environ)


def run_task(payload, registry, *, data=None, configuration=None, environ=None):
    """Answer every question of one envelope, each on its own. Nothing here invents a value for a refused question."""
    started = time.perf_counter()
    task = validate_task(payload)
    area = task["area"]
    response = {"schema": ANSWERS_SCHEMA, "id": "task:" + digest(task)[:24], "area": area,
                "request_sha256": digest(task), "answers": {}, "provider": None, "state_ref": None,
                "execution_authorized": False, "profile": "LOCAL_UNGOVERNED"}
    provider = provider_for(registry, area)
    if provider is None:
        for name, question in task["questions"].items():
            response["answers"][name] = refusal(NO_PROVIDER, f"no installed provider declares area {area!r}",
                                                question["type"])
        response["quality"] = area_quality(area, None, configuration, environ)
        response["latency_ms"] = (time.perf_counter() - started) * 1000
        return response
    response["provider"] = provider.name
    types = declared_types(provider)
    askable, answers = {}, {}
    for name, question in task["questions"].items():
        spec = types.get(question["type"])
        if spec is None:
            answers[name] = refusal(UNSUPPORTED_QUESTION_TYPE,
                                    f"provider {provider.name!r} does not answer {question['type']!r}; "
                                    f"it answers {sorted(types)}", question["type"])
            continue
        trouble = _check_fields(question, spec)
        if trouble:
            answers[name] = refusal(trouble.split(":")[0], trouble, question["type"])
            continue
        askable[name] = question
    if askable:
        method = getattr(provider, "answer_questions", None)
        if not callable(method):
            for name, question in askable.items():
                answers[name] = refusal(PROVIDER_ERROR, f"provider {provider.name!r} declares question types but "
                                                        f"implements no answer_questions()", question["type"])
        else:
            try:
                returned = method(copy.deepcopy(task["state"]), copy.deepcopy(askable), data, task.get("as_of"))
            except Exception as exc:                                    # noqa: BLE001
                returned = {name: refusal(PROVIDER_ERROR, f"{type(exc).__name__}: {exc}", q["type"])
                            for name, q in askable.items()}
            if not isinstance(returned, dict):
                returned = {}
            for name, question in askable.items():
                answer = returned.get(name)
                if not isinstance(answer, dict):
                    answers[name] = refusal(PROVIDER_ERROR, "the provider returned no answer for this question",
                                            question["type"])
                    continue
                answer = dict(answer)
                answer.setdefault("type", question["type"])
                answer.setdefault("status", "OK")
                if answer["type"] != question["type"]:
                    answers[name] = refusal(PROVIDER_ERROR, f"the provider answered type {answer['type']!r} to a "
                                                            f"{question['type']!r} question", question["type"])
                    continue
                if answer["status"] == "OK" and answer.get("execution_authorized", False) is not False:
                    answers[name] = refusal(PROVIDER_ERROR, "an answer claimed execution authority", question["type"])
                    continue
                answers[name] = answer
            response["state_ref"] = returned.get("__state_ref__") if isinstance(returned, dict) else None
    response["answers"] = {name: answers[name] for name in task["questions"]}     # the caller's order, always
    response["answered"] = sum(1 for a in response["answers"].values() if a.get("status") == "OK")
    response["refused"] = len(response["answers"]) - response["answered"]
    # WP31: what is known about how well this area answers travels WITH the answers. A reader of one answer -- in the
    # web, over the API, through MCP, in a Telegram message -- sees the measurement or sees that there is none,
    # instead of having to know that a report exists somewhere.
    response["quality"] = area_quality(area, provider, configuration, environ)
    response["latency_ms"] = (time.perf_counter() - started) * 1000
    return response


def catalog(registry, configuration=None):
    """What can be asked, per area: the provider and the question types it declares. This is what an orchestrator is given
    to choose from; it is also exactly what a person may write by hand.

    Each area also carries its `chooser`: the abstention rule that applies when a language model picks this area's
    configuration -- the threshold, the report it is cited from, and the bins at or above it -- or `NOT_CONFIGURED`.
    It is published for the same reason the question types are: a consumer, the web or an MCP client or a Telegram
    skill, must be able to see the rule an answer was produced under BEFORE it decides how much to trust the answer.
    A declared rule that does not resolve is published as its refusal, not hidden."""
    from . import decide                      # `decide` sits on top of this module; the rule is reached at call time
    chooser = decide.declared_rule(configuration)
    out = {}
    for area in AREAS:
        try:
            provider = provider_for(registry, area)
        except TaskError as error:
            out[area] = {"provider": None, "error": str(error), "question_types": {}, "chooser": chooser,
                         "quality": area_quality(area, None, configuration)}
            continue
        out[area] = {"provider": provider.name if provider else None,
                     "question_types": declared_types(provider) if provider else {},
                     # the values a question field may take, straight from the provider's declared slots: the fitted
                     # targets and horizons, the retained studies, the policy. A router shown only the data's columns
                     # will pick one of THOSE as a target, and a column is not a fitted target.
                     "parameters": declared_parameters(provider) if provider else {},
                     "aliases": declared_aliases(provider) if provider else {},
                     "combinations": declared_combinations(provider) if provider else [],
                     # whether this area's engine needs the caller's rows, as the PROVIDER declares it. A framework
                     # that guesses this refuses the wrong things; a framework that ignores it lets a person run an
                     # envelope the engine can only refuse (2026-09-24: a household forecast ran with nothing
                     # attached and came back PROVIDER_ERROR after ten seconds).
                     "data_requirement": declared_data_requirement(provider) if provider else UNKNOWN_DATA_REQUIREMENT,
                     # the rule a model's choice about THIS area is held to, so nobody has to ask the file
                     "chooser": copy.deepcopy(chooser) if isinstance(chooser, dict) else chooser,
                     # WP31: and what is known about how well this area answers, published beside what it can be
                     # asked. A consumer reading the catalog sees the measurement before it asks anything
                     "quality": area_quality(area, provider, configuration)}
    return out


#: what the framework says when a provider does not declare whether it needs the caller's data
UNKNOWN_DATA_REQUIREMENT = {"required": None, "why": "this provider does not declare whether it needs attached data"}


def declared_data_requirement(provider):
    """`{"required": bool|None, "why": str, "shape": str?}` as the provider declares it, else UNKNOWN."""
    method = getattr(provider, "data_requirement", None)
    if not callable(method):
        return dict(UNKNOWN_DATA_REQUIREMENT)
    try:
        declared = method()
    except Exception as error:                                              # noqa: BLE001
        return {"required": None, "why": f"the provider could not declare its data requirement: {error}"}
    if not isinstance(declared, dict) or not isinstance(declared.get("required"), bool):
        return dict(UNKNOWN_DATA_REQUIREMENT)
    return {"required": declared["required"], "why": str(declared.get("why", "")),
            **({"shape": str(declared["shape"])} if declared.get("shape") else {})}


def declared_parameters(provider):
    """The provider's slot vocabulary as {field: [allowed values]}, or {} when it declares none."""
    method = getattr(provider, "chat_slots", None)
    if not callable(method):
        return {}
    try:
        slots = method() or []
    except Exception:                                                   # noqa: BLE001
        return {}
    return {slot["name"]: list(slot["allowed"]) for slot in slots
            if isinstance(slot, dict) and slot.get("name") and isinstance(slot.get("allowed"), list)}


def declared_aliases(provider):
    """Ordinary phrasings per value, from the slots: "one hour" for horizon 60. A router that is shown only the number
    60 has no way to know a person's "una hora" means it."""
    method = getattr(provider, "chat_slots", None)
    if not callable(method):
        return {}
    try:
        slots = method() or []
    except Exception:                                                   # noqa: BLE001
        return {}
    return {slot["name"]: slot["aliases"] for slot in slots
            if isinstance(slot, dict) and slot.get("name") and isinstance(slot.get("aliases"), dict)}


def declared_combinations(provider):
    """The jointly valid values, when a provider serves more than one fitted state. A flat union of targets and horizons
    lets a router pair a target with a horizon no bundle has; the combinations are what is actually fitted."""
    method = getattr(provider, "chat_combinations", None)
    if not callable(method):
        return []
    try:
        combinations = method() or []
    except Exception:                                                   # noqa: BLE001
        return []
    return [dict(c) for c in combinations if isinstance(c, dict)]
