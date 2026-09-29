"""From a sentence to an envelope, and from answers back to a sentence -- with the model never touching a number.

The owner's picture of this layer: a language model reads what a person wrote, learns what columns the attached data has,
decides which area the question belongs to, and writes the typed envelope. Then the engines compute, and the model turns
the numbers back into a sentence. That is what is built here, with three constraints that keep it honest.

The model proposes; the contract disposes. Whatever the model writes is validated against the catalog of areas and declared
question types and against the profile of the attached data. An area nobody serves, a question type no provider declares, a
column the data does not have: each is refused by name and the envelope does not run. The person sees the proposal and may
edit it before it runs, and the proposal is recorded beside the result.

The model sees the shape of the data, never the data. The profile handed to it is column names, types and a row count. The
rows themselves are shipped to nobody in order to have a sentence parsed.

The narration is checked against the answers. Every number in the model's sentence must appear in the answers it was given;
a narration that introduces a figure is discarded and replaced by a deterministic rendering. So the narration can be wrong in
emphasis, never in fact.
"""

import csv
import io
import json
import math
import re

from .interpret import build as build_interpreter
from .outputs import (NOT_NARRATED, default as default_output, load as load_output,
                      names as output_names, narratable, response_view, select as select_output)
from .questions import AREAS, TaskError, catalog as question_catalog, validate_task

MAX_PROMPT = 4000

#: the schema `tools/measure_route.py` writes, and the only document a route reliability may be published from
ROUTE_RELIABILITY_SCHEMA = "m5phet_route_reliability.v1"

#: nobody has measured this installation's router. The honest answer, and the default
NOT_MEASURED = "NOT_MEASURED"
#: the cited file is not a route reliability report this framework wrote
ROUTE_REPORT_UNREADABLE = "ROUTE_REPORT_UNREADABLE"
#: the report measured a DIFFERENT interpreter writing the envelopes. A rate measured on another model is not a fact
#: about this one, exactly as it is not for the interpreter's own reliability
ROUTE_MEASURED_ON_ANOTHER_INTERPRETER = "ROUTE_MEASURED_ON_ANOTHER_INTERPRETER"

ROUTE_RELIABILITY_VARIABLE = "M5PHET_ROUTE_RELIABILITY_REPORT"

#: what the catalog says about a confidence for THIS path. `route` asks the model for a whole envelope as free text
#: and reads it back as JSON; there is no per-field value for a plugin to attach a probability to, and `route` never
#: asks for one -- not even from `openai_compatible`, the one shipped plugin whose endpoint can report logprobs,
#: because `route` calls `_ask` and `_ask` discards them. So with every shipped plugin the honest statement is that
#: no confidence exists for this path, and the abstention rule -- whatever threshold an operator declares -- cannot
#: be applied to it. It is published so nobody believes the gate is wider than it is.
ROUTE_CONFIDENCE_NOT_REPORTED = "CONFIDENCE_NOT_REPORTED"

#: RR05: the router's ONE measured failure mode, and what is done about it.
#:
#: `tools/measure_route.py` at N=5 over 19 sentences scored 81 CORRECT, `WRONG_AREA` 0, `WRONG_VALUE` 0 and
#: `WRONG_TYPE` **12** -- every one of the twelve on four sentences that ask for TWO things at once ("pronostica la
#: potencia **y dame un rango**", "assign hierarchical regimes to these rows" against a reference that also describes
#: clusters, "cual fue el efecto del tratamiento **y en jovenes**"). In all twelve the engine and every governed value
#: were right and the envelope asked a strict SUBSET of the types the sentence asks for: the model answered the more
#: specific half (interval, cluster_description, cate) and dropped the plain half (point_forecast, clustering, ate).
#: So this is an UNDER-ANSWER, not a misrouting, and it is repaired by asking about the half that is missing -- once,
#: in the interpreter's constrained mode where it picks among values that are already declared, which is the mode its
#: reliability was measured in (0.9434 when it chose) rather than the free-text envelope mode this path uses.
#:
#: Three rules make the repair incapable of inventing anything:
#:   1. only a question type the chosen area ALREADY DECLARES may be added; the choice list is that set minus the
#:      types already asked, plus `none`, and anything else the model returns is discarded by name;
#:   2. a type is offered only when every field it REQUIRES is already present in the validated envelope (in a
#:      question of the same area or in the state). A type needing a field nobody supplied -- `interval` needs a
#:      `confidence_level` -- is not offered, because filling it would be this layer choosing a number;
#:   3. whatever is added is re-validated by `check_proposal`, and an addition that does not validate is dropped and
#:      recorded. The envelope that runs is never worse than the one the model first wrote.
#: RB04, from the full-corpus measurement of 2026-09-28: the variable an operator sets to turn the completion pass on.
#: It is OFF by default, and the default is a MEASUREMENT and not a preference.
#:
#: The pass was built from a 4-sentence subset and repairs a real failure mode on those sentences. Measured over the
#: WHOLE declared corpus -- 19 sentences, 5 runs each, one scorer, one build, one day -- it repairs three sentences
#: and breaks two, and the two it breaks it breaks DETERMINISTICALLY: "describe el grupo de velas con cuerpo alto"
#: and "describe the cluster with a large body" went CORRECT 5/5 to WRONG_TYPE 5/5, because the model, asked whether
#: the sentence also asks for `clustering`, says yes on all ten runs. Describing a cluster does imply that rows were
#: assigned, so this is not a plumbing fault; it is the mirror image of the under-answer the pass exists to repair,
#: and on this corpus the over-answer costs more runs than the under-answer repair saves. Every one of the twelve
#: WRONG_TYPE runs in that measurement is this pass adding `clustering`.
#:
#: So the pass ships off, its measurement ships with it, and turning it on is an operator's informed decision rather
#: than a default nobody measured. `route(complete=True)` still forces it on for a caller that wants it -- the
#: measurement harness uses exactly that -- and the record always says which of the two happened.
ROUTE_COMPLETION_VARIABLE = "M5PHET_ROUTE_COMPLETION"
#: the completion pass did not run because it is not enabled. Published rather than left as a null, because a null
#: was read once in this very work as "the pass is broken" when the pass had simply not been asked to run
COMPLETION_DISABLED = "COMPLETION_NOT_ENABLED"
COMPLETION_NONE = "none"
#: the completion pass ran and the model said the sentence asks for nothing further
COMPLETION_NOTHING_FURTHER = "NOTHING_FURTHER_ASKED"
#: there was no type to offer: every declared type is already asked, or each remaining one needs a field nobody gave
COMPLETION_NOTHING_TO_OFFER = "NO_COMPLETABLE_TYPE_DECLARED"
#: the model named something that is not on the offered list. Discarded, never guessed at
COMPLETION_NOT_OFFERED = "CHOICE_WAS_NOT_OFFERED"
#: the addition was made and then refused by `check_proposal`; the envelope the model wrote is kept unchanged
COMPLETION_REJECTED = "ADDITION_DID_NOT_VALIDATE"
#: the addition was made and validates
COMPLETION_ADDED = "ADDED"
#: the interpreter could not be consulted for the second, narrow question. The first envelope stands
COMPLETION_NOT_CONSULTED = "INTERPRETER_NOT_CONSULTED"


def completion_enabled(environ=None):
    """Whether the route completion pass runs. OFF unless an operator turns it on; see ROUTE_COMPLETION_VARIABLE."""
    import os
    value = (environ if environ is not None else os.environ).get(ROUTE_COMPLETION_VARIABLE)
    return str(value).strip().lower() in ("1", "true", "yes", "on") if value is not None else False


def _fields_available(task):
    """Every field name the validated envelope already carries a value for, in its questions or in its state.

    A completion may only use these. It is the whole of rule 2: a required field that is not in here would have to be
    invented, and this layer does not invent one."""
    available = {}
    for question in (task.get("questions") or {}).values():
        if isinstance(question, dict):
            for field, value in question.items():
                if field != "type" and value is not None:
                    available.setdefault(field, value)
    for field, value in (task.get("state") or {}).items():
        if value is not None:
            available.setdefault(field, value)
    return available


def declared_choices(area):
    """`{field: [declared values]}` a completion may ASK about, as distinct from what the envelope already carries.

    RB04 widens rule 2 by exactly this much and no more. Before today a required field the envelope did not carry
    made its type unofferable, full stop -- which is why `interval` was never offered: its `confidence_level` is in
    no question and in no state, so offering it would have meant this layer choosing a number, and it rightly
    refused to. What changed is not the rule but the vocabulary: the provider DECLARES the confidence levels its
    bundles actually fitted, so the level can be chosen the way a target or a horizon is -- among declared values, in
    the interpreter's constrained mode, which is the mode its reliability was measured in.

    A field with no declared list is still not askable, and an area that fitted no level declares an empty list, so
    `interval` stays unofferable there. Nothing here can produce a value the area did not publish."""
    choices = {field: list(values) for field, values in ((area or {}).get("parameters") or {}).items() if values}
    levels = (area or {}).get("confidence_levels") or []
    if levels:
        choices[QUESTION_CONFIDENCE_LEVEL] = list(levels)
    return choices


def completable_types(task, declared, available=None, *, area=None):
    """The declared types this sentence did not ask and that CAN be asked without inventing a field value.

    A required field is satisfied either by a value the envelope ALREADY CARRIES (rule 2 as written) or by a field
    whose values this area declares, which a second constrained question can choose among. Deterministic, and
    testable without a model: no interpreter is consulted here."""
    available = _fields_available(task) if available is None else available
    askable = set(declared_choices(area))
    asked = {q.get("type") for q in (task.get("questions") or {}).values() if isinstance(q, dict)}
    out = []
    for name in sorted(declared):
        if name in asked:
            continue
        required = (declared[name] or {}).get("required") or []
        if all(field in available or field in askable for field in required):
            out.append(name)
    return out


def _completion_question(kind, declared_spec, available, chosen=None):
    """The question to add: its type, and ONLY the fields that type declares, taken from what is already there.

    `chosen` holds the values a second constrained question settled -- a confidence level among the ones the area
    fitted. They are applied only to fields the type declares, and a value the envelope already carries is never
    overwritten by one: what the sentence said outranks what a model was asked afterwards."""
    spec = declared_spec or {}
    allowed = list(spec.get("required") or []) + list(spec.get("optional") or [])
    question = {"type": kind}
    for field in allowed:
        if field in available:
            question[field] = available[field]
        elif field in (chosen or {}):
            question[field] = chosen[field]
    return question


def _choose_missing_fields(prompt, kind, declared_spec, available, choices, interpreter, record):
    """Ask, in ONE constrained question, for the required fields the envelope does not carry.

    Constrained means the model picks among values the area declared; anything else it returns is discarded by name,
    exactly as the type choice is. A field left unchosen simply stays missing, and the addition then fails
    `check_proposal` and is dropped -- which is the third rule doing its job rather than a special case here."""
    missing = [field for field in ((declared_spec or {}).get("required") or [])
               if field not in available and field in choices]
    if not missing:
        return {}
    slots = [{"name": field, "allowed": list(choices[field])} for field in missing]
    record["fields_offered"] = {field: list(choices[field]) for field in missing}
    try:
        proposed = interpreter.propose(prompt, slots)
    except Exception as error:                                          # noqa: BLE001
        record["fields_why"] = f"{type(error).__name__}: {error}"
        return {}
    chosen, discarded = {}, {}
    for field in missing:
        value = (proposed or {}).get(field) if isinstance(proposed, dict) else None
        if admits(value, choices[field]):
            chosen[field] = value
        elif isinstance(value, str):
            # a constrained slot may come back as the text of a declared number; the DECLARED value is what is kept,
            # never the model's spelling of it, and a string matching none of them is discarded like anything else
            matched = [candidate for candidate in choices[field] if str(candidate) == value.strip()]
            if len(matched) == 1:
                chosen[field] = matched[0]
            else:
                discarded[field] = value
        elif value is not None:
            discarded[field] = value
    record["fields_chosen"] = chosen
    if discarded:
        record["fields_discarded"] = discarded
    return chosen


def _completion_name(task, kind):
    name = kind
    existing = set(task.get("questions") or {})
    while name in existing:
        name += "_2"
    return name


def complete_under_answer(prompt, task, catalog, profile, interpreter):
    """Ask, once, whether the sentence also asks for a declared type the envelope left out. Returns (task, record).

    The returned task is the original object when nothing was added, so a caller that ignores the record is exactly as
    safe as before this function existed."""
    record = {"pass": "route_completion.v1", "offered": [], "choice": None, "outcome": None, "why": None,
              "added": None, "problems": [], "fields_offered": {}, "fields_chosen": {}}
    area = (catalog or {}).get((task or {}).get("area")) or {}
    declared = area.get("question_types") or {}
    available = _fields_available(task)
    choices = declared_choices(area)
    offered = completable_types(task, declared, available, area=area)
    record["offered"] = offered
    if not offered:
        record["outcome"] = COMPLETION_NOTHING_TO_OFFER
        record["why"] = ("every question type this area answers is already asked, or each remaining one needs a field "
                         "the envelope does not carry -- and filling one in would be this layer choosing a value")
        return task, record
    slots = [{"name": "also_asked_question_type", "type": "string", "allowed": offered + [COMPLETION_NONE]}]
    try:
        proposed = interpreter.propose(prompt, slots)
    except Exception as error:                                          # noqa: BLE001
        record["outcome"], record["why"] = COMPLETION_NOT_CONSULTED, f"{type(error).__name__}: {error}"
        return task, record
    choice = (proposed or {}).get("also_asked_question_type") if isinstance(proposed, dict) else None
    choice = choice.strip() if isinstance(choice, str) else choice
    record["choice"] = choice
    if choice in (None, "", COMPLETION_NONE, "null", "None"):
        record["outcome"] = COMPLETION_NOTHING_FURTHER
        return task, record
    if choice not in offered:
        record["outcome"] = COMPLETION_NOT_OFFERED
        record["why"] = f"{choice!r} was not one of the offered types {offered}"
        return task, record
    chosen = _choose_missing_fields(prompt, choice, declared.get(choice), available, choices, interpreter, record)
    candidate = json.loads(json.dumps(task))
    candidate["questions"][_completion_name(task, choice)] = _completion_question(choice, declared.get(choice),
                                                                                 available, chosen)
    still_missing = [field for field in (declared.get(choice) or {}).get("required") or []
                     if not any(spelling in candidate["questions"][_completion_name(task, choice)]
                                for spelling in _spellings(field))]
    if still_missing:
        # rule 3, applied to the addition's own completeness: a type whose required field nobody supplied is not
        # added half-formed for an engine to refuse. The envelope the model wrote stands.
        record["outcome"], record["why"] = COMPLETION_REJECTED, (
            f"{choice!r} needs {still_missing} and no declared value was chosen for it, so nothing was added; "
            f"filling it in here would be this layer choosing a value")
        return task, record
    checked, problems = check_proposal(candidate, catalog, profile)
    if checked is None:
        record["outcome"], record["problems"] = COMPLETION_REJECTED, list(problems)
        record["why"] = "the completed envelope did not validate, so the envelope the model wrote is kept unchanged"
        return task, record
    record["outcome"], record["added"] = COMPLETION_ADDED, choice
    return checked, record



# --- what the model is allowed to know about the data ------------------------------------------------------------------

def dataset_profile(data):
    """The shape of the attachment: columns, an inferred type per column, and the row count. Never the rows."""
    if data is None:
        return {"kind": "none", "columns": [], "rows": 0}
    if isinstance(data, str):
        text = data.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return dataset_profile(json.loads(text))
            except ValueError:
                pass
        try:
            rows = list(csv.DictReader(io.StringIO(text)))
            if rows and rows[0] and all(k for k in rows[0]):
                return _table_profile(rows)
        except csv.Error:
            pass
        return {"kind": "text", "columns": [], "rows": 1, "characters": len(text)}
    if isinstance(data, list) and data and all(isinstance(r, dict) for r in data):
        return _table_profile(data)
    if isinstance(data, list):
        return {"kind": "vector", "columns": [], "rows": len(data)}
    if isinstance(data, dict):
        return {"kind": "object", "columns": sorted(str(k) for k in data), "rows": 1}
    return {"kind": "unknown", "columns": [], "rows": 0}


def _table_profile(rows):
    columns = list(rows[0].keys())
    types = {}
    for column in columns:
        sample = [r.get(column) for r in rows[:50] if r.get(column) not in (None, "")]
        numeric = 0
        for value in sample:
            try:
                float(value)
                numeric += 1
            except (TypeError, ValueError):
                pass
        types[column] = "number" if sample and numeric == len(sample) else "text"
    return {"kind": "table", "columns": columns, "types": types, "rows": len(rows)}


# --- the proposal -----------------------------------------------------------------------------------------------------

def _columns_named(obj):
    """Every string the envelope names that looks like a column reference, so hallucinated columns can be caught."""
    found = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("treatment", "outcome", "target_variable", "target", "price_column"):
                if isinstance(value, str):
                    found.add(value)
            if key in ("confounders", "features", "context_variables", "columns"):
                if isinstance(value, list):
                    found.update(v for v in value if isinstance(v, str))
            found |= _columns_named(value)
    elif isinstance(obj, list):
        for item in obj:
            found |= _columns_named(item)
    return found


def _governed_values(task, governed):
    """Values sitting in provider-governed fields are engine vocabulary, not data columns, and are checked as such."""
    found = set()
    for question in task["questions"].values():
        for field in governed:
            if isinstance(question.get(field), str):
                found.add(question[field])
    for field in governed:
        if isinstance(task["state"].get(field), str):
            found.add(task["state"][field])
    return found


#: the field an interval carries, named once so the validator, the catalog and the completion pass cannot disagree
QUESTION_CONFIDENCE_LEVEL = "confidence_level"


#: RB04: the spellings one governed field has. `state.target_variable` is the owner's spelling of the forecaster's
#: `target`, and both are governed by the same declared list. Until today the alias was checked in the STATE and not
#: inside a question, so a question naming `target_variable` carried whatever a model wrote: an unfitted series passed
#: validation and was refused later, by the engine, after the person pressed run. `tools/measure_route.py` governs the
#: same spellings under `FIELD_SPELLINGS`, and `tests/test_route_validation.py` asserts the two agree, because a
#: spelling one of them governs and the other does not is a hole that reopens without anybody noticing.
GOVERNED_ALIASES = {"target_variable": "target"}


def admits(value, allowed):
    """Whether a declared vocabulary admits this value, comparing by TYPE as well as by equality.

    Plain `value in allowed` is what this replaces, and it had a hole with a measured consequence: `True == 1` in
    Python, so a slot whose fitted horizon was 1 admitted `True`, and a boolean reached an engine as a horizon.
    Musashi's `44130ae` repaired exactly that in the measurement scorer; this is the same rule in the validator, so
    the product refuses what the measurement refuses.

    The rule, and what each clause is for:
      * a boolean is admitted only by a slot that declares booleans. It is never a number here.
      * a non-finite float is admitted by nothing: infinity and NaN are not values an engine was fitted at, and a
        comparison against them silently succeeds at nothing.
      * a finite number is admitted by an equal number of either numeric type, so `60.0` IS the fitted horizon `60`.
        Refusing that would refuse a real sentence; truncating `60.5` to it would answer a different question.
      * anything else must match both the type and the value, so the string `"60"` is not the integer `60`. The
        validator stays strict there on purpose: the scorer accepts an exact integer string in order to SCORE what a
        model wrote, and `verdict_of` reads `INVALID_PROPOSAL` before it reads any value, so a refused envelope is
        never scored correct."""
    allowed = list(allowed or ())
    if isinstance(value, bool):
        return any(isinstance(candidate, bool) and candidate == value for candidate in allowed)
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return False
        return any(not isinstance(candidate, bool) and isinstance(candidate, (int, float)) and candidate == value
                   for candidate in allowed)
    return any(type(candidate) is type(value) and candidate == value for candidate in allowed)


def _spellings(field):
    """Every key a governed field may appear under: its own name first, then its aliases."""
    return [field] + [alias for alias, canonical in GOVERNED_ALIASES.items() if canonical == field]


def _governed_problems(where, holder, parameters):
    """Every governed value in one question or state, under EVERY spelling, plus any contradiction between two of them.

    Two spellings of one field holding different values is refused even when both values are admissible: which one
    the engine would read is not something a validator should leave to the engine.

    `where` is `None` for the state, whose problems have always read `state.<spelling> ...` and still do, so an
    existing reader of these messages is not broken by the repair."""
    problems = []
    for field, allowed in parameters.items():
        named = [(spelling, holder[spelling]) for spelling in _spellings(field) if spelling in holder]
        for spelling, value in named:
            if not admits(value, allowed):
                said = f"state.{spelling}" if where is None else f"{where}: {spelling}"
                problems.append(f"{said} {value!r} is not one the engine has ({allowed})")
        distinct = {json.dumps(value, sort_keys=True, default=str) for _, value in named}
        if len(named) > 1 and len(distinct) > 1:
            spoken = ", ".join(f"{spelling}={value!r}" for spelling, value in named)
            said = "state" if where is None else where
            problems.append(f"{said}: {spoken} name one governed field ({field}) with different values; which one "
                            f"the engine reads is not for it to decide, so the envelope is refused instead")
    return problems


def _confidence_level_problems(name, question, levels):
    """Whether an `interval` question's confidence level is one this area can be held to.

    The field was ungoverned until RB04: `0.99` passed with nothing fitted at that level, and `95` passed although it
    is not a probability at all. The two are not the same fault and are not treated the same, because
    `check_proposal` refuses the WHOLE envelope and `m5phet.questions` is explicit that every question is answered
    separately -- "a request asking for a point forecast and an interval gets the point forecast from an engine that
    has one and an explicit refusal for the interval from an engine that does not". Refusing both halves because one
    is unserviceable would hide which half the engine can actually do. So:

      * a value that is not a two-sided confidence level at all -- a boolean, a string, 95, a NaN -- is an envelope
        problem. The router wrote something no engine could mean, and a person asking for "un rango" should be told
        the request was misread, not handed a refusal that blames the model for lacking a distribution.
      * a well-formed level this area DID fit is fine.
      * a well-formed level this area did not fit, where the area declares a fitted vocabulary, is an envelope
        problem for the same reason an unfitted horizon is: the router picked outside a declared list.
      * a well-formed level where the area declares NO level at all is left to the provider, which refuses that one
        question by name -- `NOT_ESTIMABLE` for a bundle with no distribution -- while the rest of the envelope is
        answered. Nothing is invented either way; what differs is which layer says so."""
    if QUESTION_CONFIDENCE_LEVEL not in question:
        return []
    value = question[QUESTION_CONFIDENCE_LEVEL]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or not 0 < value < 1:
        fitted = f"this area fitted {levels}" if levels else "this area declares no fitted level"
        return [f"question {name!r}: {QUESTION_CONFIDENCE_LEVEL} {value!r} is not a two-sided confidence level; it is "
                f"a number strictly between 0 and 1, and {fitted}"]
    if levels and not admits(float(value), levels):
        return [f"question {name!r}: {QUESTION_CONFIDENCE_LEVEL} {value!r} is not one this area fitted ({levels}); a "
                f"level is a pair of quantiles somebody fitted, and a nearby one is not widened into it"]
    return []


def check_proposal(proposal, catalog, profile):
    """Validate what the model proposed. Returns (task_or_None, problems)."""
    problems = []
    try:
        task = validate_task(proposal)
    except TaskError as error:
        return None, [str(error)]
    area = catalog.get(task["area"]) or {}
    if not area.get("provider"):
        problems.append(f"no installed provider serves area {task['area']!r}")
        return None, problems
    declared = area.get("question_types") or {}
    parameters = area.get("parameters") or {}
    levels = area.get("confidence_levels") or []
    # RB04: the procedure a request selects is a name this installation either has or does not. Nothing is
    # substituted: a request asking for a surface that is not installed is told so, never answered by `default` as
    # though it had been honoured.
    if isinstance(task.get("output"), dict):
        wanted = task["output"].get("plugin")
        installed = output_names()
        if wanted not in installed:
            problems.append(f"output plugin {wanted!r} is not installed; this installation has {installed} and "
                            f"nothing is substituted for a procedure it does not have -- not even 'default'")
    for name, question in task["questions"].items():
        if question["type"] not in declared:
            problems.append(f"question {name!r}: type {question['type']!r} is not one this area answers "
                            f"({sorted(declared)})")
        # a field the provider governs -- target, horizon, policy, study -- must hold one of its declared values,
        # under EVERY spelling it has, compared by type as well as by equality. This is the check that stops a router
        # from naming a DATA column as a fitted target and getting a confident refusal, and (RB04) from reaching an
        # engine with a boolean where a horizon belongs or with two spellings of one field that disagree.
        problems += _governed_problems(f"question {name!r}", question, parameters)
        problems += _confidence_level_problems(name, question, levels)
    # `state.target_variable` is the owner's spelling of the forecaster's `target`; both are governed by the same list
    aliases = GOVERNED_ALIASES
    combinations = area.get("combinations") or []
    if combinations:
        # a target and a horizon may each be admissible and still not be fitted TOGETHER: two bundles, two horizons,
        # and the pair that no bundle has is refused here rather than by the engine after the person pressed run
        keys = sorted({k for c in combinations for k in c})
        for name, question in task["questions"].items():
            named = {k: question[k] for k in keys if k in question}
            if len(named) >= 2 and not any(all(c.get(k) == v for k, v in named.items()) for c in combinations):
                problems.append(f"question {name!r}: {named} is not a fitted combination; the engine has "
                                f"{combinations}")
    problems += _governed_problems(None, task["state"], parameters)
    # an engine that needs rows must not be handed an envelope with none: the refusal belongs here, before the run,
    # naming what to attach -- not ten seconds later as the engine's PROVIDER_ERROR
    requirement = area.get("data_requirement") or {}
    nothing_attached = profile.get("kind") == "none" or not (profile.get("columns") or profile.get("rows"))
    if requirement.get("required") and nothing_attached:
        shape = f" Expected: {requirement['shape']}" if requirement.get("shape") else ""
        problems.append(f"area {task['area']!r} needs data attached and none is: {requirement.get('why')}.{shape}")
    governed = set(parameters) | {a for a, canonical in aliases.items() if canonical in parameters}
    if profile.get("columns"):
        known = set(profile["columns"])
        for column in sorted(_columns_named(task) - _governed_values(task, governed)):
            if column not in known:
                problems.append(f"column {column!r} is not in the attached data ({profile['columns'][:12]}...)"
                                if len(profile["columns"]) > 12 else
                                f"column {column!r} is not in the attached data ({profile['columns']})")
    return (task if not problems else None), problems


def dataset_module():
    """`m5phet.datasets`, imported here and not at the top: the catalog describes attachments through
    `dataset_profile`, so the two modules would import each other."""
    from . import datasets as module
    return module


def resolve_dataset(prompt, data, catalog, decider):
    """(resolution, what the proposal carries) for a sentence that names a dataset, or (None, None).

    Nothing is resolved when a file is attached -- the attachment is the data -- and nothing is resolved when the
    catalog is empty, which is the state of every installation that has not built one; such an installation answers
    exactly as it did before WP15."""
    if data is not None or not (catalog or {}).get("datasets"):
        return None, None
    module = dataset_module()
    resolution = module.resolve(prompt, catalog, decider)
    if resolution["status"] != module.OK:
        return (resolution if resolution["status"] != module.NOT_ASKED else None), None
    return resolution, module.proposal_view(resolution)


def route(prompt, data, registry, *, interpreter=None, datasets=None, decider=None, complete=None,
          environ=None):
    """Turn a sentence into a validated envelope, or say exactly why it could not be.

    `datasets` is the dataset catalog (WP15) and `decider` the Engine or Registry Laya is asked through when
    more than one dataset fits the words. The catalog is consulted ONLY when nothing is attached: an attached file is the
    data the person chose, and no catalog may quietly replace it. When the sentence names a dataset the catalog
    holds, the resolved description -- columns and a row count, never a row -- becomes the profile every check below
    reads, so an engine that needs data is satisfied by a named dataset exactly as it is by an attachment."""
    if not isinstance(prompt, str) or not prompt.strip():
        return {"status": "REFUSED", "why": "the question is empty", "proposal": None, "problems": []}
    if len(prompt) > MAX_PROMPT:
        return {"status": "REFUSED", "why": f"the question is {len(prompt)} characters; the limit is {MAX_PROMPT}",
                "proposal": None, "problems": []}
    catalog = question_catalog(registry)
    profile = dataset_profile(data)
    interpreter = interpreter if interpreter is not None else build_interpreter()
    resolution, chosen = resolve_dataset(prompt, data, datasets, decider)
    if chosen:
        profile = dataset_module().profile_of(resolution["dataset"])
    report = {"profile": profile, "catalog": catalog, "interpreter": interpreter.identity(),
              "dataset": chosen, "dataset_resolution": resolution,
              # WP30: what the person reviewing this proposal is NOT being given. The proposal is checked against the
              # catalog and the data (that is `check_proposal`, and it refuses by name); what no shipped plugin can
              # supply here is how sure the model was, so the abstention rule does not run on this path and says so.
              "confidence": ROUTE_CONFIDENCE_NOT_REPORTED,
              "gate": ("every proposal is validated against the catalog of areas, the declared question types, the "
                       "provider-governed values and the attached data; a proposal that fails is refused by name and "
                       "does not run. No confidence is reported for this path, so a declared abstention threshold is "
                       "not applied to it.")}
    if not interpreter.available:
        return {**report, "status": "REFUSED", "proposal": None, "problems": [],
                "why": ("no interpreter is configured, so a sentence cannot be routed; write the envelope directly "
                        "(area, state, questions) or configure M5PHET_INTERPRETER_COMMAND")}
    served = {area: {"provider": spec["provider"], "question_types": spec["question_types"],
                     "allowed_values": spec.get("parameters") or {},
                     "value_meanings": spec.get("aliases") or {},
                     "fitted_combinations": spec.get("combinations") or []}
              for area, spec in catalog.items() if spec.get("provider")}
    instruction = (
        "Return ONLY one compact JSON object and no prose.\n"
        "You are routing a request to a machine-learning engine. Choose the area, describe the state, and write the "
        "questions, using ONLY the areas and question types listed. Where an area lists allowed_values for a field "
        "(target, horizon, policy_id, study, ...), that field MUST take one of those values -- they are what the fitted "
        "engine has, and a column name from the data is NOT a substitute. value_meanings gives the ordinary phrasings "
        "each value stands for (e.g. which horizon means 'one hour'). Where fitted_combinations is given, the fields of a "
        "question MUST together match ONE listed combination. Use data column names only for fields that have no "
        "allowed_values. Never invent a column, a type or a field. If the request cannot be served, return "
        '{"area": null, "why": "<one sentence>"}.\n'
        f"Request: {json.dumps(prompt)}\n"
        f"Data profile (the shape only; you are not shown rows): {json.dumps(profile)}\n"
        f"Areas and the question types each answers, with required and optional fields: {json.dumps(served)}\n"
        'Answer shape: {"area": "<area>", "state": {...}, "questions": {"<name>": {"type": "<type>", ...}}}'
    )
    try:
        raw = interpreter._ask(instruction)
    except Exception as error:                                          # noqa: BLE001
        return {**report, "status": "REFUSED", "proposal": None, "problems": [],
                "why": f"the interpreter could not be consulted ({error})"}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {**report, "status": "REFUSED", "proposal": None, "problems": [],
                "why": "the interpreter returned no JSON object"}
    try:
        proposal = json.loads(match.group(0))
    except ValueError:
        return {**report, "status": "REFUSED", "proposal": None, "problems": [],
                "why": "the interpreter returned malformed JSON"}
    if isinstance(proposal, dict) and proposal.get("area") is None:
        return {**report, "status": "REFUSED", "proposal": proposal, "problems": [],
                "why": f"the interpreter found no served area for this request: {proposal.get('why')}"}
    task, problems = check_proposal(proposal, catalog, profile)
    if resolution and resolution["status"] in (dataset_module().AMBIGUOUS, dataset_module().NOT_FOUND):
        # the person named a dataset and it is not one the catalog holds, or not one of them: that is refused by
        # name here, not silently answered from whatever else was to hand
        problems = list(problems) + [resolution["why"]]
        task = None
    completion = None
    if task is not None:
        # RR05: the twelve WRONG_TYPE runs measured on 2026-09-25 were all UNDER-answers of a sentence that asks two
        # things, and the pass recovers the second ask from the area's own declared types only. RB04 measured the
        # pass on the WHOLE corpus and it also OVER-answers, deterministically, on two sentences -- see
        # ROUTE_COMPLETION_VARIABLE -- so it is off unless it is asked for. Either way the record says which.
        if complete is None:
            complete = completion_enabled(environ)
        if complete:
            task, completion = complete_under_answer(prompt, task, catalog, profile, interpreter)
        else:
            completion = {"pass": "route_completion.v1", "outcome": COMPLETION_DISABLED,
                          "why": (f"the route completion pass is not enabled; set {ROUTE_COMPLETION_VARIABLE}=1 to "
                                  f"turn it on. Its full-corpus measurement is the reason it is off by default, not "
                                  f"a preference"),
                          "offered": [], "choice": None, "added": None, "problems": [],
                          "fields_offered": {}, "fields_chosen": {}}
    if task is not None and chosen:
        # what runs records which dataset it read, so the envelope beside the answers is replayable and the person
        # reviewing it sees the choice rather than having to trust it
        task["state"].setdefault("dataset", chosen["id"])
    return {**report, "status": "OK" if task else "INVALID_PROPOSAL", "proposal": proposal, "task": task,
            "problems": problems, "completion": completion,
            "why": None if task else "the proposal names something the catalog or the data does not have; see problems"}


# --- how often the router is right -------------------------------------------------------------------------------------

def declared_route_reliability(configuration=None, environ=None, interpreter=None):
    """How often the configured interpreter ROUTES a sentence correctly -- cited, or `NOT_MEASURED`.

    `route` is the path where the model does not choose among declared values: it writes a whole envelope, and
    `check_proposal` then refuses anything the catalog or the data does not have. That refusal bounds the damage; it
    says nothing about how often the model is right, which until 2026-09-25 nobody had measured for this path.
    `tools/measure_route.py` measures it on the sentences whose correct envelope this framework already knows, and
    `interpreter.route_reliability_report` is where an installation says which run of it applies here.

    Cited and refused exactly as `m5phet.interpret.declared_reliability` is, and for the same reasons:

    * `NOT_MEASURED` when nothing is declared -- the default, and not a criticism of anyone;
    * `ROUTE_REPORT_UNREADABLE` when the path is not a `m5phet_route_reliability.v1` document with a summary;
    * `ROUTE_MEASURED_ON_ANOTHER_INTERPRETER` when the report scored a different plugin or model.
    """
    import hashlib
    import os
    from pathlib import Path
    env = os.environ if environ is None else environ
    settings = {}
    if configuration is not None:
        block = getattr(configuration, "interpreter", configuration)
        settings = dict(block or {})
    else:
        from . import config as configuration_module
        try:
            settings = dict(configuration_module.load(environ=env).interpreter)
        except configuration_module.ConfigError:
            settings = {}
    named = settings.get("route_reliability_report") or env.get(ROUTE_RELIABILITY_VARIABLE)
    if not named:
        return NOT_MEASURED
    path = Path(os.path.expanduser(str(named)))
    try:
        raw = path.read_bytes()
        report = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        return {"refusal": ROUTE_REPORT_UNREADABLE,
                "why": f"{path} cannot be read as a route reliability report: {error}"}
    if not isinstance(report, dict) or report.get("schema") != ROUTE_RELIABILITY_SCHEMA:
        return {"refusal": ROUTE_REPORT_UNREADABLE,
                "why": f"{path} does not declare schema {ROUTE_RELIABILITY_SCHEMA!r}; a reliability is published "
                       f"only from a report this framework wrote"}
    summary = report.get("summary") or {}
    identity = report.get("interpreter") or {}
    if not summary or summary.get("reliability") is None:
        return {"refusal": ROUTE_REPORT_UNREADABLE,
                "why": f"{path} carries no summary with a reliability; nothing measured is in it"}
    if interpreter is not None:
        here = interpreter.identity()
        if (identity.get("plugin"), identity.get("model")) != (here.get("plugin"), here.get("model")):
            return {"refusal": ROUTE_MEASURED_ON_ANOTHER_INTERPRETER,
                    "why": (f"{path} measured {identity.get('plugin')!r} / {identity.get('model')!r} writing the "
                            f"envelopes and this installation runs {here.get('plugin')!r} / {here.get('model')!r}; a "
                            f"rate measured on another model is not a fact about this one")}
    return {"reliability": summary.get("reliability"),
            "reliability_when_it_proposed": summary.get("reliability_when_it_proposed"),
            "n": {"sentences": summary.get("sentences"), "runs": summary.get("runs"),
                  "runs_per_sentence": report.get("runs_per_sentence")},
            "verdicts": summary.get("verdicts"),
            "invalid_proposal_problems": summary.get("invalid_proposal_problems"),
            "protocol": report.get("protocol"),
            "corpus": report.get("corpus"),
            "measured_at": report.get("measured_at"),
            "measured_interpreter": {"plugin": identity.get("plugin"), "model": identity.get("model")},
            "confidence": ROUTE_CONFIDENCE_NOT_REPORTED,
            "report_sha256": hashlib.sha256(raw).hexdigest()}


# --- the narration -----------------------------------------------------------------------------------------------------

_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")
_PERCENT = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*(?:%|por ciento|percent\b)")
#: a key under which a number in [0, 1] is a share of something, so that "97%" is one of its renderings
_PROBABILITY_KEY = re.compile(r"probabilit|confidence|share|proportion|p_value|silhouette_share", re.IGNORECASE)
#: quantities said in words: no digit appears, so a digit-only guard never saw them (Retsu, 2026-09-24)
_RELATIVE_QUANTITY = re.compile(
    r"\b(?:casi\s+)?(?:el\s+|la\s+|un\s+|una\s+)?"
    r"(?:doble|dobla|duplica|mitad|triple|triplica|cu[aá]druple|tercio|cuarto|quinto|"
    r"twice|double|doubles|half|halves|triple|threefold|thrice|quarter|third|fourfold|tenfold)\b", re.IGNORECASE)
#: verbs and nouns that turn a reading into an instruction or a result; the answers never carry them
_ACTION_CLAIM = re.compile(
    r"\b(?:ganancias?|beneficios?|rentabilidad|utilidad(?:es)?|p[eé]rdidas?|realizad[ao]s?|"
    r"[oó]rden(?:es)?|compra[rs]?|vende[rs]?|venta|lotes?|posici[oó]n\s+enviable|"
    r"profits?|profitable|loss(?:es)?|realised|realized|orders?|buy|buys|sell|sells|lots?|execute[sd]?)\b",
    re.IGNORECASE)
_NEGATION = re.compile(r"\b(?:no|not|nunca|never|ni|nor|tampoco|neither|sin|without|isn't|aren't|does\s+not|"
                       r"do\s+not|is\s+not|are\s+not)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"[.;:\n]+")


def _numbers_in(obj):
    """Every number the answers carry, as strings in several renderings, so a narration can be checked against them.

    Returns two sets: plain renderings of every number, and percent renderings of those numbers that live under a key
    naming a probability or a share. 0.5412 kW gains no "54"; 0.9666 under `uncalibrated_probabilities` gains "96.66"."""
    plain, percent = set(), set()

    def walk(value, key=""):
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            plain.update({str(value), f"{value:.2f}", f"{value:.4f}", f"{value:.1f}"})
            if isinstance(value, float) and 0 <= value <= 1 and _PROBABILITY_KEY.search(key):
                percent.update({f"{value * 100:.1f}", f"{value * 100:.0f}", f"{value * 100:.2f}"})
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{key}.{k}" if _PROBABILITY_KEY.search(key) else str(k))
        elif isinstance(value, list):
            for v in value:
                walk(v, key)
        elif isinstance(value, str):
            plain.update(_NUMBER.findall(value))

    walk(obj)
    return plain, percent


def _matches(token, allowed):
    cleaned = token.replace(",", ".")
    if cleaned in allowed or cleaned.lstrip("-") in allowed:
        return True
    try:
        value = float(cleaned)
    except ValueError:
        return False
    return any(abs(float(a) - value) < 1e-9 for a in allowed if _NUMBER.fullmatch(a))


def narration_problems(text, answers):
    """Why a narration is not a faithful reading of the answers; empty when it is.

    Three checks, all deterministic. Every digit in the text must be a number the answers carry. A percent sign may only
    follow a rendering of a declared probability, never of a power or a price. A quantity said in words ("el doble",
    "half") and a claim of profit, loss or an order are refused outright unless the sentence they sit in negates them,
    because the answers carry no such thing and a digit-only check cannot see either."""
    plain, percent = _numbers_in(answers)
    problems = []
    for match in _PERCENT.finditer(text):
        if not _matches(match.group(1), percent):
            problems.append(f"'{match.group(0).strip()}' is a percent of nothing the answers carry as a probability")
    stripped = _PERCENT.sub(" ", text)
    for token in _NUMBER.findall(stripped):
        if not _matches(token, plain):
            problems.append(f"'{token}' is a figure the answers do not carry")
    for sentence in _SENTENCE.split(text):
        negated = bool(_NEGATION.search(sentence))
        for match in _RELATIVE_QUANTITY.finditer(sentence):
            problems.append(f"'{match.group(0).strip()}' is a quantity said in words; the answers carry no ratio")
        if not negated:
            for match in _ACTION_CLAIM.finditer(sentence):
                problems.append(f"'{match.group(0)}' claims a profit, a loss or an order; the answers carry none")
    return problems


# `NOT_NARRATED` and `narratable` moved to `m5phet.outputs` at WP04, where every procedure and the guard read the same
# view of an answer; they stay importable from here because that is where the rest of the package already asks.


def narration_is_faithful(text, answers):
    """True when every number in the text is one the answers carry and no claim goes beyond them. A narration may leave
    figures out; it may not add, scale or act on them."""
    return not narration_problems(text, answers)


def render(response):
    """A deterministic sentence per answer. Plain, and always faithful by construction.

    The rendering itself lives in the `default` output plugin (WP04); this is where the rest of the package reaches
    it, and it is what the narration guard falls back to when a model -- or a plugin -- says more than the answers do.
    A value too long to print is cut in a way that never leaves half a number behind."""
    return default_output.text(response)


def _asked_view(task, response):
    """The numbers and words the CALLER put in the envelope: the question fields and the state values.

    A narration may repeat what was asked -- the horizon, the confidence level, the policy id -- because the person
    wrote it. It may not invent a number the answers do not carry; that check is unchanged."""
    envelope = task if isinstance(task, dict) else response.get("task")
    if not isinstance(envelope, dict):
        return {}
    return {"questions": envelope.get("questions") or {}, "state": envelope.get("state") or {}}


def narrate(prompt, response, *, interpreter=None, language="es", area=None, plugin=None, task=None):
    """What the person reads: the configured output procedure's text, checked against the answers it was given.

    Two things happen here and both are guarded by the same rule. The area's output plugin (WP04) renders the answers
    -- the workbench's deterministic lines, a Telegram message, tomorrow something nobody has written yet -- and that
    text is checked against the answers before anyone sees it, because a plugin can be wrong or hostile exactly as a
    model can, and neither may introduce a figure. Then, for a procedure that declares `narrates`, the configured
    interpreter is asked for a sentence about the same answers, and it is kept only if every number in it is one the
    answers carry.

    Whenever either fails, the deterministic rendering stands. There is no path from "this text is not faithful" to
    showing it anyway."""
    interpreter = interpreter if interpreter is not None else build_interpreter()
    if plugin is None:
        # RB04: precedence is explicit argument, then the ENVELOPE's own selection, then the area's configuration,
        # then `default`. The request may choose the surface because a Telegram skill and the workbench ask the same
        # engine and read the answer differently; what it cannot do is name a procedure that is not installed, which
        # `check_proposal` refuses before any of this is reached.
        selected = ((task or {}).get("output") or {}).get("plugin") if isinstance(task, dict) else None
        plugin = (load_output(selected)({"plugin": selected}) if isinstance(selected, str) and selected.strip()
                  else select_output(area if area is not None else response.get("area")))
    fallback = render(response)
    plugin_name = getattr(plugin, "name", type(plugin).__name__)
    rendered = plugin.render(area if area is not None else response.get("area"), response, language)
    produced = rendered.get("text") or ""
    # a plugin is held to the narration guard, not trusted by being installed: the answers as the person reads them,
    # their names and the envelope's own counts are everything a rendering may state
    lying = narration_problems(produced, response_view(response))
    if lying:
        return {"text": fallback, "source": "DETERMINISTIC", "faithful": True, "interpreter": interpreter.identity(),
                "output_plugin": plugin_name, "table": None, "json": rendered.get("json"),
                "why": f"the output plugin {plugin_name!r} produced a figure or a claim the answers do not carry; it "
                       "was discarded: " + "; ".join(lying[:4]),
                "discarded": produced[:4000] or None}
    if not getattr(plugin, "narrates", False):
        return {"text": produced, "source": "PLUGIN", "faithful": True, "interpreter": interpreter.identity(),
                "output_plugin": plugin_name, "table": rendered.get("table"), "json": rendered.get("json")}
    fallback = produced or fallback
    if not interpreter.available:
        return {"text": fallback, "source": "DETERMINISTIC", "faithful": True, "interpreter": interpreter.identity(),
                "output_plugin": plugin_name, "table": rendered.get("table"), "json": rendered.get("json")}
    # The narrator reads what the person is meant to read. `sdk_answer` (the backend's verbatim object, kept for
    # parity checks) and `provenance` (digests) are not that: a raw SDK field named "confidence" was narrated as
    # "confianza 0.3403" once, a number the answers deliberately do not surface. The guard checks the same view.
    answers = narratable(response.get("answers") or {})
    # The person's own envelope is not an invention: a horizon of 60 steps or a confidence level of 0.95 is a number
    # they wrote, and a narration that repeats it is faithful. Refusing it discarded correct sentences about refused
    # questions, whose answers carry no numbers at all (2026-09-24: "'60' is a figure the answers do not carry").
    asked = _asked_view(task, response)
    instruction = (
        f"Write a short answer in {'Spanish' if language == 'es' else 'English'} for a person who asked: "
        f"{json.dumps(prompt)}\n"
        f"Use ONLY these results, quoting numbers exactly as they appear and never adding any: {json.dumps(answers)}\n"
        "Say plainly which questions were refused and why. Do not recommend an action. Plain text, no JSON."
    )
    try:
        text = interpreter._ask(instruction).strip()
    except Exception:                                                   # noqa: BLE001
        return {"text": fallback, "source": "DETERMINISTIC", "faithful": True, "interpreter": interpreter.identity(),
                "output_plugin": plugin_name, "table": rendered.get("table"), "json": rendered.get("json"),
                "why": "the interpreter could not be consulted"}
    problems = narration_problems(text, {"answers": answers, "asked": asked}) if text else \
        ["the interpreter returned nothing"]
    if not problems:
        return {"text": text, "source": "INTERPRETER", "faithful": True, "interpreter": interpreter.identity(),
                "output_plugin": plugin_name, "table": rendered.get("table"), "json": rendered.get("json")}
    return {"text": fallback, "source": "DETERMINISTIC", "faithful": True, "interpreter": interpreter.identity(),
            "output_plugin": plugin_name, "table": rendered.get("table"), "json": rendered.get("json"),
            "why": "the interpreter's narration introduced a figure or a claim the answers do not carry; it was "
                   "discarded: " + "; ".join(problems[:4]),
            "discarded": text[:4000] if text else None}
