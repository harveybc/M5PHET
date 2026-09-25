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
import re

from .interpret import build as build_interpreter
from .outputs import NOT_NARRATED, default as default_output, narratable, response_view, select as select_output
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
    for name, question in task["questions"].items():
        if question["type"] not in declared:
            problems.append(f"question {name!r}: type {question['type']!r} is not one this area answers "
                            f"({sorted(declared)})")
        # a field the provider governs -- target, horizon, policy, study -- must hold one of its declared values. This is
        # the check that stops a router from naming a DATA column as a fitted target and getting a confident refusal.
        for field, allowed in parameters.items():
            if field in question and question[field] not in allowed:
                problems.append(f"question {name!r}: {field} {question[field]!r} is not one the engine has "
                                f"({allowed})")
    # `state.target_variable` is the owner's spelling of the forecaster's `target`; both are governed by the same list
    aliases = {"target_variable": "target"}
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
    for field, allowed in parameters.items():
        for spelling in [field] + [a for a, canonical in aliases.items() if canonical == field]:
            if spelling in task["state"] and task["state"][spelling] not in allowed:
                problems.append(f"state.{spelling} {task['state'][spelling]!r} is not one the engine has ({allowed})")
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


def route(prompt, data, registry, *, interpreter=None, datasets=None, decider=None):
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
    if task is not None and chosen:
        # what runs records which dataset it read, so the envelope beside the answers is replayable and the person
        # reviewing it sees the choice rather than having to trust it
        task["state"].setdefault("dataset", chosen["id"])
    return {**report, "status": "OK" if task else "INVALID_PROPOSAL", "proposal": proposal, "task": task,
            "problems": problems,
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
    plugin = plugin if plugin is not None else select_output(area if area is not None else response.get("area"))
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
