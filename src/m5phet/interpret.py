"""Turning a person's words into declared parameters, without letting a language model decide anything.

The problem this solves is narrow and worth stating exactly. A provider knows which parameters it needs and, for each one,
which values are admissible -- its fitted targets, its trained horizons, its declared options. A person does not write in
those words. `forecast Global_active_power at 60 steps` is what the engine understands; "predict household power an hour
ahead" is what someone types.

The rule here is that the language model may only CHOOSE, never invent. It is shown the question and the allowed values, and
its answer is accepted only if every value it returns is one of those already declared. A target that is not in the bundle, a
horizon the model was not trained for, a number it inferred from context: each is refused by name. So the worst a wrong
interpretation can do is fail loudly, and the worst a hostile prompt can do is be refused.

Three further boundaries, each because the alternative is a quiet mistake:

* the deterministic pass runs first and, when it resolves everything, no model is consulted at all;
* the model never sees the uploaded data -- only the question and the vocabulary, so a dataset is not shipped anywhere to
  have a sentence parsed;
* an ambiguous question is refused with the candidates named, rather than resolved by picking the first.

The interpreter's own identity travels in the report, because which model read the question is part of how the answer came
about.
"""

import json
import os
import re
import shutil
import subprocess

STATUS_OK = "OK"
STATUS_AMBIGUOUS = "AMBIGUOUS"
STATUS_UNSUPPORTED = "UNSUPPORTED_VALUE"
STATUS_MISSING = "MISSING_PARAMETER"
#: the interpreter chose, and chose at a confidence the checkpoint was MEASURED to be unreliable at. Not an error: a
#: refusal to choose, which is why it names the parameter and the declared values rather than a stack of internals
STATUS_LOW_CONFIDENCE = "LOW_CONFIDENCE_ABSTAINED"
#: a threshold is declared and this plugin cannot say how sure the model was, so the rule cannot be applied. Refused
#: rather than passed, because passing would mean the operator believes a gate is running that is not
STATUS_CONFIDENCE_NOT_REPORTED = "CONFIDENCE_NOT_REPORTED"

#: what the catalog says for a plugin whose replies carry no probability
CONFIDENCE_NOT_REPORTED = "CONFIDENCE_NOT_REPORTED"
CONFIDENCE_REPORTED = "CONFIDENCE_REPORTED"

#: the statuses that mean "no value was chosen", as against "this engine cannot serve the request". The web shows these
#: as a refusal to choose in the review panel instead of an error, and a consumer can tell them apart by name.
ABSTENTION_STATUSES = (STATUS_LOW_CONFIDENCE, STATUS_CONFIDENCE_NOT_REPORTED)

DEFAULT_TIMEOUT_SECONDS = 90
MAX_PROMPT_CHARACTERS = 2000


class SlotError(ValueError):
    """A slot declaration this module cannot use. Raised at the provider, never at the person typing."""


def _check_slots(slots):
    if not isinstance(slots, list) or not slots:
        raise SlotError("a provider's slots must be a non-empty list")
    for slot in slots:
        if not isinstance(slot, dict) or not isinstance(slot.get("name"), str) or not slot["name"].strip():
            raise SlotError("every slot needs a name")
        allowed = slot.get("allowed")
        if not isinstance(allowed, list) or not allowed:
            raise SlotError(f"slot {slot['name']!r} must declare the values it allows; an open slot cannot be validated")
        if slot.get("type", "string") not in ("string", "integer"):
            raise SlotError(f"slot {slot['name']!r} has an unsupported type")
    return slots


def _candidates(prompt, slot):
    """Every allowed value this prompt could be naming, by exact token, declared alias or -- for integers -- by number."""
    lowered = prompt.lower()
    found = []
    for value in slot["allowed"]:
        names = [str(value)] + list((slot.get("aliases") or {}).get(str(value), []))
        for name in names:
            token = str(name).lower()
            if not token:
                continue
            pattern = r"(?<![a-z0-9_])" + re.escape(token) + r"(?![a-z0-9_])"
            if re.search(pattern, lowered):
                found.append(value)
                break
    return found


def deterministic(prompt, slots):
    """What the words themselves settle, using only the provider's declared vocabulary."""
    resolved, ambiguous, unresolved = {}, {}, []
    for slot in slots:
        found = _candidates(prompt, slot)
        unique = list(dict.fromkeys(found))
        if len(unique) == 1:
            resolved[slot["name"]] = unique[0]
        elif len(unique) > 1:
            ambiguous[slot["name"]] = unique
        else:
            unresolved.append(slot["name"])
    return resolved, ambiguous, unresolved


def unsupported_numbers(prompt, slots):
    """Numbers the person named that are NOT admissible, so an unsupported horizon is refused instead of ignored.

    Without this, "forecast 90 steps ahead" would leave the horizon unresolved, and a helpful interpreter would fill in the
    one supported value -- answering a question nobody asked."""
    problems = {}
    numbers = {int(n) for n in re.findall(r"(?<![a-z0-9_.])(\d{1,6})(?![a-z0-9_.])", prompt.lower())}
    for slot in slots:
        if slot.get("type") != "integer":
            continue
        allowed = {int(v) for v in slot["allowed"]}
        named = numbers - allowed
        # a number is only evidence about THIS slot when the slot's own words are nearby; a bare year is not a horizon
        hint = slot.get("number_hints") or []
        if named and (not hint or any(re.search(re.escape(word), prompt.lower()) for word in hint)):
            problems[slot["name"]] = {"named": sorted(named), "allowed": sorted(allowed)}
    return problems


def _vocabulary(slots, names):
    """Say what the model DOES have. A refusal that only says something is missing leaves the person guessing."""
    parts = []
    for slot in slots:
        if slot["name"] in names:
            parts.append(f"{slot['name']}: {list(slot['allowed'])}")
    return "; ".join(parts)


def unsupported_named(prompt, slots):
    """Values the person NAMED that this model does not have, using each slot's declared `known_unsupported` vocabulary.

    This is the string analogue of `unsupported_numbers`, and it exists because of a failure observed in the running product.
    Asked to forecast `Voltage` when the bundle holds only `Global_active_power`, the deterministic pass left the slot
    unresolved and the question went to the interpreter -- which was asked to choose among the allowed values and helpfully
    chose the only one. The answer came back as a confident forecast of a different series, and because a language model is
    not deterministic it did so only sometimes.

    A provider that can enumerate what it does NOT serve -- the other columns of its own bundle, the studies it did not fit --
    declares them here, and naming one is refused before any interpreter is consulted."""
    problems = {}
    lowered = prompt.lower()
    for slot in slots:
        named = []
        for value in slot.get("known_unsupported") or ():
            token = str(value).lower()
            if token and re.search(r"(?<![a-z0-9_])" + re.escape(token) + r"(?![a-z0-9_])", lowered):
                named.append(value)
        if named:
            problems[slot["name"]] = {"named": sorted(named), "allowed": list(slot["allowed"])}
    return problems


class Interpreter:
    """The optional language model, reached through the operator's configured command.

    It is given the question and the allowed values. It is not given the data, a tool, a path or a shell.

    This class is also the `command` plugin's behaviour and the base every other interpreter plugin subclasses: a
    plugin implements `_ask(text) -> str` and inherits `propose`, which is where the rule that the model may only
    choose among declared values is enforced. `build()` below is how one is selected by configuration."""

    #: the name this implementation is selected by in `interpreter.plugin`
    plugin = "command"

    #: whether this implementation can say HOW SURE the model was of the values it chose. `False` here and in every
    #: plugin that reads a model's text: a program that prints an answer prints no probability, and there is no way to
    #: recover one from the words. This is declared rather than defaulted to 1.0 for the obvious reason -- a confidence
    #: nobody measured, used to pass a threshold somebody measured, is the exact failure this whole rule exists to
    #: prevent. `m5phet.interpret.interpret` refuses when a threshold is declared and this is `False`; the catalog
    #: says `CONFIDENCE_NOT_REPORTED`, and neither pretends the rule was applied.
    reports_confidence = False

    def __init__(self, command=None, model=None, environ=None):
        env = os.environ if environ is None else environ
        self.environ = env
        self.command = command or env.get("M5PHET_INTERPRETER_COMMAND") or ""
        self.model = model or env.get("M5PHET_INTERPRETER_MODEL") or ""
        self.timeout = int(env.get("M5PHET_INTERPRETER_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))

    @property
    def available(self):
        if not self.command:
            return False
        return shutil.which(self.command.split()[0]) is not None or os.path.isfile(self.command.split()[0])

    def identity(self):
        return {"plugin": self.plugin,
                "command": self.command.split()[0] if self.command else None, "model": self.model or None,
                "available": self.available,
                "reports_confidence": self.reports_confidence,
                "reading": "the model chooses among declared values; it cannot introduce one"}

    def _ask(self, text):
        argv = self.command.split() + ["-z", text]
        done = subprocess.run(argv, text=True, capture_output=True, timeout=self.timeout,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        if done.returncode:
            raise ValueError(f"interpreter failed ({done.returncode}): {done.stderr.strip()[-160:]}")
        return done.stdout

    def instruction(self, prompt, slots):
        """The one thing the model is shown: the question, and the values it may choose between. Never the data."""
        vocabulary = {slot["name"]: slot["allowed"] for slot in slots}
        return ("Return ONLY compact JSON and no prose. Choose values that appear in the allowed lists; "
                "use null when the request does not name one. Never invent a value that is not listed.\n"
                f"Request: {json.dumps(prompt)}\n"
                f"Allowed values per field: {json.dumps(vocabulary)}\n"
                "Answer with an object having exactly these fields: " + json.dumps(sorted(vocabulary)))

    @staticmethod
    def parse(raw):
        """The JSON object inside whatever the model wrote, or a `ValueError` naming what was wrong with it."""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("the interpreter returned no JSON object")
        proposed = json.loads(match.group(0))
        if not isinstance(proposed, dict):
            raise ValueError("the interpreter returned something that is not an object")
        return proposed

    def propose(self, prompt, slots):
        """Ask for one JSON object whose every value is an allowed one, or null."""
        return self.parse(self._ask(self.instruction(prompt, slots)))

    def propose_with_confidence(self, prompt, slots):
        """`(proposed, confidences)`, where `confidences` is `None` for a plugin that cannot report one.

        `None` is the honest answer and the only one this base class can give: the reply is text. A plugin that CAN
        report a per-field confidence overrides this and returns `{field: probability}` for the fields it measured;
        a field missing from that mapping is a field it did not measure, which is not the same as a low confidence and
        is not treated as one."""
        return self.propose(prompt, slots), None


def build(settings=None, environ=None):
    """The interpreter this installation is configured to use.

    `interpreter.plugin` in `~/.config/m5phet/m5phet.json` names it and defaults to `command`; the block is handed to
    the plugin unchanged, so a plugin's own settings (a model, a base URL, a consent) need no change here. With no JSON
    configuration the `command` plugin reads `M5PHET_INTERPRETER_COMMAND` exactly as it always has, which is why an
    operator who never writes a JSON file sees no difference.

    A configuration that cannot be READ is not a reason to run with a different interpreter than the one asked for:
    `ConfigError` is raised, as everywhere else. A configuration FILE that does not exist is not that -- it means the
    environment is the configuration, which is a supported way to run and is what `config.load()` returns."""
    env = os.environ if environ is None else environ
    if settings is None:
        from . import config as configuration
        settings = configuration.load(environ=env).interpreter
    settings = dict(settings or {})
    from .interpreters import DEFAULT_PLUGIN, load as load_plugin
    return load_plugin(settings.get("plugin") or DEFAULT_PLUGIN)(settings, environ=env)


#: the schema `tools/measure_interpreter.py` writes, and the only document a reliability may be published from
RELIABILITY_REPORT_SCHEMA = "m5phet_interpreter_reliability.v1"

#: nobody has measured this installation's interpreter. The honest answer, and the default
NOT_MEASURED = "NOT_MEASURED"
#: the cited file is not a reliability report this framework wrote
RELIABILITY_REPORT_UNREADABLE = "RELIABILITY_REPORT_UNREADABLE"
#: the report measured a DIFFERENT interpreter. A number measured on another model is not a fact about this one
RELIABILITY_MEASURED_ON_ANOTHER_INTERPRETER = "RELIABILITY_MEASURED_ON_ANOTHER_INTERPRETER"

RELIABILITY_VARIABLE = "M5PHET_INTERPRETER_RELIABILITY_REPORT"


def declared_reliability(configuration=None, environ=None, interpreter=None):
    """How often the configured interpreter resolves a sentence correctly -- cited, or `NOT_MEASURED`.

    Until 2026-09-25 no report in this repository carried this number for any interpreter. Every quality figure was
    about an engine; the component that decides WHICH engine question gets asked was argued about ("it can only choose
    among declared values") and never scored. `tools/measure_interpreter.py` scores it on the 14 sentences whose
    correct resolution the framework already knows, and `interpreter.reliability_report` is where an installation says
    which run of it applies here.

    It is cited the same way the abstention threshold is, and refused the same way:

    * `NOT_MEASURED` when nothing is declared -- the default, and not a criticism of anyone;
    * `RELIABILITY_REPORT_UNREADABLE` when the path is not a `m5phet_interpreter_reliability.v1` document;
    * `RELIABILITY_MEASURED_ON_ANOTHER_INTERPRETER` when the report scored a different plugin or model. A rate
      measured on `llama3.2:3b` says nothing about `deepseek-v4-flash`, and publishing it beside this one's identity
      would be the same mistake as an uncited threshold.

    What is published is the rate, the two rates apart (strict, and when the model actually chose a value), the
    protocol, the counts and the report's digest -- so a reader can recompute it from the cited file."""
    import hashlib
    import json as _json
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
    named = settings.get("reliability_report") or env.get(RELIABILITY_VARIABLE)
    if not named:
        return NOT_MEASURED
    path = Path(os.path.expanduser(str(named)))
    try:
        raw = path.read_bytes()
        report = _json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        return {"refusal": RELIABILITY_REPORT_UNREADABLE, "why": f"{path} cannot be read as a reliability report: {error}"}
    if not isinstance(report, dict) or report.get("schema") != RELIABILITY_REPORT_SCHEMA:
        return {"refusal": RELIABILITY_REPORT_UNREADABLE,
                "why": f"{path} does not declare schema {RELIABILITY_REPORT_SCHEMA!r}; a reliability is published "
                       f"only from a report this framework wrote"}
    measured = (report.get("interpreter_path") or {})
    summary = measured.get("summary") or {}
    identity = measured.get("interpreter") or report.get("interpreter") or {}
    if not summary or summary.get("reliability") is None:
        return {"refusal": RELIABILITY_REPORT_UNREADABLE,
                "why": f"{path} carries no interpreter_path summary with a reliability; it measured the product path "
                       f"only, which is the deterministic pass and not this component"}
    if interpreter is not None:
        here = interpreter.identity()
        if (identity.get("plugin"), identity.get("model")) != (here.get("plugin"), here.get("model")):
            return {"refusal": RELIABILITY_MEASURED_ON_ANOTHER_INTERPRETER,
                    "why": (f"{path} measured {identity.get('plugin')!r} / {identity.get('model')!r} and this "
                            f"installation runs {here.get('plugin')!r} / {here.get('model')!r}; a rate measured on "
                            f"another model is not a fact about this one")}
    return {"reliability": summary.get("reliability"),
            "reliability_when_it_chose": summary.get("reliability_when_it_chose"),
            "n": {"sentences": summary.get("sentences"), "runs": summary.get("runs"),
                  "runs_per_sentence": measured.get("runs_per_sentence")},
            "verdicts": summary.get("verdicts"),
            "protocol": measured.get("protocol"),
            "corpus": "tools/verify_families.py PROSE (14 sentences; expectations in tools/measure_interpreter.py)",
            "measured_at": report.get("measured_at"),
            "measured_interpreter": {"plugin": identity.get("plugin"), "model": identity.get("model")},
            "report_sha256": hashlib.sha256(raw).hexdigest()}


def _declared(slots, names):
    """`{name: [allowed values]}` for the slots named, so a refusal carries the vocabulary as data, not only as prose."""
    return {slot["name"]: list(slot["allowed"]) for slot in slots if slot["name"] in names}


def abstention_rule(configuration=None, environ=None):
    """`(citation, unresolved)` for the installation's declared threshold; imported here to avoid an import cycle.

    `m5phet.decide` owns the rule -- it is where WP09's report is read and where a threshold the report cannot see is
    refused -- and it imports `m5phet.questions`, which this module sits underneath. So it is reached at call time.
    The interpreter and the chooser answer to the SAME declaration on purpose: one measurement, one number, cited once,
    applying wherever a language model picks a value in this product."""
    from . import decide
    return decide.declared_threshold(configuration, environ)


def interpret(prompt, slots, *, interpreter=None, configuration=None, environ=None):
    """Resolve a person's question into declared parameters, or refuse and say exactly what is missing.

    When the installation declares an abstention threshold (`interpreter.min_confidence`, cited from
    `interpreter.abstention_source`), the interpreter's own choices are held to it exactly as Laya's are. There is no
    third way between the two outcomes: either the plugin reports how sure it was and the value is kept only at or
    above the cited threshold, or it reports nothing and the question is refused as `CONFIDENCE_NOT_REPORTED`, saying
    so. What cannot happen is a value chosen by a model passing a gate that was never applied to it.

    With no threshold declared -- every installation until an operator writes one -- nothing below changes at all."""
    _check_slots(slots)
    if not isinstance(prompt, str) or not prompt.strip():
        return {"status": STATUS_MISSING, "parameters": {}, "sources": {},
                "why": "the question is empty", "interpreter": None}
    if len(prompt) > MAX_PROMPT_CHARACTERS:
        return {"status": STATUS_MISSING, "parameters": {}, "sources": {},
                "why": f"the question is {len(prompt)} characters and the limit is {MAX_PROMPT_CHARACTERS}",
                "interpreter": None}

    report = {"parameters": {}, "sources": {}, "interpreter": None,
              "reading": ("values are chosen from what the provider declared; nothing here can introduce a target, a "
                          "horizon or an option the fitted model does not have")}

    # A value the person named that the model does not have must be refused BEFORE anything tries to be helpful. Both
    # checks run first for the same reason: an interpreter asked to choose among allowed values will choose one, and a
    # confident answer about the wrong series is worse than a refusal.
    impossible = dict(unsupported_named(prompt, slots))
    impossible.update(unsupported_numbers(prompt, slots))
    if impossible:
        field, detail = sorted(impossible.items())[0]
        return {**report, "status": STATUS_UNSUPPORTED,
                "why": (f"the question names {field} {detail['named']} and this fitted model only has "
                        f"{detail['allowed']}; answering the nearest one would answer a different question")}

    resolved, ambiguous, unresolved = deterministic(prompt, slots)
    # A slot a provider declares OPTIONAL (`required: False`) is a value the engine can settle on its own -- which
    # fitted bundle answers, when the question already says enough. It is still offered to the interpreter below, so
    # words CAN choose it; what it must never do is turn a question that names everything required into a refusal.
    optional = {slot["name"] for slot in slots if slot.get("required", True) is False}
    report["parameters"] = dict(resolved)
    report["sources"] = {name: "QUESTION_TEXT" for name in resolved}
    if ambiguous:
        field, options = sorted(ambiguous.items())[0]
        return {**report, "status": STATUS_AMBIGUOUS,
                "why": f"the question names more than one {field}: {options}; say which one"}
    if not unresolved:
        return {**report, "status": STATUS_OK}

    interpreter = interpreter if interpreter is not None else build()
    report["interpreter"] = interpreter.identity()
    if not interpreter.available:
        required_missing = sorted(set(unresolved) - optional)
        if not required_missing:
            return {**report, "status": STATUS_OK}
        return {**report, "status": STATUS_MISSING,
                "why": (f"the question does not name a supported {', '.join(required_missing)}, and no interpreter is "
                        f"configured. This model has {_vocabulary(slots, required_missing)}")}
    pending = [slot for slot in slots if slot["name"] in unresolved]
    names = [slot["name"] for slot in pending]

    # The rule, resolved BEFORE the model is consulted, so a threshold nobody measured costs no call and cannot be
    # judged against an answer somebody has already seen.
    citation, unmeasured = abstention_rule(configuration, environ)
    if unmeasured is not None:
        return {**report, "status": unmeasured[0], "unresolved": names, "declared": _declared(slots, names),
                "why": (f"{unmeasured[1]}. Until the declaration is repaired nothing is chosen by a model here; name "
                        f"the values in your question instead. This model has {_vocabulary(slots, names)}")}
    if citation is not None:
        report["abstention"] = {"min_confidence": citation["min_confidence"], "report_sha256": citation["sha256"],
                                "stage": citation["stage"]}
    if citation is not None and not getattr(interpreter, "reports_confidence", False):
        # declared, not faked. The threshold is real and this plugin cannot be held to it, so the question is refused
        # and says which of the two is missing -- the confidence, never the person's words.
        return {**report, "status": STATUS_CONFIDENCE_NOT_REPORTED, "unresolved": names,
                "declared": _declared(slots, names),
                "why": (f"the question does not name a supported {', '.join(sorted(names))}, so a model would have to "
                        f"choose it. This installation declares that a model's choice counts only at or above "
                        f"{citation['min_confidence']} (measured in {citation['stage']!r}, report sha256 "
                        f"{citation['sha256'][:12]}...), and the configured interpreter "
                        f"{interpreter.identity().get('plugin')!r} reports no confidence for what it chooses, so that "
                        f"rule cannot be applied to it. Nothing was chosen. Name the value yourself -- this model has "
                        f"{_vocabulary(slots, names)} -- or configure an interpreter that reports a confidence")}

    try:
        proposed, confidences = interpreter.propose_with_confidence(prompt, pending)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        return {**report, "status": STATUS_MISSING,
                "why": f"the interpreter could not be consulted ({error}); name the missing values in your question"}
    if confidences:
        report["confidences"] = {name: value for name, value in confidences.items() if name in unresolved}
    for slot in pending:
        value = proposed.get(slot["name"])
        if value is None:
            continue
        if slot.get("type") == "integer":
            try:
                value = int(value)
            except (TypeError, ValueError):
                return {**report, "status": STATUS_UNSUPPORTED,
                        "why": f"the interpreter proposed {proposed[slot['name']]!r} for {slot['name']}, which is not a number"}
        allowed = [int(v) for v in slot["allowed"]] if slot.get("type") == "integer" else list(slot["allowed"])
        if value not in allowed:
            return {**report, "status": STATUS_UNSUPPORTED,
                    "why": (f"the interpreter proposed {value!r} for {slot['name']} and the fitted model only has "
                            f"{allowed}; a proposal outside the declared values is refused, never rounded to a neighbour")}
        # The gate runs on a value that IS one of the declared ones: "the model proposed something nobody declared"
        # is a stronger fact than "it was unsure", and it keeps its own name (`UNSUPPORTED_VALUE`, just above).
        if citation is not None:
            confidence = (confidences or {}).get(slot["name"])
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                return {**report, "status": STATUS_CONFIDENCE_NOT_REPORTED, "unresolved": [slot["name"]],
                        "declared": _declared(slots, [slot["name"]]),
                        "why": (f"the interpreter proposed a {slot['name']} and reported no confidence for it, so the "
                                f"declared threshold {citation['min_confidence']} cannot be applied to that choice. "
                                f"Nothing was chosen. Name it yourself: {_vocabulary(slots, [slot['name']])}")}
            if float(confidence) < citation["min_confidence"]:
                return {**report, "status": STATUS_LOW_CONFIDENCE, "unresolved": [slot["name"]],
                        "declared": _declared(slots, [slot["name"]]),
                        "why": (f"the interpreter chose {value!r} for {slot['name']} at confidence {confidence}, below "
                                f"the declared threshold {citation['min_confidence']}. That threshold is not a "
                                f"preference: {citation['path']} measured this checkpoint on "
                                f"{citation['measured_rows_at_or_above']} rows at or above it and found it right "
                                f"{citation['measured_accuracy_at_or_above']} of the time, and at chance below it. So "
                                f"no {slot['name']} was chosen. Name it yourself -- this model has "
                                f"{_vocabulary(slots, [slot['name']])}")}
        report["parameters"][slot["name"]] = value
        report["sources"][slot["name"]] = "INTERPRETER"
    still_missing = [slot["name"] for slot in pending if slot["name"] not in report["parameters"] and slot.get("required", True)]
    if still_missing:
        return {**report, "status": STATUS_MISSING,
                "why": (f"the question does not name a supported {', '.join(sorted(still_missing))}. "
                        f"This model has {_vocabulary(slots, still_missing)}")}
    return {**report, "status": STATUS_OK}
