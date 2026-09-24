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


class Interpreter:
    """The optional language model, reached through the operator's configured command.

    It is given the question and the allowed values. It is not given the data, a tool, a path or a shell."""

    def __init__(self, command=None, model=None, environ=None):
        env = os.environ if environ is None else environ
        self.command = command or env.get("M5PHET_INTERPRETER_COMMAND") or ""
        self.model = model or env.get("M5PHET_INTERPRETER_MODEL") or ""
        self.timeout = int(env.get("M5PHET_INTERPRETER_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))

    @property
    def available(self):
        if not self.command:
            return False
        return shutil.which(self.command.split()[0]) is not None or os.path.isfile(self.command.split()[0])

    def identity(self):
        return {"command": self.command.split()[0] if self.command else None, "model": self.model or None,
                "available": self.available,
                "reading": "the model chooses among declared values; it cannot introduce one"}

    def _ask(self, text):
        argv = self.command.split() + ["-z", text]
        done = subprocess.run(argv, text=True, capture_output=True, timeout=self.timeout,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        if done.returncode:
            raise ValueError(f"interpreter failed ({done.returncode}): {done.stderr.strip()[-160:]}")
        return done.stdout

    def propose(self, prompt, slots):
        """Ask for one JSON object whose every value is an allowed one, or null."""
        vocabulary = {slot["name"]: slot["allowed"] for slot in slots}
        instruction = (
            "Return ONLY compact JSON and no prose. Choose values that appear in the allowed lists; "
            "use null when the request does not name one. Never invent a value that is not listed.\n"
            f"Request: {json.dumps(prompt)}\n"
            f"Allowed values per field: {json.dumps(vocabulary)}\n"
            "Answer with an object having exactly these fields: " + json.dumps(sorted(vocabulary)))
        raw = self._ask(instruction)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("the interpreter returned no JSON object")
        proposed = json.loads(match.group(0))
        if not isinstance(proposed, dict):
            raise ValueError("the interpreter returned something that is not an object")
        return proposed


def interpret(prompt, slots, *, interpreter=None):
    """Resolve a person's question into declared parameters, or refuse and say exactly what is missing."""
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

    # a value the person named that the model does not have must be refused BEFORE anything tries to be helpful
    impossible = unsupported_numbers(prompt, slots)
    if impossible:
        field, detail = sorted(impossible.items())[0]
        return {**report, "status": STATUS_UNSUPPORTED,
                "why": (f"the question names {field} {detail['named']} and this fitted model only has "
                        f"{detail['allowed']}; answering the nearest one would answer a different question")}

    resolved, ambiguous, unresolved = deterministic(prompt, slots)
    report["parameters"] = dict(resolved)
    report["sources"] = {name: "QUESTION_TEXT" for name in resolved}
    if ambiguous:
        field, options = sorted(ambiguous.items())[0]
        return {**report, "status": STATUS_AMBIGUOUS,
                "why": f"the question names more than one {field}: {options}; say which one"}
    if not unresolved:
        return {**report, "status": STATUS_OK}

    interpreter = interpreter if interpreter is not None else Interpreter()
    report["interpreter"] = interpreter.identity()
    if not interpreter.available:
        return {**report, "status": STATUS_MISSING,
                "why": (f"the question does not name a supported {', '.join(sorted(unresolved))}, and no interpreter is "
                        f"configured. This model has {_vocabulary(slots, unresolved)}")}
    pending = [slot for slot in slots if slot["name"] in unresolved]
    try:
        proposed = interpreter.propose(prompt, pending)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        return {**report, "status": STATUS_MISSING,
                "why": f"the interpreter could not be consulted ({error}); name the missing values in your question"}
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
        report["parameters"][slot["name"]] = value
        report["sources"][slot["name"]] = "INTERPRETER"
    still_missing = [slot["name"] for slot in pending if slot["name"] not in report["parameters"] and slot.get("required", True)]
    if still_missing:
        return {**report, "status": STATUS_MISSING,
                "why": (f"the question does not name a supported {', '.join(sorted(still_missing))}. "
                        f"This model has {_vocabulary(slots, still_missing)}")}
    return {**report, "status": STATUS_OK}
