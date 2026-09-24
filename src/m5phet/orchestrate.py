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

from .interpret import Interpreter
from .questions import AREAS, TaskError, catalog as question_catalog, validate_task

MAX_PROMPT = 4000


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
    for name, question in task["questions"].items():
        if question["type"] not in declared:
            problems.append(f"question {name!r}: type {question['type']!r} is not one this area answers "
                            f"({sorted(declared)})")
    if profile.get("columns"):
        known = set(profile["columns"])
        for column in sorted(_columns_named(task)):
            if column not in known:
                problems.append(f"column {column!r} is not in the attached data ({profile['columns'][:12]}...)"
                                if len(profile["columns"]) > 12 else
                                f"column {column!r} is not in the attached data ({profile['columns']})")
    return (task if not problems else None), problems


def route(prompt, data, registry, *, interpreter=None):
    """Turn a sentence into a validated envelope, or say exactly why it could not be."""
    if not isinstance(prompt, str) or not prompt.strip():
        return {"status": "REFUSED", "why": "the question is empty", "proposal": None, "problems": []}
    if len(prompt) > MAX_PROMPT:
        return {"status": "REFUSED", "why": f"the question is {len(prompt)} characters; the limit is {MAX_PROMPT}",
                "proposal": None, "problems": []}
    catalog = question_catalog(registry)
    profile = dataset_profile(data)
    interpreter = interpreter if interpreter is not None else Interpreter()
    report = {"profile": profile, "catalog": catalog, "interpreter": interpreter.identity()}
    if not interpreter.available:
        return {**report, "status": "REFUSED", "proposal": None, "problems": [],
                "why": ("no interpreter is configured, so a sentence cannot be routed; write the envelope directly "
                        "(area, state, questions) or configure M5PHET_INTERPRETER_COMMAND")}
    served = {area: spec for area, spec in catalog.items() if spec.get("provider")}
    instruction = (
        "Return ONLY one compact JSON object and no prose.\n"
        "You are routing a request to a machine-learning engine. Choose the area, describe the state, and write the "
        "questions, using ONLY the areas and question types listed and ONLY column names present in the data profile. "
        "Never invent a column, a type or a field. If the request cannot be served by any listed area, return "
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
    return {**report, "status": "OK" if task else "INVALID_PROPOSAL", "proposal": proposal, "task": task,
            "problems": problems,
            "why": None if task else "the proposal names something the catalog or the data does not have; see problems"}


# --- the narration -----------------------------------------------------------------------------------------------------

_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def _numbers_in(obj):
    """Every number the answers carry, as strings in several renderings, so a narration can be checked against them."""
    found = set()

    def walk(value):
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            found.add(str(value))
            found.add(f"{value:.2f}")
            found.add(f"{value:.4f}")
            found.add(f"{value:.1f}")
            if isinstance(value, float) and 0 <= value <= 1:
                found.add(f"{value * 100:.1f}")
                found.add(f"{value * 100:.0f}")
                found.add(f"{value * 100:.2f}")
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        elif isinstance(value, str):
            for token in _NUMBER.findall(value):
                found.add(token)

    walk(obj)
    return found


def narration_is_faithful(text, answers):
    """True when every number in the text is one the answers carry. A narration may leave figures out; it may not add."""
    allowed = _numbers_in(answers)
    for token in _NUMBER.findall(text):
        cleaned = token.replace(",", ".")
        if cleaned in allowed or cleaned.lstrip("-") in allowed:
            continue
        try:
            value = float(cleaned)
        except ValueError:
            return False
        if any(abs(float(a) - value) < 1e-9 for a in allowed if _NUMBER.fullmatch(a)):
            continue
        return False
    return True


def render(response):
    """A deterministic sentence per answer. Plain, and always faithful by construction."""
    lines = []
    for name, answer in (response.get("answers") or {}).items():
        if answer.get("status") == "REFUSED":
            lines.append(f"{name}: not answered -- {answer.get('why')}")
            continue
        fields = {k: v for k, v in answer.items() if k not in ("type", "status", "execution_authorized", "sdk_answer")}
        summary = ", ".join(f"{k}={_short(v)}" for k, v in list(fields.items())[:6])
        lines.append(f"{name} ({answer.get('type')}): {summary}")
    lines.append(f"{response.get('answered', 0)} answered, {response.get('refused', 0)} refused; nothing here is an "
                 f"instruction to act.")
    return "\n".join(lines)


def _short(value):
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= 80 else text[:77] + "..."


def narrate(prompt, response, *, interpreter=None, language="es"):
    """A sentence about the answers, checked against them. Falls back to the deterministic rendering rather than let a
    number through that the engines did not produce."""
    interpreter = interpreter if interpreter is not None else Interpreter()
    fallback = render(response)
    if not interpreter.available:
        return {"text": fallback, "source": "DETERMINISTIC", "faithful": True, "interpreter": interpreter.identity()}
    answers = response.get("answers") or {}
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
                "why": "the interpreter could not be consulted"}
    if text and narration_is_faithful(text, answers):
        return {"text": text, "source": "INTERPRETER", "faithful": True, "interpreter": interpreter.identity()}
    return {"text": fallback, "source": "DETERMINISTIC", "faithful": True, "interpreter": interpreter.identity(),
            "why": "the interpreter's narration introduced a figure the answers do not carry; it was discarded",
            "discarded": text[:400] if text else None}
