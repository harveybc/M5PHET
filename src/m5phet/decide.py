"""Laya as a configuration chooser: a state, a declared option set, a recorded choice.

The owner's design (work plan revision 2, 2026-09-25) puts Laya in front of every area, not only classification. What
that means here is exact and narrow. Laya answers *state + typed choice questions -> typed answers with uncalibrated
probabilities*. So "Laya chooses X" is: the framework writes a **state** -- a structured, textual description of a
feature, a dataset, a problem -- asks a **`choice` question whose options are DECLARED by the repository that will
have to execute the choice**, and records the answer as a **decision record**.

Four things this module will not do, because the whole value of the primitive is that they cannot happen.

It never lets an option exist that nobody declared. The options travel in the question, the answer is checked back
against them, and a label outside the set is refused rather than recorded.

It never lets rows reach the model. `decision_state` renders a description; a list longer than `MAX_LIST_ITEMS` is
refused as `ROWS_IN_STATE`. A profile is what Laya sees, never the data.

It never records a number the answer did not carry. The probabilities are copied verbatim -- no rounding, no
rescaling, no renormalising -- and the number of decimals is the one the ANSWER declared.

It never records a choice that a model did not make. An answer whose `backend` is not `laya`, or which carries
`non_model_fixture`, is refused as `NON_MODEL_FIXTURE` and nothing is written. A decision made on a fixture would be
a hypothesis with a model's authority and none of its evidence.

And what it does not prove: nothing here says a choice is a good one. A decision is a hypothesis. The fit that follows
it and the closure table that scores it are the judge, and they live in other packages (WP18-WP21).
"""

import copy
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from m5phet.questions import MALFORMED_QUESTION, PROVIDER_ERROR, STATE_REQUIRED, refusal, run_task

DECISION_SCHEMA = "m5phet.decision.v1"

#: the area a decision is asked through: Laya's own, the one it answers end to end
AREA = "classification"

#: what the classification envelope's state says a decision is about. The provider needs an asset to stamp a text
#: state into an event; a configuration decision is about no instrument, so it is about this framework.
ASSET = "M5PHET"
LANGUAGE = "en"

#: decimals a float is rendered with in a state text. Declared, so the same profile always yields the same digest.
STATE_DECIMALS = 6

#: the longest list a state may carry. Past this it is data, not a description.
MAX_LIST_ITEMS = 128

#: the backend that may produce a decision. Anything else is a fixture or another model, and is refused by name.
LAYA_BACKEND = "laya"

# --- refusal codes a caller may match on -------------------------------------------------------------------------------
#: the answer came from a declared fixture or from a backend that is not Laya; no decision exists
NON_MODEL_FIXTURE = "NON_MODEL_FIXTURE"
#: the option list cannot be chosen from: fewer than two options, a duplicate key, a malformed pair
MALFORMED_OPTIONS = "MALFORMED_OPTIONS"
#: a choice question with no instruction text is not a question; the provider's own gate refuses it too
INSTRUCTIONS_REQUIRED = "INSTRUCTIONS_REQUIRED"
#: the answer named a label, or carried a probability for a key, that the declared option set does not contain
CHOICE_OUTSIDE_OPTIONS = "CHOICE_OUTSIDE_OPTIONS"
#: the answer carries no label or no probabilities, so there is nothing to record
NOT_A_DECISION = "NOT_A_DECISION"
#: the answer echoed an instruction or an option set other than the one that was asked
OPTIONS_MISMATCH = "OPTIONS_MISMATCH"
#: `decision_state` was handed rows instead of a description
ROWS_IN_STATE = "ROWS_IN_STATE"

# --- WP23: the outcome, and the refusals that keep it honest ------------------------------------------------------------
#: the schema of the record that links one decision to one closure-table row
OUTCOME_SCHEMA = "m5phet.decision_outcome.v1"
#: the row was not judged comparable by the table's own generator, so its rank ranks nothing
NOT_COMPARABLE = "NOT_COMPARABLE"
#: the row carries no rank, and the rank is the label; without it there is nothing to link
NOT_RANKED = "NOT_RANKED"
#: the decision record is not there, or is not a `m5phet.decision.v1` record at all
DECISION_NOT_FOUND = "DECISION_NOT_FOUND"
#: the record's bytes no longer hash to the name it is filed under: it was altered after it was written
DIGEST_MISMATCH = "DIGEST_MISMATCH"
#: the object handed in is not a row as `evaluation/compare_stages.py` emits it
MALFORMED_TABLE_ROW = "MALFORMED_TABLE_ROW"

#: the verdict `evaluation/compare_stages.py` writes into a row it found commensurable with the reference stage
COMPARABLE = "COMPARABLE"
#: the fields an outcome reads out of the row. Everything else in the row is bound by `table_row_sha256` instead.
TABLE_ROW_FIELDS = ("stage", "status", "comparability", "rank")


class DecisionError(ValueError):
    """A state cannot be rendered, or a record cannot be written or read back. Raised; never returned as a decision."""


# --- the state text ----------------------------------------------------------------------------------------------------

def decision_state(kind, payload, *, decimals=STATE_DECIMALS):
    """Render a structured payload as the deterministic text Laya is shown.

    Keys are sorted at every level, lists keep the caller's order, floats carry `decimals` decimals and integers stay
    integers (a lag of 74 is not 74.000000). The same payload -- whatever order its keys were built in -- always
    yields the same text and therefore the same `state_sha256`. That digest is what a decision record is bound to.
    """
    if not isinstance(kind, str) or not kind.strip():
        raise DecisionError("a decision kind must be a non-empty string")
    if not isinstance(payload, dict):
        raise DecisionError("a decision state is rendered from a mapping, not from " + type(payload).__name__)
    return "\n".join([f"decision_kind: {kind}", *_render(payload, 0, decimals)])


def state_sha256(state_text):
    """The digest of the state text exactly as it was shown to the model."""
    if not isinstance(state_text, str):
        raise DecisionError("a state text is a string")
    return hashlib.sha256(state_text.encode("utf-8")).hexdigest()


def _render(value, level, decimals):
    pad = "  " * level
    lines = []
    if isinstance(value, dict):
        for key in sorted(value):
            if not isinstance(key, str):
                raise DecisionError(f"a state key must be a string; {key!r} is {type(key).__name__}")
            item = value[key]
            if isinstance(item, (dict, list, tuple)):
                if not item:
                    lines.append(f"{pad}{key}: " + ("{}" if isinstance(item, dict) else "[]"))
                else:
                    lines.append(f"{pad}{key}:")
                    lines.extend(_render(item, level + 1, decimals))
            else:
                lines.append(f"{pad}{key}: {_scalar(item, decimals)}")
        return lines
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_LIST_ITEMS:
            raise DecisionError(f"{ROWS_IN_STATE}: a state describes, it does not carry rows; this list has "
                                f"{len(value)} entries and the limit is {MAX_LIST_ITEMS}")
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                lines.append(f"{pad}-")
                lines.extend(_render(item, level + 1, decimals))
            else:
                lines.append(f"{pad}- {_scalar(item, decimals)}")
        return lines
    raise DecisionError(f"a state carries mappings, lists and scalars; not {type(value).__name__}")


def _scalar(value, decimals):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DecisionError(f"a state carries finite numbers; {value!r} is not one")
        return f"{value:.{decimals}f}"
    if isinstance(value, str):
        # one fact per line: a newline inside a value is escaped rather than breaking the layout, so the rendering
        # stays one-to-one with the payload and the digest stays stable
        return value.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")
    raise DecisionError(f"a state carries null, booleans, numbers and strings; not {type(value).__name__}")


# --- asking ------------------------------------------------------------------------------------------------------------

def ask(engine_or_registry, state_text, questions, *, as_of=None, kind, record_dir=None):
    """Ask Laya to choose among DECLARED options for one state, and return one entry per question.

    `engine_or_registry` is either an `m5phet.web.engine.Engine` -- preferred, because a classification envelope then
    takes the private worker route exactly as the workbench does, reaching the real checkpoint -- or a bare `Registry`,
    in which case the envelope goes through `run_task` to whatever provider declares the classification area.

    `questions` maps a name to `{"options": [[key, label], ...], "instructions": str}`. Everything that can be checked
    without a model is checked here, BEFORE anything is asked: an empty state, an option list with fewer than two
    entries or a duplicate key, a question with no instructions. Each is refused under its own name and the remaining
    questions are still asked.

    `as_of` stamps the record. It is deliberately NOT put into the envelope: the classification provider refuses to
    replay a text state at a historical clock (`HISTORICAL_REPLAY_NEEDS_EVENT`), because a state text carries no
    clocks of its own. A decision is made now, about a description; it is not a replay of a news item.

    Returns `{name: entry}`. An entry is either `{"status": "OK", "decision": <record>, "record_path": str|None}` or a
    typed refusal (`{"status": "REFUSED", "refusal": ..., "why": ...}`). A refusal the provider itself raised --
    `TOKEN_BUDGET_EXCEEDED` among them -- is passed through verbatim, so its name survives this layer.
    """
    if not isinstance(kind, str) or not kind.strip():
        raise DecisionError("a decision kind must be a non-empty string")
    if not isinstance(questions, dict) or not questions:
        raise DecisionError("`questions` must be a non-empty mapping of name -> {options, instructions}")
    stamped = as_of if as_of is not None else datetime.now(timezone.utc).isoformat()

    out, askable, declared = {}, {}, {}
    state_ok = isinstance(state_text, str) and state_text.strip()
    for name, question in questions.items():
        if not isinstance(name, str) or not name.strip():
            raise DecisionError("every decision question needs a non-empty name")
        problem = _check_question(question)
        if problem is not None:
            out[name] = refusal(problem[0], problem[1], "choice")
            continue
        if not state_ok:
            out[name] = refusal(STATE_REQUIRED, "a decision needs a non-empty state text describing what is being "
                                                "chosen about; nothing was asked", "choice")
            continue
        options = [[str(key), str(label)] for key, label in question["options"]]
        declared[name] = options
        askable[name] = {"type": "choice", "options": options, "instructions": question["instructions"]}

    if askable:
        envelope = {"area": AREA, "state": {"news": state_text, "asset": ASSET, "language": LANGUAGE},
                    "questions": askable}
        try:
            response = _run(engine_or_registry, envelope)
        except Exception as error:                                              # noqa: BLE001
            response = None
            for name in askable:
                out[name] = refusal(PROVIDER_ERROR, f"the classification engine could not be reached: "
                                                    f"{type(error).__name__}: {error}", "choice")
        if response is not None:
            answers = response.get("answers") or {}
            checkpoint = response.get("state_ref")
            for name in askable:
                out[name] = _entry(answers.get(name), name=name, kind=kind, options=declared[name],
                                   instructions=askable[name]["instructions"], state_text=state_text,
                                   checkpoint=checkpoint, as_of=stamped, record_dir=record_dir)
    return {name: out[name] for name in questions}                      # the caller's order, always


def _run(engine_or_registry, envelope):
    """The worker route through an Engine, the in-process route through a Registry. Nothing else is accepted."""
    execute = getattr(engine_or_registry, "execute_task", None)
    if callable(execute):
        return execute("", envelope, [], language=LANGUAGE)["response"]
    if callable(getattr(engine_or_registry, "names", None)):
        return run_task(envelope, engine_or_registry)
    raise DecisionError("decide.ask takes an m5phet.web.engine.Engine or an m5phet.runtime.Registry")


def _check_question(question):
    """`None` when the question can be asked, else `(refusal_code, why)`. Nothing here needs a model."""
    if not isinstance(question, dict):
        return MALFORMED_QUESTION, "a decision question is a mapping with `options` and `instructions`"
    extra = sorted(set(question) - {"options", "instructions", "type"})
    if extra:
        return MALFORMED_QUESTION, f"a decision question does not take {extra}"
    if question.get("type", "choice") != "choice":
        return MALFORMED_QUESTION, f"a decision is a `choice`, not {question['type']!r}"
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        return INSTRUCTIONS_REQUIRED, ("a choice with no instruction text is not a question; say what is being "
                                       "chosen and on what grounds")
    options = question.get("options")
    if not isinstance(options, (list, tuple)):
        return MALFORMED_OPTIONS, "`options` must be an ordered list of [key, label] pairs"
    pairs = []
    for pair in options:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return MALFORMED_OPTIONS, f"every option is a [key, label] pair; {pair!r} is not"
        key, label = pair
        if not isinstance(key, str) or not key.strip() or not isinstance(label, str) or not label.strip():
            return MALFORMED_OPTIONS, f"an option key and its label are non-empty strings; {pair!r} is not"
        pairs.append(key)
    if len(pairs) < 2:
        return MALFORMED_OPTIONS, ("a decision needs at least two options to choose between; one option is not a "
                                   "choice, it is an instruction")
    duplicates = sorted({key for key in pairs if pairs.count(key) > 1})
    if duplicates:
        return MALFORMED_OPTIONS, f"the option keys {duplicates} appear more than once; a choice needs distinct keys"
    return None


def _entry(answer, *, name, kind, options, instructions, state_text, checkpoint, as_of, record_dir):
    """One answer becomes a decision, or a refusal. Every path that is not a real, in-set Laya choice refuses."""
    if not isinstance(answer, dict):
        return refusal(PROVIDER_ERROR, "the provider returned no answer for this question", "choice")
    if answer.get("status") != "OK":
        return copy.deepcopy(answer)                    # the provider's own refusal, name and reason untouched
    backend = answer.get("backend")
    if backend != LAYA_BACKEND or answer.get("non_model_fixture"):
        return refusal(NON_MODEL_FIXTURE, f"this answer came from backend {backend!r}"
                                          f"{' (a declared fixture)' if answer.get('non_model_fixture') else ''}; a "
                                          f"decision is only ever recorded from {LAYA_BACKEND!r}, because a choice "
                                          f"made by a non-model establishes nothing", "choice")
    if answer.get("options") is not None and [list(pair) for pair in answer["options"]] != options:
        return refusal(OPTIONS_MISMATCH, "the answer echoes an option set other than the one that was asked", "choice")
    if answer.get("instructions") is not None and answer["instructions"] != instructions:
        return refusal(OPTIONS_MISMATCH, "the answer echoes an instruction other than the one that was asked", "choice")
    chosen = answer.get("label")
    probabilities = answer.get("uncalibrated_probabilities")
    if not isinstance(chosen, str) or not isinstance(probabilities, dict) or not probabilities:
        return refusal(NOT_A_DECISION, "the answer carries no label or no uncalibrated probabilities", "choice")
    keys = [key for key, _label in options]
    if chosen not in keys:
        return refusal(CHOICE_OUTSIDE_OPTIONS, f"the answer chose {chosen!r}, which is not one of the declared "
                                               f"options {keys}", "choice")
    foreign = sorted(set(probabilities) - set(keys))
    if foreign:
        return refusal(CHOICE_OUTSIDE_OPTIONS, f"the answer carries probabilities for {foreign}, which the declared "
                                               f"option set {keys} does not contain", "choice")
    decision = {"schema": DECISION_SCHEMA,
                "kind": kind,
                "state_sha256": state_sha256(state_text),
                "question": name,
                "options": [list(pair) for pair in options],
                "chosen": chosen,
                # the model's own numbers, copied: no rounding, no rescaling, no renormalising
                "probabilities": copy.deepcopy(probabilities),
                "probability_decimals": answer.get("probability_decimals"),
                "checkpoint": checkpoint,
                "backend": backend,
                "as_of": as_of,
                "execution_authorized": False}
    entry = {"status": "OK", "decision": decision, "record_path": None}
    if record_dir is not None:
        entry["record_path"] = str(record(decision, record_dir))
    return entry


# --- the record ----------------------------------------------------------------------------------------------------------

def canonical_bytes(decision):
    return json.dumps(decision, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def decision_sha256(decision):
    """The digest of the decision's canonical JSON; also the name of the file it is written to."""
    return hashlib.sha256(canonical_bytes(decision)).hexdigest()


def validate_decision(decision):
    """Raise unless this is a complete decision record made by Laya and claiming no authority."""
    if not isinstance(decision, dict):
        raise DecisionError("a decision record is a mapping")
    if decision.get("schema") != DECISION_SCHEMA:
        raise DecisionError(f"schema {decision.get('schema')!r} is not {DECISION_SCHEMA!r}")
    expected = {"schema", "kind", "state_sha256", "question", "options", "chosen", "probabilities",
                "probability_decimals", "checkpoint", "backend", "as_of", "execution_authorized"}
    if set(decision) != expected:
        raise DecisionError(f"a {DECISION_SCHEMA} record carries exactly {sorted(expected)}; this one carries "
                            f"{sorted(decision)}")
    if decision["backend"] != LAYA_BACKEND:
        raise DecisionError(f"{NON_MODEL_FIXTURE}: backend {decision['backend']!r} is not {LAYA_BACKEND!r}; a "
                            f"decision is never recorded from anything else")
    if decision["execution_authorized"] is not False:
        raise DecisionError("a decision claims no execution authority; `execution_authorized` must be false")
    for field in ("kind", "question", "chosen", "state_sha256"):
        if not isinstance(decision[field], str) or not decision[field].strip():
            raise DecisionError(f"`{field}` must be a non-empty string")
    options = decision["options"]
    if not isinstance(options, list) or len(options) < 2:
        raise DecisionError("a decision carries the option set it was chosen from, at least two entries")
    keys = [pair[0] for pair in options]
    if decision["chosen"] not in keys:
        raise DecisionError(f"{CHOICE_OUTSIDE_OPTIONS}: {decision['chosen']!r} is not one of {keys}")
    probabilities = decision["probabilities"]
    if not isinstance(probabilities, dict) or not probabilities:
        raise DecisionError("a decision carries the uncalibrated probabilities the answer carried")
    if set(probabilities) - set(keys):
        raise DecisionError(f"{CHOICE_OUTSIDE_OPTIONS}: probabilities for {sorted(set(probabilities) - set(keys))}")
    if any(type(p) not in (int, float) or not math.isfinite(p) for p in probabilities.values()):
        raise DecisionError("a probability is a finite number")
    return decision


def record(decision, directory):
    """Write one decision as a content-addressed JSON file `<sha256>.json` and return its path.

    Content-addressed, so writing the same decision twice is the same file and a record cannot be silently revised:
    a different decision is a different name.
    """
    validate_decision(decision)
    payload = canonical_bytes(decision)
    folder = Path(os.path.expanduser(str(directory)))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{decision_sha256(decision)}.json"
    path.write_bytes(payload)
    return path


def load(path):
    """Read a decision back and verify that its content still hashes to the name it is filed under."""
    path = Path(os.path.expanduser(str(path)))
    raw = path.read_bytes()
    try:
        decision = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise DecisionError(f"{path} is not a readable decision record: {error}") from None
    validate_decision(decision)
    digest = hashlib.sha256(raw).hexdigest()
    recomputed = decision_sha256(decision)
    if digest != recomputed:
        raise DecisionError(f"{path} is not in canonical form: its bytes digest to {digest}, its content to "
                            f"{recomputed}")
    if path.stem != recomputed:
        raise DecisionError(f"digest mismatch: {path.name} holds a record whose digest is {recomputed}; the record "
                            f"has been altered since it was written")
    return decision


# --- the outcome: a decision linked to the row that judged it (WP23) -------------------------------------------------------

def outcome(record_path, table_row, *, out_dir):
    """Link one decision record to ONE closure-table row, and write the link as a content-addressed outcome record.

    **A person's opinion is never a label; only a table row is.** That sentence is the whole rule of this function, and
    every refusal below is a way of enforcing it. A decision record is a hypothesis: Laya chose an option out of a
    declared set, with uncalibrated probabilities, having seen a description and no rows. It becomes evidence about
    Laya only when the configuration it led to was fitted and *measured*, and the measurement was found commensurable
    with the alternatives it is ranked against. Nothing else — not the author's judgement that the choice looked
    sensible, not the plausibility of the label, not a reviewer's agreement — may be written into an outcome, because
    a corpus built from those would train the checkpoint on the opinions of whoever assembled it.

    `table_row` is a row exactly as `evaluation/compare_stages.py` emits it: the mapping carrying `stage`, `status`,
    `comparability`, `rank`, `metric`, `model_error`, the `naive` reference and `skill`. The outcome copies four of
    those fields and binds all the rest with `table_row_sha256`, the digest of the row's canonical JSON — so the
    numbers cannot be quoted out of the outcome, and cannot be changed behind it either.

    Refusals, each by name, nothing written for any of them:

    * `NOT_COMPARABLE` — the row's `comparability` is not `COMPARABLE`. A rank among stages measured on different
      holdouts, or against a different metric, is an ordering of incommensurable numbers;
    * `NOT_RANKED` — the row carries no rank. The rank *is* the label;
    * `DECISION_NOT_FOUND` — the path holds no readable `m5phet.decision.v1` record;
    * `DIGEST_MISMATCH` — the record's bytes no longer hash to the name it is filed under;
    * `MALFORMED_TABLE_ROW` — the object is not a closure-table row.

    `best_ranked_option` is the option key of the stage ranked first among the stages that share this decision's
    `kind` and `question`. One call sees one row, so it can be settled here only when this row **is** that stage:
    when `rank == 1` the field is written and equals `chosen`; otherwise it is absent, and
    `evaluation/decision_calibration.py` settles it for the group by reading every outcome that shares the kind and
    the question — refusing the group when two stages tie at rank 1 under different keys, because then no single
    option was ranked first.

    Returns `{"status": "OK", "outcome": <record>, "record_path": str}` or a typed refusal, in the shape `ask` uses.
    """
    problem = _check_table_row(table_row)
    if problem is not None:
        return refusal(MALFORMED_TABLE_ROW, problem)

    decision, code, why = _load_decision(record_path)
    if decision is None:
        return refusal(code, why)

    comparability = table_row.get("comparability")
    if comparability != COMPARABLE:
        return refusal(NOT_COMPARABLE, f"the closure-table row for stage {table_row.get('stage')!r} is "
                                       f"{comparability!r}, not {COMPARABLE!r}; a decision is labelled by a measured "
                                       f"row that was found commensurable with the stages it is ranked against, and "
                                       f"by nothing else")
    rank = table_row.get("rank")
    if not isinstance(rank, int) or isinstance(rank, bool):
        return refusal(NOT_RANKED, f"the closure-table row for stage {table_row.get('stage')!r} carries rank "
                                   f"{rank!r}; the rank is the label, so a row without one labels nothing")

    linked = {"schema": OUTCOME_SCHEMA,
              "decision_sha256": decision_sha256(decision),
              "kind": decision["kind"],
              "question": decision["question"],
              "chosen": decision["chosen"],
              "options": [list(pair) for pair in decision["options"]],
              # verbatim from the decision: the argmax and the probability it claimed are what WP23 calibrates, and a
              # report that had to re-open the decision record to find them could be run against a different one
              "probabilities": copy.deepcopy(decision["probabilities"]),
              "table_row_sha256": table_row_sha256(table_row),
              "stage": str(table_row["stage"]),
              "rank": rank,
              "comparability": comparability}
    if rank == 1:
        linked["best_ranked_option"] = decision["chosen"]

    validate_outcome(linked)
    return {"status": "OK", "outcome": linked, "record_path": str(record_outcome(linked, out_dir))}


def _check_table_row(table_row):
    """`None` when this is a closure-table row, else why it is not. Nothing here judges the numbers in it."""
    if not isinstance(table_row, dict):
        return f"a closure-table row is a mapping, not {type(table_row).__name__}"
    missing = [field for field in TABLE_ROW_FIELDS if field not in table_row]
    if missing:
        return (f"a row as evaluation/compare_stages.py emits it carries {list(TABLE_ROW_FIELDS)}; this one is "
                f"missing {missing}")
    return None


def _load_decision(path):
    """`(decision, None, None)`, or `(None, code, why)` naming which of the two record failures happened."""
    path = Path(os.path.expanduser(str(path)))
    try:
        raw = path.read_bytes()
    except OSError as error:
        return None, DECISION_NOT_FOUND, f"{path} cannot be read as a decision record: {error}"
    try:
        decision = json.loads(raw.decode("utf-8"))
        validate_decision(decision)
    except (ValueError, UnicodeDecodeError) as error:
        return None, DECISION_NOT_FOUND, f"{path} is not a {DECISION_SCHEMA} record: {error}"
    recomputed = decision_sha256(decision)
    if hashlib.sha256(raw).hexdigest() != recomputed:
        return None, DIGEST_MISMATCH, (f"{path} is not in canonical form: its bytes and its content do not hash to "
                                       f"the same digest")
    if path.stem != recomputed:
        return None, DIGEST_MISMATCH, (f"{path.name} holds a record whose digest is {recomputed}; it has been altered "
                                       f"since it was written, and an altered decision cannot be given an outcome")
    return decision, None, None


def table_row_sha256(table_row):
    """The digest of the row's canonical JSON — every field of it, including the numbers the outcome does not copy."""
    return hashlib.sha256(json.dumps(table_row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def outcome_sha256(linked):
    return hashlib.sha256(canonical_bytes(linked)).hexdigest()


def validate_outcome(linked):
    """Raise unless this is a complete outcome record: a real decision, a comparable row, and a rank."""
    if not isinstance(linked, dict):
        raise DecisionError("an outcome record is a mapping")
    if linked.get("schema") != OUTCOME_SCHEMA:
        raise DecisionError(f"schema {linked.get('schema')!r} is not {OUTCOME_SCHEMA!r}")
    required = {"schema", "decision_sha256", "kind", "question", "chosen", "options", "probabilities",
                "table_row_sha256", "stage", "rank", "comparability"}
    extra = set(linked) - required - {"best_ranked_option"}
    if extra or required - set(linked):
        raise DecisionError(f"a {OUTCOME_SCHEMA} record carries {sorted(required)} and optionally "
                            f"`best_ranked_option`; this one carries {sorted(linked)}")
    for field in ("decision_sha256", "kind", "question", "chosen", "table_row_sha256", "stage", "comparability"):
        if not isinstance(linked[field], str) or not linked[field].strip():
            raise DecisionError(f"`{field}` must be a non-empty string")
    if linked["comparability"] != COMPARABLE:
        raise DecisionError(f"{NOT_COMPARABLE}: an outcome exists only for a {COMPARABLE} row")
    if not isinstance(linked["rank"], int) or isinstance(linked["rank"], bool) or linked["rank"] < 1:
        raise DecisionError(f"{NOT_RANKED}: a rank is a position, the first being 1; {linked['rank']!r} is not one")
    options = linked["options"]
    if not isinstance(options, list) or len(options) < 2:
        raise DecisionError("an outcome carries the option set the decision was chosen from, at least two entries")
    keys = [pair[0] for pair in options]
    if linked["chosen"] not in keys:
        raise DecisionError(f"{CHOICE_OUTSIDE_OPTIONS}: {linked['chosen']!r} is not one of {keys}")
    probabilities = linked["probabilities"]
    if not isinstance(probabilities, dict) or not probabilities or set(probabilities) - set(keys):
        raise DecisionError("an outcome carries the decision's uncalibrated probabilities over its declared options")
    best = linked.get("best_ranked_option")
    if best is not None and (linked["rank"] != 1 or best != linked["chosen"]):
        raise DecisionError("`best_ranked_option` is written only by the stage ranked first, and is then its own "
                            "chosen key; any other value would be a claim this row cannot make")
    return linked


def record_outcome(linked, directory):
    """Write one outcome as a content-addressed JSON file `<sha256>.json` and return its path."""
    validate_outcome(linked)
    payload = canonical_bytes(linked)
    folder = Path(os.path.expanduser(str(directory)))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{outcome_sha256(linked)}.json"
    path.write_bytes(payload)
    return path


def load_outcome(path):
    """Read an outcome back and verify that its content still hashes to the name it is filed under."""
    path = Path(os.path.expanduser(str(path)))
    raw = path.read_bytes()
    try:
        linked = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise DecisionError(f"{path} is not a readable outcome record: {error}") from None
    validate_outcome(linked)
    recomputed = outcome_sha256(linked)
    if hashlib.sha256(raw).hexdigest() != recomputed or path.stem != recomputed:
        raise DecisionError(f"digest mismatch: {path.name} holds an outcome whose digest is {recomputed}; the record "
                            f"has been altered since it was written")
    return linked
