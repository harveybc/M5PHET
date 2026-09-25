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

It never lets a choice be recorded at a confidence the checkpoint was measured to be wrong at. WP09 measured this
checkpoint on 450 independently labelled rows: below 0.8 its argmax is at chance, in the 0.8-0.9 bin it is right 71 %
of the time, above 0.9 it is right in every row measured. `ask(..., min_confidence=, abstention_source=)` turns that
measurement into a rule -- an answer below the threshold is `LOW_CONFIDENCE_ABSTAINED`, recorded with `chosen: null`
-- and the threshold itself must be cited from the report that measured it, checked against that report's own bins.
A threshold nobody measured refuses every question as `UNCITED_THRESHOLD`, because a gate chosen after seeing which
value lets a study through is tuning on the outcome, which is the one thing this module exists to prevent.

And what it does not prove: nothing here says a choice is a good one. A decision is a hypothesis. The fit that follows
it and the closure table that scores it are the judge, and they live in other packages (WP18-WP21).

One thing it does do since WP23's structural finding (2026-09-25): it records a PERSON's choice too, through
`human_choice`, in the same store and under the same checks — the option set the executing repository declared, the
state that was described, and the ground in `why` — but with `chosen_by: HUMAN`, no backend, no checkpoint and no
probabilities. The reason is not symmetry. A closure table ranks pipelines, and the first table ranked a hand-written
stage first; that stage carried no decision record, so no option was the label the others could be scored against, and
writing Laya's chosen option against the hand row would have claimed Laya chose what a person used. A human record
fixes that by saying what the winning stage actually used, and it is never scored: `evaluation/decision_calibration.py`
counts it, may take its label, and excludes it from every rate.
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

# --- who chose (WP23) ---------------------------------------------------------------------------------------------------
# A closure table ranks pipelines, not choosers, and WP23's rule is that every stage entering the table must carry
# decision records for the same (kind, question) set. A stage a person configured therefore needs a record too, or the
# rank-1 row carries no label and the calibration report can only ever say NO_BEST_RANKED_OPTION. That record must be
# distinguishable from Laya's at a glance and by validation, because the two say completely different things: Laya's
# carries a head's uncalibrated probabilities that WP23 calibrates, a person's carries none and is calibrated by nothing.
#: the chooser of a record written from a Laya answer; the value assumed for a record that names no chooser
CHOSEN_BY_LAYA = "LAYA"
#: the chooser of a record a person wrote by hand: no backend, no checkpoint, no probabilities, and a stated ground
CHOSEN_BY_HUMAN = "HUMAN"
#: the chooser of a record a SEARCH wrote (WP06 stage 5): a declared budget of fits was spent over a declared space and
#: the option that minimised one declared objective was kept. It is neither a person nor Laya, and saying it is either
#: would be false in both directions: a search produced no distribution over the options (so it is not Laya's record,
#: which is calibrated by its probabilities), and nobody deliberated (so it is not a person's, whose ground is a
#: judgement). Its ground is mechanical and must name the two things that make it reviewable: the objective it
#: minimised and the budget it was allowed to spend.
CHOSEN_BY_SEARCH = "SEARCH"
CHOOSERS = (CHOSEN_BY_LAYA, CHOSEN_BY_HUMAN, CHOSEN_BY_SEARCH)

#: the choosers that carry no probabilities: a person produced no distribution, and a search produced a ranking of
#: measured objective values, which is not one either. Only Laya's head emits probabilities, and only they are calibrated.
CHOOSERS_WITHOUT_PROBABILITIES = (CHOSEN_BY_HUMAN, CHOSEN_BY_SEARCH)

#: a human record carries no probability at all rather than a flat or invented one: a person did not produce a
#: distribution, and writing 1.0 for the chosen option would put a certainty in the corpus that nobody claimed
HUMAN_PROBABILITIES = {}
#: the same, for a search: the objective values it measured are in the closure table, not in a probability field
SEARCH_PROBABILITIES = {}
#: `why` is empty, or a human record was written without the ground of the choice
WHY_REQUIRED = "WHY_REQUIRED"
#: a search record was written without naming the objective its choice minimised
OBJECTIVE_REQUIRED = "OBJECTIVE_REQUIRED"
#: a search record was written without naming the budget it was allowed to spend
BUDGET_REQUIRED = "BUDGET_REQUIRED"
#: a field only one kind of chooser may carry was found on the other kind
CHOOSER_FIELDS = "CHOOSER_FIELDS"

#: the two sentences a search's `why` must carry, written by `search_choice` and checked by `validate_decision`. They
#: are fixed strings so the check is a property of the record and not of anybody's phrasing.
OBJECTIVE_MARKER = "The objective minimised was "
BUDGET_MARKER = "The budget spent was "

# --- WP20: the abstention threshold, and the measurement it must be cited from ------------------------------------------
#: the schema of the evaluation report a confidence threshold may be cited from. `evaluation/report.py` writes it.
QUALITY_REPORT_SCHEMA = "m5phet-evaluation-report/1"

#: the metric set inside that report which measures whether a probability means anything, and the object in it that
#: holds the measured bins. A threshold is a claim about those bins and is checked against them.
CALIBRATION_METRIC_SET = "calibration"
RELIABILITY = "reliability"

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
#: the answer's top probability is below the declared threshold; no choice was made and none was recorded as one
LOW_CONFIDENCE_ABSTAINED = "LOW_CONFIDENCE_ABSTAINED"
#: a threshold was passed with no measurement cited for it. Nobody may pass a number they made up
UNCITED_THRESHOLD = "UNCITED_THRESHOLD"
#: the cited file is not an evaluation report that measured this checkpoint's reliability
ABSTENTION_SOURCE_UNREADABLE = "ABSTENTION_SOURCE_UNREADABLE"
#: the cited report measures nothing at this threshold: it is not one of the bin edges the report resolved, or no row
#: was measured at or above it. A threshold the measurement cannot see is as made up as one with no citation at all
THRESHOLD_NOT_MEASURED = "THRESHOLD_NOT_MEASURED"
#: an abstention is not a choice, so it never becomes a training label
ABSTENTION_HAS_NO_OUTCOME = "ABSTENTION_HAS_NO_OUTCOME"
#: a report was cited and no threshold declared with it. It gates nothing, and a rule that gates nothing while looking
#: like one is worse than none: an operator reading the file would believe the choices are being checked
ABSTENTION_SOURCE_WITHOUT_THRESHOLD = "ABSTENTION_SOURCE_WITHOUT_THRESHOLD"
#: what the catalog says when this installation declares no rule at all. Not a refusal -- a fact about the file
NOT_CONFIGURED = "NOT_CONFIGURED"

#: the environment variables the rule is read from when there is no JSON file. `m5phet.config` exports the JSON keys
#: into exactly these, so the two ways of configuring say the same thing and neither is privileged.
MIN_CONFIDENCE_VARIABLE = "M5PHET_INTERPRETER_MIN_CONFIDENCE"
ABSTENTION_SOURCE_VARIABLE = "M5PHET_INTERPRETER_ABSTENTION_SOURCE"

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


# --- the abstention threshold (WP20) -----------------------------------------------------------------------------------

def abstention_threshold(min_confidence, abstention_source):
    """Resolve a declared confidence threshold against the measurement it is cited from.

    Returns `(citation, None)`, or `(None, (refusal_code, why))`. Nothing here asks anything; this runs before a
    single question is sent, so a threshold nobody measured costs no model call.

    **The whole point of this function is that the number cannot be invented.** WP09 measured the zero-shot checkpoint
    on 450 independently labelled rows and found it at chance below 0.8, 71 % correct in the 0.8-0.9 bin and 100 %
    correct above 0.9. A threshold is a claim about exactly those bins, so it is checked against them:

    * `UNCITED_THRESHOLD` — a threshold with no `abstention_source`. A number with no measurement behind it is an
      opinion about the model dressed as a gate, and tuning it until a study validates is tuning on the outcome;
    * `ABSTENTION_SOURCE_UNREADABLE` — the cited path is not a `m5phet-evaluation-report/1` document carrying a
      `calibration` metric set with its `reliability` bins;
    * `THRESHOLD_NOT_MEASURED` — the threshold is not one of the bin edges the cited report resolved, or the report
      measured no row at or above it. A threshold between two bin edges is a number the measurement cannot see.

    The citation carries the report's own digest, its stage, its protocol digest and corpus seal, the bins at or above
    the threshold, and how many of those rows were correct — so a record says where its threshold came from and a
    reader can recompute it from the cited file.
    """
    if min_confidence is None:
        return None, None
    if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
        return None, (UNCITED_THRESHOLD, f"a confidence threshold is a number between 0 and 1; "
                                         f"{min_confidence!r} is not one")
    threshold = float(min_confidence)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        return None, (UNCITED_THRESHOLD, f"a confidence threshold lies in (0, 1]; {threshold!r} does not")
    if abstention_source is None:
        return None, (UNCITED_THRESHOLD,
                      f"the threshold {threshold} cites no measurement. A decision gate is a claim about how often "
                      f"this checkpoint is right at a confidence, so it is passed with the report that measured it "
                      f"(`abstention_source`); a number nobody measured is not a rule, it is a preference")

    path = Path(os.path.expanduser(str(abstention_source)))
    try:
        raw = path.read_bytes()
        report = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        return None, (ABSTENTION_SOURCE_UNREADABLE, f"{path} cannot be read as an evaluation report: {error}")
    version = report.get("version") if isinstance(report, dict) else None
    if version != QUALITY_REPORT_SCHEMA:
        return None, (ABSTENTION_SOURCE_UNREADABLE,
                      f"{path} declares version {version!r}, not {QUALITY_REPORT_SCHEMA!r}; a threshold is cited from "
                      f"a report this framework wrote")
    sets = report.get("metric_sets")
    measured = None
    if isinstance(sets, list):
        for entry in sets:
            if isinstance(entry, dict) and entry.get("name") == CALIBRATION_METRIC_SET:
                measured = ((entry.get("values") or {}).get(RELIABILITY)
                            if isinstance(entry.get("values"), dict) else None)
    bins = measured.get("bins") if isinstance(measured, dict) else None
    if not isinstance(bins, list) or not bins:
        return None, (ABSTENTION_SOURCE_UNREADABLE,
                      f"{path} carries no {CALIBRATION_METRIC_SET!r} metric set with {RELIABILITY!r} bins, so it "
                      f"measured no relation between this checkpoint's confidence and how often it is right")

    edges, above = [], []
    for entry in bins:
        if not isinstance(entry, dict) or not isinstance(entry.get("bin"), (list, tuple)) or len(entry["bin"]) != 2:
            return None, (ABSTENTION_SOURCE_UNREADABLE, f"{path} carries a reliability bin that is not a [low, high] "
                                                        f"pair with a count: {entry!r}")
        low, high = float(entry["bin"][0]), float(entry["bin"][1])
        edges.append(low)
        if low >= threshold - 1e-12:
            above.append({"bin": [round(low, 6), round(high, 6)], "count": int(entry.get("count") or 0),
                          "accuracy": entry.get("accuracy")})
    if not any(abs(edge - threshold) <= 1e-12 for edge in edges):
        return None, (THRESHOLD_NOT_MEASURED,
                      f"{path} resolved the bin edges {sorted(round(edge, 6) for edge in edges)} and the threshold "
                      f"{threshold} is not one of them; a threshold inside a bin splits rows the report never "
                      f"separated, so the report says nothing about it")
    rows = sum(entry["count"] for entry in above)
    if rows <= 0:
        return None, (THRESHOLD_NOT_MEASURED,
                      f"{path} measured no row at or above {threshold}; a threshold above everything the report saw "
                      f"is a gate whose behaviour was never observed")
    correct = sum(entry["count"] * float(entry["accuracy"] or 0.0) for entry in above)
    return {"min_confidence": threshold,
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "stage": report.get("stage"),
            "protocol_digest": report.get("protocol_digest"),
            "corpus_seal": report.get("corpus_seal"),
            "measured_rows_at_or_above": rows,
            "measured_correct_at_or_above": round(correct, 6),
            "measured_accuracy_at_or_above": round(correct / rows, 6),
            "bins_at_or_above": above}, None


def configured_abstention(configuration=None, environ=None):
    """`(min_confidence, abstention_source)` as this INSTALLATION declares them, or `(None, None)`.

    The rule stopped being an argument a caller might remember to pass on 2026-09-25, because that is what it had been:
    `ask(..., min_confidence=0.8)` gated the one study whose author had read WP09, and every other call -- the dataset
    chooser, the pipeline's four questions, the trading decisions -- asked the same checkpoint with no gate at all. A
    measurement that only applies where somebody remembered it is not a rule about the model, it is a habit.

    So it is read from `interpreter.min_confidence` / `interpreter.abstention_source` in `~/.config/m5phet/m5phet.json`,
    falling back to the variables `m5phet.config` exports them into. Nothing is validated here; whatever is declared is
    handed to `abstention_threshold`, which refuses an unmeasured number exactly as it refuses one a caller invented.
    Configuration is where a threshold is DECLARED, never where it is excused."""
    env = os.environ if environ is None else environ
    settings = {}
    if configuration is not None:
        block = getattr(configuration, "interpreter", configuration)     # a `Configuration`, or the block itself
        settings = dict(block or {})
    else:
        from . import config as configuration_module
        try:
            settings = dict(configuration_module.load(environ=env).interpreter)
        except configuration_module.ConfigError:
            settings = {}                   # a file that cannot be read is refused where it is loaded, not silently here
    minimum = settings.get("min_confidence")
    if minimum is None and MIN_CONFIDENCE_VARIABLE in env:
        raw = env[MIN_CONFIDENCE_VARIABLE].strip()
        try:
            minimum = float(raw)
        except ValueError:
            minimum = raw                   # not a number: `abstention_threshold` names it UNCITED_THRESHOLD
    source = settings.get("abstention_source") or env.get(ABSTENTION_SOURCE_VARIABLE) or None
    return minimum, source


def declared_threshold(configuration=None, environ=None):
    """`(citation, unresolved)` for the configured rule, `(None, None)` when this installation declares none."""
    minimum, source = configured_abstention(configuration, environ)
    if minimum is None and source is None:
        return None, None
    if minimum is None:
        return None, (ABSTENTION_SOURCE_WITHOUT_THRESHOLD,
                      f"{source} is cited as the measurement behind an abstention threshold and no threshold is "
                      f"declared (`interpreter.min_confidence`); a citation with no number gates nothing, and a "
                      f"configuration that looks like a rule and is not one is worse than no rule")
    return abstention_threshold(minimum, source)


def citation_view(citation):
    """What a consumer is told about a resolved rule: the number and where it was measured. Never the local path.

    `/api/catalog` is read by the web, by an MCP client and by a Telegram skill, so it carries the report's digest,
    stage, protocol and seal -- enough to identify the measurement and recompute the threshold from it -- and not the
    place on this machine where the file happens to sit."""
    return {"min_confidence": citation["min_confidence"],
            "source": {"report_sha256": citation["sha256"], "stage": citation["stage"],
                       "protocol": citation["protocol_digest"], "seal": citation["corpus_seal"],
                       "bins_at_or_above": copy.deepcopy(citation["bins_at_or_above"]),
                       "measured_rows_at_or_above": citation["measured_rows_at_or_above"],
                       "measured_accuracy_at_or_above": citation["measured_accuracy_at_or_above"]}}


def declared_rule(configuration=None, environ=None):
    """The catalog's view of this installation's rule: `NOT_CONFIGURED`, the citation, or the refusal it resolves to.

    A refusal is published rather than swallowed, because an operator who wrote a threshold the cited report cannot see
    has a machine that refuses every choice, and the catalog is where he finds out why without reading a traceback."""
    citation, unresolved = declared_threshold(configuration, environ)
    if unresolved is not None:
        return {"refusal": unresolved[0], "why": unresolved[1]}
    if citation is None:
        return NOT_CONFIGURED
    return citation_view(citation)


def _abstains(probabilities, citation):
    """`(top_option, top_probability)` when the answer's argmax is below the cited threshold, else `None`.

    The top probability is the maximum over the declared options, which is the quantity WP09's reliability bins were
    built from: the report binned each row by the confidence of the option the checkpoint put first. A tie at the top
    has no single argmax, so it is below any threshold this report can justify and abstains under the first key in
    option order."""
    if citation is None:
        return None
    top = max(probabilities.values())
    holders = sorted(key for key, value in probabilities.items() if value == top)
    if len(holders) == 1 and top >= citation["min_confidence"]:
        return None
    return holders[0], top


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

def ask(engine_or_registry, state_text, questions, *, as_of=None, kind, record_dir=None, min_confidence=None,
        abstention_source=None, configuration=None):
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

    `min_confidence` is the declared abstention threshold (WP20). With one, an answer whose top probability is below
    it is NOT a choice: the entry is `LOW_CONFIDENCE_ABSTAINED` and the abstention is recorded as a decision record
    with `chosen: null`, so a question the checkpoint could not answer leaves a trace instead of a guess. It is passed
    with `abstention_source`, the path to the report that MEASURED this checkpoint at this confidence; a threshold
    with no citation refuses every question as `UNCITED_THRESHOLD` and nothing is asked. See `abstention_threshold`.

    **A caller who passes neither gets the INSTALLATION's rule**, read from `interpreter.min_confidence` and
    `interpreter.abstention_source` (`configured_abstention`). That is the whole point of putting it in configuration:
    the gate applies to the dataset chooser, the pipeline's questions and anything written next, not only where an
    author happened to remember it. Configuration declares the rule; it does not excuse it -- a threshold with no
    citation refuses from the file exactly as it refuses from an argument. Pass `configuration` to override what is
    read (a `Configuration`, or the `interpreter` block itself); an installation that declares nothing gates nothing,
    which is how every caller behaved before.

    Returns `{name: entry}`. An entry is either `{"status": "OK", "decision": <record>, "record_path": str|None}` or a
    typed refusal (`{"status": "REFUSED", "refusal": ..., "why": ...}`). A refusal the provider itself raised --
    `TOKEN_BUDGET_EXCEEDED` among them -- is passed through verbatim, so its name survives this layer. An abstention
    is a refusal that also carries `decision` and `record_path`, because an abstention is recorded and is not a choice.
    """
    if not isinstance(kind, str) or not kind.strip():
        raise DecisionError("a decision kind must be a non-empty string")
    if not isinstance(questions, dict) or not questions:
        raise DecisionError("`questions` must be a non-empty mapping of name -> {options, instructions}")
    if min_confidence is None and abstention_source is not None:
        raise DecisionError("an `abstention_source` with no `min_confidence` gates nothing; pass the threshold it "
                            "was cited for, or neither")
    stamped = as_of if as_of is not None else datetime.now(timezone.utc).isoformat()
    if min_confidence is None and abstention_source is None:
        citation, unresolved = declared_threshold(configuration)
    else:
        citation, unresolved = abstention_threshold(min_confidence, abstention_source)

    if unresolved is not None:
        # nothing is asked: a gate nobody measured would decide which answers count, and that decision would be the
        # author's, made after seeing which threshold lets the study through
        return {name: refusal(unresolved[0], unresolved[1], "choice") for name in questions}

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
                                   checkpoint=checkpoint, as_of=stamped, record_dir=record_dir, citation=citation)
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


def _entry(answer, *, name, kind, options, instructions, state_text, checkpoint, as_of, record_dir, citation=None):
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

    abstained = _abstains(probabilities, citation)
    if abstained is not None:
        # below the declared threshold this checkpoint is at chance, and a choice made at chance is a guess wearing a
        # model's authority. The answer is kept in full -- every probability, and the label the head put first -- and
        # `chosen` is null, because nothing was chosen.
        top_option, top = abstained
        decision["chosen"] = None
        decision["abstention"] = {"refusal": LOW_CONFIDENCE_ABSTAINED, "top_option": top_option,
                                  "top_probability": top, "answer_label": chosen,
                                  "threshold": copy.deepcopy(citation)}
        why = (f"the answer's top probability was {top} on option {top_option!r}, below the declared threshold "
               f"{citation['min_confidence']}. That threshold comes from {citation['path']} (sha256 "
               f"{citation['sha256']}, stage {citation['stage']!r}), which measured "
               f"{citation['measured_correct_at_or_above']} of {citation['measured_rows_at_or_above']} rows correct "
               f"at or above it; below it the same report puts this checkpoint at chance. No choice was recorded for "
               f"{name!r}.")
        entry = refusal(LOW_CONFIDENCE_ABSTAINED, why, "choice")
        entry["decision"] = decision
        entry["record_path"] = str(record(decision, record_dir)) if record_dir is not None else None
        return entry

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


def chooser_of(decision):
    """`LAYA`, `HUMAN` or `SEARCH`. A record that names no chooser is Laya's: every record written before WP23 is one."""
    if not isinstance(decision, dict):
        raise DecisionError("a decision record is a mapping")
    chosen_by = decision.get("chosen_by", CHOSEN_BY_LAYA)
    if chosen_by not in CHOOSERS:
        raise DecisionError(f"`chosen_by` is one of {list(CHOOSERS)}; {chosen_by!r} is none of them, and a record "
                            f"whose chooser is unknown cannot be read as evidence about any of them")
    return chosen_by


def validate_decision(decision):
    """Raise unless this is a complete decision record, claiming no authority, and saying who made the choice.

    Four shapes are records, and nothing else is. A **Laya choice**: `chosen_by` absent or `LAYA`, backend `laya`, an
    option key in `chosen`, the head's own uncalibrated probabilities. An **abstention** (WP20): the same, with
    `chosen: null` and an `abstention` block naming the threshold and the measurement it was cited from -- a question
    that was asked and not answered, which is a fact about the checkpoint and never a choice. A **person's choice**
    (WP23): `chosen_by: HUMAN`, no backend and no checkpoint because a person is neither, `probabilities: {}` because
    there is no head to read them off, and a `why` in the person's own words, so a stage a person configured can enter
    a closure table carrying records like any other. A **search's choice** (WP06 stage 5): `chosen_by: SEARCH`, the
    same absences as a person's, and a `why` that must name the objective it minimised and the budget it spent -- a
    search is neither a person nor a model, and the two facts that make its winner readable are those.

    The shapes are validated apart on purpose, and neither can borrow a field from the other: a record that looked like
    the wrong kind of evidence would be read as the wrong kind of evidence later.
    """
    if not isinstance(decision, dict):
        raise DecisionError("a decision record is a mapping")
    if decision.get("schema") != DECISION_SCHEMA:
        raise DecisionError(f"schema {decision.get('schema')!r} is not {DECISION_SCHEMA!r}")
    chosen_by = chooser_of(decision)
    expected = {"schema", "kind", "state_sha256", "question", "options", "chosen", "probabilities",
                "probability_decimals", "checkpoint", "backend", "as_of", "execution_authorized"}
    if chosen_by in CHOOSERS_WITHOUT_PROBABILITIES:
        expected |= {"chosen_by", "why"}
    else:
        if "chosen_by" in decision:
            expected |= {"chosen_by"}
        if "abstention" in decision:
            expected |= {"abstention"}
    if set(decision) != expected:
        raise DecisionError(f"a {DECISION_SCHEMA} record chosen by {chosen_by} carries exactly {sorted(expected)}; "
                            f"this one carries {sorted(decision)}")
    if decision["execution_authorized"] is not False:
        raise DecisionError("a decision claims no execution authority; `execution_authorized` must be false")
    for field in ("kind", "question", "state_sha256"):
        if not isinstance(decision[field], str) or not decision[field].strip():
            raise DecisionError(f"`{field}` must be a non-empty string")
    options = decision["options"]
    if not isinstance(options, list) or len(options) < 2:
        raise DecisionError("a decision carries the option set it was chosen from, at least two entries")
    keys = [pair[0] for pair in options]

    abstention = decision.get("abstention")
    if abstention is None:
        if not isinstance(decision["chosen"], str) or not decision["chosen"].strip():
            raise DecisionError("`chosen` must be a non-empty string; only an abstention carries no choice")
        if decision["chosen"] not in keys:
            raise DecisionError(f"{CHOICE_OUTSIDE_OPTIONS}: {decision['chosen']!r} is not one of {keys}")
    else:
        if chosen_by != CHOSEN_BY_LAYA:
            raise DecisionError(f"{LOW_CONFIDENCE_ABSTAINED} is a fact about a model's confidence; a {chosen_by} "
                                f"choice has no probability to fall below a threshold")
        if decision["chosen"] is not None:
            raise DecisionError(f"{LOW_CONFIDENCE_ABSTAINED}: an abstention carries `chosen: null`; a record that "
                                f"abstained and also chose {decision['chosen']!r} would be both at once")
        if not isinstance(abstention, dict) or abstention.get("refusal") != LOW_CONFIDENCE_ABSTAINED:
            raise DecisionError(f"`abstention` names the refusal {LOW_CONFIDENCE_ABSTAINED!r} it was made under")
        threshold = abstention.get("threshold")
        if not isinstance(threshold, dict) or not threshold.get("path") or not threshold.get("sha256"):
            raise DecisionError(f"{UNCITED_THRESHOLD}: an abstention carries the measurement its threshold came "
                                f"from -- the cited report's path and digest -- or it is a number nobody measured")
        if abstention.get("top_option") not in keys:
            raise DecisionError(f"{CHOICE_OUTSIDE_OPTIONS}: the abstention's top option "
                                f"{abstention.get('top_option')!r} is not one of {keys}")

    probabilities = decision["probabilities"]
    if chosen_by in CHOOSERS_WITHOUT_PROBABILITIES:
        if probabilities != HUMAN_PROBABILITIES:
            raise DecisionError(f"{CHOOSER_FIELDS}: a record chosen by {chosen_by} carries no probabilities; it "
                                f"produced no distribution, and {probabilities!r} would be a claim nobody made")
        for field in ("backend", "checkpoint", "probability_decimals"):
            if decision[field] is not None:
                raise DecisionError(f"{CHOOSER_FIELDS}: `{field}` is {decision[field]!r} on a record chosen by "
                                    f"{chosen_by}; neither a person nor a search is a backend or ran a checkpoint")
        if not isinstance(decision["why"], str) or not decision["why"].strip():
            raise DecisionError(f"{WHY_REQUIRED}: a {chosen_by} choice carries the ground it was made on; a bare key "
                                f"in a corpus is an opinion nobody can review")
        if chosen_by == CHOSEN_BY_SEARCH:
            missing = [name for name, marker in ((OBJECTIVE_REQUIRED, OBJECTIVE_MARKER),
                                                 (BUDGET_REQUIRED, BUDGET_MARKER)) if marker not in decision["why"]]
            if missing:
                raise DecisionError(f"{missing[0]}: a record chosen by {CHOSEN_BY_SEARCH} names, in `why`, the "
                                    f"objective it minimised and the budget it was allowed to spend; this one's "
                                    f"`why` carries no {missing[0].split('_')[0].lower()} sentence "
                                    f"({', '.join(sorted(missing))} missing). A search whose objective or budget is "
                                    f"not written down cannot be reviewed and its choice is not evidence")
        return decision
    if decision["backend"] != LAYA_BACKEND:
        raise DecisionError(f"{NON_MODEL_FIXTURE}: backend {decision['backend']!r} is not {LAYA_BACKEND!r}; a "
                            f"decision is never recorded from anything else")
    if not isinstance(probabilities, dict) or not probabilities:
        raise DecisionError("a decision carries the uncalibrated probabilities the answer carried")
    if set(probabilities) - set(keys):
        raise DecisionError(f"{CHOICE_OUTSIDE_OPTIONS}: probabilities for {sorted(set(probabilities) - set(keys))}")
    if any(type(p) not in (int, float) or not math.isfinite(p) for p in probabilities.values()):
        raise DecisionError("a probability is a finite number")
    return decision


def human_choice(*, kind, question, options, chosen, state_text, why, as_of=None, record_dir=None):
    """Record a choice a PERSON made, in the same shape and the same store as Laya's, and with the same checks.

    WP23's clause, in one function: a stage that a person configured enters the closure table like any other, and the
    table's rank-1 row is a label only if the stage that produced it says which option it used. So a person's choice is
    written down — with the option set it was really chosen from (a key outside it is refused here exactly as it is for
    Laya), the state text that described what was being chosen about, and `why`.

    What it is NOT: evidence about Laya, and never a probability. `evaluation/decision_calibration.py` reads these
    records for the label they carry and excludes them from every rate it computes, because scoring a person's choice
    against the table would measure the person, not the chooser this framework is calibrating.

    Returns `{"status": "OK", "decision": <record>, "record_path": str|None}`, or a typed refusal in the shape `ask`
    uses, so a caller can handle both the same way.
    """
    if not isinstance(kind, str) or not kind.strip():
        raise DecisionError("a decision kind must be a non-empty string")
    if not isinstance(question, str) or not question.strip():
        raise DecisionError("a decision question must be a non-empty string")
    if not isinstance(why, str) or not why.strip():
        return refusal(WHY_REQUIRED, "a human choice carries the ground it was made on; say why this option and not "
                                     "the others", "choice")
    problem = _check_question({"options": options, "instructions": why})
    if problem is not None:
        return refusal(problem[0], problem[1], "choice")
    if not isinstance(state_text, str) or not state_text.strip():
        return refusal(STATE_REQUIRED, "a decision needs a non-empty state text describing what is being chosen "
                                       "about; nothing was recorded", "choice")
    pairs = [[str(key), str(label)] for key, label in options]
    keys = [key for key, _label in pairs]
    if chosen not in keys:
        return refusal(CHOICE_OUTSIDE_OPTIONS, f"the choice {chosen!r} is not one of the declared options {keys}; a "
                                               f"person may not add an option to a set the executing repository "
                                               f"declared", "choice")
    decision = {"schema": DECISION_SCHEMA,
                "kind": kind,
                "state_sha256": state_sha256(state_text),
                "question": question,
                "options": pairs,
                "chosen": str(chosen),
                "probabilities": dict(HUMAN_PROBABILITIES),
                "probability_decimals": None,
                "checkpoint": None,
                "backend": None,
                "chosen_by": CHOSEN_BY_HUMAN,
                "why": why,
                "as_of": as_of if as_of is not None else datetime.now(timezone.utc).isoformat(),
                "execution_authorized": False}
    validate_decision(decision)
    entry = {"status": "OK", "decision": decision, "record_path": None}
    if record_dir is not None:
        entry["record_path"] = str(record(decision, record_dir))
    return entry


def search_choice(*, kind, question, options, chosen, state_text, objective, budget, why, as_of=None,
                  record_dir=None):
    """Record a choice a SEARCH made, in the same shape and the same store as Laya's and a person's.

    WP06 stage 5, in one function. A search over declared configurations is a third chooser: it is not Laya, because
    it produced no distribution over the options and there is nothing to calibrate; and it is not a person, because
    nobody deliberated -- every option it kept it kept because a fit was run and one number came back smaller. The
    record therefore carries `chosen_by: SEARCH`, no probabilities, no backend and no checkpoint, and a `why` that
    must name the two facts without which a search result cannot be read at all:

    * the **objective** -- which single scalar was minimised, on which population. A search that does not say what it
      optimised has reported the winner of an unnamed contest;
    * the **budget** -- how many candidates it was allowed to evaluate. The best of 60 evaluations and the best of 6
      are different claims, and the difference is not visible in the winner.

    Both are composed into `why` through fixed markers that `validate_decision` checks, so a record cannot be written
    with one of them missing and cannot be edited into one afterwards without failing its own digest.

    What it is NOT: evidence about Laya. `evaluation/decision_calibration.py` counts these records, may take the label
    of a search-chosen stage the table ranked first, and excludes them from every rate -- scoring a search against the
    table that scored it would only measure the table twice.

    Returns `{"status": "OK", "decision": <record>, "record_path": str|None}`, or a typed refusal in the shape `ask`
    uses, so a caller can handle both the same way.
    """
    if not isinstance(kind, str) or not kind.strip():
        raise DecisionError("a decision kind must be a non-empty string")
    if not isinstance(question, str) or not question.strip():
        raise DecisionError("a decision question must be a non-empty string")
    if not isinstance(why, str) or not why.strip():
        return refusal(WHY_REQUIRED, "a search choice carries the ground it was made on; say what was searched and "
                                     "how the winner was kept", "choice")
    if not isinstance(objective, str) or not objective.strip():
        return refusal(OBJECTIVE_REQUIRED, "a search choice names the single scalar it minimised and the population "
                                           "it was measured on; without it the winner is the winner of nothing",
                       "choice")
    if not isinstance(budget, str) or not budget.strip():
        return refusal(BUDGET_REQUIRED, "a search choice names the budget it was allowed to spend; the best of 60 "
                                        "evaluations and the best of 6 are different claims", "choice")
    problem = _check_question({"options": options, "instructions": why})
    if problem is not None:
        return refusal(problem[0], problem[1], "choice")
    if not isinstance(state_text, str) or not state_text.strip():
        return refusal(STATE_REQUIRED, "a decision needs a non-empty state text describing what is being chosen "
                                       "about; nothing was recorded", "choice")
    pairs = [[str(key), str(label)] for key, label in options]
    keys = [key for key, _label in pairs]
    if chosen not in keys:
        return refusal(CHOICE_OUTSIDE_OPTIONS, f"the choice {chosen!r} is not one of the declared options {keys}; a "
                                               f"search may not add an option to a set the executing repository "
                                               f"declared, and a point it could not evaluate is a refusal, never a "
                                               f"new option", "choice")
    grounds = f"{why.strip()} {OBJECTIVE_MARKER}{objective.strip()}. {BUDGET_MARKER}{budget.strip()}."
    decision = {"schema": DECISION_SCHEMA,
                "kind": kind,
                "state_sha256": state_sha256(state_text),
                "question": question,
                "options": pairs,
                "chosen": str(chosen),
                "probabilities": dict(SEARCH_PROBABILITIES),
                "probability_decimals": None,
                "checkpoint": None,
                "backend": None,
                "chosen_by": CHOSEN_BY_SEARCH,
                "why": grounds,
                "as_of": as_of if as_of is not None else datetime.now(timezone.utc).isoformat(),
                "execution_authorized": False}
    validate_decision(decision)
    entry = {"status": "OK", "decision": decision, "record_path": None}
    if record_dir is not None:
        entry["record_path"] = str(record(decision, record_dir))
    return entry


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
    if decision.get("abstention") is not None:
        return refusal(ABSTENTION_HAS_NO_OUTCOME,
                       f"the record at {record_path} is an abstention: the checkpoint's top probability was below "
                       f"the declared threshold and nothing was chosen. A row of the table cannot rank a choice that "
                       f"was not made, and calling the stage's configuration Laya's choice would put a person's "
                       f"fallback under the model's name")

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
              # who chose, carried into the outcome so the calibration report never has to re-open the decision to
              # find out whether this link is evidence about the chooser or only the label of a stage a person built
              "chosen_by": chooser_of(decision),
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
    optional = {"best_ranked_option", "chosen_by"}
    extra = set(linked) - required - optional
    if extra or required - set(linked):
        raise DecisionError(f"a {OUTCOME_SCHEMA} record carries {sorted(required)} and optionally "
                            f"{sorted(optional)}; this one carries {sorted(linked)}")
    if linked.get("chosen_by", CHOSEN_BY_LAYA) not in CHOOSERS:
        raise DecisionError(f"`chosen_by` is one of {list(CHOOSERS)}; {linked['chosen_by']!r} is neither")
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
    if linked.get("chosen_by") in CHOOSERS_WITHOUT_PROBABILITIES:
        if probabilities != HUMAN_PROBABILITIES:
            raise DecisionError(f"{CHOOSER_FIELDS}: an outcome of a record chosen by {linked['chosen_by']} carries "
                                f"no probabilities, because the record it links carries none")
    elif not isinstance(probabilities, dict) or not probabilities or set(probabilities) - set(keys):
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
