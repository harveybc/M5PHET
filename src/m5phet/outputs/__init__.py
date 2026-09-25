"""WP04: the output component -- what an area's answers ARE, and how they are shown.

Two things travel together here and are deliberately kept apart in the code.

The **header** is what an area returns, declared once: the `output_kind` its answers carry, the fields that hold the
numbers and their units, the statuses an answer may have and the refusal codes it may come back with. It is a property
of the area and its provider, not of the screen the answer lands on, so every plugin reports the same header. A person
who has to integrate this reads the header and knows the shape before asking anything.

The **procedure** is how those answers are rendered for a particular surface: the workbench's deterministic lines and
its checked narration, a Telegram message bounded to a few thousand characters, tomorrow an export nobody has written
yet. That is what differs between plugins, and it is selected per area by `areas.<area>.output.plugin` in the JSON
configuration, under the entry-point group `m5phet.outputs`.

The rule that holds for every procedure, shipped or external: **a plugin cannot introduce a figure.** Whatever text a
plugin returns is checked against the answers it was given, exactly as a language model's narration is, and a plugin
whose text carries a number the answers do not is discarded in favour of the deterministic rendering. A rendering is
allowed to leave numbers out and to say less; it is not allowed to say more. That is why the truncation helper below
never cuts a number in half: half of 0.5412255525588989 is a figure the answers do not carry.
"""

import json
import re

from ..questions import (MALFORMED_QUESTION, MISSING_FIELD, NO_PROVIDER, NOT_ESTIMABLE, PROVIDER_ERROR,
                         STATE_REQUIRED, UNSUPPORTED_QUESTION_TYPE)

ENTRY_POINT_GROUP = "m5phet.outputs"

DEFAULT_PLUGIN = "default"

#: the shipped procedures. Declared as entry points in `pyproject.toml` and found that way; this map is the fallback
#: for an installation whose metadata is older than its code.
BUILT_IN = {
    "default": ("m5phet.outputs.default", "DefaultOutput"),
    "telegram": ("m5phet.outputs.telegram", "TelegramOutput"),
}

#: fields an answer carries for the machinery and not for the person: the backend's verbatim object (kept for parity
#: checks, and once narrated as if it were a result) and the digests.
NOT_NARRATED = ("sdk_answer", "provenance")

#: refusal codes any area may return, because the envelope itself produces them before a provider is reached
ENVELOPE_REFUSALS = (NO_PROVIDER, UNSUPPORTED_QUESTION_TYPE, MALFORMED_QUESTION, MISSING_FIELD, PROVIDER_ERROR)

#: What each area's answers are, per the owner's per-area statement of "Output" (work plan §2) and what the five
#: installed providers actually declare and emit. `question_types` is checked against the providers' own
#: `question_types()` in `tests/test_outputs.py`: a header that drifts from its provider is a header that lies.
AREA_HEADERS = {
    "classification": {
        "output_kind": "typed_questions",
        "question_types": ["choice"],
        "unit_fields": ["label", "uncalibrated_probabilities", "calibration", "uncertainty"],
        "statuses": ["OK", "REFUSED"],
        "refusal_codes": sorted(set(ENVELOPE_REFUSALS) | {STATE_REQUIRED}),
        "reading": "a label and probabilities that are UNCALIBRATED, for the exact instruction and options scored",
    },
    "forecasting": {
        "output_kind": "point_forecast",
        "question_types": ["anomaly_risk", "interval", "point_forecast"],
        "unit_fields": ["values", "unit", "scale", "targets", "horizons"],
        "statuses": ["OK", "REFUSED"],
        "refusal_codes": sorted(set(ENVELOPE_REFUSALS) | {NOT_ESTIMABLE, STATE_REQUIRED}),
        "reading": "a target- and horizon-indexed number with its unit and scale; an interval only where a bundle "
                   "has a distribution, and NOT_ESTIMABLE where none does",
    },
    "unsupervised": {
        "output_kind": "hierarchical_regimes",
        "question_types": ["cluster_description", "clustering"],
        "unit_fields": ["rows", "optimal_k", "clusters_occupied", "cluster_distribution", "matched_cluster",
                        "rows_in_cluster", "level", "target_metric"],
        "statuses": ["OK", "REFUSED"],
        "refusal_codes": sorted(set(ENVELOPE_REFUSALS) | {STATE_REQUIRED}),
        "reading": "the supplied rows assigned under a fitted reference, and the cluster satisfying a declared "
                   "metric; nothing is fitted or selected by the question",
    },
    "rl": {
        "output_kind": "policy_action",
        "question_types": ["next_action", "value_estimation"],
        "unit_fields": ["action", "unit", "action_space", "expected_return", "uncertainty_bounds", "policy_id"],
        "statuses": ["OK", "REFUSED"],
        "refusal_codes": sorted(set(ENVELOPE_REFUSALS) | {NOT_ESTIMABLE, STATE_REQUIRED}),
        "reading": "a proposed action in the policy's own scale and a critic value under the training reward; a "
                   "score is not an order and execution_authorized is always false",
    },
    "causal": {
        "output_kind": "causal_effect",
        "question_types": ["ate", "cate", "counterfactual_path", "impulse_response", "sensitivity"],
        "unit_fields": ["estimand", "effect_size", "unit", "confidence_interval", "confidence_level", "assumptions", "table"],
        "statuses": ["OK", "REFUSED"],
        "refusal_codes": sorted(set(ENVELOPE_REFUSALS) | {NOT_ESTIMABLE}),
        "reading": "an estimand, an effect with its unit and interval, and the assumptions it rests on; never a "
                   "p-value that was not computed, and NOT_ESTIMABLE where the study cannot condition",
    },
}

#: an envelope whose area is not one of the five -- or an answer set rendered outside an area, as the narration tests
#: do -- still has a header, and it says only what is true of any envelope
GENERIC_HEADER = {
    "output_kind": "typed_answers",
    "question_types": [],
    "unit_fields": [],
    "statuses": ["OK", "REFUSED"],
    "refusal_codes": sorted(ENVELOPE_REFUSALS),
    "reading": "named typed answers; every question is answered or refused on its own",
}


class OutputPluginError(ValueError):
    """An output plugin that cannot be selected. Named, never replaced by another one that happens to be installed."""


def header(area):
    """What this area returns. The same for every plugin: a surface does not change what an engine answers."""
    return {"area": area, **{k: (list(v) if isinstance(v, list) else v)
                             for k, v in AREA_HEADERS.get(area, GENERIC_HEADER).items()}}


def headers():
    return {area: header(area) for area in AREA_HEADERS}


def _from_entry_points():
    from importlib.metadata import entry_points
    found = {}
    try:
        for entry in entry_points(group=ENTRY_POINT_GROUP):
            found.setdefault(entry.name, entry)
    except Exception:                                                   # noqa: BLE001
        return {}
    return found


def names():
    """Every output procedure this installation can select, shipped or external."""
    return sorted(set(_from_entry_points()) | set(BUILT_IN))


def load(name):
    """The class registered under `name`, or a refusal saying which names exist. Nothing is substituted."""
    if not isinstance(name, str) or not name.strip():
        raise OutputPluginError(f"an output plugin is named by a string; got {name!r}")
    entry = _from_entry_points().get(name)
    if entry is not None:
        try:
            return entry.load()
        except Exception as error:                                      # noqa: BLE001
            raise OutputPluginError(f"output plugin {name!r} is registered but could not be loaded: "
                                    f"{type(error).__name__}: {error}") from None
    if name in BUILT_IN:
        module_name, attribute = BUILT_IN[name]
        from importlib import import_module
        return getattr(import_module(module_name), attribute)
    raise OutputPluginError(f"no output plugin named {name!r} is installed; this installation has {names()}")


def select(area=None, settings=None, environ=None):
    """The procedure configured for this area: `areas.<area>.output.plugin`, defaulting to `default`.

    With no JSON configuration -- or none for this area -- the default plugin is today's behaviour exactly, which is
    why an operator who writes no file sees no change."""
    if settings is None:
        from .. import config as configuration
        loaded = configuration.load(environ=environ)
        settings = loaded.output(area) if area else {}
    settings = dict(settings or {})
    return load(settings.get("plugin") or DEFAULT_PLUGIN)(settings)


# --- what a person may be shown ----------------------------------------------------------------------------------------

def narratable(answers):
    """The answers as the person should read them: every field except the backend's verbatim object and the digests.

    A raw SDK field named `confidence` was once rendered as "confianza 0.3403", a number the answers deliberately do
    not surface. Every procedure and every guard reads this same view, so what is shown and what is checked cannot
    come apart."""
    return {name: ({k: v for k, v in answer.items() if k not in NOT_NARRATED} if isinstance(answer, dict) else answer)
            for name, answer in (answers or {}).items()}


def response_view(response):
    """Everything a rendering may state: the answers as read, their names, the counts the envelope itself carries,
    and what was MEASURED about how well this area answers.

    The counts are here because a deterministic rendering says "1 answered, 1 refused" and that is a fact of the
    response, not an invention. The names are here because a question may be called `q1`. The quality block (WP31) is
    here for exactly the same reason and for no other: a macro-F1 or a held-out MAE is a figure, and a rendering may
    state it because the ANSWER carries it -- not because quality numbers are exempt from the guard. A response that
    carries no quality admits no quality figure, and a line inventing one is discarded like any other invention."""
    answers = narratable(response.get("answers") or {})
    view = {"answers": answers, "names": sorted(answers),
            "answered": response.get("answered", 0), "refused": response.get("refused", 0)}
    if response.get("quality") is not None:
        view["quality"] = response["quality"]
    return view


def quality_line(response):
    """The one line an output procedure renders about this area's measured quality, or None when none travels.

    None is not the same as "nothing was measured": a response that carries `quality: NOT_MEASURED` renders a line
    saying so, which is the whole point of WP31. None means this response carries no quality block at all -- an
    answer produced by a path that does not build one -- and there the rendering is exactly what it always was."""
    from ..quality import line
    return line(response.get("quality"), response)


# --- rendering helpers shared by the procedures --------------------------------------------------------------------------

#: a number, possibly signed and fractional, at the very end of a string -- which after a cut may be HALF a number
_TRAILING_NUMBER = re.compile(r"-?\d+(?:[.,]\d*)?$")


def clip(text, limit, mark="..."):
    """Shorten to `limit` characters without ever leaving half a number behind.

    `0.5412255525588989` cut at column 9 is `0.5412255`, which is a figure the answers do not carry and which the
    guard would -- correctly -- refuse. So the cut drops the trailing numeric token entirely: a rendering may say
    less, never something else."""
    if not isinstance(text, str) or len(text) <= limit:
        return text
    if limit <= len(mark):
        return mark[:max(0, limit)]
    return _TRAILING_NUMBER.sub("", text[:limit - len(mark)]) + mark


def value_text(value, limit=80):
    """One field's value, with its numbers exactly as the engine returned them, bounded in length."""
    if isinstance(value, str):
        return clip(" ".join(value.split()), limit)
    return clip(json.dumps(value, ensure_ascii=False, default=str), limit)


def one_line(text):
    """A single line: a rendering that must put one answer per line cannot let a reason's newline split it."""
    return " ".join(str(text).split())
