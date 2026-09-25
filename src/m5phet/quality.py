"""WP31: what is known about how well each area answers -- read from what was measured, computed nowhere here.

The bank had come to know things about itself that a person reading an answer could not see. WP09 scored the
classification checkpoint on 450 independently labelled rows; WP06/WP07 scored a forecast bundle against a naive
baseline on 9 824 sealed rows and measured its interval's coverage; WP19 computed internal indices for a fitted
regime reference. All of it lived in reports nobody reading an answer was shown.

This module puts one `quality` block on every area's answer and every area's catalog entry. Four rules make it worth
reading:

**Nothing is computed here.** Every number comes out of a record somebody else measured: the provider's own
`capabilities()['quality']` (the classification record, whose shape and refusals belong to `news_signal`), an
`m5phet-evaluation-report/1` document written by `M5PHET/evaluation`, or a bundle's manifest. This module reads,
checks and renders; if it ever computed a number, the number would have no protocol.

**A number that cannot name its protocol is not published.** A report is accepted only when it declares the
evaluation package's own version, its protocol digest, its corpus seal, its counts and the provenance of its labels.
Anything else is a refusal by name, published as such, exactly as an uncited abstention threshold is.

**A measurement of another thing is refused rather than borrowed.** A report whose family is not this area's is
refused (`QUALITY_MEASURED_ON_ANOTHER_FAMILY`), and for forecasting -- where a bundle's manifest records the digest
of the report that measured it -- a report no configured bundle references is refused
(`QUALITY_MEASURED_ON_ANOTHER_STATE`). An area may serve several fitted states and have a measurement for one of
them; the block then names which one, because "the forecaster's MAE is 0.53" would be a claim about three bundles
made from a measurement of one.

**Nothing known is said plainly.** `NOT_MEASURED` is the default and not a criticism of anyone; where the thing
cannot be measured at all -- a causal estimate has no held-out truth, a proposed action is not a realised return --
the block carries `REFUSED` with the reason the evaluation package owns, never a softer restatement of it.
"""

import json
import os
from pathlib import Path

#: the document `M5PHET/evaluation` writes, and the only kind of file a quality may be read from here
REPORT_VERSION = "m5phet-evaluation-report/1"

#: nobody has measured this area in this installation. The honest default
NOT_MEASURED = "NOT_MEASURED"

MEASURED = "MEASURED"
REFUSED = "REFUSED"

#: the cited file is not an evaluation report this framework wrote, or does not carry what one must carry
QUALITY_REPORT_UNREADABLE = "QUALITY_REPORT_UNREADABLE"
#: the report measured a different FAMILY. Forecast error is not a regime index and neither is an accuracy
QUALITY_MEASURED_ON_ANOTHER_FAMILY = "QUALITY_MEASURED_ON_ANOTHER_FAMILY"
#: the report measured a fitted state none of the configured ones references. A number about another bundle is not a
#: number about this engine, however close the two were fitted
QUALITY_MEASURED_ON_ANOTHER_STATE = "QUALITY_MEASURED_ON_ANOTHER_STATE"
#: the provider's own quality record refused itself (an incomplete record, an unknown schema). Its words are kept
QUALITY_RECORD_REFUSED = "QUALITY_RECORD_REFUSED"

#: the areas whose quality is a REFUSAL, and the name of the refusal the evaluation package owns. The reason itself is
#: never written here: `m5phet_evaluation.scoring.REFUSED_METRICS` is the single place it is worded, so it cannot be
#: re-explained more softly in a second file
AREA_REFUSAL = {"causal": "causal_accuracy", "rl": "policy_profitability"}

#: the `family` an evaluation report must declare to be about this area
AREA_FAMILY = {"forecasting": "forecast", "unsupervised": "regimes", "classification": "classification"}

#: where an operator declares the report for an area whose provider publishes no quality of its own. Classification
#: has none of these: its record is the provider's (`NEWS_SIGNAL_QUALITY`), read through `capabilities()['quality']`
QUALITY_VARIABLE = {"forecasting": "M5PHET_FORECASTING_QUALITY_REPORT",
                    "unsupervised": "M5PHET_UNSUPERVISED_QUALITY_REPORT"}

#: what the classification record calls itself; anything else is not the record this reader understands
CLASSIFICATION_SCHEMA = "news_signal.quality.v1"


def _refusal(area, code, why, **extra):
    return {"status": REFUSED, "area": area, "refusal": code, "why": why, **extra}


def _not_measured(area, why, **extra):
    return {"status": NOT_MEASURED, "area": area, "why": why, **extra}


# --- the two areas nothing can measure -----------------------------------------------------------------------------

def refused_metric_reason(name):
    """The reason `m5phet_evaluation` gives for refusing this metric, or the statement that it is not installed here.

    The reason is imported and never copied. A refusal restated in a second file is a refusal that can drift into a
    softer wording, and the evaluation package says so itself."""
    try:
        from m5phet_evaluation.scoring import REFUSED_METRICS
    except Exception:                                                   # noqa: BLE001
        return None
    return REFUSED_METRICS.get(name)


def _refused_area(area):
    name = AREA_REFUSAL[area]
    reason = refused_metric_reason(name)
    if reason is None:
        return _refusal(area, name,
                        f"this quantity does not exist and its reason is owned by "
                        f"m5phet_evaluation.scoring.REFUSED_METRICS[{name!r}], which is not installed here",
                        reason_source=f"m5phet_evaluation.scoring.REFUSED_METRICS[{name!r}] (not installed)")
    return _refusal(area, name, reason, reason_source=f"m5phet_evaluation.scoring.REFUSED_METRICS[{name!r}]")


# --- classification: the provider's own record ----------------------------------------------------------------------

def from_provider_record(area, record):
    """The provider's `capabilities()['quality']`, as this framework publishes it beside an answer.

    `news_signal` already refuses an incomplete record and says `NOT_MEASURED` when none is declared; both are kept
    with its words. Nothing is recomputed and no field is defaulted: a record missing a corpus id never got here."""
    if record in (None, NOT_MEASURED):
        return _not_measured(area, "this provider declares no measured quality record")
    if isinstance(record, str):
        return _refusal(area, QUALITY_RECORD_REFUSED, record)
    if not isinstance(record, dict) or record.get("schema") != CLASSIFICATION_SCHEMA:
        return _refusal(area, QUALITY_REPORT_UNREADABLE,
                        f"the provider published a quality that is not a {CLASSIFICATION_SCHEMA} record")
    calibration = record.get("calibration") or {}
    skill = record.get("skill") or {}
    values = {"macro_f1": record.get("macro_f1")}
    if calibration.get("expected_calibration_error") is not None:
        values["expected_calibration_error"] = calibration["expected_calibration_error"]
    if calibration.get("brier") is not None:
        values["brier"] = calibration["brier"]
    return {"status": MEASURED, "area": area, "kind": "classification", "values": values,
            "calibration_status": calibration.get("status"),
            "baseline": record.get("naive"), "skill": skill.get("skill"),
            "skill_status": skill.get("status"), "error_definition": skill.get("error_definition"),
            "n": {"scored_rows": record.get("n")},
            "corpus_id": record.get("corpus_id"), "corpus_seal": record.get("corpus_seal"),
            "protocol_digest": record.get("protocol_digest"),
            "label_provenance": record.get("label_provenance"),
            "measured_on": record.get("corpus_id"),
            "flags": [], "reading": record.get("reading"),
            "source": f"the provider's own quality record ({CLASSIFICATION_SCHEMA})"}


# --- forecasting and representation: an evaluation report the operator declares --------------------------------------

def declared_report_path(area, configuration=None, environ=None):
    """`areas.<area>.quality.report`, else the area's environment variable, else None."""
    env = os.environ if environ is None else environ
    if configuration is not None:
        declared = (getattr(configuration, "quality", lambda _name: {})(area) or {}).get("report")
        if declared:
            return str(declared)
    variable = QUALITY_VARIABLE.get(area)
    return env.get(variable) if variable else None


def read_report(path):
    """`(report, digest, problem)` for a declared evaluation report. Nothing partial is ever returned."""
    import hashlib
    target = Path(os.path.expanduser(str(path)))
    try:
        raw = target.read_bytes()
        report = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        return None, None, f"{target} cannot be read as an evaluation report: {error}"
    if not isinstance(report, dict) or report.get("version") != REPORT_VERSION:
        return None, None, (f"{target} does not declare version {REPORT_VERSION!r}; a quality is published only from "
                            f"a report M5PHET/evaluation wrote")
    missing = [field for field in ("protocol_digest", "corpus_seal", "sealed_row_count", "label_provenance",
                                  "metric_sets") if report.get(field) in (None, "")]
    if missing:
        return None, None, (f"{target} carries no {missing}; a quality claim without its protocol, its seal, its "
                            f"counts and the provenance of its labels is not a measurement")
    return report, hashlib.sha256(raw).hexdigest(), None


def bundle_evaluations(bundle_dir):
    """What each configured forecast bundle's manifest says about the evaluation that measured it.

    The bundles record `provenance.quality` (this package never scores a model, so it is `UNMEASURED`) and, since
    WP07, `provenance.evaluation`: the protocol digest, the corpus seal, the sealed rows and the sha256 of the report
    that holds the numbers. That is what binds a declared report to THIS engine rather than to a bundle fitted beside
    it."""
    found = []
    if not bundle_dir:
        return found
    root = Path(os.path.expanduser(str(bundle_dir)))
    if (root / "manifest.json").is_file():
        paths = [root / "manifest.json"]
    elif root.is_dir():
        paths = sorted(root.glob("*/manifest.json"))
    else:
        paths = []
    for path in paths:
        try:
            manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        provenance = manifest.get("provenance") or {}
        found.append({"state_id": manifest.get("state_id"), "title": manifest.get("title"),
                      "bundle": Path(path).parent.name,
                      "declared_quality": provenance.get("quality"),
                      "evaluation": provenance.get("evaluation")})
    return found


def _metric_set(report, name):
    for metrics in report.get("metric_sets") or ():
        if metrics.get("name") == name:
            return metrics
    return None


def from_evaluation_report(area, report, digest, *, measured_on=None, source=None):
    """One `m5phet-evaluation-report/1` document, as the quality block of its area."""
    family = report.get("family")
    if family != AREA_FAMILY.get(area):
        return _refusal(area, QUALITY_MEASURED_ON_ANOTHER_FAMILY,
                        f"the report measured family {family!r} and this area's engines answer "
                        f"{AREA_FAMILY.get(area)!r}; a measurement of another family is not a fact about this one")
    block = {"status": MEASURED, "area": area, "kind": family,
             "n": {"scored_rows": report.get("sealed_row_count"), "sealed_rows": report.get("sealed_row_count")},
             "protocol_digest": report.get("protocol_digest"), "corpus_seal": report.get("corpus_seal"),
             "corpus_id": report.get("corpus_seal"), "label_provenance": report.get("label_provenance"),
             "measured_on": measured_on, "flags": list(report.get("flags") or ()),
             "report_sha256": digest, "source": source or "an evaluation report declared for this area"}
    if family == "forecast":
        primary = _metric_set(report, "forecast") or {}
        values = dict(primary.get("values") or {})
        block["values"] = {k: values[k] for k in ("mae", "rmse", "skill_mae", "skill_rmse") if k in values}
        block["baseline"] = primary.get("baseline")
        block["skill"] = values.get("skill_mae")
        block["scale"] = report.get("scale")
        block["target"] = report.get("target")
        block["horizon"] = report.get("horizon")
        block["n"] = dict(block["n"], **{k: v for k, v in (primary.get("counts") or {}).items()})
        coverage = _metric_set(report, "interval_coverage")
        if coverage:
            block["interval_coverage"] = dict(coverage.get("values") or {})
    elif family == "regimes":
        primary = _metric_set(report, "regimes_internal_indices") or {}
        block["values"] = dict(primary.get("values") or {})
        block["indices_omitted"] = report.get("indices_omitted") or {}
        block["baseline"] = primary.get("baseline")
        block["n"] = dict(block["n"], **{k: v for k, v in (primary.get("counts") or {}).items()})
        block["reading"] = ("internal indices of a fitted reference; none of them is an accuracy, and an unsupervised "
                            "assignment has no ground truth")
    else:
        primary = (report.get("metric_sets") or [{}])[0]
        block["values"] = dict(primary.get("values") or {})
        block["baseline"] = primary.get("baseline")
    return block


def _from_declared_report(area, configuration=None, environ=None, bundle_dir=None):
    path = declared_report_path(area, configuration, environ)
    if not path:
        return None
    report, digest, problem = read_report(path)
    if problem:
        return _refusal(area, QUALITY_REPORT_UNREADABLE, problem)
    # the report's DIGEST travels, never its path: a local filesystem path published in an answer says where this
    # operator keeps his files and proves nothing about the number. The same rule the interpreter's reliability keeps.
    measured_on, source = None, "an evaluation report declared for this area, cited by its sha256"
    if area == "forecasting":
        bound = [entry for entry in bundle_evaluations(bundle_dir)
                 if (entry.get("evaluation") or {}).get("report_sha256") == digest]
        if not bound:
            return _refusal(area, QUALITY_MEASURED_ON_ANOTHER_STATE,
                            f"no configured bundle's manifest references the report at {path} (sha256 "
                            f"{digest[:12]}...); a number measured on another fitted state is not a fact about the "
                            f"ones this installation serves")
        measured_on = bound[0].get("state_id") or bound[0].get("bundle")
        source = (f"the evaluation report the bundle {measured_on!r} names in its manifest "
                  f"(provenance.evaluation.report_sha256)")
    return from_evaluation_report(area, report, digest, measured_on=measured_on, source=source)


# --- what every caller asks for --------------------------------------------------------------------------------------

def for_area(area, capabilities=None, configuration=None, environ=None):
    """The `quality` block for one area: a measurement with its protocol, a named refusal, or `NOT_MEASURED`."""
    env = os.environ if environ is None else environ
    if area in AREA_REFUSAL:
        return _refused_area(area)
    if area == "classification":
        record = (capabilities or {}).get("quality") if isinstance(capabilities, dict) else None
        return from_provider_record(area, record)
    if area not in AREA_FAMILY:
        return _not_measured(area, f"{area!r} is not an area this framework measures")
    bundle_dir = None
    if area == "forecasting":
        core = configuration.core("forecasting") if configuration is not None else {}
        bundle_dir = core.get("bundle_dir") or env.get("M5PHET_FORECAST_BUNDLE")
    declared = _from_declared_report(area, configuration, env, bundle_dir)
    if declared is not None:
        return declared
    if area == "forecasting":
        # no report is declared, but a bundle may still SAY where its numbers live. Publishing that pointer is not
        # publishing a number: the block stays NOT_MEASURED and names the protocol and seal a reader can go and find
        references = [entry for entry in bundle_evaluations(bundle_dir) if entry.get("evaluation")]
        if references:
            return _not_measured(area,
                                 "no evaluation report is declared for this area; the configured bundles name the "
                                 "protocol and the seal their numbers were measured under, and the report itself is "
                                 "where those numbers may be quoted from",
                                 evaluation_references=references)
    return _not_measured(area, "no evaluation report is declared for this area, so nothing is known here about how "
                               "well it answers")


def for_areas(areas, capabilities=None, configuration=None, environ=None):
    """One block per area, for a catalog. `capabilities` maps an area to its provider's capabilities."""
    published = capabilities or {}
    return {area: for_area(area, published.get(area), configuration, environ) for area in areas}


# --- one line, and only one ------------------------------------------------------------------------------------------

#: how long a rendered quality line may be. `clip` never leaves half a number behind, so a cut line still passes the
#: narration guard
LINE_LIMIT = 620

#: how much of a digest is shown. The full digest is in the block; the line names enough of it to find the report
DIGEST_SHOWN = 15


def _value(value, limit=80):
    from .outputs import value_text
    return value_text(value, limit)


def _digest(value):
    from .outputs import clip
    return clip(str(value), DIGEST_SHOWN) if value else "NOT_CARRIED"


def _measured_line(block, response=None):
    kind = block.get("kind")
    values = block.get("values") or {}
    parts = []
    if kind == "classification":
        parts.append(f"macro-F1 {_value(values.get('macro_f1'))}")
        if "expected_calibration_error" in values:
            parts.append(f"ECE {_value(values['expected_calibration_error'])} "
                         f"({block.get('calibration_status') or 'calibration status not declared'})")
    elif kind == "forecast":
        scale = block.get("scale")
        parts.append(f"MAE {_value(values.get('mae'))}" + (f" [{_value(scale, 120)}]" if scale else ""))
        if block.get("interval_coverage") and _has_interval(response):
            coverage = block["interval_coverage"]
            parts.append(f"interval coverage {_value(coverage.get('empirical_coverage'))} at nominal "
                         f"{_value(coverage.get('nominal_level'))}")
    elif kind == "regimes":
        if values:
            parts.append(", ".join(f"{name} {_value(value)}" for name, value in sorted(values.items())))
        else:
            parts.append("no internal index was defined over these rows")
    else:
        parts.append(", ".join(f"{name} {_value(value)}" for name, value in sorted(values.items())) or "measured")
    baseline = block.get("baseline") or {}
    if block.get("skill") is not None:
        # named where the report names it (`last_value`), and otherwise stated as the error definition the record
        # carries -- never invented. A skill quoted "vs macro_f1" would name a metric as though it were a baseline.
        if baseline.get("name"):
            parts.append(f"skill {_value(block['skill'])} vs {_value(baseline['name'], 60)}")
        elif block.get("error_definition"):
            parts.append(f"skill {_value(block['skill'])} ({_value(block['error_definition'], 60)}; the naive "
                         f"references are in quality.baseline)")
        else:
            parts.append(f"skill {_value(block['skill'])} over the naive reference its own report declares")
    counts = block.get("n") or {}
    rows = counts.get("scored_rows")
    measured_on = block.get("measured_on")
    tail = [f"{rows} scored rows" if rows is not None else "row count not carried"]
    if measured_on:
        tail.append(f"measured on {_value(measured_on, 40)}")
    from .outputs import clip
    tail.append(f"labels {clip(str(block.get('label_provenance') or 'NOT_CARRIED'), 90)}")
    tail.append(f"protocol {_digest(block.get('protocol_digest'))}")
    tail.append(f"seal {_digest(block.get('corpus_seal'))}")
    if block.get("flags"):
        tail.append("flags " + ",".join(str(flag) for flag in block["flags"]))
    return "; ".join(parts) + " — " + ", ".join(tail)


def _has_interval(response):
    answers = (response or {}).get("answers") or {}
    return any(isinstance(answer, dict) and answer.get("type") == "interval" and answer.get("status") == "OK"
               for answer in answers.values())


def _refusal_body(block):
    """A refusal, with the reason the evaluation package owns -- unless the guard cannot admit that wording.

    `policy_profitability`'s reason ends "profit belongs to an execution record from the system that actually
    traded". That is a statement that profit does NOT exist here, and the narration guard -- which refuses the words
    profit, loss and order in any sentence that does not negate them -- reads its last clause as a claim and refuses
    it, correctly by its own rule. The rule is not relaxed and the reason is not reworded: the line then names the
    refusal and says where its reason is, and the reason itself travels verbatim in the block beside the answer. A
    rendering may say less; it may not say something else, and it may not be exempted from the guard because it is
    about quality."""
    from .orchestrate import narration_problems
    refusal = block.get("refusal")
    reason = block.get("why") or "no reason was recorded"
    candidate = f"{refusal} — {reason}"
    if not narration_problems(candidate, {"quality": block}):
        return candidate
    return (f"{refusal} — this quantity is not measured here and no answer states one; its reason cannot be put in a "
            f"line the narration guard admits, and travels verbatim with this answer in quality.why")


def line(block, response=None, *, label="quality"):
    """Exactly one line about how well this area was measured to answer, or that nothing is known.

    Every figure in it is one the block carries verbatim, which is why it survives the narration guard: the answer
    carries the quality, so the rendering may state it. Nothing here is exempt from that guard, and a line stating a
    number the block does not carry is discarded exactly as a model's invented number is."""
    from .outputs import clip, one_line
    if not isinstance(block, dict):
        return None
    status = block.get("status")
    if status == MEASURED:
        body = _measured_line(block, response)
    elif status == REFUSED:
        body = _refusal_body(block)
    elif status == NOT_MEASURED:
        body = f"{NOT_MEASURED} — {block.get('why') or 'nothing is known about how well this area answers'}"
    else:
        return None
    return clip(one_line(f"{label} ({block.get('area')}): {body}"), LINE_LIMIT)
