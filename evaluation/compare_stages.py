"""One closure table per area across the doctoral stages, built from reports instead of typed from memory.

A doctoral comparison is a sequence of stages — baseline representation, designed representation, searched
representation, calendar-augmented — and the temptation it creates is always the same: line the stages up in a table and
let the reader assume the rows are commensurable. They usually are not. A stage measured on a corpus that was re-sealed,
a stage that reported RMSE where the previous one reported MAE, a stage whose naive reference was recomputed, a stage
that produced no measurement at all and is shown as an empty cell that reads like a zero — each of those produces a
table that looks exactly like a valid one.

So this generator reads, and refuses, and computes almost nothing:

* every number in the table comes out of an evaluation report (``m5phet-evaluation-report/1``). The generator computes
  the skill column — and only where the report does not already carry the package's own skill — and the comparability
  verdicts. Nothing else is derived, and no reported number is transformed;
* two stages are ``COMPARABLE`` only when they share the holdout identity (the corpus seal, which covers the rows *and*
  their labels), the metric, the target, the horizon and the naive reference. Otherwise the verdict is
  ``NOT_COMPARABLE: <field> differs (...)``, naming the field, and the stage is left unranked;
* a stage with no measurement in an area is ``NO_NEW_MEASUREMENT`` with the reason, never a blank. For the three areas
  whose quality this package refuses by name (``regime_accuracy``, ``causal_accuracy``, ``policy_profitability``) the
  reason is the package's own text, quoted from ``REFUSED_METRICS``. A refused area may still carry a **declared
  internal index** — ``regimes`` does, and ``ranked_metric_is_not_a_quality_claim`` says so in the area and in the
  rendering — and then the rows are ranked on that index under its own name while the refusal stays exactly where it
  was. Ranking an internal index is not measuring quality: a tighter clustering is a tighter clustering, and the
  refused metric is refused in the same row. An area whose ``metric_keys`` are empty (``causal``, ``policy``) has
  nothing to rank and every row of it stays ``NO_NEW_MEASUREMENT``;
* a literature value appears only when a report carries one, with its source. Otherwise the cell says ``NOT_CARRIED``,
  which is a different statement from "there is no literature";
* skill is the package's definition, ``1 - model_error / naive_error`` on the same rows, read from the report when the
  scorer computed it. Where the family's metric is a score and not an error, skill is ``NOT_DEFINED`` with the reason
  rather than a ratio of two scores dressed up as one.

The rendering reads no clock and sorts everything, so two runs over the same reports produce the same bytes and the
table can be diffed — a table that cannot be diffed cannot be reviewed.

Usage::

    python -m evaluation.compare_stages --report baseline=A.json --report B.json --out table.json --markdown table.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:                     # runnable from a checkout, as the rest of this package is
    sys.path.insert(0, str(_SRC))

from m5phet_evaluation.protocol import FAMILIES                                                          # noqa: E402
from m5phet_evaluation.report import REPORT_VERSION                                                      # noqa: E402
from m5phet_evaluation.scoring import REFUSED_METRICS                                                    # noqa: E402

COMPARISON_VERSION = "m5phet-evaluation-stage-comparison/1"

#: fixed everywhere a number is rendered, and declared in the header, so no cell is silently rounded to flatter a stage
DECIMALS = 6

MEASURED = "MEASURED"
NO_NEW_MEASUREMENT = "NO_NEW_MEASUREMENT"
COMPARABLE = "COMPARABLE"
NOT_COMPARABLE = "NOT_COMPARABLE"
NOT_CARRIED = "NOT_CARRIED"
NOT_DEFINED = "NOT_DEFINED"
NOT_RANKED = "NOT_RANKED"
NOT_ON_SAME_ROWS = "NOT_ON_SAME_ROWS"

#: annotation keys a report may carry beside the fields build_report writes. build_report never writes them; a producer
#: that knows the stage, the target, the horizon, the scale or a literature value adds them, and this generator reads
#: them. Anything absent stays NOT_CARRIED, which is why the column exists.
ANNOTATIONS = ("stage", "target", "horizon", "scale", "literature")


class StageComparisonError(ValueError):
    """The inputs cannot be compared as they stand. Names the file and the field; never falls back to a partial table."""


@dataclasses.dataclass(frozen=True)
class ClosureReading:
    """How one area's closure numbers are found inside a report, declared here instead of guessed per file.

    ``metric_keys`` is a preference order, not a filter: the first key any metric set carries is the one the table
    reports, and the choice is written into the table so a stage reporting another metric is visible rather than
    silently rescued. ``orientation`` says whether the reported quantity is an error (lower is better, and skill is
    defined against a naive error) or a score (higher is better, and skill is not defined by this package).
    """

    family: str
    orientation: str
    metric_keys: tuple
    skill_keys: dict
    refusal: str | None


READINGS = {
    "forecast": ClosureReading("forecast", "error", ("mae", "rmse"),
                               {"mae": "skill_mae", "rmse": "skill_rmse"}, None),
    "classification": ClosureReading("classification", "score", ("macro_f1", "accuracy"), {}, None),
    # WP19 declares that the stage table carries the representation area's internal indices under their own names
    # while `regime_accuracy` stays refused, and WP29 needs the two stages of one corpus ranked against each other.
    # `silhouette` is the one index ranked, because a table ranks one metric and this package will not mix a
    # higher-is-better index and a lower-is-better one into a single column. It is an internal index of the
    # assignment, not an accuracy; nothing about it says the clusters mean anything.
    "regimes": ClosureReading("regimes", "score", ("silhouette",), {}, "regime_accuracy"),
    "causal": ClosureReading("causal", "score", (), {}, "causal_accuracy"),
    "policy": ClosureReading("policy", "score", (), {}, "policy_profitability"),
}

SCORE_HAS_NO_SKILL = (
    f"{NOT_DEFINED}: this package defines skill as 1 - model_error / naive_error on the same rows "
    "(m5phet_evaluation.scoring.score_forecast). {metric} is a score, not an error, and this generator does not "
    "transform a reported number into another scale to produce a skill. Declare an error metric in the protocol if a "
    "skill is wanted for this area.")


@dataclasses.dataclass(frozen=True)
class Stage:
    """One report, bound to the stage whose measurement it is. Read-only: nothing here writes back to a report."""

    stage: str
    area: str
    source: str
    payload: dict

    @property
    def corpus_seal(self) -> str:
        return str(self.payload.get("corpus_seal", ""))

    @property
    def sealed_row_count(self):
        return self.payload.get("sealed_row_count")

    def annotation(self, key: str):
        """Report-level annotation, then the same key inside any metric set's values; absent is None, never a default."""
        if self.payload.get(key) is not None:
            return self.payload[key]
        for metrics in self.payload.get("metric_sets") or ():
            values = metrics.get("values") or {}
            if values.get(key) is not None:
                return values[key]
        return None


def annotate(report, **fields) -> dict:
    """The report as JSON, plus the declared annotations. One writer for fixtures and for real producers, so the key
    names in a fixture cannot drift away from the key names this generator reads."""
    payload = json.loads(report.to_json()) if hasattr(report, "to_json") else json.loads(json.dumps(report, default=str))
    for key, value in fields.items():
        if key not in ANNOTATIONS:
            raise StageComparisonError(f"annotate: {key!r} is not one of the declared annotations {ANNOTATIONS}")
        if value is not None:
            payload[key] = value
    return payload


def load_stage(path, *, stage: str | None = None) -> Stage:
    """Read one report and name its stage: the argument, then the report's own annotation, then the file name."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise StageComparisonError(f"{path.name}: cannot be read as an evaluation report ({exc})") from None
    if not isinstance(payload, dict):
        raise StageComparisonError(f"{path.name}: an evaluation report is a JSON object")
    version = payload.get("version")
    if version != REPORT_VERSION:
        raise StageComparisonError(
            f"{path.name}: version {version!r} is not {REPORT_VERSION!r}. A report written by another version is not a "
            "weaker report but an unknown one, and reading it would invent agreement between two formats")
    family = payload.get("family")
    if family not in FAMILIES:
        raise StageComparisonError(f"{path.name}: family {family!r} is not one of {FAMILIES}")
    named = stage or payload.get("stage") or path.stem
    return Stage(stage=str(named), area=str(family), source=path.name, payload=payload)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _literature(stage: Stage) -> dict:
    carried = stage.annotation("literature")
    if not isinstance(carried, dict):
        return {"value": NOT_CARRIED, "metric": NOT_CARRIED, "scale": NOT_CARRIED, "source": NOT_CARRIED}
    value = _number(carried.get("value"))
    return {
        "value": value if value is not None else NOT_CARRIED,
        "metric": carried.get("metric") or NOT_CARRIED,
        "scale": carried.get("scale") or NOT_CARRIED,
        "source": carried.get("source") or NOT_CARRIED,
    }


def _select_metric(stage: Stage, reading: ClosureReading):
    """The reported metric, chosen by declared preference across the metric sets the report carries."""
    sets = stage.payload.get("metric_sets") or ()
    for key in reading.metric_keys:
        for index, metrics in enumerate(sets):
            value = _number((metrics.get("values") or {}).get(key))
            if value is not None:
                return index, key, value
    return None, None, None


def _naive(metrics: dict, metric_key: str) -> dict:
    baseline = metrics.get("baseline")
    if not isinstance(baseline, dict):
        return {"name": NOT_CARRIED, "error": NOT_CARRIED, "same_rows_as_model": False,
                "note": "the metric set carries no baseline, so there is no naive reference on these rows"}
    value = _number(baseline.get(metric_key))
    same_rows = baseline.get("same_rows_as_model") is True
    return {
        "name": baseline.get("name") or NOT_CARRIED,
        "error": value if value is not None else NOT_CARRIED,
        "rows": baseline.get("rows"),
        "same_rows_as_model": same_rows,
        "note": None if same_rows else ("the baseline does not declare that it was computed on the scored rows; a "
                                        "baseline evaluated on other rows produces a skill that looks identical to a "
                                        "real one"),
    }


def _skill(stage: Stage, reading: ClosureReading, index: int, metric_key: str, model_error: float, naive: dict):
    """The package's skill, quoted when the scorer computed it, computed with the package's own formula when not."""
    if reading.orientation != "error":
        return None, SCORE_HAS_NO_SKILL.format(metric=metric_key)
    skill_key = reading.skill_keys.get(metric_key)
    values = (stage.payload["metric_sets"][index].get("values") or {})
    if skill_key and skill_key in values:
        carried = _number(values[skill_key])
        if carried is not None:
            return carried, f"report:metric_sets[{index}].values.{skill_key}"
    naive_error = naive["error"] if isinstance(naive["error"], float) else None
    if naive_error is None:
        return None, f"{NOT_DEFINED}: no naive reference is carried for {metric_key} on these rows"
    if not naive["same_rows_as_model"]:
        return None, (f"{NOT_DEFINED}: the naive reference is {NOT_ON_SAME_ROWS}, and a skill against a baseline "
                      "computed elsewhere is the comparison this package exists to prevent")
    if naive_error == 0:
        return None, (f"{NOT_DEFINED}: the naive error is zero on these rows, so skill is undefined rather than perfect")
    return 1 - model_error / naive_error, "computed: 1 - model_error / naive_error"


def _row_for(stage: Stage | None, reading: ClosureReading, stage_name: str, *, reason: str | None = None) -> dict:
    """One stage's line in one area's table. Every field is filled; nothing here can come out blank."""
    row = {
        "stage": stage_name, "area": reading.family, "status": NO_NEW_MEASUREMENT, "reason": None, "refusal": None,
        "metric": NOT_CARRIED, "scale": NOT_CARRIED, "target": NOT_CARRIED, "horizon": NOT_CARRIED,
        "model_error": None, "orientation": reading.orientation,
        "naive": {"name": NOT_CARRIED, "error": NOT_CARRIED, "same_rows_as_model": False, "note": None},
        "skill": None, "skill_source": f"{NOT_DEFINED}: no measurement",
        "literature": {"value": NOT_CARRIED, "metric": NOT_CARRIED, "scale": NOT_CARRIED, "source": NOT_CARRIED},
        "comparability": NO_NEW_MEASUREMENT, "rank": None, "conditions": None,
    }
    if stage is None:
        row["reason"] = reason or f"no report for area {reading.family!r} in stage {stage_name!r}"
        return row

    row["source"] = stage.source
    row["conditions"] = _conditions(stage)
    row["literature"] = _literature(stage)
    for key in ("target", "horizon", "scale"):
        carried = stage.annotation(key)
        row[key] = carried if carried is not None else NOT_CARRIED

    if reading.refusal is not None:
        # the refusal is carried whether or not the area also has an index to rank: a reader of this row must see that
        # the area's quality metric does not exist, in the row, and not only in the area's header
        row["refusal"] = {"metric": reading.refusal, "reason": REFUSED_METRICS[reading.refusal]}
        row["skill_source"] = f"{NOT_DEFINED}: {reading.refusal} is refused by name"
        if not reading.metric_keys:
            row["reason"] = (f"{reading.refusal} is refused by this package: {REFUSED_METRICS[reading.refusal]}. The "
                             "report carries what this area can report; none of it is a quality claim.")
            row["metric"] = f"{reading.refusal} (refused)"
            return row

    index, metric_key, model_error = _select_metric(stage, reading)
    if metric_key is None:
        row["reason"] = (f"no declared quality metric is carried by this report (looked for "
                         f"{', '.join(reading.metric_keys)} in every metric set)")
        return row

    metrics = stage.payload["metric_sets"][index]
    naive = _naive(metrics, metric_key)
    skill, skill_source = _skill(stage, reading, index, metric_key, model_error, naive)
    row.update({"status": MEASURED, "metric": metric_key, "model_error": model_error, "naive": naive,
                "skill": skill, "skill_source": skill_source,
                "metric_set": metrics.get("name"), "metric_set_index": index})
    return row


def _conditions(stage: Stage) -> dict:
    """What the report says a number cannot be quoted without. Copied, never summarised away."""
    sets = stage.payload.get("metric_sets") or ()
    scored = min((int((m.get("counts") or {}).get("scored_rows", 0)) for m in sets), default=0)
    return {
        "scored_rows": scored,
        "sealed_row_count": stage.sealed_row_count,
        "minimum_rows": stage.payload.get("minimum_rows"),
        "label_provenance": stage.payload.get("label_provenance"),
        "label_source": stage.payload.get("label_source") or NOT_CARRIED,
        "protocol_digest": stage.payload.get("protocol_digest"),
        "corpus_seal": stage.corpus_seal,
        "sealed_at": stage.payload.get("sealed_at"),
        "report_generated_at": stage.payload.get("generated_at"),
        "flags": list(stage.payload.get("flags") or ()),
        "report_file": stage.source,
    }


def _identity(row: dict) -> dict:
    return {"holdout": ((row.get("conditions") or {}).get("corpus_seal"),
                        (row.get("conditions") or {}).get("sealed_row_count")),
            "metric": row["metric"], "target": row["target"], "horizon": row["horizon"],
            "naive": row["naive"]["name"]}


def _verdict(row: dict, reference: dict) -> tuple:
    """COMPARABLE, or the first field that differs, named. Order is fixed so the reason does not depend on dict order."""
    if row is reference:
        return COMPARABLE, []
    mine, theirs = _identity(row), _identity(reference)
    bound = []
    if mine["holdout"] != theirs["holdout"]:
        (seal_a, rows_a), (seal_b, rows_b) = mine["holdout"], theirs["holdout"]
        return (f"{NOT_COMPARABLE}: holdout differs (corpus_seal {str(seal_a)[:12]} over {rows_a} sealed rows vs "
                f"{str(seal_b)[:12]} over {rows_b} in stage {reference['stage']!r}) — the two stages were not measured "
                "on the same rows, and the seal covers the rows and their labels"), bound
    if mine["metric"] != theirs["metric"]:
        return (f"{NOT_COMPARABLE}: metric differs ({mine['metric']} vs {theirs['metric']} in stage "
                f"{reference['stage']!r}) — two error scales ranked as one"), bound
    for field in ("target", "horizon"):
        if mine[field] == NOT_CARRIED and theirs[field] == NOT_CARRIED:
            # the holdout matched above, so the same sealed labels are the same target at the same horizon
            bound.append(field)
            continue
        if mine[field] != theirs[field]:
            return (f"{NOT_COMPARABLE}: {field} differs ({mine[field]} vs {theirs[field]} in stage "
                    f"{reference['stage']!r})"), bound
    if mine["naive"] != theirs["naive"]:
        return (f"{NOT_COMPARABLE}: naive reference differs ({mine['naive']} vs {theirs['naive']} in stage "
                f"{reference['stage']!r}) — a skill is a ratio against one declared baseline"), bound
    return COMPARABLE, bound


def compare(stages, *, not_measured=()) -> dict:
    """One table per area, every stage present in every table, verdicts and ranks against one reference stage.

    `not_measured` enters stages that were ATTEMPTED and produced no report, as `{"stage", "area", "reason"}`. A stage
    that was tried and refused is not the same thing as a stage nobody ran, and leaving it out of the table makes the
    two indistinguishable: the reader sees four candidates in the design artifact and two rows, and cannot tell whether
    the others lost, were skipped, or were quietly dropped. Such a row carries NO_NEW_MEASUREMENT, the refusal's own
    words, and no number — it is never ranked and never compared, because there is nothing to compare.
    """
    stages = list(stages)
    declared = []
    for entry in not_measured:
        if not isinstance(entry, dict) or not entry.get("stage") or not entry.get("reason"):
            raise StageComparisonError(f"a not-measured stage is {{'stage', 'area', 'reason'}}; {entry!r} is not")
        area = entry.get("area")
        if area not in READINGS:
            raise StageComparisonError(f"not-measured stage {entry['stage']!r}: area {area!r} is not one of "
                                       f"{sorted(READINGS)}")
        clash = next((stage for stage in stages if stage.stage == entry["stage"] and stage.area == area), None)
        if clash is not None:
            raise StageComparisonError(
                f"stage {entry['stage']!r} is declared not measured in area {area!r} and also carries the report "
                f"{clash.source}: a stage cannot both have a measurement and lack one")
        declared.append({"stage": str(entry["stage"]), "area": area, "reason": str(entry["reason"])})
    if not stages:
        raise StageComparisonError("compare: at least one evaluation report is required")
    seen = {}
    for stage in stages:
        key = (stage.stage, stage.area)
        if key in seen:
            raise StageComparisonError(
                f"stage {stage.stage!r} reports area {stage.area!r} twice ({seen[key]} and {stage.source}): two reports "
                "for one stage and one area cannot be reduced to one line without choosing between them")
        seen[key] = stage.source

    names = sorted({stage.stage for stage in stages} | {entry["stage"] for entry in declared})
    areas = []
    for family in sorted({stage.area for stage in stages} | {entry["area"] for entry in declared}):
        reading = READINGS[family]
        by_stage = {stage.stage: stage for stage in stages if stage.area == family}
        reasons = {entry["stage"]: entry["reason"] for entry in declared if entry["area"] == family}
        rows = [_row_for(by_stage.get(name), reading, name, reason=reasons.get(name)) for name in names]

        reference = next((row for row in rows if row["status"] == MEASURED), None)
        bound_by_seal = set()
        for row in rows:
            if row["status"] != MEASURED:
                continue
            verdict, bound = _verdict(row, reference)
            row["comparability"] = verdict
            row["compared_to"] = reference["stage"]
            bound_by_seal.update(bound)

        ranked = [row for row in rows if row["status"] == MEASURED and row["comparability"] == COMPARABLE]
        ranked.sort(key=lambda row: (row["model_error"] if reading.orientation == "error" else -row["model_error"],
                                     row["stage"]))
        position, previous = 0, object()
        for index, row in enumerate(ranked, start=1):
            if row["model_error"] != previous:              # ties share a rank; nothing here breaks a tie by luck
                position, previous = index, row["model_error"]
            row["rank"] = position

        areas.append({
            "area": family,
            "orientation": reading.orientation,
            "ranked_metric_keys": list(reading.metric_keys),
            "ranked_metric_is_not_a_quality_claim": (
                None if reading.refusal is None or not reading.metric_keys else
                f"the rows of this area are ranked on {'/'.join(reading.metric_keys)}, a declared internal index of "
                f"the assignment itself. {reading.refusal} is refused by name and stays refused in every row: a rank "
                f"here orders the indices, and says nothing about whether the clusters mean anything"),
            "reference_stage": reference["stage"] if reference else None,
            "refused_metric": reading.refusal,
            "refusal_reason": REFUSED_METRICS[reading.refusal] if reading.refusal else None,
            "bound_by_seal": sorted(bound_by_seal),
            "rows": rows,
        })

    return {
        "version": COMPARISON_VERSION,
        "decimals": DECIMALS,
        "report_version": REPORT_VERSION,
        "stages": names,
        "computed_here": ["skill (only where the report carries none)", "comparability verdicts", "rank"],
        "read_from_reports": ["model error", "metric", "scale", "target", "horizon", "naive reference and its error",
                              "skill where the scorer computed it", "literature value and source", "conditions"],
        "inputs": sorted(({"stage": stage.stage, "area": stage.area, "report_file": stage.source,
                           "protocol_digest": stage.payload.get("protocol_digest"),
                           "corpus_seal": stage.corpus_seal,
                           "report_generated_at": stage.payload.get("generated_at")} for stage in stages),
                         key=lambda item: (item["area"], item["stage"], item["report_file"])),
        "not_measured": sorted(declared, key=lambda entry: (entry["area"], entry["stage"])),
        "areas": areas,
    }


# --------------------------------------------------------------------------------------------------------------------
# rendering: sorted, fixed decimals, no clock
# --------------------------------------------------------------------------------------------------------------------

COLUMNS = ("stage", "status", "metric", "scale", "target", "horizon", "model error", "naive reference", "naive error",
           "skill", "skill source", "literature value", "literature source", "comparability", "rank")


def _cell(value) -> str:
    if value is None:
        return NOT_CARRIED
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{DECIMALS}f}"
    text = " ".join(str(value).split())
    return text.replace("|", "\\|") or NOT_CARRIED


def _value_cell(row: dict, value) -> str:
    """A cell of a stage with no measurement says so; an empty cell reads as a zero or as agreement."""
    if row["status"] != MEASURED and value is None:
        return NO_NEW_MEASUREMENT
    return _cell(value)


def render_markdown(table: dict) -> str:
    lines = [
        "# Doctoral comparison across stages",
        "",
        f"Generated by `evaluation/compare_stages.py` (`{table['version']}`) from "
        f"{len(table['inputs'])} evaluation report(s) (`{table['report_version']}`).",
        "",
        f"Every number below is read from a report. The generator computes exactly three things: the skill column "
        f"where the report carries none, the comparability verdicts, and the rank. Numbers are rendered with "
        f"{table['decimals']} decimals, fixed. Areas are sorted by family, stages by name, and no clock is read, so two "
        "runs over the same reports produce the same bytes.",
        "",
        f"Stages: {', '.join(f'`{name}`' for name in table['stages'])}.",
        "",
    ]
    for area in table["areas"]:
        lines.append(f"## Area: {area['area']}")
        lines.append("")
        if area["refused_metric"]:
            lines.append(f"**{area['refused_metric']} is refused by this package.** {area['refusal_reason']}")
            lines.append("")
        if area.get("ranked_metric_is_not_a_quality_claim"):
            lines.append(f"**The rank below is not a quality claim.** "
                         f"{area['ranked_metric_is_not_a_quality_claim']}.")
            lines.append("")
        lines.append("| " + " | ".join(COLUMNS) + " |")
        lines.append("|" + "|".join("---" for _ in COLUMNS) + "|")
        for row in area["rows"]:
            literature = row["literature"]
            lines.append("| " + " | ".join([
                _cell(row["stage"]),
                _cell(row["status"]),
                _cell(row["metric"]),
                _cell(row["scale"]),
                _cell(row["target"]),
                _cell(row["horizon"]),
                _value_cell(row, row["model_error"]),
                _value_cell(row, row["naive"]["name"] if row["status"] == MEASURED else None),
                _value_cell(row, row["naive"]["error"] if row["status"] == MEASURED else None),
                _value_cell(row, row["skill"]),
                _cell(row["skill_source"]),
                _cell(literature["value"]),
                _cell(literature["source"]),
                _cell(row["comparability"]),
                str(row["rank"]) if row["rank"] is not None else NOT_RANKED,   # a rank is a position, not a measurement
            ]) + " |")
        lines.append("")
        if area["reference_stage"]:
            lines.append(f"Comparability is judged against the reference stage `{area['reference_stage']}` "
                         "(the first measured stage in name order).")
        else:
            lines.append(f"No stage carries a measurement in this area, so there is no reference stage and every row "
                         f"is {NO_NEW_MEASUREMENT}.")
        if area["bound_by_seal"]:
            lines.append(f"Not carried by these reports, and therefore bound by the shared corpus seal rather than "
                         f"checked as a field: {', '.join(area['bound_by_seal'])}. The seal covers the rows and their "
                         "labels, so an identical seal is the same target at the same horizon on the same rows.")
        lines.append("")
        lines.append("Conditions (no number above can be quoted without them):")
        lines.append("")
        for row in area["rows"]:
            if row["conditions"] is None:
                lines.append(f"- **{row['stage']}** — {NO_NEW_MEASUREMENT}: {_cell(row['reason'])}")
                continue
            conditions = row["conditions"]
            flags = ", ".join(conditions["flags"]) if conditions["flags"] else "none"
            line = (f"- **{row['stage']}** — {conditions['scored_rows']} of {conditions['sealed_row_count']} sealed rows "
                    f"scored (declared minimum {conditions['minimum_rows']}) · labels "
                    f"{conditions['label_provenance']} from {_cell(conditions['label_source'])} · protocol "
                    f"{str(conditions['protocol_digest'])[:12]} · seal {str(conditions['corpus_seal'])[:12]} sealed "
                    f"{conditions['sealed_at']} · report written {conditions['report_generated_at']} · flags: {flags} "
                    f"· file `{conditions['report_file']}`")
            if row["status"] != MEASURED and row["reason"]:
                line += f" · {NO_NEW_MEASUREMENT}: {_cell(row['reason'])}"
            if row["naive"].get("note"):
                line += f" · naive: {_cell(row['naive']['note'])}"
            lines.append(line)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _split_report_arg(text: str) -> tuple:
    """``STAGE=PATH`` when the text before the first ``=`` names no directory and no file that exists; else ``PATH``."""
    if "=" in text:
        head, _, tail = text.partition("=")
        if head and "/" not in head and not Path(text).exists():
            return head.strip(), tail
    return None, text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.compare_stages",
        description="One closure table per area across doctoral stages, built from evaluation reports.")
    parser.add_argument("--report", action="append", required=True, metavar="[STAGE=]PATH",
                        help="an evaluation report; repeat once per stage and area")
    parser.add_argument("--not-measured", action="append", default=[], metavar="STAGE=AREA:REASON",
                        help="a stage that was attempted and produced no report; it enters the table as "
                             "NO_NEW_MEASUREMENT with this reason, unranked and uncompared")
    parser.add_argument("--out", metavar="PATH", help="write the table as JSON")
    parser.add_argument("--markdown", metavar="PATH", help="write the table as Markdown (stdout when omitted)")
    args = parser.parse_args(argv)

    stages = []
    for text in args.report:
        stage, path = _split_report_arg(text)
        stages.append(load_stage(path, stage=stage))
    declared = []
    for text in args.not_measured:
        name, _, rest = text.partition("=")
        area, _, reason = rest.partition(":")
        if not name.strip() or not area.strip() or not reason.strip():
            parser.error(f"--not-measured takes STAGE=AREA:REASON; {text!r} is not that")
        declared.append({"stage": name.strip(), "area": area.strip(), "reason": reason.strip()})
    table = compare(stages, not_measured=declared)
    markdown = render_markdown(table)

    if args.out:
        Path(args.out).write_text(json.dumps(table, sort_keys=True, indent=2) + "\n")
    if args.markdown:
        Path(args.markdown).write_text(markdown)
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
