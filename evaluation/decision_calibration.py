"""Is Laya's configuration chooser worth listening to? The only honest answer is a count, and today the count is zero.

WP23 of the 2026-09-24 work plan, revision 2. `m5phet.decide` records what Laya chose among the options a repository
declared, with the uncalibrated probabilities the checkpoint's head produced. Those records accumulate quickly and mean
nothing on their own: a choice is a hypothesis. `m5phet.decide.outcome` turns one into evidence by linking it to a row
of the closure table — a *measured*, `COMPARABLE`, *ranked* row — and this module reads those links and reports two
things per decision kind and question:

* **agreement** — how often Laya's argmax was the option whose pipeline the table ranked first. Not how often it was
  plausible, not how often a reviewer would have chosen the same, not how often the author thought it sensible: a
  person's opinion is never a label; only a table row is;
* **reliability** — whether the probability the argmax carried means anything. The outcomes are put in bins of 0.1 by
  that probability, and each bin's observed agreement is shown beside the mean probability claimed in it. The expected
  calibration error is the weighted average of the gaps.

And one refusal, which is the whole point of the package today. Under `MINIMUM_LINKED` linked outcomes a (kind,
question) is `NO_NEW_MEASUREMENT`, with the count that is missing. An agreement rate over nine links is a number that
looks exactly like a measurement and is not one; the report would rather say "22 missing" than print it.

`--inventory` states the position honestly from the other side: how many decision records exist today per kind and
question, and how many of them have an outcome. At the time this module was written the answer was: several kinds,
dozens of records, **zero** outcomes — because no Laya-chosen pipeline has been fitted and ranked yet. That is not a
defect of the report; it is the state of the work, and the report's job is to say so instead of filling the gap.

This module **reads**. It never writes a record, never repairs one, and computes nothing that is not defined here in
words. Its rendering sorts everything and reads no clock, so two runs over the same records produce the same bytes.

Usage::

    python -m evaluation.decision_calibration --outcomes <dir> --out report.json --markdown report.md
    python -m evaluation.decision_calibration --inventory <decisions dir> [<dir> ...] --out report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

CALIBRATION_VERSION = "m5phet-decision-calibration/1"

#: written by `m5phet.decide.outcome`; a test binds this constant to that module's, so the two cannot drift apart
OUTCOME_SCHEMA = "m5phet.decision_outcome.v1"
#: written by `m5phet.decide.record`
DECISION_SCHEMA = "m5phet.decision.v1"

#: the plan's number. Under it, nothing is reported for a (kind, question) but the count and what is missing.
MINIMUM_LINKED = 30

#: reliability bins. Ten of them, the last one closed at 1.0 so a probability of exactly 1 is not its own bin.
BIN_WIDTH = 0.1
BIN_COUNT = 10

#: fixed everywhere a number is rendered, so no rate is silently rounded into agreement
DECIMALS = 6

MEASURED = "MEASURED"
NO_NEW_MEASUREMENT = "NO_NEW_MEASUREMENT"

#: the outcomes of one (kind, question) were chosen from different option sets, so they are not one question
OPTION_SETS_DIFFER = "OPTION_SETS_DIFFER"
#: no outcome of this (kind, question) came from a stage the table ranked first, so no option is the label
NO_BEST_RANKED_OPTION = "NO_BEST_RANKED_OPTION"
#: two stages ranked first under different option keys; no single option was ranked first
AMBIGUOUS_BEST_RANKED_OPTION = "AMBIGUOUS_BEST_RANKED_OPTION"
#: an outcome whose probabilities have no single maximum: there is no argmax to agree or disagree with
AMBIGUOUS_ARGMAX = "ambiguous_argmax"
#: an outcome whose argmax probability is not in [0, 1]; it belongs to no reliability bin
PROBABILITY_OUT_OF_RANGE = "probability_out_of_range"
#: an outcome of a record a PERSON wrote. It carries the option a stage used — a label — and no probability, so it is
#: counted, it may settle the best-ranked option, and it is never scored: an agreement rate that included it would be
#: measuring the person who configured the stage, not the chooser this report exists to calibrate.
HUMAN_NOT_SCORED = "human_choice_not_scored"

#: `m5phet.decide.CHOSEN_BY_LAYA` / `CHOSEN_BY_HUMAN`; a record naming no chooser is Laya's, as every record written
#: before WP23 is. A test binds these to that module's constants so the two cannot drift apart.
CHOSEN_BY_LAYA = "LAYA"
CHOSEN_BY_HUMAN = "HUMAN"

#: WP23's two cases for a stage that entered the closure table, said of one (kind, question) at a time
USABLE_FOR_CALIBRATION = "USABLE_FOR_CALIBRATION"
COMPARABLE_BUT_NO_DECISION_RECORD = "COMPARABLE_BUT_NO_DECISION_RECORD"

ECE_DEFINITION = (
    "expected calibration error = sum over non-empty bins of (n_bin / n_scored) * |agreement_bin - "
    "mean_argmax_probability_bin|, where agreement_bin is the fraction of that bin's outcomes whose argmax was the "
    "best-ranked option. It is a gap between what the head claimed and what the closure table found; it is not an "
    "accuracy, and it says nothing about whether the ranked-first pipeline was any good.")

THE_RULE = (
    "A decision record becomes a label only when a closure-table row exists for the pipeline it led to, that row is "
    "COMPARABLE, and it carries a rank. The label is the rank — never a person's opinion of the choice.")


class CalibrationError(ValueError):
    """The inputs cannot be read as decision or outcome records. Names the file; never falls back to a partial count."""


# --------------------------------------------------------------------------------------------------------------------
# reading: this package reads records, it never writes one
# --------------------------------------------------------------------------------------------------------------------

def _canonical(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _read_record(path: Path, schema: str, required):
    """One content-addressed record, or the reason it is not one. The writer's rules live in `m5phet.decide`; the two
    checks repeated here are the ones a reader cannot do without: the schema, and that the bytes still hash to the
    name the record is filed under."""
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return None, f"cannot be read as JSON ({exc})"
    if not isinstance(payload, dict):
        return None, "a record is a JSON object"
    if payload.get("schema") != schema:
        return None, f"schema {payload.get('schema')!r} is not {schema!r}"
    missing = [field for field in required if field not in payload]
    if missing:
        return None, f"missing {missing}"
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    if path.stem != digest or hashlib.sha256(raw).hexdigest() != digest:
        return None, (f"the record's content hashes to {digest}, which is not the name it is filed under; it has been "
                      "altered since it was written")
    return payload, None


def _read_directory(directories, schema, required):
    """Every `*.json` under the given directories, sorted, each either a record or a named reason it is not."""
    records, unreadable = [], []
    for directory in directories:
        folder = Path(str(directory)).expanduser()
        if not folder.is_dir():
            raise CalibrationError(f"{folder} is not a directory of records")
        for path in sorted(folder.rglob("*.json")):
            payload, why = _read_record(path, schema, required)
            if payload is None:
                unreadable.append({"file": path.name, "directory": str(folder), "why": why})
            else:
                records.append(payload)
    return records, unreadable


OUTCOME_FIELDS = ("decision_sha256", "kind", "question", "chosen", "options", "probabilities", "table_row_sha256",
                  "stage", "rank", "comparability")
DECISION_FIELDS = ("kind", "question", "chosen", "options", "probabilities")


def load_outcomes(directories):
    """`(outcomes, unreadable)` — every `m5phet.decision_outcome.v1` record under the directories, in a fixed order."""
    outcomes, unreadable = _read_directory(directories, OUTCOME_SCHEMA, OUTCOME_FIELDS)
    outcomes.sort(key=lambda item: (item["kind"], item["question"], item["rank"], item["decision_sha256"]))
    return outcomes, unreadable


def load_decisions(directories):
    """`(decisions, unreadable)` — every `m5phet.decision.v1` record under the directories, in a fixed order."""
    decisions, unreadable = _read_directory(directories, DECISION_SCHEMA, DECISION_FIELDS)
    decisions.sort(key=lambda item: (item["kind"], item["question"], hashlib.sha256(_canonical(item)).hexdigest()))
    return decisions, unreadable


# --------------------------------------------------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------------------------------------------------

def argmax(probabilities):
    """`(key, probability)` when one option strictly carries the maximum, else `(None, probability)`.

    A tie is not broken. Two options at the same probability mean the head separated nothing, and choosing between
    them by option order would invent the very preference this report exists to measure.
    """
    if not probabilities:
        return None, None
    best = max(probabilities.values())
    holders = sorted(key for key, value in probabilities.items() if value == best)
    return (holders[0] if len(holders) == 1 else None), best


def bin_of(probability):
    """The index of the reliability bin a probability falls in, or `None` when it is not a probability at all."""
    if probability is None or not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        return None
    return min(BIN_COUNT - 1, int(math.floor(round(probability * BIN_COUNT, 9))))


def bin_label(index):
    lower, upper = index * BIN_WIDTH, (index + 1) * BIN_WIDTH
    closing = "]" if index == BIN_COUNT - 1 else ")"
    return f"[{lower:.2f}, {upper:.2f}{closing}"


def read_table(path):
    """The stages a closure table measured, with their ranks — the other half of WP23's rule.

    This report reads outcomes, and an outcome only exists for a stage that *has* a decision record. A stage that
    entered the table and carries none is therefore invisible here unless the table itself is read, and that stage is
    exactly the one WP23 asks about: it is `COMPARABLE` in the table and unusable for calibration. Nothing is scored
    from this file; it contributes names, ranks and verdicts.
    """
    path = Path(str(path)).expanduser()
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise CalibrationError(f"{path.name}: cannot be read as a closure table ({exc})") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("areas"), list):
        raise CalibrationError(f"{path.name}: a closure table as evaluation/compare_stages.py emits it carries `areas`")
    stages = []
    for area in payload["areas"]:
        for row in area.get("rows") or ():
            stages.append({"stage": str(row.get("stage")), "area": str(area.get("area")),
                           "status": row.get("status"), "comparability": row.get("comparability"),
                           "rank": row.get("rank")})
    stages.sort(key=lambda item: (item["area"], item["stage"]))
    return {"file": path.name, "version": payload.get("version"), "stages": stages,
            "ranked_first": [item for item in stages if item["rank"] == 1]}


def _coverage(stages_with_records, table):
    """Per stage of the table: whether this (kind, question) can be calibrated from it, or only measured by it."""
    if not table:
        return []
    known = set(stages_with_records)
    return [{"stage": item["stage"], "area": item["area"], "rank": item["rank"],
             "comparability": item["comparability"],
             "verdict": USABLE_FOR_CALIBRATION if item["stage"] in known else COMPARABLE_BUT_NO_DECISION_RECORD}
            for item in table["stages"]]


def chooser_of(entry):
    """`LAYA` or `HUMAN` for one outcome. An outcome naming no chooser links a record that named none: Laya's."""
    return entry.get("chosen_by", CHOSEN_BY_LAYA)


def _group_report(kind, question, outcomes, *, table=None):
    """One (kind, question): its counts, what is missing, and — only when nothing is missing — its numbers.

    Human-chosen outcomes are counted apart throughout. They are labels, not evidence: they may settle the
    best-ranked option (WP23's clause — a stage a person configured must say which option it used, or the rank-1 row
    labels nothing), and they are excluded from the agreement rate, the reliability bins and the minimum, because
    `MINIMUM_LINKED` exists to stop a rate being printed over too few *scored* outcomes.
    """
    n_linked = len(outcomes)
    scorable = [entry for entry in outcomes if chooser_of(entry) != CHOSEN_BY_HUMAN]
    human = [entry for entry in outcomes if chooser_of(entry) == CHOSEN_BY_HUMAN]
    missing = max(0, MINIMUM_LINKED - len(scorable))
    reasons = []

    option_sets = sorted({json.dumps(entry["options"], sort_keys=False) for entry in outcomes})
    options = json.loads(option_sets[0]) if len(option_sets) == 1 else None
    if len(option_sets) > 1:
        reasons.append(f"{OPTION_SETS_DIFFER}: {len(option_sets)} different option sets were declared under this kind "
                       "and question, so these outcomes do not answer one question and cannot share an agreement rate")

    declared_best = sorted({entry["best_ranked_option"] for entry in outcomes if entry.get("best_ranked_option")})
    best = declared_best[0] if len(declared_best) == 1 else None
    stages = sorted({entry["stage"] for entry in outcomes})
    coverage = _coverage(stages, table)
    if not declared_best:
        first = [entry["stage"] for entry in (table or {}).get("ranked_first", ())]
        named = ("; the table's first-ranked stage is "
                 + ", ".join(f"{stage!r}, which carries no decision record for this question "
                             f"({COMPARABLE_BUT_NO_DECISION_RECORD})" for stage in first)) if first else ""
        reasons.append(f"{NO_BEST_RANKED_OPTION}: no linked outcome came from a stage the table ranked first, so no "
                       f"option is the label the others are scored against{named}")
    elif len(declared_best) > 1:
        reasons.append(f"{AMBIGUOUS_BEST_RANKED_OPTION}: {declared_best} were each ranked first; no single option was")

    if missing:
        reasons.append(f"{len(scorable)} scorable linked outcome(s) of {n_linked}, {MINIMUM_LINKED} required — "
                       f"{missing} missing")

    group = {"kind": kind, "question": question, "n_linked": n_linked, "n_scorable_linked": len(scorable),
             "n_human_linked": len(human), "missing": missing,
             "status": MEASURED if not reasons else NO_NEW_MEASUREMENT,
             "reason": None if not reasons else f"{NO_NEW_MEASUREMENT}: " + "; ".join(reasons),
             "options": options, "best_ranked_option": best, "n_scored": 0,
             "excluded": {AMBIGUOUS_ARGMAX: 0, PROBABILITY_OUT_OF_RANGE: 0, HUMAN_NOT_SCORED: len(human)},
             "agreements": None, "agreement_rate": None, "bins": [], "expected_calibration_error": None,
             "stages": stages,
             "human_stages": sorted({entry["stage"] for entry in human}),
             "stage_coverage": coverage}
    if reasons:
        return group

    buckets = {}
    scored = agreements = 0
    for entry in scorable:
        chosen, probability = argmax(entry["probabilities"])
        if chosen is None:
            group["excluded"][AMBIGUOUS_ARGMAX] += 1
            continue
        index = bin_of(probability)
        if index is None:
            group["excluded"][PROBABILITY_OUT_OF_RANGE] += 1
            continue
        agreed = chosen == best
        scored += 1
        agreements += int(agreed)
        bucket = buckets.setdefault(index, {"n": 0, "agreements": 0, "total_probability": 0.0})
        bucket["n"] += 1
        bucket["agreements"] += int(agreed)
        bucket["total_probability"] += probability

    group["n_scored"] = scored
    if scored == 0:
        group["status"] = NO_NEW_MEASUREMENT
        group["reason"] = (f"{NO_NEW_MEASUREMENT}: none of the {len(scorable)} scorable linked outcomes carries a "
                           "single argmax with a probability in [0, 1], so there is nothing to score")
        return group

    group["agreements"] = agreements
    group["agreement_rate"] = agreements / scored
    error = 0.0
    for index in sorted(buckets):
        bucket = buckets[index]
        agreement = bucket["agreements"] / bucket["n"]
        mean_probability = bucket["total_probability"] / bucket["n"]
        error += (bucket["n"] / scored) * abs(agreement - mean_probability)
        group["bins"].append({"bin": bin_label(index), "lower": index * BIN_WIDTH, "upper": (index + 1) * BIN_WIDTH,
                              "n": bucket["n"], "agreements": bucket["agreements"], "agreement": agreement,
                              "mean_argmax_probability": mean_probability})
    group["expected_calibration_error"] = error
    return group


def calibrate(outcomes, *, inventory=None, unreadable=(), table=None) -> dict:
    """The whole report: one entry per (kind, question), sorted, plus the inventory when one was asked for."""
    grouped = {}
    for entry in outcomes:
        grouped.setdefault((entry["kind"], entry["question"]), []).append(entry)

    report = {
        "version": CALIBRATION_VERSION,
        "outcome_schema": OUTCOME_SCHEMA,
        "decision_schema": DECISION_SCHEMA,
        "minimum_linked": MINIMUM_LINKED,
        "bin_width": BIN_WIDTH,
        "decimals": DECIMALS,
        "rule": THE_RULE,
        "expected_calibration_error_definition": ECE_DEFINITION,
        "linked_outcomes": len(outcomes),
        "unreadable": list(unreadable),
        "groups": [_group_report(kind, question, grouped[(kind, question)], table=table)
                   for kind, question in sorted(grouped)],
    }
    if inventory is not None:
        report["inventory"] = inventory
    if table is not None:
        report["table"] = table
    return report


def inventory(directories, *, outcomes=()) -> dict:
    """How many decision records exist per kind and question today, and how many of them have an outcome.

    The unlinked count is the honest statement of where WP23 stands: a decision record with no outcome is a
    hypothesis nobody has measured, and no number of them adds up to evidence about the chooser.
    """
    decisions, unreadable = load_decisions(directories)
    linked_digests = {entry["decision_sha256"] for entry in outcomes}

    counts = {}
    for decision in decisions:
        key = (decision["kind"], decision["question"])
        entry = counts.setdefault(key, {"records": 0, "linked": 0})
        entry["records"] += 1
        entry["linked"] += int(hashlib.sha256(_canonical(decision)).hexdigest() in linked_digests)

    groups = [{"kind": kind, "question": question, "records": entry["records"], "linked": entry["linked"],
               "unlinked": entry["records"] - entry["linked"]}
              for (kind, question), entry in sorted(counts.items())]
    return {"directories": sorted(str(Path(str(directory)).expanduser()) for directory in directories),
            "records": sum(group["records"] for group in groups),
            "linked": sum(group["linked"] for group in groups),
            "unlinked": sum(group["unlinked"] for group in groups),
            "groups": groups,
            "unreadable": unreadable}


# --------------------------------------------------------------------------------------------------------------------
# rendering: sorted, fixed decimals, no clock
# --------------------------------------------------------------------------------------------------------------------

def _cell(value) -> str:
    if value is None:
        return NO_NEW_MEASUREMENT
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value) if isinstance(value, int) else f"{float(value):.{DECIMALS}f}"
    return " ".join(str(value).split()).replace("|", "\\|")


def render_markdown(report: dict) -> str:
    lines = [
        "# Calibrating Laya's decisions against the closure table",
        "",
        f"Generated by `evaluation/decision_calibration.py` (`{report['version']}`) from "
        f"{report['linked_outcomes']} linked outcome record(s) (`{report['outcome_schema']}`).",
        "",
        report["rule"],
        "",
        f"A (kind, question) with fewer than {report['minimum_linked']} linked outcomes is "
        f"{NO_NEW_MEASUREMENT}, and the count that is missing is printed instead of a rate. Numbers are rendered with "
        f"{report['decimals']} decimals, fixed; groups, bins and stages are sorted and no clock is read, so two runs "
        "over the same records produce the same bytes.",
        "",
        f"{report['expected_calibration_error_definition']}",
        "",
    ]

    if not report["groups"]:
        lines += [f"**No linked outcome exists.** Nothing is reported per decision kind, because a decision record "
                  f"with no closure-table row is a hypothesis, not a label.", ""]

    for group in report["groups"]:
        lines.append(f"## `{group['kind']}` · question `{group['question']}`")
        lines.append("")
        lines.append(f"- linked outcomes: **{group['n_linked']}** — {group['n_scorable_linked']} scorable, "
                     f"{group['n_human_linked']} chosen by a person and never scored (minimum "
                     f"{report['minimum_linked']} scorable, {group['missing']} missing)")
        lines.append(f"- best-ranked option: {('`' + group['best_ranked_option'] + '`') if group['best_ranked_option'] else NO_NEW_MEASUREMENT}")
        lines.append(f"- status: **{group['status']}**")
        if group["reason"]:
            lines.append(f"- reason: {_cell(group['reason'])}")
        lines.append("")
        if group["stage_coverage"]:
            lines.append("")
            lines.append("| stage | rank | comparability | this question |")
            lines.append("|---|---|---|---|")
            for item in group["stage_coverage"]:
                lines.append(f"| {_cell(item['stage'])} | {_cell(item['rank'])} | {_cell(item['comparability'])} | "
                             f"{_cell(item['verdict'])} |")
        lines.append("")
        if group["status"] != MEASURED:
            lines.append("No agreement rate, no reliability bins and no calibration error are reported for this "
                         "group: the counts above are the whole of what is known.")
            lines.append("")
            continue
        lines.append(f"- scored outcomes: {group['n_scored']} "
                     f"(excluded: {group['excluded'][AMBIGUOUS_ARGMAX]} with no single argmax, "
                     f"{group['excluded'][PROBABILITY_OUT_OF_RANGE]} with a probability outside [0, 1], "
                     f"{group['excluded'][HUMAN_NOT_SCORED]} chosen by a person)")
        lines.append(f"- agreement with the best-ranked option: **{group['agreements']}/{group['n_scored']}** = "
                     f"{_cell(group['agreement_rate'])}")
        lines.append(f"- expected calibration error: **{_cell(group['expected_calibration_error'])}**")
        lines.append("")
        lines.append("| bin | n | agreements | observed agreement | mean argmax probability |")
        lines.append("|---|---|---|---|---|")
        for entry in group["bins"]:
            lines.append(f"| {entry['bin']} | {entry['n']} | {entry['agreements']} | "
                         f"{_cell(entry['agreement'])} | {_cell(entry['mean_argmax_probability'])} |")
        lines.append("")

    if report.get("inventory") is not None:
        book = report["inventory"]
        lines.append("## Inventory: the decision records that exist today")
        lines.append("")
        lines.append(f"{book['records']} decision record(s) under "
                     f"{', '.join('`' + directory + '`' for directory in book['directories'])}; "
                     f"{book['linked']} linked to a closure-table row, {book['unlinked']} not.")
        lines.append("")
        if book["records"] and not book["linked"]:
            lines.append("**None of them is a label: no decision record has an outcome yet**, because no Laya-chosen "
                         "pipeline has been fitted and ranked. The records below are hypotheses.")
            lines.append("")
        lines.append("| kind | question | records | linked | unlinked |")
        lines.append("|---|---|---|---|---|")
        for group in book["groups"]:
            lines.append(f"| {_cell(group['kind'])} | {_cell(group['question'])} | {group['records']} | "
                         f"{group['linked']} | {group['unlinked']} |")
        lines.append("")
        if book["unreadable"]:
            lines.append(f"{len(book['unreadable'])} file(s) under those directories are not decision records: "
                         + ", ".join(f"`{entry['file']}` ({_cell(entry['why'])})" for entry in book["unreadable"]))
            lines.append("")

    if report["unreadable"]:
        lines.append(f"{len(report['unreadable'])} file(s) in the outcome directories are not outcome records: "
                     + ", ".join(f"`{entry['file']}` ({_cell(entry['why'])})" for entry in report["unreadable"]))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.decision_calibration",
        description="Agreement and reliability of Laya's decisions against the closure table that judged them.")
    parser.add_argument("--outcomes", action="append", metavar="DIR",
                        help="a directory of m5phet.decision_outcome.v1 records; repeat for several")
    parser.add_argument("--inventory", nargs="+", metavar="DIR",
                        help="directories of m5phet.decision.v1 records to count per kind and question")
    parser.add_argument("--table", metavar="PATH",
                        help="a closure table as evaluation/compare_stages.py emits it; its stages and ranks are read "
                             "so the report can name the stages that entered the table with no decision record")
    parser.add_argument("--out", metavar="PATH", help="write the report as JSON")
    parser.add_argument("--markdown", metavar="PATH", help="write the report as Markdown (stdout when omitted)")
    args = parser.parse_args(argv)

    if not args.outcomes and not args.inventory:
        parser.error("nothing to read: give --outcomes, --inventory, or both")

    outcomes, unreadable = load_outcomes(args.outcomes) if args.outcomes else ([], [])
    book = inventory(args.inventory, outcomes=outcomes) if args.inventory else None
    table = read_table(args.table) if args.table else None
    report = calibrate(outcomes, inventory=book, unreadable=unreadable, table=table)
    markdown = render_markdown(report)

    if args.out:
        Path(args.out).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    if args.markdown:
        Path(args.markdown).write_text(markdown)
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
