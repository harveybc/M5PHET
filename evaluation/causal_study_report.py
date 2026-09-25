"""WP22 step 6: the row a fitted causal study occupies in the closure table, written as a report and not as a cell.

The closure table already holds two arms of the calendar-event study: `naive`, the mean response by surprise sign, and
`local_projection`, the OLS impulse response -- both scored as *forecasts*, by their out-of-sample error on the realised
outcome of the held-out releases. WP22 asks for a third: `dml`, the EconML study of the same surprise. Writing that row
is not a matter of finding the DML's error, because a fitted causal study does not carry one. It carries an effect and
an interval, and `m5phet_evaluation` has a family for exactly that -- `causal` -- whose accuracy metric it refuses by
name (`causal_accuracy`: there is no held-out counterfactual to score against).

So this module writes a `causal`-family `m5phet-evaluation-report/1` for a study that `prepare-study` fitted, and
`compare_stages` renders it in the causal area with `causal_accuracy` marked refused and no rank. That is the honest
shape of the DML arm: present in the table, named, with its effect and interval visible and its comparability stated
rather than assumed. **It is not comparable with the forecast arms**, and the reason is a field: the metric. The two
forecast stages report `mae` on realised outcomes over the held-out releases; this stage reports an effect per unit of
treatment over the releases the study was fitted on. Nothing here converts one into the other, because nothing can: the
retained study carries no outcome model, so there is no prediction it could make for a held-out release, and inventing
one from the effect alone would be manufacturing the very number the table exists to check.

What the report carries, and where each field comes from:

* the **population** is the `event_key` of every row of the table the study was fitted on, and the **labels** are those
  rows' realised outcome. They are sealed, so a study quoted beside another is quoted over a corpus whose identity can
  be checked;
* the **estimate, interval, assumptions and diagnostics** are copied out of the `prepare-study` result verbatim. No
  number in this report is computed here;
* the **identification caveat** the spec declared is copied in as an annotation rule, so the report cannot be read
  without the sentence saying the publication clock was never observed;
* `causal_accuracy` stays refused. The report says what the study measured; it makes no quality claim, and the table
  generator prints the package's own refusal text in the metric cell.

Deterministic: no clock is read. The report is generated as of the last release in the sealed population, so two runs
over the same artifacts write the same bytes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from m5phet_evaluation import freeze, protocol, report, scoring                                           # noqa: E402

#: the annotations `compare_stages.py` reads off the top level of a report
ANNOTATIONS = ("stage", "target", "horizon", "scale")

#: the metrics this report promises before any of them is read. A causal metric set carries the estimand, the estimate,
#: the interval and the assumptions -- and no accuracy, which is refused by name by the package itself.
METRICS = ("estimand", "estimate", "interval", "assumptions")

#: there is no baseline an effect is a ratio against; the package's baseline field is a name, and this is the honest one
BASELINE = "none: an effect has no naive reference on the same rows, and causal_accuracy is refused"


class CausalReportError(ValueError):
    """The study or its table cannot produce a report as they stand. Names the field; never fills one in."""


def _rows(table_path, outcome):
    """The (event_key, realised outcome) pairs of the table a study was fitted on, in file order."""
    with Path(table_path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise CausalReportError(f"{table_path}: the study's table has no rows, so there is no population to seal")
    for field in ("event_key", "published_at", outcome):
        if field not in rows[0]:
            raise CausalReportError(f"{table_path}: the table carries {sorted(rows[0])} and this report needs "
                                    f"{field!r}; a population cannot be sealed from columns that are not there")
    keys = [row["event_key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise CausalReportError(f"{table_path}: an event key appears twice, which makes every count ambiguous")
    return rows


def build(study, table_path, *, stage, outcome, horizon, scale, caveat=None, minimum_rows=100,
          label_source=None, label_producer=None):
    """One causal-family report for one fitted study, every number copied out of its `prepare-study` result."""
    result = (study or {}).get("result") or {}
    if result.get("status") != "OK":
        raise CausalReportError(f"the study result is {result.get('status')!r} ({result.get('reason')!r}); a report is "
                                f"written for a study that was fitted, and a refusal is not a measurement")
    payload = result["payload"]
    rows = _rows(table_path, outcome)
    population = tuple(row["event_key"] for row in rows)
    truth = {row["event_key"]: float(row[outcome]) for row in rows}
    sealed_at = max(row["published_at"] for row in rows)
    diagnostics = dict(payload["diagnostics"])

    rules = [
        f"the label of an event is its REALISED {outcome} over {horizon} from the release instant, read off the price "
        f"bars by feature_eng_m5phet.events; no label was written by hand",
        "the population is the rows the study was FITTED on, not a held-out set: a causal estimate has no held-out "
        "counterfactual, which is why causal_accuracy is refused by this package rather than reported",
        f"the treatment is {study.get('spec', {}).get('treatment_kind', 'binary')} and the effect is read for the "
        f"contrast the study declares",
    ]
    if caveat:
        rules.append(f"identification caveat carried verbatim from the study spec: {caveat}")

    declared = protocol.EvaluationProtocol(
        family="causal",
        population=population,
        # the labels come from the PRICE BARS, not from the table: the table's path identifies a design -- which
        # columns, which window -- and two studies of the same releases under different designs are two studies over
        # ONE corpus, which is exactly what the seal has to be able to say
        label_source=str(label_source or table_path),
        label_producer=(label_producer or
                        "feature_eng_m5phet.event_study_dataset, from the event rows the price bars and the economic "
                        "calendar produced; no label was written by hand"),
        label_provenance="REALISED_OUTCOME",
        annotation_rules=tuple(rules),
        ambiguity_adjudication=("none was needed: a label is a realised price path, not a judgement, and an event "
                                "whose path had a gap in it was excluded by name before the table was built"),
        split=(("analysis", population),),
        split_frozen_at=sealed_at,
        split_frozen_by=("feature_eng_m5phet.local_projections._split_by_time, by publication instant, before any "
                         "coefficient was read"),
        metrics=METRICS,
        baseline=BASELINE,
        minimum_rows=int(minimum_rows),
    )
    seal = freeze.seal_corpus(truth, protocol=declared, sealed_at=sealed_at)
    metrics = scoring.score_causal(protocol=declared, seal=seal, corpus=truth,
                                   estimand=payload["estimand"], estimate=payload["estimate"],
                                   interval=payload["interval"], assumptions=payload["assumptions"],
                                   diagnostics=diagnostics)
    built = report.build_report(protocol=declared, seal=seal, metric_sets=(metrics,), generated_at=sealed_at)
    document = json.loads(built.to_json())
    document.update({"stage": stage, "target": outcome, "horizon": horizon, "scale": scale})
    # `build_report` binds the protocol by its digest and does not copy the rules into the file. A reader holding only
    # this report would then have to take the caveat on trust, so the rules -- the caveat among them -- are written
    # out here as well. They are the same strings the digest was taken over.
    document["annotation_rules"] = list(declared.annotation_rules)
    document["identification_caveat"] = caveat
    document["causal_study"] = {
        "study_id": study.get("study_id"),
        "state_ref": study.get("state_ref"),
        "spec_sha256": study.get("spec_sha256"),
        "estimator": diagnostics.get("engine"),
        "treatment_kind": diagnostics.get("treatment_kind", "binary"),
        "uncertainty_method": diagnostics.get("uncertainty_method"),
        "conditional_effects": payload.get("conditional_effects"),
        "identification_caveat": caveat,
        "execution_authorized": False,
        "reading": ("this report says what the study estimated and under which assumptions. It is NOT comparable with "
                    "the forecast arms of the same table: those report an out-of-sample error on realised outcomes "
                    "over held-out releases, and this reports an effect per unit of treatment over the rows the study "
                    "was fitted on. The retained study carries no outcome model, so it has no prediction for a "
                    "held-out release and no error can be computed for it on those rows"),
    }
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.causal_study_report",
        description="Write the causal-family evaluation report of one study fitted by `prepare-study`, so the closure "
                    "table can carry it beside the forecast arms with its comparability stated.")
    parser.add_argument("--study", type=Path, required=True,
                        help="the JSON `prepare-study` printed for this study")
    parser.add_argument("--table", type=Path, required=True,
                        help="the CSV the study was fitted on; its event keys are the sealed population")
    parser.add_argument("--stage", required=True, help="the stage name this row occupies in the closure table")
    parser.add_argument("--outcome", required=True, help="the outcome column the study declared")
    parser.add_argument("--horizon", required=True, help="the horizon annotation, e.g. h+30min")
    parser.add_argument("--scale", required=True, help="what one unit of the reported effect means")
    parser.add_argument("--caveat", default=None, help="the identification caveat, carried verbatim into the report")
    parser.add_argument("--label-source", default=None,
                        help="where the LABELS came from -- the price bars and their digest. Defaults to the table's "
                             "own path, which identifies a design rather than a corpus")
    parser.add_argument("--label-producer", default=None)
    parser.add_argument("--minimum-rows", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        study = json.loads(args.study.read_text(encoding="utf-8"))
        document = build(study, args.table, stage=args.stage, outcome=args.outcome, horizon=args.horizon,
                         scale=args.scale, caveat=args.caveat, minimum_rows=args.minimum_rows,
                         label_source=args.label_source, label_producer=args.label_producer)
    except (OSError, ValueError) as trouble:
        print(f"REFUSED {type(trouble).__name__}: {trouble}", file=sys.stderr)
        return 2
    args.out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.stage}: causal report over {document['sealed_row_count']} sealed row(s) written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
