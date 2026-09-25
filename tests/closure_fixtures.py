"""Real closure-table rows for the WP23 tests, built by the evaluation package instead of typed by hand.

A decision outcome links a decision record to *a row of the closure table*, so a test that types a row by hand tests
nothing: it would keep passing after `evaluation/compare_stages.py` changed the shape or the meaning of the field it
reads. Every row below therefore comes out of `compare_stages.compare()` over reports written by
`m5phet_evaluation.build_report`, exactly as a real table's rows do — ranks included, since the rank is the label WP23
calibrates against.

The forecast corpus is the one `tests/test_compare_stages.py` uses, six rows whose naive (last-value) MAE is exactly
1.0, so a stage whose predictions are the realised values plus a constant `offset` has model error `offset` and skill
`1 - offset`. Ranks are then the offsets in increasing order, which anybody can check without running the code.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                       # the generator is run from a checkout, not from an installed wheel
    sys.path.insert(0, str(ROOT))
EVALUATION_SRC = ROOT / "evaluation" / "src"
if str(EVALUATION_SRC) not in sys.path:
    sys.path.insert(0, str(EVALUATION_SRC))

from evaluation import compare_stages                                                                    # noqa: E402
from m5phet_evaluation import EvaluationProtocol, build_report, score_forecast, seal_corpus               # noqa: E402

ROWS = ("r1", "r2", "r3", "r4", "r5", "r6")
TRUTH = {"r1": 10.0, "r2": 12.0, "r3": 11.0, "r4": 13.0, "r5": 12.0, "r6": 14.0}
#: the last-value naive is wrong by exactly 1.0 on every row, so the naive MAE is 1.0 and skill reads off the offset
NAIVE = {"r1": 9.0, "r2": 11.0, "r3": 12.0, "r4": 12.0, "r5": 13.0, "r6": 13.0}
FROZEN = "2026-09-24T00:00:00Z"

#: a second realised series, and therefore a second seal: two stages measured on it are NOT_COMPARABLE with the first
OTHER_TRUTH = {row: value + 100.0 for row, value in TRUTH.items()}


def forecast_protocol(**overrides) -> EvaluationProtocol:
    declared = {
        "family": "forecast",
        "population": ROWS,
        "label_source": "household-power-holdout-2026-09",
        "label_producer": "the meter record, read after the fact",
        "label_provenance": "REALISED_OUTCOME",
        "annotation_rules": ("No row is annotated: the label is the realised value of the target at the horizon.",),
        "ambiguity_adjudication": "No row is ambiguous: a realised value is either recorded or the row is absent.",
        "split": {"test": ROWS},
        "split_frozen_at": FROZEN,
        "split_frozen_by": "Row order fixed by the corpus file digest before any representation was fitted.",
        "metrics": ("mae", "rmse", "skill_mae"),
        "baseline": "last_value",
        "minimum_rows": 3,
    }
    declared.update(overrides)
    return EvaluationProtocol(**declared)


def forecast_report(offset, *, truth=None):
    """One stage's report: predictions are the realised values plus `offset`, so the model MAE is `offset`."""
    truth = dict(truth or TRUTH)
    protocol = forecast_protocol()
    seal = seal_corpus(truth, protocol=protocol, sealed_at=FROZEN)
    predictions = {row: value + offset for row, value in truth.items()}
    baseline = NAIVE if truth == TRUTH else {row: value - 1.0 for row, value in truth.items()}
    metrics = score_forecast(protocol=protocol, seal=seal, truth=truth, predictions=predictions,
                             baseline_predictions=baseline, baseline_name="last_value")
    return build_report(protocol=protocol, seal=seal, metric_sets=[metrics], generated_at=FROZEN)


def table(tmp_path, stages):
    """`stages` maps a stage name to an offset, or to `(offset, truth)` for a stage on another holdout."""
    paths = []
    for name in sorted(stages):
        declared = stages[name]
        offset, truth = declared if isinstance(declared, tuple) else (declared, None)
        path = Path(tmp_path) / f"{name}.json"
        path.write_text(json.dumps(compare_stages.annotate(forecast_report(offset, truth=truth)),
                                   sort_keys=True, indent=2) + "\n")
        paths.append(path)
    built = compare_stages.compare([compare_stages.load_stage(path) for path in paths])
    area = next(entry for entry in built["areas"] if entry["area"] == "forecast")
    return {row["stage"]: row for row in area["rows"]}


def ranked_rows(tmp_path, count, *, prefix="stage"):
    """`count` comparable stages of one holdout, returned in rank order (rank 1 first). Every rank is distinct."""
    stages = {f"{prefix}_{index:03d}": round(0.05 * (index + 1), 4) for index in range(count)}
    rows = table(tmp_path, stages)
    return sorted(rows.values(), key=lambda row: row["rank"])
