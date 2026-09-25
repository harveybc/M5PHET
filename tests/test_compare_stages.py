"""The comparison table across doctoral stages: what it is allowed to say, and what it must refuse to say.

These tests are written against the generator as a reader. Every fixture report below is produced by the evaluation
package's own writer (`build_report(...).to_json()`), never typed by hand, so a change to the report format breaks these
tests instead of silently leaving the table reading fields that no longer exist.

Four dishonest tables are closed here. A table that ranks two stages measured on different holdouts. A table that ranks
an MAE against an RMSE. A table that leaves a cell blank where a stage has no measurement, which reads as a zero or as
agreement. And a table whose bytes move between two runs, which cannot be diffed and therefore cannot be reviewed.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                       # the generator is run from a checkout, not from an installed wheel
    sys.path.insert(0, str(ROOT))
EVALUATION_SRC = ROOT / "evaluation" / "src"
if str(EVALUATION_SRC) not in sys.path:
    sys.path.insert(0, str(EVALUATION_SRC))

from evaluation import compare_stages                                                                    # noqa: E402
from m5phet_evaluation import (ABSTAINED, EvaluationProtocol, build_report, score_classification,        # noqa: E402
                               score_forecast, score_regimes, seal_corpus)

ROWS = ("r1", "r2", "r3", "r4", "r5", "r6")
#: the realised values, the last-value naive, and two stages whose errors are exactly 0.5 and 0.25 per row,
#: so the skills below (0.5 and 0.75) are arithmetic anybody can check without running the code
TRUTH = {"r1": 10.0, "r2": 12.0, "r3": 11.0, "r4": 13.0, "r5": 12.0, "r6": 14.0}
NAIVE = {"r1": 9.0, "r2": 11.0, "r3": 12.0, "r4": 12.0, "r5": 13.0, "r6": 13.0}
STAGE_A = {row: value + 0.5 for row, value in TRUTH.items()}
STAGE_B = {row: value - 0.25 for row, value in TRUTH.items()}

FROZEN = "2026-09-24T00:00:00Z"


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


def forecast_report(predictions, *, truth=None, protocol=None):
    """One stage's report, built through the package: protocol, seal taken first, scorer, report."""
    truth = dict(truth or TRUTH)
    protocol = protocol or forecast_protocol()
    seal = seal_corpus(truth, protocol=protocol, sealed_at=FROZEN)
    metrics = score_forecast(protocol=protocol, seal=seal, truth=truth, predictions=predictions,
                             baseline_predictions=NAIVE, baseline_name="last_value")
    return build_report(protocol=protocol, seal=seal, metric_sets=[metrics], generated_at=FROZEN)


def regimes_report():
    protocol = EvaluationProtocol(
        family="regimes", population=ROWS, label_source=None, label_producer=None,
        label_provenance="AUTHOR_WRITTEN_SMOKE",
        annotation_rules=("No row is annotated: an unsupervised assignment has no ground truth.",),
        ambiguity_adjudication="No adjudication exists, because no row carries a correct regime.",
        split={"test": ROWS}, split_frozen_at=FROZEN, split_frozen_by="Fixed with the forecast holdout.",
        metrics=("rand_index", "adjusted_rand_index"),
        baseline="none: no baseline exists for an unsupervised assignment", minimum_rows=3)
    corpus = {row: f"window-{row}" for row in ROWS}
    seal = seal_corpus(corpus, protocol=protocol, sealed_at=FROZEN)
    first = {"r1": "a", "r2": "a", "r3": "b", "r4": "b", "r5": "c", "r6": "c"}
    second = {"r1": "a", "r2": "a", "r3": "b", "r4": "c", "r5": "c", "r6": "c"}
    metrics = score_regimes(protocol=protocol, seal=seal, corpus=corpus, assignment_a=first, assignment_b=second)
    return build_report(protocol=protocol, seal=seal, metric_sets=[metrics], generated_at=FROZEN)


def classification_report():
    protocol = EvaluationProtocol(
        family="classification", population=ROWS, label_source=None, label_producer=None,
        label_provenance="AUTHOR_WRITTEN_SMOKE",
        annotation_rules=("A row is relevant when the release names the target series.",),
        ambiguity_adjudication="Decided by the author against the written rule; no second reader existed.",
        split={"test": ROWS}, split_frozen_at=FROZEN, split_frozen_by="Fixed with the forecast holdout.",
        metrics=("macro_f1", "accuracy"), baseline="majority_class", minimum_rows=3)
    truth = {"r1": "relevant", "r2": "relevant", "r3": "irrelevant",
             "r4": "irrelevant", "r5": "relevant", "r6": "irrelevant"}
    predictions = {"r1": "relevant", "r2": "irrelevant", "r3": "irrelevant",
                   "r4": "relevant", "r5": ABSTAINED, "r6": "irrelevant"}
    seal = seal_corpus(truth, protocol=protocol, sealed_at=FROZEN)
    metrics = score_classification(protocol=protocol, seal=seal, truth=truth, predictions=predictions,
                                   labels=("relevant", "irrelevant"))
    return build_report(protocol=protocol, seal=seal, metric_sets=[metrics], generated_at=FROZEN)


def write(tmp_path, name, report, **annotations) -> Path:
    path = tmp_path / f"{name}.json"
    payload = compare_stages.annotate(report, **annotations)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return path


def rows_of(table, area):
    return {row["stage"]: row for row in next(a for a in table["areas"] if a["area"] == area)["rows"]}


# --------------------------------------------------------------------------------------------------------------------
# two stages on one holdout: the only case in which a ranking means anything
# --------------------------------------------------------------------------------------------------------------------

def test_two_comparable_stages_are_ranked_and_their_skill_is_the_packages_own_number(tmp_path):
    files = [write(tmp_path, "baseline_representation", forecast_report(STAGE_A)),
             write(tmp_path, "designed_representation", forecast_report(STAGE_B))]
    table = compare_stages.compare([compare_stages.load_stage(path) for path in files])
    rows = rows_of(table, "forecast")

    assert set(rows) == {"baseline_representation", "designed_representation"}
    for row in rows.values():
        assert row["status"] == compare_stages.MEASURED
        assert row["comparability"] == compare_stages.COMPARABLE
        assert row["metric"] == "mae" and row["naive"]["name"] == "last_value"
        assert row["naive"]["same_rows_as_model"] is True

    assert rows["baseline_representation"]["model_error"] == pytest.approx(0.5)
    assert rows["designed_representation"]["model_error"] == pytest.approx(0.25)
    assert rows["baseline_representation"]["naive"]["error"] == pytest.approx(1.0)
    assert rows["baseline_representation"]["skill"] == pytest.approx(0.5)
    assert rows["designed_representation"]["skill"] == pytest.approx(0.75)
    # the package computed skill_mae itself; the generator must quote it, not recompute a second definition
    assert rows["designed_representation"]["skill_source"] == "report:metric_sets[0].values.skill_mae"

    assert rows["designed_representation"]["rank"] == 1
    assert rows["baseline_representation"]["rank"] == 2
    # a rank is a position, not a measurement, so it is not rendered on the metric scale
    line = next(l for l in compare_stages.render_markdown(table).splitlines() if l.startswith("| designed_representation"))
    assert [cell.strip() for cell in line.strip("|").split("|")][-1] == "1"


def test_the_comparability_verdict_names_the_holdout_when_the_seals_differ(tmp_path):
    moved = dict(TRUTH, r6=15.0)                     # one realised value edited: same rows, another sealed corpus
    files = [write(tmp_path, "baseline_representation", forecast_report(STAGE_A)),
             write(tmp_path, "designed_representation", forecast_report(STAGE_B, truth=moved))]
    rows = rows_of(compare_stages.compare([compare_stages.load_stage(p) for p in files]), "forecast")

    verdict = rows["designed_representation"]["comparability"]
    assert verdict.startswith(f"{compare_stages.NOT_COMPARABLE}: holdout differs")
    assert "corpus_seal" in verdict
    assert rows["designed_representation"]["rank"] is None       # an unranked stage, never a rank against other rows
    assert rows["baseline_representation"]["comparability"] == compare_stages.COMPARABLE


def test_the_comparability_verdict_names_the_metric_when_the_stages_report_different_ones(tmp_path):
    import dataclasses
    protocol = forecast_protocol()
    seal = seal_corpus(TRUTH, protocol=protocol, sealed_at=FROZEN)
    measured = score_forecast(protocol=protocol, seal=seal, truth=TRUTH, predictions=STAGE_B,
                              baseline_predictions=NAIVE, baseline_name="last_value")
    # a stage that reported RMSE only: the same holdout, a different error scale, so the two cannot be ranked together
    rmse_only = dataclasses.replace(
        measured,
        values={"rmse": measured.values["rmse"], "skill_rmse": measured.values["skill_rmse"]},
        baseline={"name": "last_value", "rmse": measured.baseline["rmse"], "rows": 6, "same_rows_as_model": True})
    files = [write(tmp_path, "baseline_representation", forecast_report(STAGE_A)),
             write(tmp_path, "searched_representation",
                   build_report(protocol=protocol, seal=seal, metric_sets=[rmse_only], generated_at=FROZEN))]
    rows = rows_of(compare_stages.compare([compare_stages.load_stage(p) for p in files]), "forecast")

    verdict = rows["searched_representation"]["comparability"]
    assert verdict.startswith(f"{compare_stages.NOT_COMPARABLE}: metric differs")
    assert "mae" in verdict and "rmse" in verdict
    assert rows["searched_representation"]["metric"] == "rmse"
    assert rows["searched_representation"]["rank"] is None


# --------------------------------------------------------------------------------------------------------------------
# what the table must refuse to present as a measurement
# --------------------------------------------------------------------------------------------------------------------

def test_a_refused_area_is_no_new_measurement_and_carries_the_packages_own_reason(tmp_path):
    files = [write(tmp_path, "baseline_representation", forecast_report(STAGE_A)),
             write(tmp_path, "baseline_representation_regimes", regimes_report(), stage="baseline_representation")]
    table = compare_stages.compare([compare_stages.load_stage(p) for p in files])
    row = rows_of(table, "regimes")["baseline_representation"]

    assert row["status"] == compare_stages.NO_NEW_MEASUREMENT
    assert row["refusal"]["metric"] == "regime_accuracy"
    assert "stability, never correctness" in row["refusal"]["reason"]
    assert row["model_error"] is None and row["skill"] is None
    # never a blank: the rendered cells say the words, because an empty cell reads as a zero or as agreement
    markdown = compare_stages.render_markdown(table)
    regimes_block = markdown.split("## Area: regimes")[1]
    assert compare_stages.NO_NEW_MEASUREMENT in regimes_block
    assert "| |" not in markdown and "|  |" not in markdown


def test_a_stage_that_did_not_measure_an_area_is_no_new_measurement_rather_than_absent(tmp_path):
    files = [write(tmp_path, "baseline_representation", forecast_report(STAGE_A)),
             write(tmp_path, "designed_representation", forecast_report(STAGE_B)),
             write(tmp_path, "designed_regimes", regimes_report(), stage="designed_representation")]
    rows = rows_of(compare_stages.compare([compare_stages.load_stage(p) for p in files]), "regimes")

    assert set(rows) == {"baseline_representation", "designed_representation"}
    absent = rows["baseline_representation"]
    assert absent["status"] == compare_stages.NO_NEW_MEASUREMENT
    assert "no report" in absent["reason"]


def test_classification_reports_a_score_and_refuses_to_call_a_ratio_of_scores_a_skill(tmp_path):
    files = [write(tmp_path, "baseline_representation", classification_report())]
    row = rows_of(compare_stages.compare([compare_stages.load_stage(p) for p in files]), "classification")["baseline_representation"]

    assert row["status"] == compare_stages.MEASURED and row["metric"] == "macro_f1"
    assert row["naive"]["name"] == "majority_class" and row["naive"]["same_rows_as_model"] is True
    assert row["skill"] is None
    assert row["skill_source"].startswith(compare_stages.NOT_DEFINED)
    assert "is a score, not an error" in row["skill_source"]


# --------------------------------------------------------------------------------------------------------------------
# the literature column, and the bytes
# --------------------------------------------------------------------------------------------------------------------

def test_a_literature_value_is_shown_with_its_source_and_its_absence_is_named(tmp_path):
    files = [write(tmp_path, "baseline_representation", forecast_report(STAGE_A)),
             write(tmp_path, "designed_representation", forecast_report(STAGE_B),
                   literature={"value": 0.31, "metric": "mae", "scale": "kWh",
                               "source": "Hebrail & Berard, UCI household power, persistence baseline table 3"})]
    rows = rows_of(compare_stages.compare([compare_stages.load_stage(p) for p in files]), "forecast")

    assert rows["baseline_representation"]["literature"]["value"] == compare_stages.NOT_CARRIED
    assert rows["baseline_representation"]["literature"]["source"] == compare_stages.NOT_CARRIED
    assert rows["designed_representation"]["literature"]["value"] == pytest.approx(0.31)
    assert "Hebrail" in rows["designed_representation"]["literature"]["source"]

    markdown = compare_stages.render_markdown(compare_stages.compare([compare_stages.load_stage(p) for p in files]))
    assert compare_stages.NOT_CARRIED in markdown


def test_the_markdown_is_byte_stable_across_two_runs(tmp_path):
    files = [write(tmp_path, "baseline_representation", forecast_report(STAGE_A), target="global_active_power",
                   horizon="h+1", scale="kWh"),
             write(tmp_path, "designed_representation", forecast_report(STAGE_B), target="global_active_power",
                   horizon="h+1", scale="kWh"),
             write(tmp_path, "calendar_augmented_regimes", regimes_report(), stage="calendar_augmented")]
    argv = ["--report", f"baseline_representation={files[0]}", "--report", str(files[1]), "--report", str(files[2])]

    first, second = tmp_path / "one.md", tmp_path / "two.md"
    assert compare_stages.main(argv + ["--markdown", str(first), "--out", str(tmp_path / "one.json")]) == 0
    assert compare_stages.main(argv + ["--markdown", str(second), "--out", str(tmp_path / "two.json")]) == 0
    assert first.read_bytes() == second.read_bytes()
    assert (tmp_path / "one.json").read_bytes() == (tmp_path / "two.json").read_bytes()

    text = first.read_text()
    assert "6 decimals" in text and "0.500000" in text and "0.750000" in text
    assert "global_active_power" in text and "h+1" in text


def test_the_stage_name_comes_from_the_argument_then_the_annotation_then_the_file_name(tmp_path):
    plain = write(tmp_path, "searched_representation", forecast_report(STAGE_B))
    annotated = write(tmp_path, "run_17", forecast_report(STAGE_A), stage="baseline_representation")
    named = compare_stages.load_stage(annotated, stage="calendar_augmented")

    assert compare_stages.load_stage(plain).stage == "searched_representation"
    assert compare_stages.load_stage(annotated).stage == "baseline_representation"
    assert named.stage == "calendar_augmented"


def test_a_file_that_is_not_a_report_of_this_package_is_refused_by_name(tmp_path):
    foreign = tmp_path / "foreign.json"
    foreign.write_text(json.dumps({"version": "somebody-elses-report/9", "family": "forecast"}))
    with pytest.raises(compare_stages.StageComparisonError, match="somebody-elses-report/9"):
        compare_stages.load_stage(foreign)

    unknown = tmp_path / "unknown_family.json"
    payload = compare_stages.annotate(forecast_report(STAGE_A))
    payload["family"] = "astrology"
    unknown.write_text(json.dumps(payload))
    with pytest.raises(compare_stages.StageComparisonError, match="astrology"):
        compare_stages.load_stage(unknown)


def test_two_reports_for_one_stage_and_one_area_are_refused_rather_than_silently_dropped(tmp_path):
    files = [write(tmp_path, "baseline_representation", forecast_report(STAGE_A)),
             write(tmp_path, "baseline_again", forecast_report(STAGE_B), stage="baseline_representation")]
    with pytest.raises(compare_stages.StageComparisonError, match="twice"):
        compare_stages.compare([compare_stages.load_stage(p) for p in files])


# --------------------------------------------------------------------------------------------------------------------
# a stage that was attempted and refused: in the table, with its reason, never with a number (WP06 stages 3-4)
# --------------------------------------------------------------------------------------------------------------------

REFUSAL = ("the stage's window 2892 exceeds the sealing window 197; the sealed population would not hold its history "
           "(7129 origins instead of 9824, seal 97155a063e25 instead of 33820b552ddf)")


def test_an_attempted_stage_with_no_report_is_in_the_table_with_its_refusal(tmp_path):
    table = compare_stages.compare(
        [compare_stages.load_stage(write(tmp_path, "designed", forecast_report(STAGE_B))),
         compare_stages.load_stage(write(tmp_path, "baseline", forecast_report(STAGE_A)))],
        not_measured=[{"stage": "candidate_seasonal_lag_2892", "area": "forecast", "reason": REFUSAL}])

    rows = {row["stage"]: row for row in table["areas"][0]["rows"]}
    assert sorted(rows) == ["baseline", "candidate_seasonal_lag_2892", "designed"]
    refused = rows["candidate_seasonal_lag_2892"]
    assert refused["status"] == compare_stages.NO_NEW_MEASUREMENT
    assert refused["reason"] == REFUSAL
    assert refused["model_error"] is None and refused["rank"] is None
    assert refused["comparability"] == compare_stages.NO_NEW_MEASUREMENT
    # the stages that WERE measured are ranked as before: an unmeasured row takes no position
    assert rows["designed"]["rank"] == 1 and rows["baseline"]["rank"] == 2
    assert table["not_measured"] == [{"stage": "candidate_seasonal_lag_2892", "area": "forecast", "reason": REFUSAL}]
    # and the markdown says so instead of leaving a cell that reads like a zero
    markdown = compare_stages.render_markdown(table)
    assert "NO_NEW_MEASUREMENT" in markdown and "2892" in markdown


def test_a_stage_cannot_both_carry_a_report_and_be_declared_unmeasured(tmp_path):
    stages = [compare_stages.load_stage(write(tmp_path, "designed", forecast_report(STAGE_B)))]
    with pytest.raises(compare_stages.StageComparisonError, match="cannot both have a measurement"):
        compare_stages.compare(stages, not_measured=[{"stage": "designed", "area": "forecast", "reason": REFUSAL}])


def test_an_unmeasured_stage_of_an_unknown_area_is_refused(tmp_path):
    stages = [compare_stages.load_stage(write(tmp_path, "designed", forecast_report(STAGE_B)))]
    with pytest.raises(compare_stages.StageComparisonError, match="is not one of"):
        compare_stages.compare(stages, not_measured=[{"stage": "x", "area": "weather", "reason": REFUSAL}])


def test_the_cli_takes_a_not_measured_stage(tmp_path):
    out = tmp_path / "table.json"
    code = compare_stages.main(["--report", f"designed={write(tmp_path, 'designed', forecast_report(STAGE_B))}",
                                "--not-measured", f"candidate_seasonal_lag_2892=forecast:{REFUSAL}",
                                "--out", str(out), "--markdown", str(tmp_path / "table.md")])
    assert code == 0
    table = json.loads(out.read_text())
    assert [entry["stage"] for entry in table["not_measured"]] == ["candidate_seasonal_lag_2892"]


# --------------------------------------------------------------------------------------------------------------------
# WP19/WP29: the representation area carries a declared internal index, and it is ranked WITHOUT becoming a quality claim
# --------------------------------------------------------------------------------------------------------------------

def regimes_report_with_silhouette(silhouette):
    """A regimes report that also carries the internal index `evaluate_regimes` computes on the holdout.

    The index is put into the metric set the package itself built, exactly as `feature_eng_m5phet.evaluate_regimes`
    writes it, so this test reads the field the real producer writes and not one invented here.
    """
    payload = json.loads(regimes_report().to_json())
    payload["metric_sets"][0]["values"]["silhouette"] = silhouette
    return payload


def test_a_regimes_stage_carrying_a_silhouette_is_ranked_on_it_and_still_refuses_regime_accuracy(tmp_path):
    files = [write(tmp_path, "hand_a", regimes_report_with_silhouette(0.11), stage="hand_baseline"),
             write(tmp_path, "laya_b", regimes_report_with_silhouette(0.42), stage="laya_chosen")]
    table = compare_stages.compare([compare_stages.load_stage(path) for path in files])
    area = next(a for a in table["areas"] if a["area"] == "regimes")
    rows = rows_of(table, "regimes")

    # ranked, higher silhouette first, because the area's orientation is a score
    assert rows["laya_chosen"]["rank"] == 1 and rows["hand_baseline"]["rank"] == 2
    assert rows["laya_chosen"]["comparability"] == compare_stages.COMPARABLE
    assert rows["laya_chosen"]["metric"] == "silhouette"
    assert rows["laya_chosen"]["model_error"] == 0.42

    # and the refusal is exactly where it was: in the area, and in every row
    assert area["refused_metric"] == "regime_accuracy"
    assert rows["laya_chosen"]["refusal"]["metric"] == "regime_accuracy"
    # skill is still not defined: silhouette is a score, and this package will not divide two scores into one
    assert rows["laya_chosen"]["skill"] is None
    assert compare_stages.NOT_DEFINED in rows["laya_chosen"]["skill_source"]
    markdown = compare_stages.render_markdown(table).split("## Area: regimes")[1]
    assert "The rank below is not a quality claim." in markdown


def test_a_regimes_stage_with_no_internal_index_is_still_no_new_measurement(tmp_path):
    files = [write(tmp_path, "plain", regimes_report(), stage="hand_baseline")]
    row = rows_of(compare_stages.compare([compare_stages.load_stage(path) for path in files]), "regimes")
    assert row["hand_baseline"]["status"] == compare_stages.NO_NEW_MEASUREMENT
    assert row["hand_baseline"]["rank"] is None


def test_an_area_whose_metric_is_refused_and_declares_no_index_is_never_ranked():
    for family in ("causal", "policy"):
        assert compare_stages.READINGS[family].metric_keys == ()
        assert compare_stages.READINGS[family].refusal is not None
