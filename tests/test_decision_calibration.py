"""WP23 step 2: what the calibration report may say about Laya's choices, and what it must refuse to say.

Every number below is worked out by hand in the test, on a fixture of 40 decision records linked to 40 real closure
table rows across two decision kinds. The point of the fixture is not that the numbers are hard — they are not — but
that the report is checked against arithmetic a reviewer can do on paper, instead of against whatever the code
happened to produce.

Four dishonest reports are closed here. A report that quotes an agreement rate over a handful of links and lets it
read like a measurement. A report that calls Laya's *own* choice its label, instead of the rank of the pipeline the
choice led to. A report whose bytes move between two runs, which cannot be diffed and therefore cannot be reviewed.
And an inventory that shows the records that exist without saying that none of them is linked to a measured row.
"""

import json
import sys
from pathlib import Path

import pytest

import closure_fixtures
from m5phet import decide

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import decision_calibration                                                              # noqa: E402

PREPROCESSING_OPTIONS = [["a", "keep the level"], ["b", "normalise per feature"], ["c", "log return"]]
METHOD_OPTIONS = [["agglomerative", "hierarchical merging"], ["kmeans", "centroid clustering"],
                  ["dbscan", "density clustering"]]

#: the probability the argmax option carries, and therefore the reliability bin the outcome falls in
LOW, HIGH = 0.35, 0.85


def probabilities(options, chosen, argmax_probability):
    """The argmax carries `argmax_probability`; the rest share the remainder in the declared order, never tying it."""
    keys = [key for key, _label in options]
    others = [key for key in keys if key != chosen]
    remainder = round(1.0 - argmax_probability, 6)
    first = round(remainder * 0.52, 6)
    assert first < argmax_probability                    # the chosen option must really be the argmax
    return {chosen: argmax_probability, others[0]: first, others[1]: round(remainder - first, 6)}


def decision(kind, question, options, chosen, argmax_probability, salt):
    return {"schema": decide.DECISION_SCHEMA, "kind": kind, "question": question,
            "state_sha256": decide.state_sha256(f"state {salt}"),
            "options": [list(pair) for pair in options], "chosen": chosen,
            "probabilities": probabilities(options, chosen, argmax_probability),
            "probability_decimals": 4, "checkpoint": "laya-checkpoint:bd12df887789", "backend": "laya",
            "as_of": "2026-09-25T00:00:00+00:00", "execution_authorized": False}


def link(tmp_path, plan, *, kind, question, options, prefix):
    """`plan` is one `(chosen, argmax_probability)` per stage, in rank order: the first entry is the ranked-first stage.

    Returns the outcome entries. Every row is a real `compare_stages` row and every rank is the generator's own.
    """
    rows = closure_fixtures.ranked_rows(tmp_path, len(plan), prefix=prefix)
    entries = []
    for index, ((chosen, argmax_probability), row) in enumerate(zip(plan, rows)):
        path = decide.record(decision(kind, question, options, chosen, argmax_probability, f"{prefix}-{index}"),
                             tmp_path / "decisions")
        entry = decide.outcome(path, row, out_dir=tmp_path / "outcomes")
        assert entry["status"] == "OK", entry
        entries.append(entry)
    return entries


def preprocessing_plan():
    """32 stages. 20 at p=0.35 of which 8 chose `b`; 12 at p=0.85 of which 9 chose `b`. `b` is ranked first."""
    plan = [("b", LOW)] * 8 + [("a", LOW)] * 12 + [("b", HIGH)] * 9 + [("c", HIGH)] * 3
    assert len(plan) == 32 and plan[0] == ("b", LOW)          # the ranked-first stage is a `b`, so `b` is the label
    return plan


def method_plan():
    """8 stages only: under the declared minimum, so this kind is NO_NEW_MEASUREMENT whatever its agreement is."""
    return [("agglomerative", HIGH)] * 5 + [("kmeans", LOW)] * 3


def fixture_report(tmp_path):
    link(tmp_path, preprocessing_plan(), kind="feature_preprocessing", question="preprocessing",
         options=PREPROCESSING_OPTIONS, prefix="fp")
    link(tmp_path, method_plan(), kind="regime_method", question="regime_method",
         options=METHOD_OPTIONS, prefix="rm")
    outcomes, unreadable = decision_calibration.load_outcomes([tmp_path / "outcomes"])
    assert unreadable == [] and len(outcomes) == 40
    return decision_calibration.calibrate(outcomes)


def group_of(report, kind):
    return next(group for group in report["groups"] if group["kind"] == kind)


# --------------------------------------------------------------------------------------------------------------------
# 40 links across two kinds: the agreement rate, the reliability bins and the calibration error, by hand
# --------------------------------------------------------------------------------------------------------------------

def test_agreement_and_calibration_error_over_forty_linked_outcomes(tmp_path):
    report = fixture_report(tmp_path)
    assert [group["kind"] for group in report["groups"]] == ["feature_preprocessing", "regime_method"]
    assert report["linked_outcomes"] == 40

    group = group_of(report, "feature_preprocessing")
    assert group["question"] == "preprocessing"
    assert group["status"] == decision_calibration.MEASURED
    assert group["n_linked"] == 32 and group["n_scored"] == 32
    # the label is the rank, never the chooser's own preference: `b` is ranked first, so `b` is what agreement means
    assert group["best_ranked_option"] == "b"
    assert group["agreements"] == 8 + 9 == 17
    assert group["agreement_rate"] == pytest.approx(17 / 32)          # 0.53125

    bins = {entry["bin"]: entry for entry in group["bins"]}
    assert set(bins) == {"[0.30, 0.40)", "[0.80, 0.90)"}              # only non-empty bins are shown
    assert bins["[0.30, 0.40)"]["n"] == 20
    assert bins["[0.30, 0.40)"]["agreement"] == pytest.approx(8 / 20)          # 0.40 observed at 0.35 claimed
    assert bins["[0.30, 0.40)"]["mean_argmax_probability"] == pytest.approx(LOW)
    assert bins["[0.80, 0.90)"]["n"] == 12
    assert bins["[0.80, 0.90)"]["agreement"] == pytest.approx(9 / 12)          # 0.75 observed at 0.85 claimed
    assert bins["[0.80, 0.90)"]["mean_argmax_probability"] == pytest.approx(HIGH)

    # ECE = (20/32)*|0.40 - 0.35| + (12/32)*|0.75 - 0.85| = 0.03125 + 0.0375
    assert group["expected_calibration_error"] == pytest.approx(0.06875)


def test_a_kind_under_the_minimum_is_NO_NEW_MEASUREMENT_and_names_how_many_are_missing(tmp_path):
    group = group_of(fixture_report(tmp_path), "regime_method")

    assert group["status"] == decision_calibration.NO_NEW_MEASUREMENT
    assert group["n_linked"] == 8 and group["missing"] == 22
    assert "22 missing" in group["reason"]
    # and not one number is reported for it
    assert group["agreement_rate"] is None
    assert group["expected_calibration_error"] is None
    assert group["bins"] == []


def test_twenty_nine_linked_outcomes_are_one_short(tmp_path):
    link(tmp_path, [("b", HIGH)] * 29, kind="feature_preprocessing", question="preprocessing",
         options=PREPROCESSING_OPTIONS, prefix="fp")
    outcomes, _ = decision_calibration.load_outcomes([tmp_path / "outcomes"])
    group = group_of(decision_calibration.calibrate(outcomes), "feature_preprocessing")

    assert group["n_linked"] == 29
    assert group["status"] == decision_calibration.NO_NEW_MEASUREMENT
    assert group["missing"] == 1
    assert "1 missing" in group["reason"]
    assert "1 missing" in decision_calibration.render_markdown(
        decision_calibration.calibrate(outcomes))


def test_thirty_is_enough_and_twenty_nine_is_not(tmp_path):
    assert decision_calibration.MINIMUM_LINKED == 30
    link(tmp_path, [("b", HIGH)] * 30, kind="feature_preprocessing", question="preprocessing",
         options=PREPROCESSING_OPTIONS, prefix="fp")
    outcomes, _ = decision_calibration.load_outcomes([tmp_path / "outcomes"])
    group = group_of(decision_calibration.calibrate(outcomes), "feature_preprocessing")

    assert group["status"] == decision_calibration.MEASURED
    assert group["n_linked"] == 30 and group["missing"] == 0
    assert group["agreement_rate"] == pytest.approx(1.0)
    # every outcome claimed 0.85 and every one of them agreed: the gap is 0.15, and so is the calibration error
    assert group["expected_calibration_error"] == pytest.approx(0.15)


# --------------------------------------------------------------------------------------------------------------------
# what the report refuses to reduce to a number
# --------------------------------------------------------------------------------------------------------------------

def test_a_group_whose_outcomes_declared_different_option_sets_is_not_one_question(tmp_path):
    link(tmp_path, [("b", HIGH)] * 30, kind="feature_preprocessing", question="preprocessing",
         options=PREPROCESSING_OPTIONS, prefix="fp")
    link(tmp_path, [("agglomerative", HIGH)] * 2, kind="feature_preprocessing", question="preprocessing",
         options=METHOD_OPTIONS, prefix="other")
    outcomes, _ = decision_calibration.load_outcomes([tmp_path / "outcomes"])
    group = group_of(decision_calibration.calibrate(outcomes), "feature_preprocessing")

    assert group["status"] == decision_calibration.NO_NEW_MEASUREMENT
    assert decision_calibration.OPTION_SETS_DIFFER in group["reason"]
    assert group["agreement_rate"] is None


def test_a_group_with_no_first_ranked_stage_has_no_label_to_agree_with(tmp_path):
    link(tmp_path, [("b", HIGH)] * 31, kind="feature_preprocessing", question="preprocessing",
         options=PREPROCESSING_OPTIONS, prefix="fp")
    # drop the outcome of the ranked-first stage: nothing then says which option was ranked first
    for path in sorted((tmp_path / "outcomes").glob("*.json")):
        if json.loads(path.read_text()).get("rank") == 1:
            path.unlink()
    outcomes, _ = decision_calibration.load_outcomes([tmp_path / "outcomes"])
    group = group_of(decision_calibration.calibrate(outcomes), "feature_preprocessing")

    assert group["status"] == decision_calibration.NO_NEW_MEASUREMENT
    assert decision_calibration.NO_BEST_RANKED_OPTION in group["reason"]
    assert group["best_ranked_option"] is None and group["agreement_rate"] is None


# --------------------------------------------------------------------------------------------------------------------
# a report that can be diffed
# --------------------------------------------------------------------------------------------------------------------

def test_the_markdown_and_the_json_are_byte_stable_across_runs(tmp_path):
    report = fixture_report(tmp_path)
    again = decision_calibration.calibrate(decision_calibration.load_outcomes([tmp_path / "outcomes"])[0])

    assert json.dumps(report, sort_keys=True) == json.dumps(again, sort_keys=True)
    first = decision_calibration.render_markdown(report).encode("utf-8")
    second = decision_calibration.render_markdown(again).encode("utf-8")
    assert first == second
    assert b"2026" not in first.split(b"\n")[0]               # no clock is read, so no date can drift into the header
    assert "0.068750" in first.decode("utf-8")                # rendered at the declared decimals, not rounded to two


def test_the_cli_writes_both_files_and_returns_zero(tmp_path):
    fixture_report(tmp_path)
    code = decision_calibration.main(["--outcomes", str(tmp_path / "outcomes"),
                                      "--out", str(tmp_path / "report.json"),
                                      "--markdown", str(tmp_path / "report.md")])
    assert code == 0
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["linked_outcomes"] == 40
    assert (tmp_path / "report.md").read_text() == decision_calibration.render_markdown(payload)


# --------------------------------------------------------------------------------------------------------------------
# the inventory: what exists today, and the sentence that none of it is linked
# --------------------------------------------------------------------------------------------------------------------

def test_the_inventory_counts_the_records_that_exist_and_how_many_have_no_outcome(tmp_path):
    folder = tmp_path / "records"
    for index in range(7):
        decide.record(decision("feature_preprocessing", "preprocessing", PREPROCESSING_OPTIONS, "a", LOW,
                               f"p{index}"), folder)
    for index in range(2):
        decide.record(decision("regime_parameters", f"regime_parameters_{index}", METHOD_OPTIONS, "kmeans", HIGH,
                               f"r{index}"), folder)
    (folder / "not_a_record.json").write_text("{}")

    inventory = decision_calibration.inventory([folder], outcomes=[])
    assert inventory["records"] == 9
    assert inventory["linked"] == 0 and inventory["unlinked"] == 9
    assert [(group["kind"], group["question"], group["records"], group["unlinked"])
            for group in inventory["groups"]] == [
        ("feature_preprocessing", "preprocessing", 7, 7),
        ("regime_parameters", "regime_parameters_0", 1, 1),
        ("regime_parameters", "regime_parameters_1", 1, 1)]
    assert [entry["file"] for entry in inventory["unreadable"]] == ["not_a_record.json"]

    markdown = decision_calibration.render_markdown(decision_calibration.calibrate([], inventory=inventory))
    assert "no decision record has an outcome yet" in markdown


def test_the_inventory_marks_a_record_linked_once_an_outcome_exists(tmp_path):
    entries = link(tmp_path, [("b", HIGH)] * 3, kind="feature_preprocessing", question="preprocessing",
                   options=PREPROCESSING_OPTIONS, prefix="fp")
    assert len(entries) == 3
    outcomes, _ = decision_calibration.load_outcomes([tmp_path / "outcomes"])
    inventory = decision_calibration.inventory([tmp_path / "decisions"], outcomes=outcomes)

    assert inventory["records"] == 3 and inventory["linked"] == 3 and inventory["unlinked"] == 0


def test_the_cli_reports_the_inventory_beside_an_empty_calibration(tmp_path):
    folder = tmp_path / "records"
    decide.record(decision("feature_grouping", "grouping_cut", PREPROCESSING_OPTIONS, "a", LOW, "g0"), folder)
    code = decision_calibration.main(["--inventory", str(folder), "--out", str(tmp_path / "report.json"),
                                      "--markdown", str(tmp_path / "report.md")])
    assert code == 0
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["groups"] == [] and payload["linked_outcomes"] == 0
    assert payload["inventory"]["records"] == 1 and payload["inventory"]["linked"] == 0


def test_the_outcome_schema_the_report_reads_is_the_one_decide_writes():
    assert decision_calibration.OUTCOME_SCHEMA == decide.OUTCOME_SCHEMA


# --------------------------------------------------------------------------------------------------------------------
# WP29: thirty outcomes over thirty corpora are thirty contests, not one
# --------------------------------------------------------------------------------------------------------------------

def contest(tmp_path, name, plan, shift, *, kind="regime_method", question="regime_method", options=None):
    """One corpus: its OWN holdout (the realised series shifted by `shift`, so its own seal), its own table, its own
    ranking. `plan` is one `(chosen, argmax_probability)` per stage in rank order, best first."""
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    options = options or METHOD_OPTIONS
    truth = {row: value + shift for row, value in closure_fixtures.TRUTH.items()}
    stages = {f"{name}_{index:03d}": (round(0.05 * (index + 1), 4), truth) for index in range(len(plan))}
    rows = sorted(closure_fixtures.table(folder, stages).values(), key=lambda row: row["rank"])
    entries = []
    for index, ((chosen, argmax_probability), row) in enumerate(zip(plan, rows)):
        path = decide.record(decision(kind, question, options, chosen, argmax_probability, f"{name}-{index}"),
                             folder / "decisions")
        entry = decide.outcome(path, row, out_dir=folder / "outcomes")
        assert entry["status"] == "OK", entry
        entries.append(entry)
    return entries


def test_each_outcome_is_scored_against_its_own_contests_winner(tmp_path):
    # two corpora that ranked DIFFERENT options first. In each one Laya's argmax is the winner, so the honest
    # agreement is 2/2 -- while scoring both against a single "best option" would score one of them against the
    # option the other corpus happened to prefer, and report 1/2.
    entries = (contest(tmp_path, "corpus_one", [("kmeans", HIGH), ("dbscan", LOW)], 0.0)
               + contest(tmp_path, "corpus_two", [("dbscan", HIGH), ("kmeans", LOW)], 100.0))
    outcomes = [entry["outcome"] for entry in entries]
    group = decision_calibration.calibrate(outcomes)["groups"][0]

    assert group["n_contests"] == 2
    assert group["best_ranked_option"] == decision_calibration.PER_CONTEST
    # each contest settled its own winner, and they are not the same option. Under one shared best-ranked option the
    # group would have been refused AMBIGUOUS_BEST_RANKED_OPTION and no outcome of either corpus would ever be scored
    assert sorted(group["best_ranked_option_by_contest"].values()) == ["dbscan", "kmeans"]
    assert decision_calibration.AMBIGUOUS_BEST_RANKED_OPTION not in (group["reason"] or "")
    # and four outcomes are still four outcomes: the minimum is what stops a rate here, nothing else
    assert group["n_scorable_linked"] == 4 and group["missing"] == 26
    assert group["excluded"][decision_calibration.CONTEST_NOT_SETTLED] == 0


def test_a_group_reaches_a_rate_only_over_thirty_outcomes_and_says_over_how_many_contests(tmp_path):
    entries = []
    for index in range(15):                                # 15 corpora x 2 stages = 30 scorable outcomes
        chosen = "kmeans" if index % 2 else "dbscan"
        other = "dbscan" if index % 2 else "kmeans"
        entries += contest(tmp_path, f"corpus_{index:02d}", [(chosen, HIGH), (other, LOW)], 10.0 * index)
    report = decision_calibration.calibrate([entry["outcome"] for entry in entries])
    group = report["groups"][0]

    assert group["n_linked"] == 30 and group["n_contests"] == 15 and group["missing"] == 0
    assert group["status"] == decision_calibration.MEASURED
    # in every corpus Laya's rank-1 argmax is that corpus's winner and its rank-2 argmax is not, so 15 of 30 agree.
    # The winners are not one option: eight corpora ranked one first and seven the other.
    assert sorted(set(group["best_ranked_option_by_contest"].values())) == ["dbscan", "kmeans"]
    assert len(group["best_ranked_option_by_contest"]) == 15
    assert group["n_scored"] == 30 and group["agreements"] == 15 and group["agreement_rate"] == 0.5
    assert group["expected_calibration_error"] is not None
    markdown = decision_calibration.render_markdown(report)
    assert "contests (distinct holdouts these ranks were taken on): **15**" in markdown


def test_an_outcome_whose_contest_ranked_no_recorded_option_first_is_excluded_and_counted(tmp_path):
    # a contest in which only the SECOND-ranked stage carries a decision record: nothing there is the label
    settled = contest(tmp_path, "settled", [("kmeans", HIGH), ("dbscan", LOW)], 0.0)
    folder = tmp_path / "unsettled"
    folder.mkdir()
    other_truth = {row: value + 500.0 for row, value in closure_fixtures.TRUTH.items()}
    rows = sorted(closure_fixtures.table(folder, {"u_a": (0.25, other_truth),
                                                  "u_b": (0.5, other_truth)}).values(),
                  key=lambda row: row["rank"])
    path = decide.record(decision("regime_method", "regime_method", METHOD_OPTIONS, "kmeans", HIGH, "u"),
                         folder / "decisions")
    orphan = decide.outcome(path, rows[1], out_dir=folder / "outcomes")       # rank 2, and no rank-1 record exists
    assert orphan["status"] == "OK" and orphan["outcome"]["rank"] == 2

    group = decision_calibration.calibrate([entry["outcome"] for entry in settled] + [orphan["outcome"]])["groups"][0]
    assert group["n_linked"] == 3 and group["n_contests"] == 2
    assert group["excluded"][decision_calibration.CONTEST_NOT_SETTLED] == 1
    assert decision_calibration.NO_BEST_RANKED_OPTION in group["reason"]
