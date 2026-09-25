"""WP23's clause: a stage a PERSON configured carries decision records too, or the rank-1 row labels nothing.

The structural finding of 2026-09-25 was not that Laya chose badly. It was that the closure table ranked a hand-written
stage first, that stage carried no decision record because a person wrote it, and the calibration report could
therefore only ever say `NO_BEST_RANKED_OPTION` — there was no option to score the others against. Writing Laya's
chosen option against the hand row would have claimed Laya chose what the person used.

So a person's choice becomes a record of its own, and these tests fix what that record may and may not contain:

* it names its chooser (`chosen_by: HUMAN`) and carries the ground of the choice (`why`);
* it carries **no** probabilities, no backend and no checkpoint — a person produced no distribution, and a flat one
  written in its place would put a certainty in the corpus that nobody claimed;
* it is chosen from the option set the executing repository declared, exactly as Laya's is: a key outside it is
  refused rather than recorded;
* its outcome may settle the best-ranked option, and is never scored: an agreement rate that included it would measure
  the person who configured the stage, not the chooser this framework is calibrating.
"""

import json

import pytest

import closure_fixtures
from evaluation import decision_calibration
from m5phet import decide

OPTIONS = [["a", "keep the level"], ["b", "normalise per feature"], ["c", "log return"]]
STATE = decide.decision_state("feature_preprocessing", {"feature": "Global_active_power", "scale": "kW"})
WHY = "the fitting harness standardises every column on the TRAIN rows only, which is what this option declares"


def human(tmp_path, *, chosen="b", question="preprocessing", why=WHY, options=None, state_text=STATE):
    return decide.human_choice(kind="feature_preprocessing", question=question, options=options or OPTIONS,
                               chosen=chosen, state_text=state_text, why=why,
                               as_of="2026-09-25T00:00:00+00:00", record_dir=tmp_path / "decisions")


# --------------------------------------------------------------------------------------------------------------------
# the record a person may write
# --------------------------------------------------------------------------------------------------------------------

def test_a_human_choice_is_recorded_with_its_chooser_its_ground_and_no_probability(tmp_path):
    entry = human(tmp_path)

    assert entry["status"] == "OK"
    decision = entry["decision"]
    assert decision["chosen_by"] == decide.CHOSEN_BY_HUMAN == "HUMAN"
    assert decision["chosen"] == "b" and decision["why"] == WHY
    assert decision["probabilities"] == {} and decision["probability_decimals"] is None
    assert decision["backend"] is None and decision["checkpoint"] is None
    assert decision["execution_authorized"] is False
    assert decision["state_sha256"] == decide.state_sha256(STATE)
    # content-addressed and reloadable exactly like Laya's
    assert decide.load(entry["record_path"]) == decision
    assert json.loads(open(entry["record_path"]).read()) == decision


def test_a_record_that_names_no_chooser_is_laya_s(tmp_path):
    laya = {"schema": decide.DECISION_SCHEMA, "kind": "feature_preprocessing", "state_sha256": "0" * 64,
            "question": "preprocessing", "options": OPTIONS, "chosen": "b",
            "probabilities": {"a": 0.3, "b": 0.5, "c": 0.2}, "probability_decimals": 4,
            "checkpoint": "laya-checkpoint:bd12df887789", "backend": "laya",
            "as_of": "2026-09-25T00:00:00+00:00", "execution_authorized": False}

    assert decide.chooser_of(laya) == decide.CHOSEN_BY_LAYA == "LAYA"
    assert decide.validate_decision(laya) is laya


def test_a_key_outside_the_declared_options_is_refused_and_nothing_is_written(tmp_path):
    entry = human(tmp_path, chosen="d")

    assert entry["status"] == "REFUSED" and entry["refusal"] == decide.CHOICE_OUTSIDE_OPTIONS
    assert not (tmp_path / "decisions").exists()


def test_a_human_choice_without_a_ground_is_refused_as_WHY_REQUIRED(tmp_path):
    assert human(tmp_path, why="   ")["refusal"] == decide.WHY_REQUIRED == "WHY_REQUIRED"


def test_one_option_is_not_a_choice_for_a_person_either(tmp_path):
    assert human(tmp_path, options=[["a", "keep the level"]])["refusal"] == decide.MALFORMED_OPTIONS


def test_a_human_record_carrying_probabilities_or_a_backend_is_not_a_record_at_all(tmp_path):
    decision = human(tmp_path)["decision"]

    with_probabilities = dict(decision, probabilities={"a": 0.0, "b": 1.0, "c": 0.0})
    with pytest.raises(decide.DecisionError, match=decide.CHOOSER_FIELDS):
        decide.validate_decision(with_probabilities)

    with_backend = dict(decision, backend="laya")
    with pytest.raises(decide.DecisionError, match=decide.CHOOSER_FIELDS):
        decide.validate_decision(with_backend)

    with pytest.raises(decide.DecisionError, match="chosen_by"):
        decide.validate_decision(dict(decision, chosen_by="THE_AUTHOR"))


# --------------------------------------------------------------------------------------------------------------------
# the outcome: a label, never a score
# --------------------------------------------------------------------------------------------------------------------

def test_a_human_outcome_carries_its_chooser_and_settles_the_best_ranked_option(tmp_path):
    rows = closure_fixtures.table(tmp_path, {"designed": 0.25, "hand": 0.5})
    entry = human(tmp_path, chosen="b")

    linked = decide.outcome(entry["record_path"], rows["hand"], out_dir=tmp_path / "outcomes")
    assert linked["status"] == "OK"
    assert linked["outcome"]["chosen_by"] == "HUMAN" and linked["outcome"]["probabilities"] == {}
    assert "best_ranked_option" not in linked["outcome"]          # `hand` is ranked second here

    first = decide.outcome(entry["record_path"], rows["designed"], out_dir=tmp_path / "outcomes")
    assert first["outcome"]["best_ranked_option"] == "b" and first["outcome"]["rank"] == 1
    assert decide.load_outcome(first["record_path"]) == first["outcome"]


def test_a_laya_outcome_still_carries_its_own_chooser(tmp_path):
    rows = closure_fixtures.table(tmp_path, {"designed": 0.25, "hand": 0.5})
    laya = decide.record({"schema": decide.DECISION_SCHEMA, "kind": "feature_preprocessing", "state_sha256": "0" * 64,
                          "question": "preprocessing", "options": OPTIONS, "chosen": "a",
                          "probabilities": {"a": 0.5, "b": 0.3, "c": 0.2}, "probability_decimals": 4,
                          "checkpoint": "laya-checkpoint:bd12df887789", "backend": "laya",
                          "as_of": "2026-09-25T00:00:00+00:00", "execution_authorized": False},
                         tmp_path / "decisions")

    linked = decide.outcome(laya, rows["designed"], out_dir=tmp_path / "outcomes")
    assert linked["outcome"]["chosen_by"] == "LAYA"


# --------------------------------------------------------------------------------------------------------------------
# the report: counted, never scored
# --------------------------------------------------------------------------------------------------------------------

def _report(tmp_path, *, table=None):
    outcomes, unreadable = decision_calibration.load_outcomes([tmp_path / "outcomes"])
    assert unreadable == []
    return decision_calibration.calibrate(outcomes, table=table)


def test_the_human_label_stops_NO_BEST_RANKED_OPTION_and_is_not_scored(tmp_path):
    rows = closure_fixtures.table(tmp_path, {"hand": 0.25, "laya": 0.5})
    # the person's stage is ranked first and says which option it used
    person = human(tmp_path, chosen="b")
    decide.outcome(person["record_path"], rows["hand"], out_dir=tmp_path / "outcomes")
    # Laya's own choice, on the stage ranked second
    laya = decide.record({"schema": decide.DECISION_SCHEMA, "kind": "feature_preprocessing", "state_sha256": "1" * 64,
                          "question": "preprocessing", "options": OPTIONS, "chosen": "a",
                          "probabilities": {"a": 0.5, "b": 0.3, "c": 0.2}, "probability_decimals": 4,
                          "checkpoint": "laya-checkpoint:bd12df887789", "backend": "laya",
                          "as_of": "2026-09-25T00:00:00+00:00", "execution_authorized": False},
                         tmp_path / "decisions")
    decide.outcome(laya, rows["laya"], out_dir=tmp_path / "outcomes")

    group = _report(tmp_path)["groups"][0]
    assert group["best_ranked_option"] == "b"
    assert decision_calibration.NO_BEST_RANKED_OPTION not in (group["reason"] or "")
    assert group["n_linked"] == 2 and group["n_scorable_linked"] == 1 and group["n_human_linked"] == 1
    assert group["excluded"][decision_calibration.HUMAN_NOT_SCORED] == 1
    assert group["human_stages"] == ["hand"]
    # still under the minimum, and the missing count is of SCORABLE outcomes: a person's choice is not evidence
    assert group["status"] == decision_calibration.NO_NEW_MEASUREMENT and group["missing"] == 29


def test_the_report_names_the_stage_that_entered_the_table_with_no_decision_record(tmp_path):
    rows = closure_fixtures.table(tmp_path, {"hand": 0.25, "laya": 0.5})
    table_path = tmp_path / "table.json"
    table_path.write_text(json.dumps({"version": "m5phet-evaluation-stage-comparison/1",
                                      "areas": [{"area": "forecast", "rows": list(rows.values())}]}))
    laya = decide.record({"schema": decide.DECISION_SCHEMA, "kind": "feature_preprocessing", "state_sha256": "1" * 64,
                          "question": "preprocessing", "options": OPTIONS, "chosen": "a",
                          "probabilities": {"a": 0.5, "b": 0.3, "c": 0.2}, "probability_decimals": 4,
                          "checkpoint": "laya-checkpoint:bd12df887789", "backend": "laya",
                          "as_of": "2026-09-25T00:00:00+00:00", "execution_authorized": False},
                         tmp_path / "decisions")
    decide.outcome(laya, rows["laya"], out_dir=tmp_path / "outcomes")

    table = decision_calibration.read_table(table_path)
    assert [item["stage"] for item in table["ranked_first"]] == ["hand"]
    group = _report(tmp_path, table=table)["groups"][0]

    verdicts = {item["stage"]: item["verdict"] for item in group["stage_coverage"]}
    assert verdicts == {"hand": decision_calibration.COMPARABLE_BUT_NO_DECISION_RECORD,
                        "laya": decision_calibration.USABLE_FOR_CALIBRATION}
    assert decision_calibration.NO_BEST_RANKED_OPTION in group["reason"]
    assert "'hand'" in group["reason"] and decision_calibration.COMPARABLE_BUT_NO_DECISION_RECORD in group["reason"]
    assert decision_calibration.COMPARABLE_BUT_NO_DECISION_RECORD in decision_calibration.render_markdown(
        _report(tmp_path, table=table))


def test_the_calibration_report_and_decide_agree_on_the_chooser_names():
    assert decision_calibration.CHOSEN_BY_LAYA == decide.CHOSEN_BY_LAYA
    assert decision_calibration.CHOSEN_BY_HUMAN == decide.CHOSEN_BY_HUMAN
