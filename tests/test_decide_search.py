"""WP06 stage 5: a SEARCH is a third chooser, and it is neither a person nor Laya.

The search that closes WP06 evaluates candidate representations, fits each one, scores every fit on the sealed holdout
and keeps the one whose objective is smallest. The option it keeps has to enter the decision store like any other, or
the stage it produced enters the closure table with no record and WP23's clause (every stage in a table carries
records for the same (kind, question) set) is broken again by the very package that found it.

But it cannot be written as either of the two records that exist:

* not as Laya's, because a search produced no distribution over the options. `decision_calibration` calibrates the
  probabilities of a head; a search has none, and a flat one invented for it would be a claim nobody made;
* not as a person's, because nobody deliberated. A person's `why` is a judgement a reviewer weighs; a search's is
  mechanical, and the two facts that make it readable at all are the objective it minimised and the budget it spent.
  The best of 60 evaluations and the best of 6 are different claims and the winner does not show which.

These tests fix that third shape: `chosen_by: SEARCH`, no probabilities, no backend, no checkpoint, a `why` that must
carry both the objective and the budget, an option key from the declared set or a refusal, and an outcome that is
counted, may settle the best-ranked option, and is never scored.
"""

import json

import pytest

import closure_fixtures
from evaluation import decision_calibration
from m5phet import decide

OPTIONS = [["hand_household_w60", "the hand window of 60 rows"],
           ["searched_w101", "the searched window of 101 rows"],
           ["seasonal_lag_74", "the ACF peak window of 74 rows"]]
STATE = decide.decision_state("representation_search",
                              {"dataset": "household_dev_slice.csv", "seal": "33820b552ddf", "sealed_rows": 9824})
WHY = ("a genetic search over declared representation specs evaluated every point by fitting it and scoring it on the "
       "sealed holdout")
OBJECTIVE = "the held-out mean absolute error in kW on seal 33820b552ddf (9824 sealed rows), minimised"
BUDGET = "12 individuals x 5 generations, 47 distinct specs evaluated"


def searched(tmp_path, *, chosen="searched_w101", why=WHY, objective=OBJECTIVE, budget=BUDGET, options=None,
             state_text=STATE):
    return decide.search_choice(kind="representation", question="candidate", options=options or OPTIONS,
                                chosen=chosen, state_text=state_text, objective=objective, budget=budget, why=why,
                                as_of="2026-09-25T00:00:00+00:00", record_dir=tmp_path / "decisions")


# --------------------------------------------------------------------------------------------------------------------
# the record a search may write
# --------------------------------------------------------------------------------------------------------------------

def test_a_search_choice_is_recorded_with_its_chooser_its_objective_its_budget_and_no_probability(tmp_path):
    entry = searched(tmp_path)

    assert entry["status"] == "OK"
    decision = entry["decision"]
    assert decision["chosen_by"] == decide.CHOSEN_BY_SEARCH == "SEARCH"
    assert decision["chosen"] == "searched_w101"
    assert decision["probabilities"] == {} and decision["probability_decimals"] is None
    assert decision["backend"] is None and decision["checkpoint"] is None
    assert decision["execution_authorized"] is False
    assert decision["state_sha256"] == decide.state_sha256(STATE)
    assert decide.OBJECTIVE_MARKER + OBJECTIVE in decision["why"]
    assert decide.BUDGET_MARKER + BUDGET in decision["why"]
    # content-addressed and reloadable exactly like Laya's and like a person's
    assert decide.load(entry["record_path"]) == decision
    assert json.loads(open(entry["record_path"]).read()) == decision


def test_the_three_choosers_stay_distinct(tmp_path):
    assert decide.CHOOSERS == ("LAYA", "HUMAN", "SEARCH")
    search = searched(tmp_path)["decision"]
    human = decide.human_choice(kind="representation", question="candidate", options=OPTIONS,
                                chosen="hand_household_w60", state_text=STATE,
                                why="the window the household resource declares", as_of="2026-09-25T00:00:00+00:00",
                                record_dir=tmp_path / "decisions")["decision"]
    laya = {"schema": decide.DECISION_SCHEMA, "kind": "representation", "state_sha256": "0" * 64,
            "question": "candidate", "options": OPTIONS, "chosen": "seasonal_lag_74",
            "probabilities": {"hand_household_w60": 0.3, "searched_w101": 0.3, "seasonal_lag_74": 0.4},
            "probability_decimals": 4, "checkpoint": "laya-checkpoint:bd12df887789", "backend": "laya",
            "as_of": "2026-09-25T00:00:00+00:00", "execution_authorized": False}

    assert (decide.chooser_of(search), decide.chooser_of(human), decide.chooser_of(laya)) == ("SEARCH", "HUMAN", "LAYA")
    # and no record is readable as another chooser's: the search record's own `why` is not a person's ground
    assert search["why"] != human["why"]
    assert decide.decision_sha256(search) != decide.decision_sha256(human)
    with pytest.raises(decide.DecisionError, match="chosen_by"):
        decide.validate_decision(dict(search, chosen_by="THE_OPTIMISER"))


def test_a_search_record_without_an_objective_or_without_a_budget_is_refused_and_nothing_is_written(tmp_path):
    assert searched(tmp_path, objective="  ")["refusal"] == decide.OBJECTIVE_REQUIRED == "OBJECTIVE_REQUIRED"
    assert searched(tmp_path, budget="")["refusal"] == decide.BUDGET_REQUIRED == "BUDGET_REQUIRED"
    assert searched(tmp_path, why=" ")["refusal"] == decide.WHY_REQUIRED
    assert not (tmp_path / "decisions").exists()


def test_a_search_record_whose_why_lost_the_objective_or_the_budget_is_not_a_record(tmp_path):
    decision = searched(tmp_path)["decision"]

    for stripped in (WHY, WHY + " " + decide.OBJECTIVE_MARKER + OBJECTIVE + "."):
        with pytest.raises(decide.DecisionError, match="REQUIRED"):
            decide.validate_decision(dict(decision, why=stripped))


def test_a_key_outside_the_declared_options_is_refused_for_a_search_too(tmp_path):
    entry = searched(tmp_path, chosen="window_401")

    assert entry["status"] == "REFUSED" and entry["refusal"] == decide.CHOICE_OUTSIDE_OPTIONS
    assert "never a new option" in entry["why"]
    assert not (tmp_path / "decisions").exists()


def test_a_search_record_carrying_probabilities_or_a_backend_is_not_a_record_at_all(tmp_path):
    decision = searched(tmp_path)["decision"]

    with pytest.raises(decide.DecisionError, match=decide.CHOOSER_FIELDS):
        decide.validate_decision(dict(decision, probabilities={"searched_w101": 1.0}))
    with pytest.raises(decide.DecisionError, match=decide.CHOOSER_FIELDS):
        decide.validate_decision(dict(decision, backend="laya"))
    with pytest.raises(decide.DecisionError, match=decide.CHOOSER_FIELDS):
        decide.validate_decision(dict(decision, checkpoint="laya-checkpoint:bd12df887789"))


def test_a_search_never_abstains(tmp_path):
    """An abstention is a fact about a head's confidence; a search that evaluated nothing has a refusal, not a doubt."""
    decision = searched(tmp_path)["decision"]
    abstained = dict(decision, chosen=None,
                     abstention={"refusal": decide.LOW_CONFIDENCE_ABSTAINED, "top_option": "searched_w101",
                                 "threshold": {"path": "report.json", "sha256": "0" * 64}})
    with pytest.raises(decide.DecisionError):
        decide.validate_decision(abstained)


# --------------------------------------------------------------------------------------------------------------------
# the outcome: a label, never a score
# --------------------------------------------------------------------------------------------------------------------

def test_a_search_outcome_carries_its_chooser_and_settles_the_best_ranked_option(tmp_path):
    rows = closure_fixtures.table(tmp_path, {"searched": 0.25, "hand": 0.5})
    entry = searched(tmp_path)

    second = decide.outcome(entry["record_path"], rows["hand"], out_dir=tmp_path / "outcomes")
    assert second["outcome"]["chosen_by"] == "SEARCH" and second["outcome"]["probabilities"] == {}
    assert "best_ranked_option" not in second["outcome"]

    first = decide.outcome(entry["record_path"], rows["searched"], out_dir=tmp_path / "outcomes")
    assert first["outcome"]["best_ranked_option"] == "searched_w101" and first["outcome"]["rank"] == 1
    assert decide.load_outcome(first["record_path"]) == first["outcome"]


def test_a_search_outcome_is_counted_apart_and_never_scored(tmp_path):
    rows = closure_fixtures.table(tmp_path, {"searched": 0.25, "laya": 0.5})
    entry = searched(tmp_path)
    decide.outcome(entry["record_path"], rows["searched"], out_dir=tmp_path / "outcomes")
    laya = decide.record({"schema": decide.DECISION_SCHEMA, "kind": "representation", "state_sha256": "1" * 64,
                          "question": "candidate", "options": OPTIONS, "chosen": "hand_household_w60",
                          "probabilities": {"hand_household_w60": 0.5, "searched_w101": 0.3, "seasonal_lag_74": 0.2},
                          "probability_decimals": 4, "checkpoint": "laya-checkpoint:bd12df887789", "backend": "laya",
                          "as_of": "2026-09-25T00:00:00+00:00", "execution_authorized": False},
                         tmp_path / "decisions")
    decide.outcome(laya, rows["laya"], out_dir=tmp_path / "outcomes")

    outcomes, unreadable = decision_calibration.load_outcomes([tmp_path / "outcomes"])
    assert unreadable == []
    group = decision_calibration.calibrate(outcomes)["groups"][0]

    assert group["best_ranked_option"] == "searched_w101"
    assert decision_calibration.NO_BEST_RANKED_OPTION not in (group["reason"] or "")
    assert group["n_linked"] == 2 and group["n_scorable_linked"] == 1
    assert group["n_search_linked"] == 1 and group["n_human_linked"] == 0
    assert group["excluded"][decision_calibration.SEARCH_NOT_SCORED] == 1
    assert group["search_stages"] == ["searched"]
    # a search's own choice is not evidence about Laya: the missing count is of SCORABLE outcomes
    assert group["status"] == decision_calibration.NO_NEW_MEASUREMENT and group["missing"] == 29
    assert "1 by a search, neither ever scored" in decision_calibration.render_markdown(
        decision_calibration.calibrate(outcomes))


def test_the_calibration_report_and_decide_agree_on_the_third_chooser_name():
    assert decision_calibration.CHOSEN_BY_SEARCH == decide.CHOSEN_BY_SEARCH
    assert set(decision_calibration.CHOOSERS_NOT_SCORED) == set(decide.CHOOSERS_WITHOUT_PROBABILITIES)
