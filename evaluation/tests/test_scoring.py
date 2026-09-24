"""Each family computes what it can and names what it cannot. Every expected number here was worked out by hand."""

import math

import pytest
from conftest import LABELS, PREDICTIONS, ROWS, TRUTH, build_protocol

from m5phet_evaluation import (ABSTAINED, REFUSED_METRICS, NotEvaluable, PopulationMismatch, ProtocolError,
                               causal_accuracy, policy_profitability, regime_accuracy, score_causal,
                               score_classification, score_forecast, score_policy, score_regimes, seal_corpus)

FORECAST_ROWS = ("f1", "f2", "f3", "f4")
FORECAST_TRUTH = {"f1": 10.0, "f2": 12.0, "f3": 11.0, "f4": 13.0}
FORECAST_PREDICTIONS = {"f1": 11.0, "f2": 12.0, "f3": 12.0, "f4": 12.0}
FORECAST_BASELINE = {"f1": 9.0, "f2": 10.0, "f3": 12.0, "f4": 11.0}


def forecast_protocol(**overrides):
    declared = {"family": "forecast", "population": FORECAST_ROWS, "split": {"test": FORECAST_ROWS},
                "baseline": "naive_last_value", "metrics": ("mae", "rmse", "skill_mae")}
    declared.update(overrides)
    return build_protocol(**declared)


def other_family_protocol(family, rows, **overrides):
    declared = {"family": family, "population": rows, "split": {"test": rows}, "baseline": "none_declared"}
    declared.update(overrides)
    return build_protocol(**declared)


# --- classification ---------------------------------------------------------------------------------------------------
def test_classification_numbers_and_their_denominators(classification_case):
    _, _, metrics = classification_case
    values, counts = metrics.values, metrics.counts
    assert counts == {"declared_rows": 6, "scored_rows": 5, "abstained_rows": 1,
                      "support_by_class": {"relevant": 2, "irrelevant": 3}}
    assert values["confusion"] == {"relevant": {"relevant": 1, "irrelevant": 1},
                                   "irrelevant": {"relevant": 1, "irrelevant": 2}}
    assert values["accuracy"] == pytest.approx(3 / 5)
    assert values["coverage"] == pytest.approx(5 / 6)
    assert values["abstention_rate"] == pytest.approx(1 / 6)
    assert values["per_class"]["relevant"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2, "predicted": 2}
    assert values["per_class"]["irrelevant"]["f1"] == pytest.approx(2 / 3)
    assert values["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 2)


def test_the_majority_baseline_is_computed_on_the_scored_rows(classification_case):
    """Scored on the same five rows the model kept, which is why its accuracy ties the model instead of flattering it."""
    _, _, metrics = classification_case
    assert metrics.baseline["class"] == "irrelevant"
    assert metrics.baseline["rows"] == metrics.counts["scored_rows"] == 5
    assert metrics.baseline["same_rows_as_model"] is True
    assert metrics.baseline["accuracy"] == pytest.approx(3 / 5) == pytest.approx(metrics.values["accuracy"])
    assert metrics.baseline["macro_f1"] == pytest.approx((0.75 + 0.0) / 2)


def test_classification_refuses_absences_unknown_labels_and_total_abstention(protocol):
    seal = seal_corpus(TRUTH, protocol=protocol)
    without_r6 = {row: PREDICTIONS[row] for row in ROWS[:5]}
    with pytest.raises(PopulationMismatch, match="the predictions was computed on different rows"):
        score_classification(protocol=protocol, seal=seal, truth=TRUTH, predictions=without_r6, labels=LABELS)
    with pytest.raises(ProtocolError, match="neither a declared class"):
        score_classification(protocol=protocol, seal=seal, truth=TRUTH,
                             predictions=dict(PREDICTIONS, r1="maybe"), labels=LABELS)
    # a truth class nobody declared would vanish from the macro average instead of lowering it
    third_class = dict(TRUTH, r5="unclear")
    with pytest.raises(ProtocolError, match="which is not a declared class"):
        score_classification(protocol=protocol, seal=seal_corpus(third_class, protocol=protocol), truth=third_class,
                             predictions=PREDICTIONS, labels=LABELS)
    with pytest.raises(NotEvaluable, match="every row abstained"):
        score_classification(protocol=protocol, seal=seal, truth=TRUTH,
                             predictions={row: ABSTAINED for row in ROWS}, labels=LABELS)


def test_a_scorer_refuses_a_protocol_from_another_family(protocol):
    seal = seal_corpus(TRUTH, protocol=protocol)
    with pytest.raises(ProtocolError, match="declares family 'classification'"):
        score_forecast(protocol=protocol, seal=seal, truth=TRUTH, predictions=PREDICTIONS,
                       baseline_predictions=PREDICTIONS, baseline_name="majority_class")


# --- forecast ---------------------------------------------------------------------------------------------------------
def test_forecast_errors_and_skill_against_the_declared_baseline():
    declared = forecast_protocol()
    seal = seal_corpus(FORECAST_TRUTH, protocol=declared)
    metrics = score_forecast(protocol=declared, seal=seal, truth=FORECAST_TRUTH, predictions=FORECAST_PREDICTIONS,
                             baseline_predictions=FORECAST_BASELINE, baseline_name="naive_last_value")
    assert metrics.values["mae"] == pytest.approx(0.75)
    assert metrics.values["rmse"] == pytest.approx(math.sqrt(0.75))
    assert metrics.baseline["mae"] == pytest.approx(1.5)
    assert metrics.baseline["rows"] == metrics.counts["scored_rows"] == 4
    assert metrics.values["skill_mae"] == pytest.approx(0.5)
    assert metrics.values["skill_rmse"] == pytest.approx(1 - math.sqrt(0.75) / math.sqrt(2.5))


def test_a_baseline_on_different_rows_is_refused():
    """The comparison this package exists to prevent: the same skill number, computed over two populations."""
    declared = forecast_protocol()
    seal = seal_corpus(FORECAST_TRUTH, protocol=declared)
    elsewhere = {"f1": 9.0, "f2": 10.0, "f3": 12.0, "f9": 11.0}
    with pytest.raises(PopulationMismatch, match="computed on different rows"):
        score_forecast(protocol=declared, seal=seal, truth=FORECAST_TRUTH, predictions=FORECAST_PREDICTIONS,
                       baseline_predictions=elsewhere, baseline_name="naive_last_value")


def test_an_undeclared_baseline_is_refused_and_a_perfect_one_leaves_skill_undefined():
    declared = forecast_protocol()
    seal = seal_corpus(FORECAST_TRUTH, protocol=declared)
    with pytest.raises(ProtocolError, match="chosen to be beaten"):
        score_forecast(protocol=declared, seal=seal, truth=FORECAST_TRUTH, predictions=FORECAST_PREDICTIONS,
                       baseline_predictions=FORECAST_BASELINE, baseline_name="zero_forecast")
    metrics = score_forecast(protocol=declared, seal=seal, truth=FORECAST_TRUTH, predictions=FORECAST_PREDICTIONS,
                             baseline_predictions=dict(FORECAST_TRUTH), baseline_name="naive_last_value")
    assert metrics.values["skill_mae"] is None
    assert any("undefined" in note for note in metrics.notes)


# --- regimes ----------------------------------------------------------------------------------------------------------
def test_regime_agreement_is_stability_and_says_it_is_not_correctness():
    rows = ("g1", "g2", "g3", "g4")
    declared = other_family_protocol("regimes", rows, metrics=("rand_index",))
    corpus = {row: f"input-digest-{row}" for row in rows}
    seal = seal_corpus(corpus, protocol=declared)
    first = {"g1": 0, "g2": 0, "g3": 1, "g4": 1}
    identical = score_regimes(protocol=declared, seal=seal, corpus=corpus, assignment_a=first, assignment_b=dict(first))
    assert identical.values["rand_index"] == pytest.approx(1.0)
    assert identical.values["adjusted_rand_index"] == pytest.approx(1.0)
    shifted = score_regimes(protocol=declared, seal=seal, corpus=corpus, assignment_a=first,
                            assignment_b={"g1": "x", "g2": "y", "g3": "y", "g4": "y"})
    assert shifted.values["rand_index"] == pytest.approx(0.5)
    assert shifted.values["clusters_second"] == 2
    assert shifted.baseline is None
    assert any("no ground truth" in note for note in shifted.notes)


def test_regime_stability_across_different_rows_is_refused():
    rows = ("g1", "g2", "g3", "g4")
    declared = other_family_protocol("regimes", rows, metrics=("rand_index",))
    corpus = {row: f"input-digest-{row}" for row in rows}
    seal = seal_corpus(corpus, protocol=declared)
    first = {"g1": 0, "g2": 0, "g3": 1, "g4": 1}
    with pytest.raises(PopulationMismatch, match="the second assignment was computed on different rows"):
        score_regimes(protocol=declared, seal=seal, corpus=corpus, assignment_a=first,
                      assignment_b={"g1": 0, "g2": 0, "g3": 1, "g9": 1})


def test_regime_accuracy_refuses_by_name():
    with pytest.raises(NotEvaluable, match="regime_accuracy") as refusal:
        regime_accuracy()
    assert REFUSED_METRICS["regime_accuracy"] in str(refusal.value)
    assert "no ground truth" in str(refusal.value)


# --- causal -----------------------------------------------------------------------------------------------------------
def causal_case():
    rows = ("c1", "c2", "c3", "c4")
    declared = other_family_protocol("causal", rows, metrics=("estimate", "interval"))
    corpus = {row: {"treated": row in ("c1", "c2"), "outcome": 0.1} for row in rows}
    return declared, corpus, seal_corpus(corpus, protocol=declared)


def test_causal_reports_the_estimate_its_interval_and_its_assumptions():
    declared, corpus, seal = causal_case()
    metrics = score_causal(protocol=declared, seal=seal, corpus=corpus, estimand="ATE of a surprise on 6h return",
                           estimate=0.4, interval=(0.1, 0.7), assumptions=("no unmeasured confounding", "positivity"),
                           diagnostics={"overlap": "checked"})
    assert metrics.values["estimate"] == pytest.approx(0.4)
    assert metrics.values["interval"] == [0.1, 0.7]
    assert metrics.counts["assumptions_declared"] == 2
    assert "accuracy" not in metrics.values and metrics.baseline is None
    assert any("no held-out truth" in note for note in metrics.notes)


def test_causal_refuses_an_effect_without_assumptions_or_with_an_inverted_interval():
    declared, corpus, seal = causal_case()
    with pytest.raises(ProtocolError, match="identifying assumptions"):
        score_causal(protocol=declared, seal=seal, corpus=corpus, estimand="ATE", estimate=0.4,
                     interval=(0.1, 0.7), assumptions=())
    with pytest.raises(ProtocolError, match="out of order"):
        score_causal(protocol=declared, seal=seal, corpus=corpus, estimand="ATE", estimate=0.4,
                     interval=(0.7, 0.1), assumptions=("positivity",))


def test_causal_accuracy_refuses_by_name():
    with pytest.raises(NotEvaluable, match="causal_accuracy") as refusal:
        causal_accuracy(estimate=0.4)
    assert "counterfactual" in str(refusal.value)
    assert REFUSED_METRICS["causal_accuracy"] in str(refusal.value)


# --- policy -----------------------------------------------------------------------------------------------------------
def policy_case():
    rows = ("p1", "p2", "p3", "p4")
    declared = other_family_protocol("policy", rows, metrics=("mean_absolute_action", "nonzero_share"))
    corpus = {row: f"state-digest-{row}" for row in rows}
    return declared, corpus, seal_corpus(corpus, protocol=declared)


def test_policy_reports_action_statistics_only():
    declared, corpus, seal = policy_case()
    actions = {"p1": (0.0,), "p2": (0.5,), "p3": (-0.5,), "p4": (0.0,)}
    metrics = score_policy(protocol=declared, seal=seal, corpus=corpus, actions=actions, bounds=(-0.3, 0.3))
    assert metrics.values["action_dimension"] == 1
    assert metrics.values["mean_absolute_action"] == pytest.approx(0.25)
    assert metrics.values["nonzero_share"] == pytest.approx(0.5)
    assert metrics.values["mean_turnover_in_population_order"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert metrics.values["bound_violations"] == 2
    assert not any(key in metrics.values for key in ("return", "pnl", "profit", "sharpe"))
    assert any("realised return" in note for note in metrics.notes)


def test_policy_refuses_a_ragged_action_space():
    declared, corpus, seal = policy_case()
    with pytest.raises(ProtocolError, match="same action dimension"):
        score_policy(protocol=declared, seal=seal, corpus=corpus,
                     actions={"p1": (0.0,), "p2": (0.5, 0.1), "p3": (-0.5,), "p4": (0.0,)})


def test_policy_profitability_refuses_by_name():
    with pytest.raises(NotEvaluable, match="policy_profitability") as refusal:
        policy_profitability(actions={})
    assert "not a realised return" in str(refusal.value)
    assert REFUSED_METRICS["policy_profitability"] in str(refusal.value)
