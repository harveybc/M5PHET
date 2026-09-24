"""Metrics per family, each of which refuses instead of guessing.

Every scorer here starts from the same three gates: the family it was written for, the seal taken before the scores were
seen, and the population declared in the protocol. A metric that survives those gates carries the counts it rests on, so
no ratio is ever read without its denominator, and a value that is undefined stays None rather than becoming a zero that
looks like a measurement of something.

Two families have no accuracy at all and say so by name instead of quietly offering a surrogate. A causal estimate has no
held-out truth, because the counterfactual for a row is never observed. A policy proposal has no profit, because a
proposed action is not a realised return. Both refusals live in REFUSED_METRICS with the reason attached, so the reason
travels with the refusal wherever it is caught.

Nothing here is applied to a real model in this package. This is the instrument, not a measurement.
"""

from __future__ import annotations

import dataclasses
import math

from .freeze import require_intact
from .protocol import NotEvaluable, PopulationMismatch, ProtocolError

#: a declared abstention. A row missing from the predictions is an absence, not this, and the two are never merged
ABSTAINED = "ABSTAINED"

#: quantities this package will not compute, with the reason each one does not exist. Caught code can read the reason
REFUSED_METRICS = {
    "causal_accuracy": (
        "a causal estimate has no held-out truth: the counterfactual outcome of a row is never observed, so there is no "
        "row against which the estimate can be scored right or wrong. What can be reported is the estimate, its interval, "
        "the assumptions under which it is identified, and its sensitivity to those assumptions"),
    "policy_profitability": (
        "a proposed action is not a realised return: no order was placed, no fill, slippage, financing or timing exists, "
        "and the market did not respond to it. What can be reported is the action statistics; profit belongs to an "
        "execution record from the system that actually traded"),
    "regime_accuracy": (
        "an unsupervised assignment has no ground truth: cluster identities are arbitrary and no row carries a correct "
        "regime, so agreement between two assignments is stability, never correctness. If independently produced regime "
        "labels exist, declare them in the protocol and score them as a classification"),
}


def refuse(name: str):
    """Raise the named refusal. Single source for the reason, so a refusal cannot be re-explained more softly elsewhere."""
    raise NotEvaluable(f"{name}: {REFUSED_METRICS[name]}")


@dataclasses.dataclass(frozen=True)
class MetricSet:
    """Values plus the counts they rest on plus the population they were computed over, which the report re-checks."""

    name: str
    family: str
    population: tuple
    values: dict
    counts: dict
    baseline: dict | None = None
    notes: tuple = ()

    def as_dict(self) -> dict:
        return {"name": self.name, "family": self.family, "rows_declared": len(self.population),
                "values": self.values, "counts": self.counts, "baseline": self.baseline, "notes": list(self.notes)}


def _family(protocol, expected: str) -> None:
    if protocol.family != expected:
        raise ProtocolError(f"this scorer computes {expected} metrics, but the protocol declares family {protocol.family!r}")


def _finite(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProtocolError(f"{what}: a finite number is required, got {value!r}")
    return float(value)


def _same_rows(rows, expected, what: str) -> None:
    """The refusal every silent comparison depends on: a model scored on one set of rows and a baseline on another."""
    if not isinstance(rows, dict):
        raise ProtocolError(f"{what}: a row->value mapping is required")
    expected = tuple(expected)
    missing = [r for r in expected if r not in rows]
    extra = [r for r in rows if r not in expected]
    if missing or extra:
        raise PopulationMismatch(
            f"{what} was computed on different rows: {len(missing)} expected rows absent "
            f"{[str(r) for r in missing[:5]]}, {len(extra)} unexpected rows present {[str(r) for r in extra[:5]]}")


def _labels(labels) -> tuple:
    names = tuple(labels or ())
    if not names or any(not isinstance(label, str) or not label.strip() for label in names):
        raise ProtocolError("labels: a nonempty list of nonempty class names is required")
    if len(set(names)) != len(names):
        raise ProtocolError("labels: duplicate class name")
    if ABSTAINED in names:
        raise ProtocolError(f"labels: {ABSTAINED!r} is the abstention marker and cannot also be a class")
    return names


def _confusion(truth, predicted, rows, labels) -> dict:
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for row in rows:
        matrix[truth[row]][predicted[row]] += 1
    return matrix


def _per_class(matrix, labels):
    per_class, in_scope = {}, []
    for label in labels:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[t][label] for t in labels)
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        elif support or predicted:
            f1 = 0.0            # the class occurs in this corpus and was never got right: a real zero, not a gap
        else:
            f1 = None           # absent from both truth and predictions: undefined, and left out of the macro average
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": support, "predicted": predicted}
        if f1 is not None:
            in_scope.append(label)
    return per_class, tuple(in_scope)


def _classification_values(truth, predicted, rows, labels) -> dict:
    matrix = _confusion(truth, predicted, rows, labels)
    per_class, in_scope = _per_class(matrix, labels)
    correct = sum(matrix[label][label] for label in labels)
    return {
        "confusion": matrix,
        "per_class": per_class,
        "macro_f1": sum(per_class[label]["f1"] for label in in_scope) / len(in_scope),
        "macro_f1_classes": list(in_scope),
        "accuracy": correct / len(rows),
        "correct": correct,
    }


def score_classification(*, protocol, seal, truth, predictions, labels) -> MetricSet:
    """Confusion, per-class precision/recall/F1, macro-F1, accuracy, coverage, abstention, and the majority-class
    baseline over the SAME scored rows, because a baseline computed over all rows while the model keeps only the ones it
    was confident about is the comparison this whole package exists to prevent."""
    _family(protocol, "classification")
    require_intact(seal, truth, protocol)
    names = _labels(labels)
    _same_rows(truth, protocol.population, "the labelled corpus")
    _same_rows(predictions, protocol.population, "the predictions")
    for row in protocol.population:
        if truth[row] not in names:
            raise ProtocolError(f"row {row!r} carries truth label {truth[row]!r}, which is not a declared class")
        if predictions[row] != ABSTAINED and predictions[row] not in names:
            raise ProtocolError(f"row {row!r} predicts {predictions[row]!r}, which is neither a declared class nor {ABSTAINED!r}")

    scored = tuple(row for row in protocol.population if predictions[row] != ABSTAINED)
    total = len(protocol.population)
    if not scored:
        raise NotEvaluable("every row abstained: with no scored row there is no accuracy and no macro-F1 to report")

    values = _classification_values(truth, predictions, scored, names)
    values["coverage"] = len(scored) / total
    values["abstention_rate"] = (total - len(scored)) / total

    support = {label: 0 for label in names}
    for row in scored:
        support[truth[row]] += 1
    top = max(support.values())
    # ties break by declared label order and are recorded, so the baseline is reproducible rather than dict-order luck
    tied = [label for label in names if support[label] == top]
    majority = tied[0]
    baseline_values = _classification_values(truth, {row: majority for row in scored}, scored, names)
    baseline = {"name": "majority_class", "class": majority, "tied_classes": tied,
                "rows": len(scored), "same_rows_as_model": True,
                "accuracy": baseline_values["accuracy"], "macro_f1": baseline_values["macro_f1"]}
    return MetricSet(name="classification", family="classification", population=protocol.population, values=values,
                     counts={"declared_rows": total, "scored_rows": len(scored), "abstained_rows": total - len(scored),
                             "support_by_class": support},
                     baseline=baseline,
                     notes=("Coverage and abstention are reported beside every ratio: accuracy over the scored rows is "
                            "not accuracy over the population.",))


def score_forecast(*, protocol, seal, truth, predictions, baseline_predictions, baseline_name) -> MetricSet:
    """MAE, RMSE and skill against the declared naive baseline on the same rows. A baseline evaluated on other rows is
    refused outright, because the resulting skill number looks identical to a real one."""
    _family(protocol, "forecast")
    require_intact(seal, truth, protocol)
    if not isinstance(baseline_name, str) or baseline_name.strip() != protocol.baseline:
        raise ProtocolError(
            f"the baseline must be the one declared in the protocol ({protocol.baseline!r}), not {baseline_name!r}: "
            "a baseline chosen after the fact is a baseline chosen to be beaten")
    _same_rows(truth, protocol.population, "the labelled corpus")
    _same_rows(predictions, protocol.population, "the forecasts")
    rows = protocol.population
    _same_rows(baseline_predictions, rows, f"the naive baseline {protocol.baseline!r}")

    errors, baseline_errors = [], []
    for row in rows:
        actual = _finite(truth[row], f"truth for row {row!r}")
        errors.append(abs(_finite(predictions[row], f"forecast for row {row!r}") - actual))
        baseline_errors.append(abs(_finite(baseline_predictions[row], f"baseline for row {row!r}") - actual))
    n = len(rows)
    mae = sum(errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    baseline_mae = sum(baseline_errors) / n
    baseline_rmse = math.sqrt(sum(e * e for e in baseline_errors) / n)
    notes = []
    if baseline_mae > 0:
        skill_mae = 1 - mae / baseline_mae
    else:
        skill_mae = None        # a perfect baseline leaves nothing to improve on; a skill of 1 here would be arithmetic
        notes.append("The baseline error is zero on these rows, so skill is undefined rather than perfect.")
    skill_rmse = 1 - rmse / baseline_rmse if baseline_rmse > 0 else None
    return MetricSet(name="forecast", family="forecast", population=protocol.population,
                     values={"mae": mae, "rmse": rmse, "skill_mae": skill_mae, "skill_rmse": skill_rmse},
                     counts={"declared_rows": n, "scored_rows": n, "baseline_rows": n},
                     baseline={"name": protocol.baseline, "mae": baseline_mae, "rmse": baseline_rmse,
                               "rows": n, "same_rows_as_model": True},
                     notes=tuple(notes))


def _pair_counts(assignment, rows) -> dict:
    groups: dict = {}
    for row in rows:
        groups.setdefault(assignment[row], []).append(row)
    return groups


def score_regimes(*, protocol, seal, corpus, assignment_a, assignment_b) -> MetricSet:
    """Stability between two assignments of the same rows. Nothing here is correctness: see REFUSED_METRICS."""
    _family(protocol, "regimes")
    require_intact(seal, corpus, protocol)
    rows = protocol.population
    _same_rows(corpus, rows, "the sealed population")
    _same_rows(assignment_a, rows, "the first assignment")
    # stability across two different row sets is not stability; it is two unrelated fits described as one number
    _same_rows(assignment_b, rows, "the second assignment")

    groups_a, groups_b = _pair_counts(assignment_a, rows), _pair_counts(assignment_b, rows)
    n = len(rows)
    agree = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_a = assignment_a[rows[i]] == assignment_a[rows[j]]
            same_b = assignment_b[rows[i]] == assignment_b[rows[j]]
            agree += same_a == same_b
    pairs = n * (n - 1) // 2
    rand = agree / pairs if pairs else None
    overlap = sum(math.comb(len([r for r in ra if r in set(rb)]), 2)
                  for ra in groups_a.values() for rb in groups_b.values())
    sum_a = sum(math.comb(len(g), 2) for g in groups_a.values())
    sum_b = sum(math.comb(len(g), 2) for g in groups_b.values())
    expected = (sum_a * sum_b / pairs) if pairs else None
    maximum = 0.5 * (sum_a + sum_b) if expected is not None else None
    adjusted = (overlap - expected) / (maximum - expected) if expected is not None and maximum != expected else None

    notes = [REFUSED_METRICS["regime_accuracy"]]
    if protocol.independent_labels:
        notes.append("Independent labels are declared in this protocol; score them with score_classification. Agreement "
                     "between two fits stays stability even then.")
    if adjusted is None:
        notes.append("The adjusted index is undefined on these assignments (a degenerate grouping), so it is not reported.")
    return MetricSet(name="regimes", family="regimes", population=protocol.population,
                     values={"rand_index": rand, "adjusted_rand_index": adjusted,
                             "clusters_first": len(groups_a), "clusters_second": len(groups_b),
                             "largest_cluster_share_first": max(len(g) for g in groups_a.values()) / n,
                             "largest_cluster_share_second": max(len(g) for g in groups_b.values()) / n},
                     counts={"declared_rows": n, "scored_rows": n, "compared_pairs": pairs},
                     baseline=None, notes=tuple(notes))


def regime_accuracy(*_args, **_kwargs):
    """Refuses by name. Kept as a function so the mistake has somewhere to land and get an explanation."""
    refuse("regime_accuracy")


def score_causal(*, protocol, seal, corpus, estimand, estimate, interval, assumptions, diagnostics=None) -> MetricSet:
    """The estimate, its interval and the assumptions it is identified under. No accuracy: see causal_accuracy."""
    _family(protocol, "causal")
    require_intact(seal, corpus, protocol)
    _same_rows(corpus, protocol.population, "the sealed analysis sample")
    if not isinstance(estimand, str) or not estimand.strip():
        raise ProtocolError("estimand: name the quantity, because an effect without an estimand is a number without a claim")
    value = _finite(estimate, "estimate")
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ProtocolError("interval: two finite bounds are required")
    low, high = (_finite(interval[0], "interval lower bound"), _finite(interval[1], "interval upper bound"))
    if low > high:
        raise ProtocolError(f"interval: bounds are out of order ({low}, {high})")
    declared = tuple(a.strip() for a in (assumptions or ()) if isinstance(a, str) and a.strip())
    if not declared:
        # without its assumptions an effect estimate is not identified, and reporting it as a result would be the claim
        raise ProtocolError("assumptions: a causal estimate with no declared identifying assumptions is not an estimate")
    return MetricSet(name="causal", family="causal", population=protocol.population,
                     values={"estimand": estimand.strip(), "estimate": value, "interval": [low, high],
                             "assumptions": list(declared), "diagnostics": dict(diagnostics or {})},
                     counts={"declared_rows": len(protocol.population), "scored_rows": len(protocol.population),
                             "assumptions_declared": len(declared)},
                     baseline=None,
                     notes=(REFUSED_METRICS["causal_accuracy"],
                            "The interval is the uncertainty of the estimate under these assumptions; it is not a "
                            "prediction interval and it does not cover assumption failure."))


def causal_accuracy(*_args, **_kwargs):
    """Refuses by name, with the reason a causal estimate cannot be scored against held-out rows."""
    refuse("causal_accuracy")


def score_policy(*, protocol, seal, corpus, actions, bounds=None) -> MetricSet:
    """Action statistics only. Profit is refused by name: see policy_profitability."""
    _family(protocol, "policy")
    require_intact(seal, corpus, protocol)
    rows = protocol.population
    _same_rows(corpus, rows, "the sealed decision population")
    _same_rows(actions, rows, "the proposed actions")

    vectors = []
    for row in rows:
        action = actions[row]
        action = (action,) if isinstance(action, (int, float)) and not isinstance(action, bool) else tuple(action or ())
        if not action:
            raise ProtocolError(f"row {row!r}: an empty action is not an action; propose a vector or say the policy abstained")
        vectors.append(tuple(_finite(v, f"action component for row {row!r}") for v in action))
    width = len(vectors[0])
    if any(len(v) != width for v in vectors):
        raise ProtocolError("actions: every proposal must have the same action dimension, or they are not one action space")

    per_dim = [[v[i] for v in vectors] for i in range(width)]
    violations = 0
    if bounds is not None:
        low, high = (_finite(bounds[0], "lower bound"), _finite(bounds[1], "upper bound"))
        violations = sum(1 for v in vectors if any(c < low or c > high for c in v))
    # turnover reads the population order as the decision order; out of order it measures nothing, so it is named as such
    turnover = (sum(sum(abs(b - a) for a, b in zip(vectors[i - 1], vectors[i])) for i in range(1, len(vectors)))
                / max(len(vectors) - 1, 1)) if len(vectors) > 1 else None
    return MetricSet(name="policy", family="policy", population=protocol.population,
                     values={"action_dimension": width,
                             "mean_by_dimension": [sum(col) / len(col) for col in per_dim],
                             "min_by_dimension": [min(col) for col in per_dim],
                             "max_by_dimension": [max(col) for col in per_dim],
                             "mean_absolute_action": sum(sum(abs(c) for c in v) for v in vectors) / (len(vectors) * width),
                             "nonzero_share": sum(1 for v in vectors if any(c != 0 for c in v)) / len(vectors),
                             "mean_turnover_in_population_order": turnover,
                             "bound_violations": violations if bounds is not None else None},
                     counts={"declared_rows": len(rows), "scored_rows": len(vectors)},
                     baseline=None,
                     notes=(REFUSED_METRICS["policy_profitability"],
                            "Turnover assumes the declared population order is the decision order; in any other order it "
                            "measures nothing."))


def policy_profitability(*_args, **_kwargs):
    """Refuses by name. A proposed action has no fill, no cost and no market response."""
    refuse("policy_profitability")
