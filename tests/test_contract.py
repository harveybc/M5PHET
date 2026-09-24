import copy

import pytest

from m5phet import ContractError, make_classification_result


def inputs():
    return {"tone": {"label": "positive", "uncalibrated_probabilities": {"positive": 0.75, "negative": 0.25}}}


def make(outputs=None, **overrides):
    args = dict(expected_labels={"tone": ["positive", "negative"]}, input_sha256="a" * 64,
                model_sha256="b" * 64, task_sha256="c" * 64)
    args.update(overrides)
    return make_classification_result(inputs() if outputs is None else outputs, **args)


def test_bound_uncalibrated_snapshot():
    source = inputs()
    result = make(source)
    assert result["schema"] == "m5phet.classification.v1"
    assert result["execution_authorized"] is False
    assert result["uncertainty"] == "UNCALIBRATED_CLASS_PROBABILITIES"
    source["tone"]["label"] = "negative"
    assert result["outputs"]["tone"]["label"] == "positive"
    assert make()["result_sha256"] == result["result_sha256"]


@pytest.mark.parametrize("p", [True, float("nan"), float("inf"), -0.1, 1.1, 10**400])
def test_invalid_probabilities(p):
    x = inputs()
    x["tone"]["uncalibrated_probabilities"]["positive"] = p
    with pytest.raises(ContractError):
        make(x)


@pytest.mark.parametrize("change", ["empty", "missing", "extra", "label", "argmax", "sum", "foreign_field"])
def test_population_and_semantics(change):
    x = inputs()
    if change == "empty":
        x.clear()
    elif change == "missing":
        del x["tone"]["uncalibrated_probabilities"]["negative"]
    elif change == "extra":
        x["foreign"] = copy.deepcopy(x["tone"])
    elif change == "label":
        x["tone"]["label"] = "BUY"
    elif change == "argmax":
        x["tone"]["label"] = "negative"
    elif change == "sum":
        x["tone"]["uncalibrated_probabilities"]["positive"] = 0.5
    else:
        x["tone"]["calibrated"] = True
    with pytest.raises(ContractError):
        make(x)


@pytest.mark.parametrize("labels", [{}, {"tone": "positive"}, {"tone": ["positive", "positive"]}, {"tone": [True, "negative"]}])
def test_invalid_task_schema(labels):
    with pytest.raises(ContractError):
        make(expected_labels=labels)


@pytest.mark.parametrize("field", ["input_sha256", "model_sha256", "task_sha256"])
def test_identity_required(field):
    with pytest.raises(ContractError):
        make(**{field: "unknown"})


def test_declared_rounding_not_silent_normalization():
    outputs = {"q": {"label": "a", "uncalibrated_probabilities": {"a": 0.3333, "b": 0.3333, "c": 0.3333}}}
    with pytest.raises(ContractError):
        make(outputs, expected_labels={"q": ["a", "b", "c"]})
    result = make(outputs, expected_labels={"q": ["a", "b", "c"]}, probability_decimals=4)
    assert result["outputs"] == outputs
    with pytest.raises(ContractError):
        make(probability_decimals=True)
