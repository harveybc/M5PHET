"""Design consistency only; these tests do not execute planned ML adapters."""

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "specs/use_cases.json").read_text())
CASES = {case["id"]: case for case in SPEC["use_cases"]}


def test_exact_task_families_and_use_case_population():
    assert SPEC["status"] == "DESIGN_ONLY"
    assert set(SPEC["families"]) == {
        "classification", "regression_forecasting", "representation_unsupervised",
        "reinforcement_learning", "causal_inference",
    }
    assert len(CASES) == len(SPEC["use_cases"]) == 6
    assert set(CASES) == {f"UC-{n:02}" for n in range(1, 7)}
    assert {c["family"] for c in CASES.values()} == set(SPEC["families"]) | {"data_transform"}


@pytest.mark.parametrize("case", SPEC["use_cases"], ids=lambda c: c["id"])
def test_each_use_case_has_consumer_requirements_outputs_and_refusals(case):
    assert case["consumer"]
    for key in ("inputs", "outputs", "requirements", "refusals", "evaluation"):
        assert isinstance(case[key], list) and case[key]
        assert len(case[key]) == len(set(case[key]))
        assert all(isinstance(v, str) and v for v in case[key])
    assert set(case["depends_on"]) <= set(CASES) - {case["id"]}
    plan = (ROOT / "docs/IMPLEMENTATION_PLAN.md").read_text()
    assert all(f"| {req} " in plan for req in case["requirements"])
    assert f"## {case['id']}:" in (ROOT / "docs/USE_CASES.md").read_text()
    expected = "CHOICE_RESULT_CONTRACT_ONLY" if case["id"] == "UC-01" else "NOT_IMPLEMENTED"
    assert case["implementation"] == expected


def test_calendar_is_shared_data_transform_not_sixth_ml_engine():
    assert CASES["UC-02"]["family"] == "data_transform"
    calendar = (ROOT / "docs/ECONOMIC_CALENDAR.md").read_text()
    assert CASES["UC-02"]["evaluation"] == [f"CAL{n:02}" for n in range(1, 13)]
    assert all(f"| {rule} |" in calendar for rule in CASES["UC-02"]["evaluation"])


def test_calendar_dependencies_do_not_require_news_or_forecasting_winner():
    resolved = set()
    while len(resolved) != len(CASES):
        ready = {name for name, c in CASES.items() if set(c["depends_on"]) <= resolved} - resolved
        assert ready, "cycle in use-case prerequisites"
        resolved |= ready
    assert CASES["UC-02"]["depends_on"] == []
    assert CASES["UC-05"]["depends_on"] == ["UC-02"]


def test_method_state_does_not_promote_design_tests_to_model_evidence():
    state = json.loads((ROOT / "PROJECT_METHOD_STATE.json").read_text())
    assert state["implemented"] == ["classification_result_contract"]
    assert state["model_performance_measured"] is False
    assert state["real_capital_authorized"] is False
    assert state["product_design"]["behavioral_acceptance"] == "PENDING_IMPLEMENTATION"
