"""The approved plugin architecture: specialized ML providers, optional language models.

Four acceptance criteria from the specification, plus the rule that makes them meaningful: support is a list of DECLARED,
TESTED combinations, never the Cartesian product of independent capability lists. A provider that can classify and can emit
quantiles has not thereby declared that it emits quantiles for a classification task.
"""

import copy

import pytest

from m5phet import ContractError
from m5phet.runtime import REQUEST_SCHEMA, Registry, Status, run


class Specialized:
    """A provider that declares exactly which combinations it supports and records what it was asked to do."""

    def __init__(self, name, supported, *, states=("state-1",), model_sha="m" * 64, task_id="task-1"):
        self.name = name
        self.calls = []
        self.model_sha = model_sha
        self._supported = [dict(s) for s in supported]
        self._states = list(states)
        self._task_id = task_id

    def capabilities(self):
        return {"provider": self.name,
                "operations": sorted({s["operation"] for s in self._supported}),
                "families": sorted({s["family"] for s in self._supported}),
                "output_kinds": sorted({s["output_kind"] for s in self._supported}),
                "uncertainty_methods": ["UNCALIBRATED_CLASS_PROBABILITIES"],
                "supported": copy.deepcopy(self._supported),
                "known_states": list(self._states)}

    def load(self, state_ref):
        self.calls.append(("load", state_ref))
        return {"state_ref": state_ref, "digest": "d" * 64, "model_sha256": self.model_sha, "task_id": self._task_id}

    def infer(self, request, state):
        self.calls.append(("infer", request["request_id"]))
        return {"outputs": {"tone": {"status": "OK", "payload": {"label": "neutral"},
                                     "uncertainty": "UNCALIBRATED_CLASS_PROBABILITIES"}}}

    def called(self, what):
        return [c for c in self.calls if c[0] == what]


CLASSIFY = {"operation": "infer", "family": "classification", "output_kind": "typed_questions"}
FORECAST = {"operation": "infer", "family": "regression_forecasting", "output_kind": "marginal_quantiles"}


def request(**over):
    base = {"schema_version": REQUEST_SCHEMA, "request_id": "r1", "task_id": "task-1", "operation": "infer",
            "family": "classification", "output_kind": "typed_questions", "as_of": "2026-09-24T00:00:00Z",
            "provider_ref": "local-classifier", "fitted_state_ref": "state-1",
            "output_schema": {"questions": ["tone"]}, "execution_constraints": {"partial_results": False}}
    base.update(over)
    return base


# --- support is declared, never a product ------------------------------------------------------------------------------------

def test_a_provider_must_declare_its_supported_combinations():
    class Vague(Specialized):
        def capabilities(self):
            caps = super().capabilities()
            caps.pop("supported")
            return caps

    with pytest.raises(ContractError) as exc:
        Registry().register(Vague("vague", [CLASSIFY]))
    assert "supported" in str(exc.value)


def test_the_cartesian_product_of_capability_lists_is_not_support():
    """The provider classifies typed questions and forecasts quantiles. It has NOT declared quantiles for classification."""
    registry = Registry()
    provider = Specialized("both", [CLASSIFY, FORECAST])
    registry.register(provider)
    crossed = run(request(provider_ref="both", family="classification", output_kind="marginal_quantiles"), registry)
    assert crossed["status"] == Status.UNSUPPORTED_TASK
    assert provider.called("load") == [], "an undeclared combination refuses before the model is loaded"
    assert "combination" in (crossed["why"] or "").lower()
    ok = run(request(provider_ref="both"), registry)
    assert ok["status"] == Status.OK


def test_an_undeclared_operation_for_a_declared_family_refuses_before_loading():
    registry = Registry()
    provider = Specialized("infer-only", [CLASSIFY])
    registry.register(provider)
    result = run(request(provider_ref="infer-only", operation="calibrate"), registry)
    assert result["status"] == Status.UNSUPPORTED_TASK
    assert provider.called("load") == []


# --- acceptance 1: local specialized inference with no language-model service ------------------------------------------------

def test_local_specialized_inference_works_with_no_language_model_installed(monkeypatch):
    """No LLM package, no key, no network: a specialized classifier still answers."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"laya", "openai", "transformers"}:
            raise ImportError(f"{name} is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for key in ("OPENAI_API_KEY", "LAYA_API_KEY", "M5PHET_LLM_ENDPOINT"):
        monkeypatch.delenv(key, raising=False)
    registry = Registry()
    registry.register(Specialized("local-classifier", [CLASSIFY]))
    result = run(request(), registry)
    assert result["status"] == Status.OK
    assert result["outputs"]["tone"]["payload"] == {"label": "neutral"}


# --- acceptance 2: the named provider is selected, with no hidden fallback -----------------------------------------------------

def test_an_unavailable_provider_does_not_silently_route_to_another_one():
    registry = Registry()
    working = Specialized("local-classifier", [CLASSIFY])
    registry.register(working)
    result = run(request(provider_ref="remote-laya"), registry)
    assert result["status"] == Status.UNSUPPORTED_TASK
    assert "remote-laya" in result["why"]
    assert working.called("infer") == [], "a named provider that is absent must not be replaced by whichever one is present"


def test_an_unavailable_remote_provider_does_not_block_the_working_local_one():
    registry = Registry()
    registry.register(Specialized("local-classifier", [CLASSIFY]))
    assert run(request(provider_ref="remote-laya"), registry)["status"] == Status.UNSUPPORTED_TASK
    assert run(request(provider_ref="local-classifier"), registry)["status"] == Status.OK


# --- acceptance 4: a provider swap keeps the task and population and records a changed model identity -------------------------

def test_a_provider_swap_preserves_the_task_and_records_a_changed_model_identity():
    registry = Registry()
    registry.register(Specialized("provider-a", [CLASSIFY], model_sha="a" * 64))
    registry.register(Specialized("provider-b", [CLASSIFY], model_sha="b" * 64))
    first = run(request(provider_ref="provider-a"), registry)
    second = run(request(provider_ref="provider-b"), registry)
    assert first["status"] == second["status"] == Status.OK
    assert first["binding"]["task_id"] == second["binding"]["task_id"] == "task-1"
    assert sorted(first["outputs"]) == sorted(second["outputs"]), "the evaluation population is the same questions"
    assert first["binding"]["model_sha256"] != second["binding"]["model_sha256"]
    assert first["binding"]["provider"] != second["binding"]["provider"]
    assert first["request_sha256"] != second["request_sha256"], "the request names its provider, so its identity moves"


# --- the optional language-model role is optional, and never an authority -----------------------------------------------------

def test_a_language_model_provider_is_registered_like_any_other_and_authorizes_nothing():
    registry = Registry()
    explainer = Specialized("llm-explainer", [{"operation": "infer", "family": "classification",
                                               "output_kind": "typed_questions"}])
    registry.register(explainer)
    result = run(request(provider_ref="llm-explainer"), registry)
    assert result["status"] == Status.OK
    assert result["execution_authorized"] is False
    assert result["facts"]["application_eligible"] is False
    assert registry.names() == ["llm-explainer"], "it is one provider in the same registry, not a parallel subsystem"


def test_a_language_model_cannot_answer_a_task_it_did_not_declare():
    registry = Registry()
    explainer = Specialized("llm-explainer", [{"operation": "infer", "family": "classification",
                                               "output_kind": "typed_questions"}])
    registry.register(explainer)
    result = run(request(provider_ref="llm-explainer", family="causal_inference", output_kind="effects"), registry)
    assert result["status"] == Status.UNSUPPORTED_TASK
    assert explainer.called("load") == []
