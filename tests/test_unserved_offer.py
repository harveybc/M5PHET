"""AP01: a family the product cannot serve is never OFFERED, on any door.

The hole these tests exist for was recorded by CB05 in its own §9 and then measured on 2026-09-29 against the staged
build, on its own port and its own state directory (`docs/evidence/AP01_RESTART_2026_09_29/before-gate.json`):

    GET /api/catalog        classification_backend: {"status": "CLASSIFICATION_CHECKPOINT_MISMATCH",
                                                     "validated": false, ...}
    GET /api/tasks/catalog  classification: {"provider": "laya_news", "question_types": ["choice"]}

Nothing answered wrongly -- every run of that area refused by name, and the pinned checkpoint is why. What was wrong
is one step earlier: the product offered `choice` on a family it could not serve, and it went on offering it after a
restart, so it was not a transient either. An offer is a promise, and this is the promise withdrawn.

The rules asserted here:

* an area named as unserved offers no question types, no parameters, no aliases, no combinations and no confidence
  levels, and carries its refusal's own CODE on its face;
* it keeps the name of the provider the configuration declares, because hiding that would make the configuration
  unreadable from the outside;
* `Engine.unserved_areas()` reads the finding from the EFFECTIVE backend block -- the path that would answer -- and
  is empty whenever that block is validated, so a served installation is not gated by accident;
* a hand-written proposal naming an unserved area is refused by the name its contract gave it, not as "that type is
  not one this area answers ([])", which would blame the question for the configuration;
* the web door and the MCP door offer the same thing.
"""
import json

import pytest

from m5phet.classification_backend import CHECKPOINT_MISMATCH
from m5phet.orchestrate import check_proposal
from m5phet.questions import UNAVAILABLE_READING, catalog as question_catalog, refusal, unserved_area
from m5phet.runtime import Registry

REFUSAL = {"code": CHECKPOINT_MISMATCH, "why": "the pin names one checkpoint and the path serves another",
           "mode": "remote_worker"}


class Classifier:
    """The shape of `laya_news` and nothing of its substance: it declares `choice` and is never asked anything here."""
    name, area = "laya_news", "classification"

    def capabilities(self):
        return {"provider": self.name, "operations": ["infer"], "families": ["classification"],
                "output_kinds": ["typed_questions"], "uncertainty_methods": ["UNCALIBRATED_CLASS_PROBABILITIES"],
                "supported": [{"operation": "infer", "family": "classification", "output_kind": "typed_questions"}],
                "known_states": ["laya-checkpoint:" + "a" * 64], "backend": "laya", "weights_present": True}

    def question_types(self):
        return {"choice": {"required": ["options"], "optional": ["instructions"]}}

    def answer_questions(self, state, questions, data, as_of):           # pragma: no cover - never reached here
        raise AssertionError("an unserved area must not be asked anything")


class Forecaster:
    name, area = "fake_forecaster", "forecasting"

    def capabilities(self):
        return {"provider": self.name, "operations": ["infer"], "families": ["regression_forecasting"],
                "output_kinds": ["point_forecast"], "uncertainty_methods": ["none"],
                "supported": [{"operation": "infer", "family": "regression_forecasting",
                               "output_kind": "point_forecast"}], "known_states": ["s1"]}

    def question_types(self):
        return {"point_forecast": {"required": ["horizon"], "optional": ["target"]}}

    def answer_questions(self, state, questions, data, as_of):
        return {name: {"type": "point_forecast", "values": [0.5], "unit": "kW"} for name in questions}


def registry():
    bank = Registry()
    bank.register(Classifier())
    bank.register(Forecaster())
    return bank


# --- the catalog entry ---------------------------------------------------------------------------------------------

def test_an_unserved_area_offers_nothing_and_names_its_refusal():
    areas = question_catalog(registry(), unserved={"classification": REFUSAL})
    entry = areas["classification"]
    assert entry["question_types"] == {}
    assert entry["parameters"] == {} and entry["aliases"] == {} and entry["combinations"] == []
    assert entry["confidence_levels"] == []
    assert entry["unavailable"]["code"] == CHECKPOINT_MISMATCH
    assert entry["unavailable"]["mode"] == "remote_worker"
    assert entry["reading"] == UNAVAILABLE_READING


def test_an_unserved_area_still_names_the_declared_provider():
    areas = question_catalog(registry(), unserved={"classification": REFUSAL})
    assert areas["classification"]["provider"] == "laya_news"


def test_an_unserved_area_claims_no_measured_quality():
    areas = question_catalog(registry(), unserved={"classification": REFUSAL})
    assert areas["classification"]["quality"] == "NOT_MEASURED"


def test_the_other_areas_are_untouched_by_one_areas_refusal():
    gated = question_catalog(registry(), unserved={"classification": REFUSAL})
    plain = question_catalog(registry())
    assert gated["forecasting"] == plain["forecasting"]
    assert sorted(gated) == sorted(plain)                          # an unserved area is published, never dropped


def test_nothing_is_gated_without_a_finding():
    for empty in (None, {}, {"classification": None}):
        areas = question_catalog(registry(), unserved=empty)
        assert sorted(areas["classification"]["question_types"]) == ["choice"]
        assert "unavailable" not in areas["classification"]


def test_unserved_area_invents_no_reason():
    entry = unserved_area({"provider": "laya_news", "question_types": {"choice": {}}}, {"code": "X"})
    assert entry["unavailable"] == {"code": "X", "why": None, "mode": None}


# --- what the engine passes in -------------------------------------------------------------------------------------

class FakeEffective(dict):
    pass


def engine_with(effective, monkeypatch):
    pytest.importorskip("fastapi")
    from m5phet.web.engine import Engine
    engine = Engine(registry=registry())
    monkeypatch.setattr(engine, "classification_effective", lambda: effective)
    return engine


def test_the_engine_reports_a_refused_backend_as_an_unserved_area(monkeypatch):
    engine = engine_with({"status": CHECKPOINT_MISMATCH, "validated": False, "why": "another checkpoint",
                          "mode": "remote_worker"}, monkeypatch)
    assert engine.unserved_areas() == {"classification": {"code": CHECKPOINT_MISMATCH, "why": "another checkpoint",
                                                          "mode": "remote_worker"}}
    assert engine.task_catalog()["classification"]["question_types"] == {}


@pytest.mark.parametrize("effective", [None, {"status": "VALIDATED", "validated": True, "mode": "fixture"}])
def test_a_served_installation_is_not_gated(effective, monkeypatch):
    engine = engine_with(effective, monkeypatch)
    assert engine.unserved_areas() == {}
    assert sorted(engine.task_catalog()["classification"]["question_types"]) == ["choice"]


# --- the proposal door ---------------------------------------------------------------------------------------------

PROPOSAL = {"area": "classification", "state": {"asset": "EURUSD", "language": "en"},
            "questions": {"economia": {"type": "choice", "options": [["euro_area", "Euro area"],
                                                                     ["other", "Another economy"]]}}}


def test_a_proposal_for_an_unserved_area_is_refused_by_the_contracts_own_name():
    areas = question_catalog(registry(), unserved={"classification": REFUSAL})
    task, problems = check_proposal(PROPOSAL, areas, {"kind": "text", "columns": [], "rows": 1})
    assert task is None
    assert len(problems) == 1
    assert CHECKPOINT_MISMATCH in problems[0]
    assert "is not one this area answers" not in problems[0]


def test_the_same_proposal_passes_when_the_area_is_served():
    task, problems = check_proposal(PROPOSAL, question_catalog(registry()),
                                    {"kind": "text", "columns": [], "rows": 1})
    assert problems == [] and task is not None


# --- the MCP door --------------------------------------------------------------------------------------------------

def test_the_mcp_catalog_is_the_gated_one_when_it_has_an_engine(monkeypatch):
    from m5phet.mcp_server import Server
    engine = engine_with({"status": CHECKPOINT_MISMATCH, "validated": False, "why": "another checkpoint",
                          "mode": "remote_worker"}, monkeypatch)
    payload = json.loads(Server(registry(), engine=engine).call("m5phet_catalog", {})["content"][0]["text"])
    assert payload["areas"]["classification"]["question_types"] == {}
    assert payload["areas"]["classification"]["unavailable"]["code"] == CHECKPOINT_MISMATCH


def test_the_mcp_catalog_without_an_engine_has_no_contract_to_consult_and_says_nothing_new():
    from m5phet.mcp_server import Server
    payload = json.loads(Server(registry()).call("m5phet_catalog", {})["content"][0]["text"])
    assert sorted(payload["areas"]["classification"]["question_types"]) == ["choice"]


# --- and the refusal is not a number ------------------------------------------------------------------------------

def test_no_answer_shape_is_produced_for_an_unserved_area():
    """The gate withdraws an offer. It must never fabricate an answer, not even an empty one."""
    areas = question_catalog(registry(), unserved={"classification": REFUSAL})
    text = json.dumps(areas["classification"])
    assert "values" not in text and "probabilities" not in text
    assert refusal("X", "y", "choice")["status"] == "REFUSED"          # the refusal shape is still the answer shape


# --- and a measured number never travels with an answer no model gave ----------------------------------------------

RECORD = {"schema": "news_signal.quality.v1", "macro_f1": 0.37775954555995367, "n": 450,
          "calibration": {"expected_calibration_error": 0.13151177777777778, "brier": 0.6472929861999999,
                          "status": "UNCALIBRATED"},
          "skill": {"skill": 0.2533114546719444, "status": "OK", "error_definition": "error = 1 - macro_f1"},
          "naive": {"majority_class": 0.16666666666666666, "keyword_baseline": 0.17599601009211993,
                    "metric": "macro_f1", "same_rows": True},
          "corpus_id": "7e4789e5", "corpus_seal": "31257d47", "protocol_digest": "b9aefb3c",
          "label_provenance": "INDEPENDENT_LABELS", "reading": "macro-F1 on 450 sealed rows"}


def test_real_weights_keep_their_measured_record():
    from m5phet.quality import MEASURED, for_area
    block = for_area("classification", {"provider": "laya_news", "backend": "laya", "weights_present": True,
                                        "quality": RECORD})
    assert block["status"] == MEASURED
    assert block["values"]["macro_f1"] == RECORD["macro_f1"]


def test_a_declared_fixture_does_not_publish_a_models_measurement():
    """Measured on 2026-09-29: the fixture answered `other` with `euro_area 0.0` and the answer carried
    `MEASURED macro_f1 0.3778, n 450` -- a real checkpoint's retained record, beside an answer no model gave."""
    from m5phet.quality import NOT_MEASURED, QUALITY_RECORD_NOT_OF_THE_ANSWERING_PATH, for_area
    block = for_area("classification", {"provider": "laya_news", "backend": "fixture", "weights_present": False,
                                        "quality": RECORD})
    assert block["status"] == NOT_MEASURED
    assert block["quality_record_withheld"]["reason"] == QUALITY_RECORD_NOT_OF_THE_ANSWERING_PATH
    assert "values" not in block and "skill" not in block
    assert str(RECORD["macro_f1"]) not in json.dumps(block)          # named, not quoted
    assert block["quality_record_withheld"]["corpus_id"] == "7e4789e5"      # findable


def test_absent_weights_alone_are_enough_to_withhold_it():
    from m5phet.quality import NOT_MEASURED, for_area
    block = for_area("classification", {"provider": "laya_news", "weights_present": False, "quality": RECORD})
    assert block["status"] == NOT_MEASURED


def test_a_provider_that_declares_neither_is_not_contradicted():
    """This rule refuses to let a model's number travel with a declared non-model's answer. It does not GUESS what
    answered, so a provider that declares no backend and no weights flag keeps whatever it published."""
    from m5phet.quality import MEASURED, for_area
    block = for_area("classification", {"provider": "laya_news", "quality": RECORD})
    assert block["status"] == MEASURED


def test_nothing_is_invented_when_there_was_no_record():
    from m5phet.quality import NOT_MEASURED, for_area
    block = for_area("classification", {"provider": "laya_news", "backend": "fixture", "weights_present": False})
    assert block["status"] == NOT_MEASURED
    assert "quality_record_withheld" not in block
