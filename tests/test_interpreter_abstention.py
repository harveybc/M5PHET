"""The measured rule, applied where a language model chooses: in configuration, in the catalog, in the interpreter.

WP09 measured one thing about this product's language models that nobody had measured before: below 0.8 confidence
the classification checkpoint's argmax is at chance (0.327 correct against a 0.333 chance rate), at or above it 48 of
55 rows were correct. WP20 turned that into a gate on `m5phet.decide`. What these tests pin is the three ways that gate
was still escapable, and are not any more.

It was an argument. `ask(..., min_confidence=0.8)` gated the one call whose author had read the report; the dataset
chooser and the pipeline's questions asked the same checkpoint with no gate. It is now read from the installation's
configuration, so a caller who passes nothing gets the rule rather than no rule.

It was invisible. A consumer -- the web, an MCP client, a Telegram skill -- could not see whether an answer had been
produced under a threshold or under none. The catalog now carries it, with the report's digest, stage, protocol and
seal, or `NOT_CONFIGURED`.

It did not apply to the interpreter at all. `m5phet.interpret` accepted whatever the model picked among the declared
values with no confidence whatsoever. It now applies -- and where a plugin CANNOT report a confidence, that is said
(`CONFIDENCE_NOT_REPORTED`) and the question is refused, because the one thing that must never happen is a value
chosen by a model passing a gate that was never applied to it.

None of this claims abstaining is wise. That is WP09's claim. Here the question is only whether the rule is the
installation's, whether it is published, and whether it is applied or honestly declared inapplicable.
"""

import json

import pytest

from m5phet import config as configuration_module, decide
from m5phet.interpret import (CONFIDENCE_NOT_REPORTED, CONFIDENCE_REPORTED, STATUS_CONFIDENCE_NOT_REPORTED,
                              STATUS_LOW_CONFIDENCE, STATUS_MISSING, STATUS_OK, Interpreter, interpret)
from m5phet.questions import catalog as question_catalog, refusal
from m5phet.runtime import Registry

#: the household bundle's own slots, the pair every sentence in `tools/verify_families.py` has to resolve
FORECAST = [{"name": "target", "allowed": ["Global_active_power", "Voltage"]},
            {"name": "horizon", "allowed": [60, 120], "type": "integer"}]

#: the real report's reliability bins (`~/.local/state/m5phet/wp09-quality-20260925/report_laya_zero_shot.json`),
#: written to a file the tests own so the suite measures the same numbers on a machine that never ran WP09
MEASURED_BINS = [(0.0, 0.1, 0, None), (0.1, 0.2, 0, None), (0.2, 0.3, 0, None),
                 (0.3, 0.4, 134, 0.3208955223880597), (0.4, 0.5, 152, 0.35526315789473684),
                 (0.5, 0.6, 58, 0.29310344827586204), (0.6, 0.7, 22, 0.2727272727272727),
                 (0.7, 0.8, 29, 0.3103448275862069), (0.8, 0.9, 24, 0.7083333333333334),
                 (0.9, 1.0, 31, 1.0)]


def wp09_report(path):
    path.write_text(json.dumps(
        {"version": "m5phet-evaluation-report/1", "stage": "laya_zero_shot", "family": "classification",
         "protocol_digest": "b9aefb3c" + "0" * 56, "corpus_seal": "31257d47" + "0" * 56,
         "metric_sets": [{"name": "calibration",
                          "values": {"reliability": {"bin_count": len(MEASURED_BINS),
                                                     "bins": [{"bin": [low, high], "count": count,
                                                               "accuracy": accuracy}
                                                              for low, high, count, accuracy in MEASURED_BINS]}}}]}),
        encoding="utf-8")
    return path


def configured(tmp_path, monkeypatch, interpreter=None, **rule):
    """An installation whose `m5phet.json` declares `rule`, and nothing else. Never the operator's own file."""
    document = {"schema": "m5phet.config.v1", "interpreter": {"plugin": "command", **(interpreter or {}), **rule}}
    path = tmp_path / "m5phet.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setenv(configuration_module.PATH_VARIABLE, str(path))
    monkeypatch.delenv(decide.MIN_CONFIDENCE_VARIABLE, raising=False)
    monkeypatch.delenv(decide.ABSTENTION_SOURCE_VARIABLE, raising=False)
    return configuration_module.load()


@pytest.fixture
def source(tmp_path):
    return wp09_report(tmp_path / "report_laya_zero_shot.json")


@pytest.fixture(autouse=True)
def _no_inherited_rule(monkeypatch, tmp_path):
    """No test here reads the operator's own configuration, and an installation with none declares no rule."""
    monkeypatch.setenv(configuration_module.PATH_VARIABLE, str(tmp_path / "absent.json"))
    monkeypatch.delenv(decide.MIN_CONFIDENCE_VARIABLE, raising=False)
    monkeypatch.delenv(decide.ABSTENTION_SOURCE_VARIABLE, raising=False)


# --- the two interpreters: one that can be held to the rule, one that cannot --------------------------------------------

class Confident(Interpreter):
    """A plugin that CAN say how sure it was -- the shape `openai_compatible` reads back out of its logprobs."""

    reports_confidence = True

    def __init__(self, answer, confidences):
        super().__init__(command="fixture", model="reports-confidence-v1", environ={})
        self.answer, self.confidences, self.asked = dict(answer), dict(confidences), []

    @property
    def available(self):
        return True

    def propose_with_confidence(self, prompt, slots):
        self.asked.append(prompt)
        return dict(self.answer), dict(self.confidences)


class Silent(Interpreter):
    """A plugin that cannot: it prints an answer, like `command` and `ollama`, and no probability comes with it."""

    def __init__(self, answer):
        super().__init__(command="fixture", model="text-only-v1", environ={})
        self.answer, self.asked = dict(answer), []

    @property
    def available(self):
        return True

    def propose(self, prompt, slots):
        self.asked.append(prompt)
        return dict(self.answer)


SURE = {"target": 0.97, "horizon": 0.93}
UNSURE = {"target": 0.41, "horizon": 0.93}


# --- 1. the rule is the installation's, not an argument -----------------------------------------------------------------

def test_the_threshold_comes_from_the_configuration_and_gates_a_model_that_reports_one(tmp_path, monkeypatch, source):
    model = Confident({"target": "Global_active_power"}, UNSURE)
    configured(tmp_path, monkeypatch, min_confidence=0.8, abstention_source=str(source))
    report = interpret("predict household power one hour ahead", FORECAST, interpreter=model)
    assert report["status"] == STATUS_LOW_CONFIDENCE
    assert report["parameters"].get("target") is None, "a value below the threshold is not kept as if it were chosen"
    assert model.asked, "the model WAS asked; what is refused is its choice, not the question"


def test_above_the_configured_threshold_the_same_model_resolves_normally(tmp_path, monkeypatch, source):
    model = Confident({"target": "Global_active_power"}, SURE)
    configured(tmp_path, monkeypatch, min_confidence=0.8, abstention_source=str(source))
    report = interpret("predict household power at 60 steps", FORECAST, interpreter=model)
    assert report["status"] == STATUS_OK
    assert report["parameters"] == {"horizon": 60, "target": "Global_active_power"}
    assert report["sources"]["target"] == "INTERPRETER"
    assert report["confidences"] == {"target": 0.97}
    assert report["abstention"]["min_confidence"] == 0.8


def test_the_environment_declares_the_same_rule_as_the_file(tmp_path, monkeypatch, source):
    """The operator who never writes a JSON file is not outside the rule: `m5phet.config` exports it into these."""
    monkeypatch.setenv(decide.MIN_CONFIDENCE_VARIABLE, "0.8")
    monkeypatch.setenv(decide.ABSTENTION_SOURCE_VARIABLE, str(source))
    citation, unresolved = decide.declared_threshold()
    assert unresolved is None and citation["min_confidence"] == 0.8
    assert citation["measured_rows_at_or_above"] == 55 and citation["measured_correct_at_or_above"] == 48.0


def test_a_caller_that_passes_the_rule_explicitly_still_wins(tmp_path, monkeypatch, source):
    """Configuration is the default, not an override: WP20's own call sites keep meaning what they say."""
    configured(tmp_path, monkeypatch, min_confidence=0.9, abstention_source=str(source))
    citation, _ = decide.abstention_threshold(0.8, str(source))
    assert citation["min_confidence"] == 0.8


# --- 2. configuration does not excuse an invented number ----------------------------------------------------------------

def test_a_configured_threshold_with_no_citation_refuses_and_asks_nothing(tmp_path, monkeypatch):
    model = Confident({"target": "Global_active_power"}, SURE)
    configured(tmp_path, monkeypatch, min_confidence=0.8)
    report = interpret("predict household power one hour ahead", FORECAST, interpreter=model)
    assert report["status"] == decide.UNCITED_THRESHOLD
    assert model.asked == [], "a threshold nobody measured costs no model call"
    assert "Global_active_power" in report["why"], "the person is still told what this model has"


def test_a_configured_threshold_the_report_never_resolved_is_refused_by_its_own_name(tmp_path, monkeypatch, source):
    model = Confident({"target": "Global_active_power"}, SURE)
    configured(tmp_path, monkeypatch, min_confidence=0.85, abstention_source=str(source))
    report = interpret("predict household power one hour ahead", FORECAST, interpreter=model)
    assert report["status"] == decide.THRESHOLD_NOT_MEASURED
    assert model.asked == []


def test_a_citation_with_no_threshold_gates_nothing_and_says_so(tmp_path, monkeypatch, source):
    configured(tmp_path, monkeypatch, abstention_source=str(source))
    rule = decide.declared_rule()
    assert rule["refusal"] == decide.ABSTENTION_SOURCE_WITHOUT_THRESHOLD


def test_a_configured_rule_reaches_decide_ask_without_the_caller_passing_it(tmp_path, monkeypatch):
    """The chooser's half of the same declaration: `decide.ask` reads it, so every call site is gated, not one."""
    configured(tmp_path, monkeypatch, min_confidence=0.8)
    out = decide.ask(Registry(), "a state text", {"transform": {"options": [["level", "keep it"], ["diff", "difference"]],
                                                               "instructions": "which transform?"}},
                     kind="feature_profile")
    assert out["transform"]["refusal"] == decide.UNCITED_THRESHOLD


# --- 3. a plugin that cannot report a confidence is declared, never faked ------------------------------------------------

def test_a_plugin_that_cannot_report_a_confidence_is_refused_rather_than_silently_passed(tmp_path, monkeypatch, source):
    model = Silent({"target": "Global_active_power"})
    configured(tmp_path, monkeypatch, min_confidence=0.8, abstention_source=str(source))
    report = interpret("predict household power one hour ahead", FORECAST, interpreter=model)
    assert report["status"] == STATUS_CONFIDENCE_NOT_REPORTED
    assert model.asked == [], "nothing is chosen by a model that cannot be held to the declared rule"
    assert "reports no confidence" in report["why"]
    assert "Global_active_power" in report["why"], "the refusal says what the person can name instead"
    assert report["declared"]["target"] == ["Global_active_power", "Voltage"]


def test_the_same_plugin_with_no_rule_declared_behaves_exactly_as_before():
    model = Silent({"target": "Global_active_power"})
    report = interpret("predict household power at 60 steps", FORECAST, interpreter=model)
    assert report["status"] == STATUS_OK
    assert report["parameters"] == {"horizon": 60, "target": "Global_active_power"}
    assert "confidences" not in report and "abstention" not in report
    assert model.asked, "with no rule declared the interpreter is consulted exactly as it always was"


def test_a_model_that_reports_a_confidence_for_one_field_and_not_the_other_refuses_on_the_unmeasured_one(
        tmp_path, monkeypatch, source):
    """"Not measured" is not "measured high". A field with no confidence is refused by its own name, not passed."""
    model = Confident({"target": "Global_active_power"}, {"horizon": 0.99})
    configured(tmp_path, monkeypatch, min_confidence=0.8, abstention_source=str(source))
    report = interpret("predict household power one hour ahead", FORECAST, interpreter=model)
    assert report["status"] == STATUS_CONFIDENCE_NOT_REPORTED
    assert report["unresolved"] == ["target"]


def test_no_interpreter_at_all_is_still_the_old_missing_parameter_refusal(tmp_path, monkeypatch, source):
    """The rule is about a model's choices. Where there is no model there is no choice to gate."""
    class Absent(Interpreter):
        def __init__(self):
            super().__init__(command="", model=None, environ={})
    configured(tmp_path, monkeypatch, min_confidence=0.8, abstention_source=str(source))
    report = interpret("predict household power one hour ahead", FORECAST, interpreter=Absent())
    assert report["status"] == STATUS_MISSING


def test_the_words_alone_are_never_gated(tmp_path, monkeypatch, source):
    """A parameter the person wrote is not a model's choice, and a threshold about a model says nothing about it."""
    model = Silent({})
    configured(tmp_path, monkeypatch, min_confidence=0.8, abstention_source=str(source))
    report = interpret("forecast Global_active_power at 60 steps", FORECAST, interpreter=model)
    assert report["status"] == STATUS_OK
    assert set(report["sources"].values()) == {"QUESTION_TEXT"}


# --- 4. the catalog declares it -----------------------------------------------------------------------------------------

def test_the_catalog_carries_the_rule_for_every_areas_chooser(tmp_path, monkeypatch, source):
    configuration = configured(tmp_path, monkeypatch, min_confidence=0.8, abstention_source=str(source))
    areas = question_catalog(Registry(), configuration)
    for area, entry in areas.items():
        chooser = entry["chooser"]
        assert chooser["min_confidence"] == 0.8, area
        assert chooser["source"]["report_sha256"] and chooser["source"]["stage"] == "laya_zero_shot"
        assert chooser["source"]["protocol"] and chooser["source"]["seal"]
        assert [b["bin"] for b in chooser["source"]["bins_at_or_above"]] == [[0.8, 0.9], [0.9, 1.0]]
        assert "path" not in json.dumps(chooser), "the catalog names the measurement, never where the file sits here"


def test_an_installation_that_declares_nothing_says_NOT_CONFIGURED():
    areas = question_catalog(Registry())
    assert {entry["chooser"] for entry in areas.values()} == {decide.NOT_CONFIGURED}


def test_a_declaration_that_does_not_resolve_is_published_as_its_refusal(tmp_path, monkeypatch):
    configuration = configured(tmp_path, monkeypatch, min_confidence=0.8)
    areas = question_catalog(Registry(), configuration)
    assert areas["forecasting"]["chooser"]["refusal"] == decide.UNCITED_THRESHOLD
    assert "preference" in areas["forecasting"]["chooser"]["why"]


def test_the_web_catalog_says_the_rule_and_whether_the_interpreter_can_be_held_to_it(tmp_path, monkeypatch, source):
    pytest.importorskip("fastapi")
    from m5phet.web.engine import Engine
    configuration = configured(tmp_path, monkeypatch, min_confidence=0.8, abstention_source=str(source))
    engine = Engine(registry=Registry(), configuration=configuration, environ={})
    engine.interpreter = Silent({})
    published = engine.catalog()["abstention"]
    assert published["interpreter"]["rule"]["min_confidence"] == 0.8
    assert published["interpreter"]["confidence"] == CONFIDENCE_NOT_REPORTED
    assert "refused as CONFIDENCE_NOT_REPORTED" in published["interpreter"]["why"]
    assert published["areas"]["classification"]["source"]["stage"] == "laya_zero_shot"

    engine.interpreter = Confident({}, {})
    assert engine.catalog()["abstention"]["interpreter"]["confidence"] == CONFIDENCE_REPORTED


def test_the_web_catalog_of_an_unconfigured_installation_says_NOT_CONFIGURED(tmp_path):
    pytest.importorskip("fastapi")
    from m5phet.web.engine import Engine
    engine = Engine(registry=Registry(), environ={})
    published = engine.catalog()["abstention"]
    assert published["interpreter"]["rule"] == decide.NOT_CONFIGURED
    assert set(published["areas"].values()) == {decide.NOT_CONFIGURED}


# --- 5. the person's path stays usable ----------------------------------------------------------------------------------

class SlottedProvider:
    """A provider with a vocabulary, so a sentence takes the path `Engine.execute` builds a typed request on."""

    name, area = "slotted_forecaster", "forecasting"

    def capabilities(self):
        return {"provider": self.name, "operations": ["infer"], "families": ["regression_forecasting"],
                "output_kinds": ["point_forecast"], "uncertainty_methods": ["none"],
                "supported": [{"operation": "infer", "family": "regression_forecasting",
                               "output_kind": "point_forecast"}],
                "known_states": ["s1"]}

    def question_types(self):
        return {"point_forecast": {"required": ["horizon"]}}

    def chat_slots(self):
        return [dict(slot) for slot in FORECAST]

    def chat_request(self, prompt, context, config, parameters=None):
        return {"operation": "infer", "provider_ref": self.name, "family": config["family"],
                "output_kind": config["output_kind"], "inputs": dict(parameters or {})}

    def answer_questions(self, state, questions, data, as_of):
        return {name: refusal("NOT_ESTIMABLE", "nothing is fitted here", q["type"]) for name, q in questions.items()}


def slotted_engine(tmp_path, monkeypatch, interpreter, **rule):
    pytest.importorskip("fastapi")
    from m5phet.web.engine import Engine
    configuration = configured(tmp_path, monkeypatch, **rule)
    registry = Registry()
    registry.register(SlottedProvider())
    engine = Engine(registry=registry, configuration=configuration, environ={})
    engine.interpreter = interpreter
    return engine


SENTENCE_CONFIG = {"provider": "slotted_forecaster", "family": "regression_forecasting",
                   "output_kind": "point_forecast", "input": "text", "context": "unused"}


def test_the_review_window_shows_a_low_confidence_refusal_as_a_refusal_to_choose(tmp_path, monkeypatch, source):
    engine = slotted_engine(tmp_path, monkeypatch, Confident({"target": "Global_active_power"}, UNSURE),
                            min_confidence=0.8, abstention_source=str(source))
    preview = engine.execute("predict household power one hour ahead", SENTENCE_CONFIG, [], dry_run=True)
    assert preview["status"] == "REFUSED" and preview["refusal"] == STATUS_LOW_CONFIDENCE
    assert preview["ran"] is False and preview["request"] is None and preview["execution_authorized"] is False
    assert preview["unresolved"] == ["target"], "the person is told WHICH parameter went unresolved"
    assert preview["declared"] == {"target": ["Global_active_power", "Voltage"]}
    assert "0.8" in preview["why"] and "0.41" in preview["why"]


def test_the_review_window_says_when_the_rule_could_not_be_applied_at_all(tmp_path, monkeypatch, source):
    engine = slotted_engine(tmp_path, monkeypatch, Silent({"target": "Global_active_power"}),
                            min_confidence=0.8, abstention_source=str(source))
    preview = engine.execute("predict household power one hour ahead", SENTENCE_CONFIG, [], dry_run=True)
    assert preview["refusal"] == STATUS_CONFIDENCE_NOT_REPORTED
    assert preview["declared"]["horizon"] == [60, 120]


def test_running_the_same_sentence_refuses_by_name_rather_than_answering_something_else(tmp_path, monkeypatch, source):
    engine = slotted_engine(tmp_path, monkeypatch, Confident({"target": "Global_active_power"}, UNSURE),
                            min_confidence=0.8, abstention_source=str(source))
    with pytest.raises(ValueError, match=STATUS_LOW_CONFIDENCE):
        engine.execute("predict household power one hour ahead", SENTENCE_CONFIG, [])


def test_with_nothing_configured_the_same_sentence_resolves_as_it_always_did(tmp_path, monkeypatch):
    engine = slotted_engine(tmp_path, monkeypatch, Silent({"target": "Global_active_power", "horizon": 60}))
    preview = engine.execute("predict household power one hour ahead", SENTENCE_CONFIG, [], dry_run=True)
    assert preview["ran"] is False and preview["request"]["inputs"]["target"] == "Global_active_power"
    assert preview["interpretation"]["status"] == STATUS_OK


# --- 6. the interpreter's own reliability, cited or NOT_MEASURED --------------------------------------------------------

from m5phet.interpret import (NOT_MEASURED, RELIABILITY_MEASURED_ON_ANOTHER_INTERPRETER,  # noqa: E402
                              RELIABILITY_REPORT_UNREADABLE, declared_reliability)


def reliability_report(path, *, plugin="command", model="text-only-v1"):
    """A `m5phet_interpreter_reliability.v1` document of the shape `tools/measure_interpreter.py` writes."""
    path.write_text(json.dumps(
        {"schema": "m5phet_interpreter_reliability.v1", "measured_at": "2026-09-25T12:30:00+00:00",
         "interpreter": {"plugin": plugin, "model": model},
         "interpreter_path": {
             "interpreter": {"plugin": plugin, "model": model}, "runs_per_sentence": 5,
             "protocol": "Interpreter.propose(sentence, slots) N times per sentence",
             "summary": {"sentences": 12, "runs": 60, "matched": 50, "reliability": 0.8333,
                         "reliability_when_it_chose": 0.9434,
                         "verdicts": {"CORRECT": 50, "WRONG_VALUE": 3, "DECLINED": 7,
                                      "OUTSIDE_DECLARED": 0, "INTERPRETER_FAILED": 0}}}}),
        encoding="utf-8")
    return path


def test_an_installation_that_measured_nothing_says_NOT_MEASURED():
    assert declared_reliability() == NOT_MEASURED


def test_a_cited_reliability_is_published_with_its_protocol_and_its_n(tmp_path, monkeypatch):
    report = reliability_report(tmp_path / "interpreter_reliability.json")
    configured(tmp_path, monkeypatch, reliability_report=str(report))
    model = Silent({})
    published = declared_reliability(interpreter=model)
    assert published["reliability"] == 0.8333 and published["reliability_when_it_chose"] == 0.9434
    assert published["n"] == {"sentences": 12, "runs": 60, "runs_per_sentence": 5}
    assert published["protocol"] and published["report_sha256"]
    assert published["measured_interpreter"] == {"plugin": "command", "model": "text-only-v1"}


def test_a_reliability_measured_on_another_model_is_refused_not_published(tmp_path, monkeypatch):
    """A rate measured on llama3.2:3b is not a fact about deepseek-v4-flash, and would read as one beside it."""
    report = reliability_report(tmp_path / "interpreter_reliability.json", model="some-other-model")
    configured(tmp_path, monkeypatch, reliability_report=str(report))
    published = declared_reliability(interpreter=Silent({}))
    assert published["refusal"] == RELIABILITY_MEASURED_ON_ANOTHER_INTERPRETER
    assert "some-other-model" in published["why"]


def test_a_report_that_measured_only_the_product_path_publishes_no_reliability(tmp_path, monkeypatch):
    """The deterministic pass resolving a sentence is not evidence about the model; a report of only that is refused."""
    path = tmp_path / "interpreter_reliability.json"
    path.write_text(json.dumps({"schema": "m5phet_interpreter_reliability.v1",
                                "summary": {"deterministic_runs": 60, "deterministic_matched": 60}}), encoding="utf-8")
    configured(tmp_path, monkeypatch, reliability_report=str(path))
    assert declared_reliability()["refusal"] == RELIABILITY_REPORT_UNREADABLE


def test_the_web_catalog_carries_the_reliability_beside_the_interpreters_identity(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from m5phet.web.engine import Engine
    report = reliability_report(tmp_path / "interpreter_reliability.json")
    configuration = configured(tmp_path, monkeypatch, reliability_report=str(report))
    engine = Engine(registry=Registry(), configuration=configuration, environ={})
    engine.interpreter = Silent({})
    assert engine.catalog()["interpreter"]["reliability"]["reliability"] == 0.8333

    # an installation that declared nothing -- neither the file nor the variable the file is exported into
    bare = Engine(registry=Registry(), environ={})
    bare.interpreter = Silent({})
    assert bare.catalog()["interpreter"]["reliability"] == NOT_MEASURED
