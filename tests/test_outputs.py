"""WP04: the output component -- the header an area declares, and the procedure that renders its answers.

What these tests pin. The header each area declares is the one its provider actually serves: the question types come
from `question_types()` and not from a list someone kept up to date by hand, so a provider that grows a type and a
header that does not are a failing test rather than a quiet lie in the catalog. The `telegram` procedure renders the
answer shapes the five engines really return -- the ones stored by `tools/verify_envelopes.py` -- with the numbers
exactly as they came back, refusals named with their code and reason, and the same closing sentence every time.

And the rule that outlives any particular plugin: the narration guard applies to procedures too. A plugin is not
trusted because it is installed. The last tests here install one that lies, in three different ways, and watch its
text be discarded in favour of the deterministic rendering.
"""

import json

import pytest

from m5phet.interpret import Interpreter
from m5phet.orchestrate import narrate, narration_problems, render
from m5phet.outputs import (AREA_HEADERS, OutputPluginError, header, load, names, narratable, response_view,
                            select)
from m5phet.outputs.default import DefaultOutput
from m5phet.outputs.telegram import CLOSING, LIMIT, TelegramOutput
from m5phet.questions import NOT_ESTIMABLE, STATE_REQUIRED, catalog as question_catalog, refusal
from m5phet.runtime import Registry


class Fixed(Interpreter):
    """A stand-in for the model, as the orchestration tests use."""

    def __init__(self, reply):
        super().__init__(command="fixture", model="fixture-v1", environ={})
        self.reply, self.asked = reply, []

    @property
    def available(self):
        return True

    def _ask(self, text):
        self.asked.append(text)
        return self.reply


SILENT = Interpreter(command="", environ={})


# --- the header is the provider's, not a list kept by hand ---------------------------------------------------------

#: exactly what the five installed providers return from `question_types()` (read from the running registry on
#: 2026-09-24). The fakes below declare these, so this file fails if a header and its provider ever disagree.
DECLARED = {
    "classification": {"choice": {"required": ["options"], "optional": ["instructions"]}},
    "forecasting": {"point_forecast": {"required": ["horizon"], "optional": ["target"]},
                    "interval": {"required": ["horizon", "confidence_level"], "optional": ["target"]},
                    "anomaly_risk": {"required": ["threshold"], "optional": ["horizon", "target"]}},
    "causal": {"ate": {"required": [], "optional": []},
               "cate": {"required": [], "optional": ["subgroup", "condition"]},
               # WP22 step 5: the event-study types the causal provider declares beside ate/cate
               "impulse_response": {"required": ["event", "outcome"], "optional": ["horizons"]},
               "sensitivity": {"required": ["events", "outcome", "window"], "optional": []},
               "counterfactual_path": {"required": ["window", "zero_out"], "optional": []}},
    "rl": {"next_action": {"required": [], "optional": []},
           "value_estimation": {"required": [], "optional": []}},
    "unsupervised": {"clustering": {"required": [], "optional": ["method", "expected_clusters", "level"]},
                     "cluster_description": {"required": ["target_metric"], "optional": ["level"]}},
}


def fake_provider(area):
    """A provider that declares one area and the question types the real one declares, and nothing else."""
    kind = AREA_HEADERS[area]["output_kind"]

    class Fake:
        name, _area = f"fake_{area}", area

        def capabilities(self):
            return {"provider": self.name, "operations": ["infer"], "families": [area],
                    "output_kinds": [kind], "uncertainty_methods": ["none"],
                    "supported": [{"operation": "infer", "family": area, "output_kind": kind}],
                    "known_states": ["s1"]}

        def question_types(self):
            return {k: {"required": list(v["required"]), "optional": list(v["optional"])}
                    for k, v in DECLARED[area].items()}

        def answer_questions(self, state, questions, data, as_of):
            return {n: {"type": q["type"]} for n, q in questions.items()}

    provider = Fake()
    provider.area = area
    return provider


@pytest.mark.parametrize("area", sorted(AREA_HEADERS))
def test_every_areas_header_declares_the_question_types_its_provider_answers(area):
    registry = Registry()
    registry.register(fake_provider(area))
    served = question_catalog(registry)[area]["question_types"]
    assert sorted(served) == header(area)["question_types"], f"{area}: the header and its provider disagree"
    assert set(served) == set(DECLARED[area])


@pytest.mark.parametrize("area", sorted(AREA_HEADERS))
def test_every_areas_header_names_its_output_kind_statuses_and_refusal_codes(area):
    from m5phet import questions as envelope
    declared = header(area)
    assert declared["area"] == area and declared["output_kind"]
    assert declared["statuses"] == ["OK", "REFUSED"], "an answer is answered or refused; there is no third thing"
    known = {getattr(envelope, n) for n in dir(envelope) if n.isupper() and isinstance(getattr(envelope, n), str)}
    assert declared["refusal_codes"] and set(declared["refusal_codes"]) <= known, \
        f"{area} declares a refusal code the envelope does not define"
    assert declared["unit_fields"], "an area that returns numbers says which fields carry them"
    assert declared["reading"]


def test_both_plugins_report_the_same_header_because_a_surface_does_not_change_what_an_engine_answers():
    for area in AREA_HEADERS:
        assert DefaultOutput().header(area) == TelegramOutput().header(area) == header(area)


def test_an_envelope_outside_the_five_areas_still_has_a_header():
    generic = header(None)
    assert generic["output_kind"] == "typed_answers" and generic["statuses"] == ["OK", "REFUSED"]


# --- the answer shapes the five engines really return ----------------------------------------------------------------

#: copied from `tests/test_orchestrate.py` and from the envelopes `tools/verify_envelopes.py` runs, with the engines'
#: own precision: a rendering that rounds any of these is rendering a different number.
FORECASTING = {"area": "forecasting", "answered": 1, "refused": 1, "answers": {
    "prediccion": {"type": "point_forecast", "status": "OK", "values": [0.5412255525588989], "unit": "kW",
                   "scale": "original", "targets": ["Global_active_power"], "horizons": [60],
                   "uncertainty": "none", "execution_authorized": False},
    "rango": refusal(NOT_ESTIMABLE, "this bundle cannot: it has no predictive distribution", "interval")}}

CAUSAL = {"area": "causal", "answered": 1, "refused": 1, "answers": {
    "efecto": {"type": "ate", "status": "OK", "estimand": "ATE", "effect_size": 2.0294,
               "unit": "outcome units", "confidence_interval": [1.9313, 2.1275], "confidence_level": 0.95,
               "assumptions": ["no unmeasured confounding", "positivity"], "execution_authorized": False},
    "jovenes": refusal(NOT_ESTIMABLE, "this study was fitted with no effect modifier", "cate")}}

RL = {"area": "rl", "answered": 2, "refused": 0, "answers": {
    "accion": {"type": "next_action", "status": "OK", "action": [0.05912280082702637],
               "unit": "target position fraction of the policy's own action scale", "action_space": "Box(-1, 1)",
               "policy_id": "eth_4h_sac_current_stack_anchor_v1", "execution_authorized": False},
    "retorno": {"type": "value_estimation", "status": "OK", "expected_return": 3.7995848655700684,
                "uncertainty_bounds": [3.7995848655700684, 4.006697177886963], "execution_authorized": False}}}

RL_REFUSED = {"area": "rl", "answered": 0, "refused": 1, "answers": {
    "accion": refusal(STATE_REQUIRED, "TOO_FEW_ROWS: the policy needs more bars than were supplied",
                      "next_action")}}

UNSUPERVISED = {"area": "unsupervised", "answered": 2, "refused": 0, "answers": {
    "segmentacion": {"type": "clustering", "status": "OK", "rows": 40, "optimal_k": 3, "clusters_occupied": 3,
                     "cluster_distribution": {"0": 0.55, "1": 0.45},
                     "assignments": [{"row_id": i, "cluster_path": [0]} for i in range(40)],
                     "execution_authorized": False},
    "perfil": {"type": "cluster_description", "status": "OK", "level": 1, "target_metric": "body_pipettes > 0",
               "matched_cluster": [1], "rows_in_cluster": 18,
               "centroid_features": {"body_pipettes": 12.5}, "execution_authorized": False}}}

CLASSIFICATION = {"area": "classification", "answered": 1, "refused": 0, "answers": {
    "economia": {"type": "choice", "status": "OK", "label": "euro_area",
                 "uncalibrated_probabilities": {"euro_area": 0.9666, "united_states": 0.0212, "other": 0.0122},
                 "calibration": "UNCALIBRATED", "uncertainty": "none",
                 "sdk_answer": {"label": "euro_area", "confidence": 0.3403},
                 "provenance": {"questions_sha256": "9f3a1c"}, "execution_authorized": False}}}

EVERY = [FORECASTING, CAUSAL, RL, RL_REFUSED, UNSUPERVISED, CLASSIFICATION]


@pytest.mark.parametrize("response", EVERY, ids=lambda r: r["area"] + str(r["refused"]))
def test_no_procedure_may_state_a_number_the_answers_do_not_carry(response):
    for plugin in (DefaultOutput(), TelegramOutput()):
        text = plugin.render(response["area"], response, "es")["text"]
        assert narration_problems(text, response_view(response)) == [], f"{plugin.name} on {response['area']}"


@pytest.mark.parametrize("response", EVERY, ids=lambda r: r["area"] + str(r["refused"]))
def test_telegram_writes_one_line_per_answer_and_always_ends_the_same_way(response):
    text = TelegramOutput().render(response["area"], response, "es")["text"]
    assert text.endswith(CLOSING) and len(text) <= LIMIT
    lines = text.splitlines()
    for name in response["answers"]:
        matching = [line for line in lines if line.startswith(name + " (")]
        assert len(matching) == 1, f"{name}: one answer, one line"


def test_telegram_quotes_the_numbers_exactly_as_the_engines_returned_them():
    text = TelegramOutput().render("forecasting", FORECASTING, "es")["text"]
    assert "0.5412255525588989" in text and "kW" in text
    text = TelegramOutput().render("rl", RL, "es")["text"]
    assert "0.05912280082702637" in text and "3.7995848655700684" in text and "4.006697177886963" in text


def test_telegram_names_a_refusal_with_its_code_and_its_reason_and_carries_no_number():
    text = TelegramOutput().render("rl", RL_REFUSED, "es")["text"]
    line = next(l for l in text.splitlines() if l.startswith("accion"))
    assert "REFUSED" in line and STATE_REQUIRED in line and "TOO_FEW_ROWS" in line
    assert "0 answered, 1 refused." in text


def test_telegram_shows_the_fields_the_header_declares_and_not_a_list_of_ten_thousand_row_ids():
    text = TelegramOutput().render("unsupervised", UNSUPERVISED, "es")["text"]
    assert "optimal_k=3" in text and "rows=40" in text
    assert "row_id" not in text, "the header names the fields that carry the numbers; assignments are not a message"


def test_no_procedure_shows_the_backends_verbatim_object_nor_the_digests():
    for plugin in (DefaultOutput(), TelegramOutput()):
        rendered = plugin.render("classification", CLASSIFICATION, "es")
        blob = rendered["text"] + json.dumps(rendered["json"], default=str)
        assert "0.3403" not in blob and "9f3a1c" not in blob and "sdk_answer" not in blob


def test_telegram_stays_within_its_bound_and_keeps_its_ending_when_there_is_too_much_to_say():
    crowded = {"area": "rl", "answered": 300, "refused": 0,
               "answers": {f"q{i}": dict(RL["answers"]["accion"]) for i in range(300)}}
    text = TelegramOutput().render("rl", crowded, "es")["text"]
    assert len(text) <= LIMIT and text.endswith(CLOSING)
    assert narration_problems(text, response_view(crowded)) == [], "a cut may not leave half a number behind"


def test_the_default_procedure_is_todays_rendering_unchanged():
    for response in EVERY:
        assert DefaultOutput().render(response["area"], response, "es")["text"] == render(response)
    text = render(FORECASTING)
    assert "not answered" in text and "no predictive distribution" in text and "0.5412255525588989" in text


def test_a_procedure_returns_text_a_table_and_json():
    rendered = TelegramOutput().render("causal", CAUSAL, "es")
    assert set(rendered) >= {"text", "table", "json"}
    assert rendered["json"]["header"] == header("causal")
    assert rendered["json"]["answers"] == narratable(CAUSAL["answers"])
    assert rendered["json"]["execution_authorized"] is False
    assert [row["question"] for row in rendered["table"]] == list(CAUSAL["answers"])


# --- selection ---------------------------------------------------------------------------------------------------------

def test_both_shipped_procedures_are_registered_and_loadable_by_name():
    assert {"default", "telegram"} <= set(names())
    assert isinstance(load("default"), type) and isinstance(load("telegram"), type)


def test_an_unknown_procedure_is_refused_naming_the_ones_installed():
    with pytest.raises(OutputPluginError, match="no output plugin named 'smoke-signals'"):
        load("smoke-signals")


def test_the_configuration_selects_the_procedure_per_area(tmp_path):
    assert select("rl", {"plugin": "telegram"}).name == "telegram"
    assert select("rl", {}).name == "default", "the default is today's rendering"
    assert select("rl", {"plugin": "default"}).name == "default"


def test_the_json_configuration_binds_a_procedure_to_one_area_and_leaves_the_others_alone(tmp_path):
    import os

    from m5phet.config import load as load_config
    from m5phet.web.engine import Engine
    path = tmp_path / "m5phet.json"
    path.write_text(json.dumps({"schema": "m5phet.config.v1",
                                "areas": {"rl": {"provider": "trading_policy",
                                                 "output": {"plugin": "telegram", "language": "en"}},
                                          "causal": {"provider": "causal_inference"}}}), encoding="utf-8")
    environ = dict(os.environ, HOME=str(tmp_path), M5PHET_CONFIG=str(path))
    engine = Engine(registry=Registry(), configuration=load_config(path=path, environ=environ), environ=environ)
    assert engine.output("rl").name == "telegram"
    assert engine.output("causal").name == "default"
    assert engine.output_headers()["rl"]["plugin"] == "telegram"
    assert engine.output_headers()["causal"]["question_types"] == header("causal")["question_types"]


def test_the_selected_procedure_writes_what_the_person_reads():
    out = narrate("que accion propone", RL, interpreter=SILENT, area="rl", plugin=TelegramOutput())
    assert out["source"] == "PLUGIN" and out["output_plugin"] == "telegram"
    assert out["text"].endswith(CLOSING) and "0.05912280082702637" in out["text"]


def test_a_procedure_that_asks_no_model_is_not_narrated_even_when_one_is_configured():
    fixed = Fixed("la accion propuesta es 0.0591")
    out = narrate("que accion propone", RL, interpreter=fixed, area="rl", plugin=TelegramOutput())
    assert out["source"] == "PLUGIN" and fixed.asked == [], "telegram states the numbers; it does not commission prose"


def test_the_default_procedure_still_lets_a_faithful_narration_through():
    fixed = Fixed("La previsión es 0.5412255525588989 kW; el intervalo no se pudo calcular.")
    out = narrate("q", FORECASTING, interpreter=fixed, area="forecasting", plugin=DefaultOutput())
    assert out["source"] == "INTERPRETER" and "0.5412255525588989" in out["text"]


# --- a plugin is not trusted because it is installed ----------------------------------------------------------------------

class Liar:
    """An output plugin that says more than the answers do. Three ways, all of them plausible-looking."""

    name = "liar"
    narrates = False

    def __init__(self, text):
        self.text = text

    def header(self, area):
        return header(area)

    def render(self, area, response, language="es"):
        return {"text": self.text, "table": [], "json": {}}


@pytest.mark.parametrize("lie, tell", [
    ("La previsión es 0.5412255525588989 kW con un 95% de confianza.", "95"),
    ("La previsión es 0.5412255525588989 kW, casi el doble de la anterior.", "doble"),
    ("Ganancia de 0.5412255525588989 kW realizada.", "anancia"),
    ("La previsión es 0.5412255 kW.", "0.5412255"),
])
def test_a_plugin_cannot_introduce_a_figure_or_a_claim(lie, tell):
    out = narrate("q", FORECASTING, interpreter=SILENT, area="forecasting", plugin=Liar(lie))
    assert out["source"] == "DETERMINISTIC"
    assert "the output plugin 'liar'" in out["why"] and tell in out["why"]
    assert out["discarded"] == lie
    assert out["text"] == render(FORECASTING), "what stands is the rendering that is faithful by construction"


def test_the_guard_lets_a_plugin_say_less_than_the_answers():
    out = narrate("q", FORECASTING, interpreter=SILENT, area="forecasting",
                  plugin=Liar("La previsión es 0.5412255525588989 kW; el rango no se pudo calcular."))
    assert out["source"] == "PLUGIN" and out["output_plugin"] == "liar"


def test_the_mcp_server_hands_a_client_the_selected_procedures_text():
    """The MCP route goes through the engine, so a client gets the same narration the workbench shows."""
    from m5phet.mcp_server import Server

    class FakeEngine:
        registry = Registry()
        discovery = {"registered": [], "refused": {}}

        def execute_task(self, prompt, task, attachments, language="es"):
            return {"task": task, "response": RL,
                    "narration": narrate(prompt, RL, interpreter=SILENT, area="rl", plugin=TelegramOutput()),
                    "execution_authorized": False}

    engine = FakeEngine()
    server = Server(registry=engine.registry, engine=engine)
    payload = server.call("m5phet_execute_ml_task", {"area": "rl", "state": {}, "questions": {"a": {"type": "next_action"}}})
    narration = payload["structuredContent"]["narration"]
    assert narration["output_plugin"] == "telegram" and narration["text"].endswith(CLOSING)
    assert payload["structuredContent"]["execution_authorized"] is False
