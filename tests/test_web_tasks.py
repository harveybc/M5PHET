"""The envelope through the web: a sentence becomes a proposal, an accepted envelope becomes answers and a narration."""

import json
import time

import pytest
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from m5phet.interpret import Interpreter
from m5phet.questions import refusal
from m5phet.runtime import Registry
from m5phet.web.app import create_app
from m5phet.web.engine import Engine


class Forecaster:
    name, area = "fake_forecaster", "forecasting"

    def capabilities(self):
        return {"provider": self.name, "operations": ["infer"], "families": ["regression_forecasting"],
                "output_kinds": ["point_forecast"], "uncertainty_methods": ["none"],
                "supported": [{"operation": "infer", "family": "regression_forecasting", "output_kind": "point_forecast"}],
                "known_states": ["s1"]}

    def question_types(self):
        return {"point_forecast": {"required": ["horizon"], "optional": ["target"]},
                "interval": {"required": ["horizon", "confidence_level"]}}

    def answer_questions(self, state, questions, data, as_of):
        out = {}
        for name, q in questions.items():
            if q["type"] == "point_forecast":
                out[name] = {"type": "point_forecast", "values": [0.5412] * q["horizon"], "unit": "kW",
                             "rows_seen": len(data) if isinstance(data, list) else 0}
            else:
                out[name] = refusal("NOT_ESTIMABLE", "this model emits a point estimate and no predictive distribution",
                                    q["type"])
        return out


class Fixed(Interpreter):
    def __init__(self, replies):
        super().__init__(command="fixture", model="fixture-v1", environ={})
        self.replies, self.asked = list(replies), []

    @property
    def available(self):
        return True

    def _ask(self, text):
        self.asked.append(text)
        return self.replies.pop(0)


PROPOSAL = {"area": "forecasting", "state": {"target_variable": "ventas"},
            "questions": {"proximo": {"type": "point_forecast", "horizon": 3},
                          "rango": {"type": "interval", "horizon": 3, "confidence_level": 0.95}}}
CSV = b"date,ventas\n2026-01-01,120.5\n2026-01-02,121.0\n"


@pytest.fixture
def client(tmp_path):
    registry = Registry()
    registry.register(Forecaster())
    engine = Engine(registry=registry)
    engine.interpreter = Fixed([json.dumps(PROPOSAL),
                                "El pronóstico es 0.5412 kW en los tres días; el rango no se pudo calcular."])
    app = create_app(tmp_path, engine=engine)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        yield c, engine


def chat_with_file(c):
    cid = c.post("/api/chats", json={"title": "t"}).json()["id"]
    fid = c.post(f"/api/chats/{cid}/files", files={"file": ("ventas.csv", CSV, "text/csv")}).json()["id"]
    return cid, fid


def finished(c, cid):
    for _ in range(200):
        data = c.get(f"/api/chats/{cid}").json()
        if data["messages"] and data["messages"][-1]["status"] not in ("RUNNING", "QUEUED"):
            return data
        time.sleep(0.02)
    raise AssertionError("did not finish")


def test_the_catalog_says_what_each_area_answers(client):
    c, _engine = client
    cat = c.get("/api/tasks/catalog").json()
    assert cat["areas"]["forecasting"]["provider"] == "fake_forecaster"
    assert set(cat["areas"]["forecasting"]["question_types"]) == {"point_forecast", "interval"}
    assert cat["areas"]["rl"]["provider"] is None and cat["execution_authorized"] is False


def test_a_sentence_becomes_a_proposal_that_names_the_datas_columns(client):
    c, engine = client
    cid, fid = chat_with_file(c)
    out = c.post(f"/api/chats/{cid}/tasks/propose", json={"prompt": "pronostica ventas 3 dias", "file_ids": [fid]})
    assert out.status_code == 200, out.text
    body = out.json()
    assert body["status"] == "OK" and body["task"]["area"] == "forecasting"
    assert body["profile"]["columns"] == ["date", "ventas"]
    assert "120.5" not in engine.interpreter.asked[0], "the rows never reach the model"


def test_a_proposal_naming_an_absent_column_does_not_run(client):
    c, engine = client
    engine.interpreter = Fixed([json.dumps({"area": "forecasting", "state": {"target_variable": "ingresos"},
                                            "questions": {"p": {"type": "point_forecast", "horizon": 2}}})])
    cid, fid = chat_with_file(c)
    body = c.post(f"/api/chats/{cid}/tasks/propose", json={"prompt": "x", "file_ids": [fid]}).json()
    assert body["status"] == "INVALID_PROPOSAL" and body["task"] is None
    assert any("ingresos" in p for p in body["problems"])


def test_an_accepted_envelope_runs_and_its_answers_and_narration_are_stored(client):
    c, engine = client
    # running an accepted envelope never routes, so the only thing the interpreter is asked for is the narration
    engine.interpreter = Fixed(["El pronóstico es 0.5412 kW en los tres días; el rango no se pudo calcular."])
    cid, fid = chat_with_file(c)
    sent = c.post(f"/api/chats/{cid}/tasks/run", json={"prompt": "pronostica ventas 3 dias", "task": PROPOSAL,
                                                       "file_ids": [fid], "client_id": "t1"})
    assert sent.status_code == 202, sent.text
    data = finished(c, cid)
    message = data["messages"][-1]
    assert message["status"] == "PARTIAL", message
    detail = message["detail"]
    answers = detail["response"]["answers"]
    assert answers["proximo"]["status"] == "OK" and answers["proximo"]["values"] == [0.5412] * 3
    assert answers["proximo"]["rows_seen"] == 2, "the engine got the rows; the model did not"
    assert answers["rango"]["status"] == "REFUSED" and answers["rango"]["refusal"] == "NOT_ESTIMABLE"
    assert detail["narration"]["source"] == "INTERPRETER" and "0.5412" in message["content"]
    assert detail["envelope"] == PROPOSAL, "the envelope that ran is recorded beside its result"
    assert detail["execution_authorized"] is False


def test_an_edited_envelope_is_a_new_request_not_a_replay(client):
    c, engine = client
    engine.interpreter = Fixed(["r1", "r2"])
    cid, fid = chat_with_file(c)
    first = c.post(f"/api/chats/{cid}/tasks/run", json={"prompt": "p", "task": PROPOSAL, "file_ids": [fid],
                                                        "client_id": "same"})
    finished(c, cid)
    edited = dict(PROPOSAL, questions={"proximo": {"type": "point_forecast", "horizon": 5}})
    second = c.post(f"/api/chats/{cid}/tasks/run", json={"prompt": "p", "task": edited, "file_ids": [fid],
                                                         "client_id": "same"})
    assert first.status_code == 202 and second.status_code == 409, "same client id, different envelope: refused"


def test_a_malformed_envelope_is_refused_and_recorded(client):
    c, _engine = client
    cid, fid = chat_with_file(c)
    c.post(f"/api/chats/{cid}/tasks/run", json={"prompt": "p", "task": {"area": "astrology", "state": {},
                                                                        "questions": {"q": {"type": "x"}}},
                                                "file_ids": [fid], "client_id": "bad"})
    data = finished(c, cid)
    assert data["messages"][-1]["status"] == "REFUSED"
    assert "UNSUPPORTED_AREA" in data["messages"][-1]["content"]


def test_a_narration_that_invents_a_number_is_replaced(client):
    c, engine = client
    # one reply, because on this path nothing is routed: the envelope is given, so the only question asked of the
    # interpreter is the narration. With two replies the JSON proposal was narrated and this case never ran.
    engine.interpreter = Fixed(["Ventas de 0.5412 kW con 99% de certeza."])
    cid, fid = chat_with_file(c)
    c.post(f"/api/chats/{cid}/tasks/run", json={"prompt": "p", "task": PROPOSAL, "file_ids": [fid], "client_id": "n"})
    data = finished(c, cid)
    detail = data["messages"][-1]["detail"]
    assert detail["narration"]["source"] == "DETERMINISTIC" and "99" not in data["messages"][-1]["content"]


# --- a classification envelope goes to the private worker when one is configured -------------------------------------------

def test_a_classification_envelope_is_routed_to_the_worker_and_bound_to_the_request(tmp_path, monkeypatch):
    from m5phet.questions import digest, validate_task

    registry = Registry()
    registry.register(Forecaster())
    engine = Engine(registry=registry)
    engine.interpreter = Fixed(["narración fiel: euro_area con 0.9231"])
    envelope = {"area": "classification", "state": {"asset": "EURUSD", "language": "en"},
                "questions": {"economia": {"type": "choice", "instructions": "Which economy?",
                                           "options": [["euro_area", "Euro area"], ["other", "Other"]]}}}
    sent = []

    def fake_remote(command):
        sent.append(command)
        return {"schema": "m5phet.answers.v1", "request_sha256": digest(validate_task(command["task"])),
                "area": "classification", "provider": "laya_news", "answered": 1, "refused": 0,
                "answers": {"economia": {"type": "choice", "status": "OK", "label": "euro_area",
                                         "uncalibrated_probabilities": {"euro_area": 0.9231, "other": 0.0769}}},
                "execution_authorized": False}

    engine.remote, engine._remote = "worker-host", fake_remote
    out = engine.execute_task("¿de qué economía habla?", envelope,
                              [{"name": "news.txt", "data": b"The ECB left its deposit rate unchanged."}])
    assert sent[0]["action"] == "task" and sent[0]["task"] == envelope
    assert sent[0]["data"] == "The ECB left its deposit rate unchanged.", "the news goes to the worker, not to the model"
    assert out["response"]["answers"]["economia"]["label"] == "euro_area"
    assert out["narration"]["source"] == "INTERPRETER"


def test_a_worker_answer_not_bound_to_the_envelope_is_refused(tmp_path):
    registry = Registry()
    registry.register(Forecaster())
    engine = Engine(registry=registry)
    engine.interpreter = Fixed([])
    envelope = {"area": "classification", "state": {}, "questions": {"q": {"type": "choice", "options": [["a", "A"], ["b", "B"]]}}}
    engine.remote, engine._remote = "worker-host", lambda command: {"request_sha256": "0" * 64, "answers": {},
                                                                    "execution_authorized": False}
    with pytest.raises(ValueError, match="not bound"):
        engine.execute_task("q", envelope, [])


def test_a_forecasting_envelope_never_goes_to_the_classification_worker():
    registry = Registry()
    registry.register(Forecaster())
    engine = Engine(registry=registry)
    engine.interpreter = Fixed(["texto"])
    engine.remote, engine._remote = "worker-host", lambda command: (_ for _ in ()).throw(AssertionError("routed"))
    out = engine.execute_task("p", PROPOSAL, [])
    assert out["response"]["answers"]["proximo"]["status"] == "OK"


def test_a_narration_may_repeat_what_the_person_asked(client):
    """The horizon in the envelope is the person's own number; a sentence about a REFUSED question carries no answer
    figures at all, and refusing to repeat the question's own horizon discarded correct sentences (2026-09-24)."""
    c, engine = client
    engine.interpreter = Fixed(["No se pudo pronosticar a 3 pasos: la pregunta fue rechazada."])
    cid, fid = chat_with_file(c)
    only_interval = {"area": "forecasting", "state": {"target_variable": "ventas"},
                     "questions": {"rango": {"type": "interval", "horizon": 3, "confidence_level": 0.95}}}
    c.post(f"/api/chats/{cid}/tasks/run",
           json={"prompt": "p", "task": only_interval, "file_ids": [fid], "client_id": "asked"})
    detail = finished(c, cid)["messages"][-1]["detail"]
    assert detail["narration"]["source"] == "INTERPRETER", detail["narration"].get("why")
    assert "3 pasos" in detail["narration"]["text"]
    # a number that is neither in the answers nor in the envelope is still refused
    engine.interpreter = Fixed(["No se pudo pronosticar a 7 pasos."])
    c.post(f"/api/chats/{cid}/tasks/run",
           json={"prompt": "p", "task": only_interval, "file_ids": [fid], "client_id": "asked-2"})
    second = finished(c, cid)["messages"][-1]["detail"]
    assert second["narration"]["source"] == "DETERMINISTIC" and "'7'" in second["narration"]["why"]
