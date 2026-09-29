"""AP01: what the store remembers across a restart, and what a person may do about a request the restart interrupted.

A conversation is the product's only memory. The rules asserted here are the ones a restart tests:

* a request that was in flight when the process went away is named `INTERRUPTED`; it is never left `RUNNING` (a
  message that waits for ever) and never promoted to an answer (a message that pretends);
* re-sending THAT request -- the same request id, the same prompt, the same attachments -- runs it, once, from the
  snapshot it was accepted with. Before this round `begin` handed the interrupted record straight back and started
  nothing: the caller received 202 for work that would never begin, and the only escape was to invent a new request
  id, which is a different request;
* a request that DID finish is still a replay: the same id returns the same answer and runs nothing again;
* a request id reused for different input is still refused, which is the rule that makes all of the above safe.
"""
import json

import pytest

pytest.importorskip("fastapi")
from m5phet.web.store import Store


def store(tmp_path):
    return Store(tmp_path / "state")


def test_a_request_in_flight_is_named_interrupted_and_not_left_running(tmp_path):
    bank = store(tmp_path)
    chat = bank.create("c")
    mid, job = bank.begin(chat["id"], "one", "pronostica", [])
    assert job is not None
    bank.recover()
    message = [m for m in bank.get(chat["id"])["messages"] if m["id"] == mid][0]
    assert message["status"] == "INTERRUPTED"
    assert "no answer was produced" in message["content"]


def test_resending_an_interrupted_request_runs_it_from_its_own_snapshot(tmp_path):
    bank = store(tmp_path)
    chat = bank.create("c")
    accepted = dict(bank.get(chat["id"])["config"])
    mid, _ = bank.begin(chat["id"], "one", "pronostica", [])
    bank.recover()
    # the chat's configuration moves on AFTER the request was accepted; the replay must not silently adopt it
    bank.update(chat["id"], config=dict(accepted, provider="another_provider"))
    again, job = bank.begin(chat["id"], "one", "pronostica", [])
    assert again == mid                                   # the same request, not a second one beside it
    assert job is not None                                # and this time it runs
    prompt, config, attachments, snapshot = job
    assert prompt == "pronostica" and attachments == []
    assert config == accepted and snapshot["config"] == accepted
    running = [m for m in bank.get(chat["id"])["messages"] if m["id"] == mid][0]
    assert running["status"] == "RUNNING" and running["content"] == ""


def test_the_interrupted_replay_carries_the_attachment_it_was_accepted_with(tmp_path):
    bank = store(tmp_path)
    chat = bank.create("c")
    uploaded = bank.upload(chat["id"], "window.json", b"[[1.0],[2.0]]")
    mid, _ = bank.begin(chat["id"], "one", "pronostica", [uploaded["id"]])
    bank.recover()
    again, job = bank.begin(chat["id"], "one", "pronostica", [uploaded["id"]])
    assert again == mid
    assert [item["id"] for item in job[2]] == [uploaded["id"]]
    assert job[2][0]["data"] == b"[[1.0],[2.0]]"


def test_a_finished_request_is_still_a_replay_and_runs_nothing_again(tmp_path):
    bank = store(tmp_path)
    chat = bank.create("c")
    mid, _ = bank.begin(chat["id"], "one", "pronostica", [])
    bank.finish(mid, "OK", "0.5412 kW", {"response": {"answers": {}}})
    again, job = bank.begin(chat["id"], "one", "pronostica", [])
    assert again == mid and job is None


def test_a_request_id_reused_for_different_input_is_still_refused(tmp_path):
    bank = store(tmp_path)
    chat = bank.create("c")
    mid, _ = bank.begin(chat["id"], "one", "pronostica", [])
    bank.recover()
    with pytest.raises(RuntimeError, match="different input"):
        bank.begin(chat["id"], "one", "pronostica otra cosa", [])


def test_an_interrupted_replay_waits_for_whatever_is_running_now(tmp_path):
    bank = store(tmp_path)
    chat = bank.create("c")
    interrupted, _ = bank.begin(chat["id"], "one", "pronostica", [])
    bank.recover()
    bank.begin(chat["id"], "two", "otra pregunta", [])                  # something else is in flight now
    with pytest.raises(RuntimeError, match="Another request is running"):
        bank.begin(chat["id"], "one", "pronostica", [])


def test_an_envelope_the_person_edited_is_still_a_different_request(tmp_path):
    bank = store(tmp_path)
    chat = bank.create("c")
    bank.begin(chat["id"], "one", "", [], extra={"area": "forecasting", "questions": {}})
    bank.recover()
    with pytest.raises(RuntimeError, match="different input"):
        bank.begin(chat["id"], "one", "", [], extra={"area": "forecasting", "questions": {"a": {}}})


def test_recover_is_idempotent_and_touches_nothing_terminal(tmp_path):
    bank = store(tmp_path)
    chat = bank.create("c")
    done, _ = bank.begin(chat["id"], "done", "una", [])
    bank.finish(done, "REFUSED", "NOT_ESTIMABLE", {})
    live, _ = bank.begin(chat["id"], "live", "otra", [])
    bank.recover(); bank.recover()
    states = {m["id"]: m["status"] for m in bank.get(chat["id"])["messages"]}
    assert states[done] == "REFUSED" and states[live] == "INTERRUPTED"


def test_the_whole_thing_through_the_http_door(tmp_path):
    """The same rule where a person meets it: send, lose the process, send the same request, get the answer."""
    from fastapi.testclient import TestClient
    from m5phet.interpret import Interpreter
    from m5phet.runtime import Registry
    from m5phet.web.app import create_app
    from m5phet.web.engine import Engine

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
            return {name: {"type": "point_forecast", "values": [0.5412], "unit": "kW"} for name in questions}

    class Mute(Interpreter):
        def __init__(self):
            super().__init__(command="fixture", model="fixture-v1", environ={})

        @property
        def available(self):
            return False

    registry = Registry()
    registry.register(Forecaster())
    engine = Engine(registry=registry)
    engine.interpreter = Mute()
    envelope = {"area": "forecasting", "state": {}, "questions": {"p": {"type": "point_forecast", "horizon": 1}}}

    with TestClient(create_app(tmp_path / "state", engine=engine), base_url="http://127.0.0.1") as client:
        cid = client.post("/api/chats", json={"title": "c"}).json()["id"]
        first = client.post(f"/api/chats/{cid}/tasks/run",
                            json={"prompt": "p", "task": envelope, "client_id": "same", "language": "es"})
        assert first.status_code == 202
        mid = first.json()["message_id"]
    # the process goes away with that message recorded, whatever state it reached; a NEW app on the SAME directory
    # is what a restart is, and `recover()` runs in its lifespan
    with Store(tmp_path / "state").connect() as db:
        db.execute("UPDATE messages SET status='RUNNING',content='' WHERE id=?", (mid,))
    with TestClient(create_app(tmp_path / "state", engine=engine), base_url="http://127.0.0.1") as client:
        message = [m for m in client.get(f"/api/chats/{cid}").json()["messages"] if m["id"] == mid][0]
        assert message["status"] == "INTERRUPTED"
        again = client.post(f"/api/chats/{cid}/tasks/run",
                            json={"prompt": "p", "task": envelope, "client_id": "same", "language": "es"})
        assert again.status_code == 202 and again.json()["message_id"] == mid
        for _ in range(100):
            message = [m for m in client.get(f"/api/chats/{cid}").json()["messages"] if m["id"] == mid][0]
            if message["status"] not in ("QUEUED", "RUNNING"):
                break
        assert message["status"] == "OK"
        answers = message["detail"]["response"]["answers"]
        assert answers["p"]["values"] == [0.5412] and answers["p"]["unit"] == "kW"
        assert message["detail"]["execution_authorized"] is False


# --- and the other half of the same event: the run thread that SEES the shutdown -----------------------------------

def test_a_provider_that_died_with_the_process_is_not_recorded_as_a_provider_that_refused(tmp_path):
    """AP01. Stopping the server kills the engines' subprocesses with it. Measured on 2026-09-29, the person was then
    shown `PROVIDER_ERROR: native CPU forecast process refused:` -- the shutdown wearing the engine's name."""
    import threading

    from fastapi.testclient import TestClient
    from m5phet.interpret import Interpreter
    from m5phet.runtime import Registry
    from m5phet.web.app import STOPPED_WHILE_RUNNING, create_app
    from m5phet.web.engine import Engine

    entered, stopped = threading.Event(), threading.Event()

    class Dies:
        """An engine that is alive when the request starts and dead by the time it would answer."""
        name, area = "fake_forecaster", "forecasting"

        def capabilities(self):
            return {"provider": self.name, "operations": ["infer"], "families": ["regression_forecasting"],
                    "output_kinds": ["point_forecast"], "uncertainty_methods": ["none"],
                    "supported": [{"operation": "infer", "family": "regression_forecasting",
                                   "output_kind": "point_forecast"}], "known_states": ["s1"]}

        def question_types(self):
            return {"point_forecast": {"required": ["horizon"], "optional": ["target"]}}

        def answer_questions(self, state, questions, data, as_of):
            entered.set()
            stopped.wait(timeout=10)                 # the stop arrives while this run is in flight, as it does
            raise RuntimeError("native CPU forecast process refused:")

    class Mute(Interpreter):
        def __init__(self):
            super().__init__(command="fixture", model="fixture-v1", environ={})

        @property
        def available(self):
            return False

    registry = Registry()
    registry.register(Dies())
    engine = Engine(registry=registry)
    engine.interpreter = Mute()
    envelope = {"area": "forecasting", "state": {}, "questions": {"p": {"type": "point_forecast", "horizon": 1}}}
    app = create_app(tmp_path / "state", engine=engine)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        cid = client.post("/api/chats", json={"title": "c"}).json()["id"]
        mid = client.post(f"/api/chats/{cid}/tasks/run",
                          json={"prompt": "p", "task": envelope, "client_id": "same",
                                "language": "es"}).json()["message_id"]
        entered.wait(timeout=10)
        # the stop signal, at the instant `main()`'s handler would see it: in flight, engines about to die with it
        app.state.begin_stopping()
        stopped.set()
    # leaving the context drains the pool, so the thread's exception is recorded while the server is going down
    with TestClient(app, base_url="http://127.0.0.1") as client:
        message = [m for m in client.get(f"/api/chats/{cid}").json()["messages"] if m["id"] == mid][0]
    assert message["status"] == "INTERRUPTED"
    assert message["content"] == STOPPED_WHILE_RUNNING
    assert message["detail"]["stopped_while_running"] is True
    # `run_task` turned the dead engine into a TYPED refusal -- which is why the app's `except` branch is not where
    # this arrives, and why the measured message said the forecaster had declined the question. The engine's own
    # words are kept as evidence, verbatim; they are simply no longer the person's answer.
    answer = message["detail"]["response"]["answers"]["p"]
    assert answer["refusal"] == "PROVIDER_ERROR"
    assert "native CPU forecast process refused" in answer["why"]
    assert message["detail"]["response"]["answered"] == 0


def test_a_typed_refusal_that_lands_during_a_stop_is_still_that_refusal(tmp_path):
    """The other side of the same rule: an engine that CONSIDERED the question and declined it by name keeps its
    answer, shutdown or no shutdown. Only `PROVIDER_ERROR` -- the engine never got to consider anything -- is
    re-attributed to the stop, and this is the test that keeps that boundary where it is."""
    import threading

    from fastapi.testclient import TestClient
    from m5phet.interpret import Interpreter
    from m5phet.questions import refusal
    from m5phet.runtime import Registry
    from m5phet.web.app import create_app
    from m5phet.web.engine import Engine

    entered, stopped = threading.Event(), threading.Event()

    class Declines:
        name, area = "fake_forecaster", "forecasting"

        def capabilities(self):
            return {"provider": self.name, "operations": ["infer"], "families": ["regression_forecasting"],
                    "output_kinds": ["point_forecast"], "uncertainty_methods": ["none"],
                    "supported": [{"operation": "infer", "family": "regression_forecasting",
                                   "output_kind": "point_forecast"}], "known_states": ["s1"]}

        def question_types(self):
            return {"point_forecast": {"required": ["horizon"], "optional": ["target"]}}

        def answer_questions(self, state, questions, data, as_of):
            entered.set()
            stopped.wait(timeout=10)
            return {name: refusal("NOT_ESTIMABLE", "this model emits no predictive distribution", "point_forecast")
                    for name in questions}

    class Mute(Interpreter):
        def __init__(self):
            super().__init__(command="fixture", model="fixture-v1", environ={})

        @property
        def available(self):
            return False

    registry = Registry()
    registry.register(Declines())
    engine = Engine(registry=registry)
    engine.interpreter = Mute()
    envelope = {"area": "forecasting", "state": {}, "questions": {"p": {"type": "point_forecast", "horizon": 1}}}
    app = create_app(tmp_path / "state", engine=engine)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        cid = client.post("/api/chats", json={"title": "c"}).json()["id"]
        mid = client.post(f"/api/chats/{cid}/tasks/run",
                          json={"prompt": "p", "task": envelope, "client_id": "same",
                                "language": "es"}).json()["message_id"]
        entered.wait(timeout=10)
        app.state.begin_stopping()
        stopped.set()
    with TestClient(app, base_url="http://127.0.0.1") as client:
        message = [m for m in client.get(f"/api/chats/{cid}").json()["messages"] if m["id"] == mid][0]
    assert message["status"] == "REFUSED"
    assert message["detail"]["response"]["answers"]["p"]["refusal"] == "NOT_ESTIMABLE"
    assert "stopped_while_running" not in message["detail"]
