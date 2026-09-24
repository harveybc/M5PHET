import json
import time

import pytest
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from m5phet.web.app import create_app
from m5phet.web.engine import Engine
from m5phet.runtime import Registry


class Recorder:
    name = "recording"

    def __init__(self):
        self.calls = []

    def capabilities(self):
        return {"operations": ["infer"], "families": ["classification"],
                "output_kinds": ["typed_questions"], "uncertainty_methods": ["NONE"],
                "supported": [{"operation": "infer", "family": "classification", "output_kind": "typed_questions"}],
                "known_states": ["test-state"]}

    def load(self, ref):
        return {"digest": "a" * 64}

    def infer(self, request, state):
        self.calls.append(request)
        return {"outputs": {"answer": {"status": "OK", "uncertainty": "NONE",
                "payload": {"label": "test"}}}, "population": request.get("population")}


@pytest.fixture
def setup(tmp_path):
    provider = Recorder()
    registry = Registry()
    registry.register(provider)
    engine = Engine(registry=registry)
    app = create_app(tmp_path, engine=engine)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client, provider, tmp_path


def chat(client, title="Test"):
    r = client.post("/api/chats", json={"title": title})
    assert r.status_code == 201, r.text
    return r.json()


def finished(client, cid):
    for _ in range(100):
        data = client.get(f"/api/chats/{cid}").json()
        if data["messages"] and data["messages"][-1]["status"] != "RUNNING":
            return data
        time.sleep(.02)
    raise AssertionError("job did not complete")


def test_chat_crud_settings_and_restart(setup):
    c, _, path = setup
    first, second = chat(c), chat(c, "Two")
    cfg = first["config"] | {"context": "Unique context"}
    r = c.patch(f'/api/chats/{first["id"]}', json={"title": "Renamed", "config": cfg})
    assert r.status_code == 200
    assert c.get(f'/api/chats/{second["id"]}').json()["config"]["context"] == ""
    with TestClient(create_app(path, engine=Engine(registry=Registry())), base_url="http://127.0.0.1") as fresh:
        assert fresh.get(f'/api/chats/{first["id"]}').json()["config"] == cfg
    assert c.delete(f'/api/chats/{first["id"]}').status_code == 204
    assert c.get(f'/api/chats/{first["id"]}').status_code == 404


def test_upload_identity_scope_and_validation(setup):
    c, _, _ = setup
    a, b = chat(c), chat(c)
    url = f'/api/chats/{a["id"]}/files'
    good = c.post(url, files={"file": ("../../notes.csv", b"time,value\n1,2\n", "text/csv")})
    assert good.status_code == 201, good.text
    entry = good.json()
    assert entry["name"] == "notes.csv"
    assert c.get(f'{url}/{entry["id"]}').content == b"time,value\n1,2\n"
    assert c.get(f'/api/chats/{b["id"]}/files/{entry["id"]}').status_code == 404
    assert c.post(url, files={"file": ("bad.json", b'{"a":NaN}', "application/json")}).status_code == 422
    assert c.post(url, files={"file": ("bad.json", b'{"a":1e999}', "application/json")}).status_code == 422
    assert c.post(url, files={"file": ("bad.json", b'{"a":1,"a":2}', "application/json")}).status_code == 422
    assert c.post(url, files={"file": ("bad.csv", b"a,a\n1,2\n", "text/csv")}).status_code == 422
    assert c.post(url, files={"file": ("code.py", b"exit()", "text/plain")}).status_code == 422
    assert c.post(url, files={"file": ("large.txt", b"a" * (8 * 1024 * 1024 + 1), "text/plain")}).status_code == 413


def test_native_runtime_and_send_idempotency(setup):
    c, provider, _ = setup
    row = chat(c)
    cfg = row["config"] | {"input": "typed_request", "provider": "recording", "state": "test-state"}
    assert c.patch(f'/api/chats/{row["id"]}', json={"config": cfg}).status_code == 200
    request = {"schema_version": "m5phet.task.draft2", "request_id": "native-1", "task_id": "test",
               "operation": "infer", "family": "classification", "output_kind": "typed_questions",
               "provider_ref": "recording", "fitted_state_ref": "test-state", "as_of": "2026-09-24T10:00:00Z",
               "output_schema": {"questions": ["answer"]}, "inputs": {"value": 4}}
    body = {"prompt": json.dumps(request), "file_ids": [], "client_id": "turn-1"}
    url = f'/api/chats/{row["id"]}/messages'
    assert c.post(url, json=body).status_code == 202
    result = finished(c, row["id"])["messages"][-1]
    assert result["status"] == "OK", result
    assert provider.calls == [request]
    assert result["detail"]["request"] == request
    assert result["detail"]["result"]["execution_authorized"] is False
    assert c.post(url, json=body).status_code == 202
    assert len(provider.calls) == 1
    assert c.post(url, json=body | {"prompt": "changed"}).status_code == 409


def test_missing_provider_not_fixture(setup):
    c, provider, _ = setup
    row = chat(c)
    c.patch(f'/api/chats/{row["id"]}', json={"config": row["config"] | {"provider": "missing"}})
    assert c.post(f'/api/chats/{row["id"]}/messages', json={"prompt": "Classify?", "client_id": "x"}).status_code == 202
    result = finished(c, row["id"])["messages"][-1]
    assert result["status"] == "REFUSED"
    assert "missing" in result["content"]
    assert provider.calls == []


def test_origin_host_and_remote_auth(setup, tmp_path):
    c, _, _ = setup
    assert c.post("/api/chats", json={}, headers={"origin": "https://evil.example"}).status_code == 403
    assert c.get("/api/chats", headers={"host": "evil.example"}).status_code == 403
    with TestClient(create_app(tmp_path / "private", access_token="test-owner-key", allowed_hosts=["chat.local"]),
                    base_url="http://chat.local") as remote:
        assert remote.get("/api/chats").status_code == 401
        assert remote.post("/api/login", json={"token": "wrong"}).status_code == 401
        assert remote.post("/api/login", json={"token": "test-owner-key"}).status_code == 200
        assert remote.get("/api/chats").status_code == 200


def test_typed_request_never_fits_or_silently_switches_provider(setup):
    c, provider, _ = setup
    row = chat(c)
    cfg = row["config"] | {"input": "typed_request", "provider": "recording"}
    c.patch(f'/api/chats/{row["id"]}', json={"config": cfg})
    for i, task in enumerate(({"operation": "fit"}, {"operation": "infer", "provider_ref": "other"})):
        c.post(f'/api/chats/{row["id"]}/messages', json={"prompt": json.dumps(task), "client_id": str(i)})
        assert finished(c, row["id"])["messages"][-1]["status"] == "REFUSED"
    assert provider.calls == []


def test_installed_news_adapter_fixture_native_parity():
    pytest.importorskip("news_signal")
    from news_signal.provider import LayaNewsProvider
    from m5phet.web.engine import DEFAULT_CONFIG
    from m5phet.runtime import run
    registry = Registry()
    registry.register(LayaNewsProvider(environ={"NEWS_SIGNAL_BACKEND": "fixture", "NEWS_SIGNAL_DEVICE": "cpu"}))
    engine = Engine(registry=registry)
    config = DEFAULT_CONFIG | {"context": "The European Central Bank left interest rates unchanged."}
    actual = engine.execute("Which economy is named?", config, [])
    native = run(actual["request"], registry)
    assert actual["result"]["status"] == "OK", actual
    assert actual["result"]["outputs"] == native["outputs"]
    assert actual["backend"] == "fixture"


def test_files_are_not_executable_and_prompt_is_preserved(setup):
    c, _, _ = setup
    row = chat(c)
    prompt = '<script>window.INJECTED=true</script>'
    c.post(f'/api/chats/{row["id"]}/messages', json={"prompt": prompt, "client_id": "xss"})
    outcome = finished(c, row["id"])
    assert outcome["messages"][0]["content"] == prompt
    assert "script-src 'self'" in c.get("/").headers["Content-Security-Policy"]


def test_interrupt_recovery_and_corrupt_attachment(setup):
    c, _, path = setup
    from m5phet.web.store import Store
    store = Store(path)
    row = chat(c)
    mid, job = store.begin(row["id"], "interrupted", "Question?", [])
    store.recover()
    assert store.get(row["id"])["messages"][-1]["status"] == "INTERRUPTED"
    entry = store.upload(row["id"], "context.txt", b"original")
    with store.connect() as db:
        db.execute("UPDATE files SET data=? WHERE id=?", (b"changed", entry["id"]))
    assert c.get(f'/api/chats/{row["id"]}/files/{entry["id"]}').status_code == 422
    assert c.post(f'/api/chats/{row["id"]}/messages', json={"prompt": "Q", "client_id": "next", "file_ids": [entry["id"]]}).status_code == 422


def test_remote_result_cannot_change_binding(monkeypatch):
    pytest.importorskip("news_signal")
    from news_signal.provider import LayaNewsProvider
    from m5phet.web.engine import DEFAULT_CONFIG
    registry = Registry()
    registry.register(LayaNewsProvider(environ={"NEWS_SIGNAL_BACKEND": "fixture"}))
    engine = Engine(registry=registry)
    engine.remote = "configured-worker"
    monkeypatch.setattr(engine, "_remote", lambda request: {"status": "OK", "request_sha256": "wrong", "execution_authorized": False})
    with pytest.raises(ValueError, match="not bound"):
        engine.execute("Question?", DEFAULT_CONFIG | {"context": "A test news paragraph."}, [])
