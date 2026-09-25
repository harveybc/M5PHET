"""The MCP server is a door to the same contract, not a second runtime."""

import io
import json

import pytest

from m5phet.mcp_server import Server, TOOLS, serve
from m5phet.runtime import Registry
from test_questions import Forecaster


@pytest.fixture
def server():
    registry = Registry()
    registry.register(Forecaster())
    return Server(registry)


def rpc(server, method, params=None, ident=1):
    return server.handle({"jsonrpc": "2.0", "id": ident, "method": method, "params": params or {}})


def test_the_handshake_and_the_tool_list(server):
    init = rpc(server, "initialize")
    assert init["result"]["serverInfo"]["name"] == "m5phet" and "tools" in init["result"]["capabilities"]
    listed = rpc(server, "tools/list")["result"]["tools"]
    assert [t["name"] for t in listed] == [t["name"] for t in TOOLS]
    assert all("inputSchema" in t for t in listed)


def test_the_catalog_tool_says_what_may_be_asked(server):
    out = rpc(server, "tools/call", {"name": "m5phet_catalog"})["result"]
    areas = out["structuredContent"]["areas"]
    assert areas["forecasting"]["provider"] == "fake_forecaster"
    assert set(areas["forecasting"]["question_types"]) == {"point_forecast", "interval"}
    assert out["structuredContent"]["execution_authorized"] is False


def test_executing_an_envelope_answers_and_refuses_per_question(server):
    out = rpc(server, "tools/call", {"name": "m5phet_execute_ml_task", "arguments": {
        "area": "forecasting", "state": {"dataset_id": "ds"},
        "questions": {"p": {"type": "point_forecast", "horizon": 2},
                      "i": {"type": "interval", "horizon": 2, "confidence_level": 0.9}}}})["result"]
    answers = out["structuredContent"]["answers"]
    assert answers["p"]["status"] == "OK" and answers["p"]["values"] == [1.0, 1.0]
    assert answers["i"]["status"] == "REFUSED" and answers["i"]["refusal"] == "NOT_ESTIMABLE"
    assert out["isError"] is False
    assert json.loads(out["content"][0]["text"])["answered"] == 1


def test_a_malformed_envelope_is_a_typed_rpc_error_not_a_crash(server):
    out = rpc(server, "tools/call", {"name": "m5phet_execute_ml_task",
                                     "arguments": {"area": "astrology", "state": {}, "questions": {"q": {"type": "x"}}}})
    assert "error" in out and out["error"]["code"] == -32602 and "UNSUPPORTED_AREA" in out["error"]["message"]


def test_an_unknown_tool_and_an_unknown_method_are_refused(server):
    assert rpc(server, "tools/call", {"name": "m5phet_run_shell"})["error"]["code"] == -32602
    assert rpc(server, "files/read")["error"]["code"] == -32601


def test_a_notification_gets_no_response(server):
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_the_stdio_loop_answers_one_line_per_request():
    registry = Registry()
    registry.register(Forecaster())
    stdin = io.StringIO("\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        "not json",
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    ]) + "\n")
    stdout = io.StringIO()
    serve(stdin, stdout, registry)
    lines = [json.loads(l) for l in stdout.getvalue().splitlines()]
    assert [l.get("id") for l in lines] == [1, None, 2]
    assert lines[1]["error"]["code"] == -32700
    assert lines[2]["result"]["tools"]


def test_the_server_exposes_no_tool_that_is_not_about_fitted_models():
    names = {t["name"] for t in TOOLS}
    assert names == {"m5phet_catalog", "m5phet_execute_ml_task", "m5phet_propose_task"}


def test_without_an_explicit_registry_execution_goes_through_the_workbench_engine(monkeypatch):
    """WP11 found the MCP surface answering classification from the coordinator's fixture: run_task on the local
    registry never took the worker route. The default server is the engine, so every tool call takes the same road as
    the web workbench, including the private worker and the digest binding."""
    from m5phet.web.engine import Engine
    from m5phet.runtime import Registry
    registry = Registry()
    engine = Engine(registry=registry)
    seen = {}

    def execute_task(prompt, task, attachments, language="es"):
        seen.update(task=task, attachments=attachments)
        return {"task": task, "response": {"answers": {}, "answered": 0, "refused": 0},
                "narration": {"text": "nothing", "source": "DETERMINISTIC"}, "execution_authorized": False}
    monkeypatch.setattr(engine, "execute_task", execute_task)
    server = Server(engine=engine)
    out = server.call("m5phet_execute_ml_task", {"area": "forecasting", "state": {}, "questions": {}, "data": {"x": [1]}})
    assert seen["task"] == {"area": "forecasting", "state": {}, "questions": {}}
    assert json.loads(seen["attachments"][0]["data"]) == {"x": [1]}
    assert out["structuredContent"]["execution_authorized"] is False and out["structuredContent"]["narration"]
