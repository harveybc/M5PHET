"""WP03: the interpreter is a plugin, and every plugin keeps the same invariant.

What these tests pin. Each shipped plugin builds the request its transport expects -- a `-z PROMPT` argument vector, an
ollama `/api/chat` body with the GPU off, an OpenAI-compatible `/chat/completions` body with a bearer token -- and each
is checked against a FAKE: a script on disk and a server on the loopback interface. Nothing here reaches a network, a
real model or a real key.

The two rules that matter more than the shapes are checked for all three: the model is shown the declared vocabulary
and never the data, and a value it returns that was not declared is refused rather than rounded to a neighbour. And
`openai_compatible`, the only plugin that can send a person's sentence off this machine, refuses to run at all unless
the configuration says so in as many words.
"""

import json
import os
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from m5phet.interpret import STATUS_OK, STATUS_UNSUPPORTED, build, interpret
from m5phet.interpreters import BUILT_IN, InterpreterPluginError, load, names
from m5phet.interpreters.command import CommandInterpreter
from m5phet.interpreters.ollama import OllamaInterpreter
from m5phet.interpreters.openai_compatible import CONSENT, CloudConsentRequired, OpenAICompatibleInterpreter

FORECAST = [{"name": "target", "allowed": ["Global_active_power"]},
            {"name": "horizon", "allowed": [60], "type": "integer"}]

#: what a person attached; no plugin may put any of it in a request
SECRET_ROWS = "4.216;0.418;234.840"


# --- a fake command on disk -------------------------------------------------------------------------------------------

def fake_command(tmp_path, reply, *, exit_code=0):
    """A program that records the argv it was called with and prints `reply`, so the `-z` contract is checked for real."""
    path = tmp_path / "fake_interpreter.py"
    log = tmp_path / "argv.json"
    path.write_text("#!/usr/bin/env python3\n"
                    "import json, sys\n"
                    f"open({str(log)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
                    f"sys.stdout.write({reply!r})\n"
                    f"sys.exit({exit_code})\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path, log


def test_the_command_plugin_calls_its_program_with_dash_z_and_reads_stdout(tmp_path):
    path, log = fake_command(tmp_path, '{"target": "Global_active_power", "horizon": 60}')
    plugin = CommandInterpreter({"command": f"{path}", "model": "fake-v1"}, environ={})
    assert plugin.available and plugin.identity()["plugin"] == "command"
    out = plugin._ask("what will consumption be?")
    assert json.loads(out) == {"target": "Global_active_power", "horizon": 60}
    assert json.loads(log.read_text()) == ["-z", "what will consumption be?"]


def test_the_command_plugin_reports_a_failing_program_rather_than_returning_nothing(tmp_path):
    path, _log = fake_command(tmp_path, "", exit_code=3)
    plugin = CommandInterpreter({"command": f"{path}"}, environ={})
    with pytest.raises(ValueError, match="interpreter failed"):
        plugin._ask("anything")


def test_the_command_plugin_takes_its_timeout_from_the_configuration(tmp_path):
    path, _log = fake_command(tmp_path, "{}")
    assert CommandInterpreter({"command": f"{path}", "timeout_seconds": 7}, environ={}).timeout == 7


# --- a fake HTTP server on the loopback interface ------------------------------------------------------------------

class Recorder(ThreadingHTTPServer):
    daemon_threads = True
    seen = None
    reply = {}
    status = 200


def handler_for():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):                                              # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length).decode()
            self.server.seen = {"path": self.path, "body": json.loads(body),
                                "headers": {k.lower(): v for k, v in self.headers.items()}}
            payload = json.dumps(self.server.reply).encode()
            self.send_response(self.server.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):                                   # keep the test output quiet
            return
    return Handler


@pytest.fixture
def server():
    """A loopback server that records one request. It is not a network: it is this process talking to itself."""
    httpd = Recorder(("127.0.0.1", 0), handler_for())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    httpd.base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


# --- ollama ------------------------------------------------------------------------------------------------------------

def test_the_ollama_plugin_asks_a_local_server_with_the_gpu_off_and_thinking_off(server):
    server.reply = {"message": {"content": '{"target": "Global_active_power", "horizon": 60}'}}
    plugin = OllamaInterpreter({"model": "llama3.2:3b", "base_url": server.base_url}, environ={})
    assert plugin.available and plugin.identity()["plugin"] == "ollama"
    out = plugin._ask("cuanta potencia habra en la proxima hora?")
    assert json.loads(out)["horizon"] == 60
    assert server.seen["path"] == "/api/chat"
    body = server.seen["body"]
    assert body["model"] == "llama3.2:3b" and body["stream"] is False and body["think"] is False
    assert body["options"]["num_gpu"] == 0, "no GPU on this host is admitted for reading a sentence"
    assert body["options"]["temperature"] == 0 and body["options"]["num_predict"] >= 1
    assert body["messages"] == [{"role": "user", "content": "cuanta potencia habra en la proxima hora?"}]


def test_the_ollama_plugin_keeps_only_what_follows_a_models_thinking(server):
    server.reply = {"message": {"content": "<think>the person means power</think>\n{\"target\": \"x\"}"}}
    plugin = OllamaInterpreter({"model": "qwen3:4b", "base_url": server.base_url}, environ={})
    assert plugin._ask("q") == '{"target": "x"}'


def test_the_ollama_plugin_without_a_model_is_unavailable_and_says_which_setting_is_missing():
    plugin = OllamaInterpreter({}, environ={})
    assert plugin.available is False
    with pytest.raises(ValueError, match="interpreter.model"):
        plugin._ask("q")


def test_the_ollama_plugin_reports_a_server_that_is_not_there_instead_of_returning_nothing(tmp_path):
    # port 1 on the loopback interface: nothing listens there, and the refusal must say so
    plugin = OllamaInterpreter({"model": "llama3.2:3b", "base_url": "http://127.0.0.1:1"}, environ={})
    with pytest.raises(ValueError, match="did not answer"):
        plugin._ask("q")


def test_the_ollama_plugin_publishes_no_endpoint_in_its_identity(server):
    identity = OllamaInterpreter({"model": "llama3.2:3b", "base_url": server.base_url}, environ={}).identity()
    assert server.base_url not in json.dumps(identity)
    assert identity["device"] == "CPU (num_gpu: 0)"


# --- openai_compatible: consent first ---------------------------------------------------------------------------------

def test_the_cloud_plugin_refuses_without_the_configurations_explicit_consent(server):
    plugin = OpenAICompatibleInterpreter({"base_url": server.base_url, "model": "some-cloud-model",
                                          "api_key_env": "FAKE_KEY"}, environ={"FAKE_KEY": "k"})
    assert plugin.available is False
    assert plugin.identity()["consent"] == "NOT_GIVEN"
    assert "cloud_ok" in plugin.identity()["why"]
    with pytest.raises(CloudConsentRequired):
        plugin._ask("q")
    assert server.seen is None, "nothing may leave this machine before consent is stated"


def test_the_cloud_plugin_runs_once_the_configuration_states_consent(server):
    server.reply = {"choices": [{"message": {"content": '{"target": "Global_active_power", "horizon": 60}'}}]}
    plugin = OpenAICompatibleInterpreter({"base_url": server.base_url, "model": "some-cloud-model",
                                          "api_key_env": "FAKE_KEY", "consent": CONSENT},
                                         environ={"FAKE_KEY": "s3cret-token"})
    assert plugin.available is True
    out = plugin._ask("what will consumption be?")
    assert json.loads(out)["target"] == "Global_active_power"
    assert server.seen["path"] == "/chat/completions"
    assert server.seen["headers"]["authorization"] == "Bearer s3cret-token"
    body = server.seen["body"]
    assert body["model"] == "some-cloud-model" and body["temperature"] == 0 and body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "what will consumption be?"}]


def test_the_cloud_plugin_refuses_by_name_when_the_named_variable_holds_no_key(server):
    plugin = OpenAICompatibleInterpreter({"base_url": server.base_url, "model": "m", "consent": CONSENT,
                                          "api_key_env": "M5PHET_TEST_KEY_NOT_SET"}, environ={})
    with pytest.raises(ValueError, match="M5PHET_TEST_KEY_NOT_SET"):
        plugin._ask("q")
    assert server.seen is None


def test_the_cloud_plugins_identity_carries_neither_the_key_nor_the_endpoint(server):
    identity = OpenAICompatibleInterpreter({"base_url": server.base_url, "model": "m", "consent": CONSENT,
                                            "api_key_env": "FAKE_KEY"}, environ={"FAKE_KEY": "s3cret"}).identity()
    text = json.dumps(identity)
    assert "s3cret" not in text and server.base_url not in text and "FAKE_KEY" not in text


# --- the invariant, for every plugin -------------------------------------------------------------------------------------

def test_no_plugin_shows_the_model_the_data_and_none_may_introduce_a_value(server, tmp_path):
    """The same two rules through three transports: the vocabulary is sent, the rows are not, and a value the provider
    never declared is refused rather than rounded to the one that exists."""
    undeclared = json.dumps({"target": "Voltage", "horizon": 60})
    path, log = fake_command(tmp_path, undeclared)
    server.reply = {"message": {"content": undeclared}, "choices": [{"message": {"content": undeclared}}]}
    plugins = [
        CommandInterpreter({"command": f"{path}"}, environ={}),
        OllamaInterpreter({"model": "llama3.2:3b", "base_url": server.base_url}, environ={}),
        OpenAICompatibleInterpreter({"base_url": server.base_url, "model": "m", "consent": CONSENT,
                                     "api_key_env": "FAKE_KEY"}, environ={"FAKE_KEY": "k"}),
    ]
    import inspect
    assert "data" not in inspect.signature(interpret).parameters, \
        "there is no argument by which a dataset could reach an interpreter; the invariant is structural"
    for plugin in plugins:
        report = interpret("predict something", FORECAST, interpreter=plugin)
        assert report["status"] == STATUS_UNSUPPORTED, plugin.plugin
        assert "Voltage" in report["why"] and "Global_active_power" in report["why"]
    sent = json.dumps(json.loads(log.read_text())) + json.dumps(server.seen["body"])
    assert "Global_active_power" in sent, "the declared vocabulary IS shown to the model"
    assert SECRET_ROWS not in sent, "and the rows a person attached are not, through any of the three transports"


def test_a_declared_value_chosen_through_each_transport_is_accepted(server, tmp_path):
    chosen = json.dumps({"target": "Global_active_power", "horizon": 60})
    path, _log = fake_command(tmp_path, chosen)
    server.reply = {"message": {"content": chosen}, "choices": [{"message": {"content": chosen}}]}
    for plugin in (CommandInterpreter({"command": f"{path}"}, environ={}),
                   OllamaInterpreter({"model": "llama3.2:3b", "base_url": server.base_url}, environ={}),
                   OpenAICompatibleInterpreter({"base_url": server.base_url, "model": "m", "consent": CONSENT,
                                                "api_key_env": "FAKE_KEY"}, environ={"FAKE_KEY": "k"})):
        report = interpret("what will consumption be?", FORECAST, interpreter=plugin)
        assert report["status"] == STATUS_OK, plugin.plugin
        assert report["parameters"] == {"target": "Global_active_power", "horizon": 60}
        assert report["interpreter"]["plugin"] == plugin.plugin


# --- selection ------------------------------------------------------------------------------------------------------------

def test_every_shipped_plugin_is_registered_and_loadable_by_name():
    for name in BUILT_IN:
        assert name in names()
        assert isinstance(load(name), type)


def test_an_unknown_plugin_is_refused_naming_the_ones_installed():
    with pytest.raises(InterpreterPluginError, match="no interpreter plugin named 'telepathy'"):
        load("telepathy")


#: `isinstance` against the module-level classes is not safe in the full suite: another test deliberately drops every
#: `m5phet.*` module from `sys.modules`, so a later import builds NEW class objects. The selection tests therefore
#: re-import the classes at the moment they check, which is what `build()` itself does.
def _classes():
    from m5phet.interpreters.command import CommandInterpreter
    from m5phet.interpreters.ollama import OllamaInterpreter
    from m5phet.interpreters.openai_compatible import OpenAICompatibleInterpreter
    return CommandInterpreter, OllamaInterpreter, OpenAICompatibleInterpreter


def test_the_configuration_selects_the_implementation_with_no_code_change(tmp_path):
    command_cls, ollama_cls, cloud_cls = _classes()
    environ = {"HOME": str(tmp_path)}
    assert isinstance(build({"plugin": "ollama", "model": "llama3.2:3b"}, environ=environ), ollama_cls)
    assert isinstance(build({"plugin": "openai_compatible", "model": "m"}, environ=environ), cloud_cls)
    assert isinstance(build({"plugin": "command", "command": "hermes"}, environ=environ), command_cls)
    assert isinstance(build({}, environ=environ), command_cls), "the default is the command plugin"


def test_the_json_configuration_selects_the_plugin_and_carries_its_own_settings(tmp_path):
    path = tmp_path / "m5phet.json"
    path.write_text(json.dumps({"schema": "m5phet.config.v1",
                                "interpreter": {"plugin": "ollama", "model": "llama3.2:3b",
                                                "timeout_seconds": 42}}), encoding="utf-8")
    built = build(environ={"HOME": str(tmp_path), "M5PHET_CONFIG": str(path)})
    assert isinstance(built, _classes()[1])
    assert built.model == "llama3.2:3b" and built.timeout == 42


def test_with_no_json_configuration_the_environment_still_names_the_interpreter(tmp_path):
    """Backward compatibility, exactly as WP02 kept it: the env file alone is a valid configuration."""
    assert not (tmp_path / ".config" / "m5phet" / "m5phet.json").exists()
    built = build(environ={"HOME": str(tmp_path), "M5PHET_INTERPRETER_COMMAND": "hermes --ignore-user-config",
                           "M5PHET_INTERPRETER_MODEL": "deepseek-v4-flash", "M5PHET_INTERPRETER_TIMEOUT": "120"})
    assert isinstance(built, _classes()[0]) and built.plugin == "command"
    assert built.command == "hermes --ignore-user-config" and built.model == "deepseek-v4-flash"
    assert built.timeout == 120


def test_the_engine_builds_the_plugin_the_json_names(tmp_path, monkeypatch):
    from m5phet.config import load as load_config
    from m5phet.web.engine import Engine
    path = tmp_path / "m5phet.json"
    path.write_text(json.dumps({"schema": "m5phet.config.v1",
                                "interpreter": {"plugin": "ollama", "model": "llama3.2:3b"}}), encoding="utf-8")
    environ = dict(os.environ, HOME=str(tmp_path), M5PHET_CONFIG=str(path))
    engine = Engine(registry=_empty_registry(), configuration=load_config(path=path, environ=environ),
                    environ=environ)
    assert isinstance(engine.interpreter, _classes()[1])
    assert engine.catalog()["interpreter"]["plugin"] == "ollama"


def _empty_registry():
    from m5phet.runtime import Registry
    return Registry()
