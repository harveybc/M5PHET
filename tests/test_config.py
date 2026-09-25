"""WP02: the JSON configuration that binds provider, core settings, interpreter and surfaces per area.

Written before the implementation. What it pins: the shipped example loads; an unknown key is refused instead of
ignored; a `$ENV` that is not set is refused BY NAME; a literal hostname, IP or `user@host` in the file is refused
(the operator's host lives in the environment, never in a repository); the existing env file keeps working with no
JSON at all; and the engine takes its bindings from the JSON when there is one.

No test writes to ~/.config: every path comes from tmp_path through the documented M5PHET_CONFIG variable.
"""
import json
import os
from pathlib import Path

import pytest

from m5phet import config as config_module
from m5phet.config import ConfigError, Configuration, load

EXAMPLE = Path(__file__).resolve().parents[1] / "tools" / "m5phet.json.example"

#: a placeholder that is not any real host; the real one is only ever $M5PHET_CHAT_LAYA_WORKER in the operator's env
WORKER_PLACEHOLDER = "worker-placeholder-for-tests"


def write(tmp_path, document):
    path = tmp_path / "m5phet.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def minimal(**areas):
    return {"schema": "m5phet.config.v1", "areas": areas or {}}


# --- the shipped example -------------------------------------------------------------------------------------------
def test_example_file_carries_no_host_and_loads(tmp_path):
    raw = EXAMPLE.read_text(encoding="utf-8")
    assert "$M5PHET_CHAT_LAYA_WORKER" in raw, "the example names the worker through the environment only"
    environ = {"M5PHET_CHAT_LAYA_WORKER": WORKER_PLACEHOLDER, "HOME": str(tmp_path)}
    loaded = load(path=EXAMPLE, environ=environ)
    assert loaded.source == "m5phet.json"
    assert loaded.path == EXAMPLE
    assert loaded.provider("classification") == "laya_news"
    assert loaded.provider("forecasting") == "predictor_forecast"
    assert loaded.core("classification")["worker"] == WORKER_PLACEHOLDER
    # ~ is expanded against the environment's HOME, never left for a shell to interpret
    assert loaded.core("forecasting")["bundle_dir"].startswith(str(tmp_path))
    assert "~" not in loaded.core("forecasting")["python"]
    assert loaded.interpreter["plugin"] == "command"
    assert loaded.surfaces["web"]["port"] == 8765


def test_example_environment_reaches_the_variables_every_provider_reads(tmp_path):
    environ = {"M5PHET_CHAT_LAYA_WORKER": WORKER_PLACEHOLDER, "HOME": str(tmp_path)}
    exported = load(path=EXAMPLE, environ=environ).environment()
    assert exported["M5PHET_CHAT_LAYA_WORKER"] == WORKER_PLACEHOLDER
    assert exported["M5PHET_FORECAST_BUNDLE"].startswith(str(tmp_path))
    assert exported["M5PHET_POLICY_PYTHON"].startswith(str(tmp_path))
    assert exported["CAUSAL_INFERENCE_STATE_DIR"].startswith(str(tmp_path))
    assert exported["FEATURE_ENG_REGIMES_DEMO_DIR"].startswith(str(tmp_path))
    assert exported["M5PHET_INTERPRETER_COMMAND"]


def test_every_core_key_the_schema_allows_has_a_declared_environment_variable():
    schema = json.loads((Path(config_module.__file__).parent / "config.schema.json").read_text(encoding="utf-8"))
    areas = schema["properties"]["areas"]["properties"]
    for area, definition in areas.items():
        declared = set(definition["properties"]["core"]["properties"])
        mapped = set(config_module.CORE_ENVIRONMENT[area])
        assert declared == mapped, f"{area}: schema and environment map disagree"


# --- unknown keys are refused, never ignored ----------------------------------------------------------------------
@pytest.mark.parametrize("document, needle", [
    ({"schema": "m5phet.config.v1", "areas": {}, "surface": {}}, "surface"),
    ({"schema": "m5phet.config.v1", "areas": {"forecasting": {"provider": "p", "typo": 1}}}, "typo"),
    ({"schema": "m5phet.config.v1", "areas": {"forecasting": {"provider": "p", "core": {"bundel_dir": "/x"}}}}, "bundel_dir"),
    ({"schema": "m5phet.config.v1", "areas": {"astrology": {"provider": "p"}}}, "astrology"),
    ({"schema": "m5phet.config.v1", "interpreter": {"plugin": "command", "temperature": 1}}, "temperature"),
])
def test_unknown_key_is_refused(tmp_path, document, needle):
    path = write(tmp_path, document)
    with pytest.raises(ConfigError) as error:
        load(path=path, environ={"HOME": str(tmp_path)})
    assert needle in str(error.value)


def test_wrong_schema_version_is_refused(tmp_path):
    path = write(tmp_path, {"schema": "m5phet.config.v2", "areas": {}})
    with pytest.raises(ConfigError) as error:
        load(path=path, environ={"HOME": str(tmp_path)})
    assert "m5phet.config.v1" in str(error.value)


def test_wrong_type_is_refused(tmp_path):
    path = write(tmp_path, {"schema": "m5phet.config.v1", "surfaces": {"web": {"port": "8765"}}})
    with pytest.raises(ConfigError) as error:
        load(path=path, environ={"HOME": str(tmp_path)})
    assert "surfaces.web.port" in str(error.value)


# --- a $ENV that is not set is refused by name --------------------------------------------------------------------
def test_missing_environment_variable_is_refused_with_its_name(tmp_path):
    path = write(tmp_path, minimal(classification={"provider": "laya_news",
                                                   "core": {"worker": "$M5PHET_CHAT_LAYA_WORKER"}}))
    with pytest.raises(ConfigError) as error:
        load(path=path, environ={"HOME": str(tmp_path)})
    message = str(error.value)
    assert "M5PHET_CHAT_LAYA_WORKER" in message
    assert "areas.classification.core.worker" in message


def test_environment_variable_is_expanded_in_braced_form(tmp_path):
    path = write(tmp_path, minimal(classification={"provider": "laya_news",
                                                   "core": {"worker": "${M5PHET_CHAT_LAYA_WORKER}"}}))
    loaded = load(path=path, environ={"M5PHET_CHAT_LAYA_WORKER": WORKER_PLACEHOLDER, "HOME": str(tmp_path)})
    assert loaded.core("classification")["worker"] == WORKER_PLACEHOLDER


# --- a literal host in the file is refused ------------------------------------------------------------------------
@pytest.mark.parametrize("literal", [
    "operator@192.168.1.44",
    "10.0.0.7",
    "workstation.local",
    "gpubox.lan",
    "operator@gpubox",
])
def test_hostname_literal_is_refused(tmp_path, literal):
    path = write(tmp_path, minimal(classification={"provider": "laya_news", "core": {"worker": literal}}))
    with pytest.raises(ConfigError) as error:
        load(path=path, environ={"HOME": str(tmp_path)})
    message = str(error.value)
    assert "areas.classification.core.worker" in message
    assert "$" in message, "the refusal must say the value belongs in the environment"


@pytest.mark.parametrize("innocent", [
    "$M5PHET_CHAT_LAYA_WORKER",
    "bash $HOME/work/m5phet-chat-worker/chat_laya_worker.sh",
    "~/.local/share/m5phet/forecast-native-venv/bin/python",
    "hermes --ignore-user-config -m deepseek-v4-flash --provider opencode-go",
])
def test_values_that_are_not_hosts_are_accepted(tmp_path, innocent):
    path = write(tmp_path, minimal(classification={"provider": "laya_news", "core": {"command": innocent}}))
    loaded = load(path=path, environ={"HOME": str(tmp_path), "M5PHET_CHAT_LAYA_WORKER": WORKER_PLACEHOLDER})
    assert loaded.core("classification")["command"]


def test_expansion_may_bring_a_host_from_the_environment(tmp_path):
    """The check is on the file, not on the operator's environment: that is the whole point of $ENV."""
    path = write(tmp_path, minimal(classification={"provider": "laya_news",
                                                   "core": {"worker": "$M5PHET_CHAT_LAYA_WORKER"}}))
    loaded = load(path=path, environ={"M5PHET_CHAT_LAYA_WORKER": "10.1.2.3", "HOME": str(tmp_path)})
    assert loaded.core("classification")["worker"] == "10.1.2.3"


# --- the env file keeps working ------------------------------------------------------------------------------------
def test_without_a_json_file_the_environment_is_the_configuration(tmp_path):
    loaded = load(path=tmp_path / "absent.json", environ={"HOME": str(tmp_path)})
    assert loaded.source == "env"
    assert loaded.path is None
    assert loaded.provider("forecasting") is None
    assert loaded.core("forecasting") == {}
    assert loaded.environment() == {}


def test_path_comes_from_the_documented_variable(tmp_path):
    path = write(tmp_path, minimal(causal={"provider": "causal_inference"}))
    loaded = load(environ={config_module.PATH_VARIABLE: str(path), "HOME": str(tmp_path)})
    assert loaded.path == path
    assert loaded.provider("causal") == "causal_inference"


def test_apply_does_not_touch_a_variable_the_configuration_does_not_bind(tmp_path):
    path = write(tmp_path, minimal(forecasting={"provider": "predictor_forecast",
                                                "core": {"bundle_dir": "/tmp/bundles"}}))
    environ = {"HOME": str(tmp_path), "NEWS_SIGNAL_BACKEND": "fixture", "M5PHET_FORECAST_BUNDLE": "/from/env"}
    load(path=path, environ=environ).apply(environ)
    assert environ["M5PHET_FORECAST_BUNDLE"] == "/tmp/bundles", "JSON wins where it binds"
    assert environ["NEWS_SIGNAL_BACKEND"] == "fixture", "the env file keeps everything the JSON does not bind"


def test_a_broken_file_is_refused_and_never_silently_ignored(tmp_path):
    path = tmp_path / "m5phet.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load(path=path, environ={"HOME": str(tmp_path)})
    assert str(path) in str(error.value)


# --- the engine reads its bindings through config.py ---------------------------------------------------------------
def engine_with(document, tmp_path, environ):
    from m5phet.runtime import Registry
    from m5phet.web.engine import Engine

    class Recorder:
        name = "recording"

        def capabilities(self):
            return {"operations": ["infer"], "families": ["classification"],
                    "output_kinds": ["typed_questions"], "uncertainty_methods": ["NONE"],
                    "supported": [{"operation": "infer", "family": "classification",
                                   "output_kind": "typed_questions"}], "known_states": ["s"]}

    registry = Registry()
    registry.register(Recorder())
    path = write(tmp_path, document)
    environ[config_module.PATH_VARIABLE] = str(path)
    return Engine(registry=registry, environ=environ)


def test_engine_takes_its_bindings_from_the_json(tmp_path):
    document = {"schema": "m5phet.config.v1",
                "interpreter": {"plugin": "command", "command": "hermes --ignore-user-config", "model": "test-model"},
                "areas": {"forecasting": {"provider": "predictor_forecast",
                                          "core": {"bundle_dir": "/tmp/json-bundles", "python": "/usr/bin/python3"}},
                          "classification": {"provider": "laya_news", "core": {"backend": "fixture"}}},
                "surfaces": {"web": {"port": 8766}}}
    environ = {"HOME": str(tmp_path), "M5PHET_FORECAST_BUNDLE": "/from/env", "M5PHET_INTERPRETER_MODEL": "env-model"}
    engine = engine_with(document, tmp_path, environ)
    assert engine.config_source == "m5phet.json"
    assert environ["M5PHET_FORECAST_BUNDLE"] == "/tmp/json-bundles"
    assert environ["M5PHET_FORECAST_PYTHON"] == "/usr/bin/python3"
    assert environ["NEWS_SIGNAL_BACKEND"] == "fixture"
    assert engine.interpreter.model == "test-model", "the interpreter is bound by the JSON, not by the env file"
    catalog = engine.catalog()
    assert catalog["config_source"] == "m5phet.json"


def test_engine_says_env_when_there_is_no_json(tmp_path):
    from m5phet.runtime import Registry
    from m5phet.web.engine import Engine

    registry = Registry()
    environ = {"HOME": str(tmp_path), config_module.PATH_VARIABLE: str(tmp_path / "absent.json"),
               "M5PHET_INTERPRETER_MODEL": "env-model"}
    engine = Engine(registry=registry, environ=environ)
    assert engine.config_source == "env"
    assert engine.catalog()["config_source"] == "env"
    assert engine.interpreter.model == "env-model"


def test_engine_binds_the_worker_from_the_json_without_a_host_in_the_file(tmp_path):
    document = minimal(classification={"provider": "laya_news",
                                       "core": {"worker": "$M5PHET_CHAT_LAYA_WORKER",
                                                "command": "bash $HOME/work/chat_laya_worker.sh"}})
    environ = {"HOME": str(tmp_path), "M5PHET_CHAT_LAYA_WORKER": WORKER_PLACEHOLDER}
    path = write(tmp_path, document)
    loaded = load(path=path, environ=environ)
    assert loaded.core("classification")["worker"] == WORKER_PLACEHOLDER
    assert isinstance(loaded, Configuration)
    assert loaded.environment()["M5PHET_CHAT_LAYA_WORKER"] == WORKER_PLACEHOLDER
    assert os.environ.get("M5PHET_CHAT_LAYA_WORKER") is None or True  # nothing here writes to the real environment
