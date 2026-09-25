"""WP02: one JSON file that binds, per area, the provider and its core settings, plus the interpreter and surfaces.

Why a file and not more environment variables. The operator already declares every engine, every fitted state and the
interpreter through `~/.config/m5phet/chat.env`; that works, and it keeps working -- the environment IS a valid
configuration and nothing here requires a JSON file to exist. What the environment cannot do is say, in one place a
person can read and review, WHICH plugin serves which area and with which settings. `~/.config/m5phet/m5phet.json`
(schema `m5phet.config.v1`, validated against `config.schema.json` beside this module) says exactly that.

The rules this module enforces, each of them a refusal and never a repair:

* an unknown key is refused, naming its path. A configuration that silently ignores a misspelled key gives the
  operator a machine that answers with settings he believes he changed;
* `~` and `$NAME` / `${NAME}` are expanded. A variable the environment does not hold is refused BY NAME, so the
  operator learns which one to set instead of watching a provider refuse later for an unrelated-looking reason;
* a literal hostname, IPv4 address, `user@host` or `.local` / `.lan` name in the FILE is refused. The file is meant to
  be committed as an example and read by others; the operator's host is only ever `$M5PHET_CHAT_LAYA_WORKER`. The
  check is on what the file says, never on what the environment supplies: expansion may of course yield a host.

Precedence: JSON wins where it binds, the environment keeps everything it does not. `Configuration.apply()` writes the
bound values into the environment the providers read, because each provider owns its own variables (`M5PHET_FORECAST_*`,
`M5PHET_POLICY_*`, `NEWS_SIGNAL_*`, `FEATURE_ENG_REGIMES_*`, `CAUSAL_INFERENCE_*`) and this package does not import a
single one of them. That is why the mapping below is explicit: it is the contract between the JSON key an operator
writes and the variable a provider reads.
"""
import json
import os
import re
from pathlib import Path

SCHEMA = "m5phet.config.v1"

#: where the file lives, and the variable that overrides it (tests and verification instances use the variable so the
#: owner's own instance is never touched)
DEFAULT_PATH = "~/.config/m5phet/m5phet.json"
PATH_VARIABLE = "M5PHET_CONFIG"

SCHEMA_FILE = Path(__file__).resolve().parent / "config.schema.json"

AREAS = ("classification", "forecasting", "unsupervised", "rl", "causal")

#: JSON core key -> the environment variable the area's provider actually reads. Kept beside the schema and checked
#: against it by the test suite: a key the schema allows and nothing maps would configure nothing at all.
CORE_ENVIRONMENT = {
    "classification": {"worker": "M5PHET_CHAT_LAYA_WORKER", "command": "M5PHET_CHAT_LAYA_COMMAND",
                       "backend": "NEWS_SIGNAL_BACKEND", "checkpoint": "NEWS_SIGNAL_CHECKPOINT",
                       "manifest": "NEWS_SIGNAL_MANIFEST", "device": "NEWS_SIGNAL_DEVICE",
                       "gpu_uuid": "NEWS_SIGNAL_GPU_UUID"},
    "forecasting": {"bundle_dir": "M5PHET_FORECAST_BUNDLE", "python": "M5PHET_FORECAST_PYTHON"},
    "unsupervised": {"reference_dir": "FEATURE_ENG_REGIMES_DEMO_DIR", "state_path": "FEATURE_ENG_REGIMES_STATE_PATH"},
    "rl": {"bundle": "M5PHET_POLICY_BUNDLE", "python": "M5PHET_POLICY_PYTHON", "gym_fx": "M5PHET_GYM_FX",
           "sample": "M5PHET_POLICY_SAMPLE"},
    "causal": {"studies_dir": "CAUSAL_INFERENCE_STATE_DIR", "state_refs": "CAUSAL_INFERENCE_STATE_REFS"},
}

#: WP31 -- JSON `areas.<area>.quality.report` -> the variable the quality reader falls back to, so an operator whose
#: configuration is the env file declares the measurement exactly as one with a JSON file does
QUALITY_ENVIRONMENT = {"forecasting": "M5PHET_FORECASTING_QUALITY_REPORT",
                       "unsupervised": "M5PHET_UNSUPERVISED_QUALITY_REPORT"}

#: the interpreter's own variables, which `m5phet.interpret.Interpreter` reads. The last two are the abstention rule:
#: they are exported like the rest because the environment is a valid configuration here, so an operator who never
#: writes a JSON file can still declare the threshold and the report it is cited from.
INTERPRETER_ENVIRONMENT = {"command": "M5PHET_INTERPRETER_COMMAND", "model": "M5PHET_INTERPRETER_MODEL",
                           "timeout_seconds": "M5PHET_INTERPRETER_TIMEOUT",
                           "min_confidence": "M5PHET_INTERPRETER_MIN_CONFIDENCE",
                           "abstention_source": "M5PHET_INTERPRETER_ABSTENTION_SOURCE",
                           "reliability_report": "M5PHET_INTERPRETER_RELIABILITY_REPORT",
                           "route_reliability_report": "M5PHET_ROUTE_RELIABILITY_REPORT"}

_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
_IPV4 = re.compile(r"(?<![\d.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![\d.])")
_USER_AT_HOST = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9][A-Za-z0-9.-]*$")
_LOCAL_DOMAIN = re.compile(r"(?<![/\w.-])[A-Za-z0-9][A-Za-z0-9-]*\.(?:local|lan)(?![A-Za-z0-9-])")


class ConfigError(ValueError):
    """A configuration that cannot be read as written. Never repaired, never partially applied."""


# --- validation of the restricted JSON-Schema subset ----------------------------------------------------------------
_TYPES = {"object": dict, "string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list}


def _validate(value, schema, where):
    if "const" in schema and value != schema["const"]:
        raise ConfigError(f"{where}: expected {schema['const']!r}, found {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ConfigError(f"{where}: {value!r} is not one of {schema['enum']}")
    expected = schema.get("type")
    if expected:
        wanted = _TYPES[expected]
        if expected in ("integer", "number") and isinstance(value, bool):
            raise ConfigError(f"{where}: expected {expected}, found a boolean")
        if not isinstance(value, wanted):
            raise ConfigError(f"{where}: expected {expected}, found {type(value).__name__}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ConfigError(f"{where}: must not be empty")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ConfigError(f"{where}: longer than {schema['maxLength']} characters")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ConfigError(f"{where}: {value} is below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ConfigError(f"{where}: {value} is above {schema['maximum']}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    known = ", ".join(sorted(properties)) or "none"
                    raise ConfigError(f"unknown configuration key {_join(where, key)!r}; "
                                      f"{where or 'the file'} accepts: {known}")
        for key in schema.get("required", ()):
            if key not in value:
                raise ConfigError(f"{_join(where, key)} is required")
        for key, item in value.items():
            if key in properties:
                _validate(item, properties[key], _join(where, key))


def _join(where, key):
    return f"{where}.{key}" if where else key


# --- host literals --------------------------------------------------------------------------------------------------
def _looks_like_host(token):
    if "$" in token:
        return False
    match = _IPV4.search(token)
    if match and all(int(group) <= 255 for group in match.groups()):
        return True
    if _USER_AT_HOST.match(token):
        return True
    return bool(_LOCAL_DOMAIN.search(token))


def _refuse_host_literals(value, where):
    if isinstance(value, dict):
        for key, item in value.items():
            _refuse_host_literals(item, _join(where, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _refuse_host_literals(item, f"{where}[{index}]")
    elif isinstance(value, str):
        for token in value.split():
            if _looks_like_host(token):
                raise ConfigError(
                    f"{where}: {token!r} looks like a host, an address or user@host. A host is never written into "
                    f"this file; name it through the environment instead, e.g. \"$M5PHET_CHAT_LAYA_WORKER\".")


# --- expansion ------------------------------------------------------------------------------------------------------
def _expand_string(value, environ, where):
    def replace(match):
        name = match.group(1) or match.group(2)
        if name not in environ:
            raise ConfigError(f"{where}: environment variable ${name} is not set; "
                              f"set {name} (the operator's env file is the place for it) or write a literal value")
        return environ[name]
    expanded = _VARIABLE.sub(replace, value)
    if expanded.startswith("~"):
        home = environ.get("HOME")
        if home:
            expanded = str(Path(home) / expanded[1:].lstrip("/")) if expanded != "~" else home
        else:
            expanded = os.path.expanduser(expanded)
    return expanded


def _expand(value, environ, where):
    if isinstance(value, dict):
        # a `$comment...` key is documentation, not a setting: it may name a variable ($M5PHET_CHAT_LAYA_WORKER,
        # $M5PHET_API_TOKEN_FILE) without that variable having to be set. Every such key is exempt from expansion,
        # not only the bare `$comment`, because the file carries one per section. They are still checked for host
        # literals, like every other string in the file.
        return {key: (item if str(key).startswith("$comment") else _expand(item, environ, _join(where, str(key))))
                for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, environ, f"{where}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, str):
        return _expand_string(value, environ, where)
    return value


class Configuration:
    """A loaded configuration, or the declaration that the environment is the configuration.

    `source` is `"m5phet.json"` when a file was read and `"env"` when there was none; it is what `/api/catalog`
    reports, so a person looking at the workbench can see which of the two is in force."""

    def __init__(self, data=None, path=None, environ=None):
        self.data = data or {}
        self.path = path
        self.source = "m5phet.json" if data else "env"
        self.environ = os.environ if environ is None else environ

    # --- what an area is bound to
    def area(self, name):
        return (self.data.get("areas") or {}).get(name) or {}

    def provider(self, name):
        return self.area(name).get("provider")

    def core(self, name):
        return dict(self.area(name).get("core") or {})

    def output(self, name):
        return dict(self.area(name).get("output") or {})

    def quality(self, name):
        """WP31: `areas.<area>.quality` -- where the evaluation report that MEASURED this area is read from.

        Only forecasting and unsupervised take one: classification publishes its provider's own record
        (`NEWS_SIGNAL_QUALITY`), and causal and rl refuse the quantity by name rather than declare a report for it."""
        return dict(self.area(name).get("quality") or {})

    @property
    def interpreter(self):
        return dict(self.data.get("interpreter") or {})

    @property
    def surfaces(self):
        return dict(self.data.get("surfaces") or {})

    @property
    def datasets(self):
        """WP15: `datasets.catalog` is where the dataset catalog is read from. Optional: with no binding the default
        path is used, and with no catalog there no sentence can name a dataset, which is how it worked before."""
        return dict(self.data.get("datasets") or {})

    # --- what the providers read
    def environment(self):
        """The variables this configuration binds. Everything it does not bind stays as the env file left it."""
        exported = {}
        for area in AREAS:
            mapping = CORE_ENVIRONMENT[area]
            for key, value in self.core(area).items():
                variable = mapping.get(key)
                if variable and value is not None:
                    exported[variable] = str(value)
        for key, variable in INTERPRETER_ENVIRONMENT.items():
            value = self.interpreter.get(key)
            if value is not None:
                exported[variable] = str(value)
        for area, variable in QUALITY_ENVIRONMENT.items():
            declared = self.quality(area).get("report")
            if declared is not None:
                exported[variable] = str(declared)
        return exported

    def apply(self, environ=None):
        target = self.environ if environ is None else environ
        bound = self.environment()
        target.update(bound)
        return bound


def config_path(environ=None):
    env = os.environ if environ is None else environ
    named = env.get(PATH_VARIABLE)
    if named:
        return Path(named)
    home = env.get("HOME")
    if home:
        return Path(home) / DEFAULT_PATH[2:]
    return Path(os.path.expanduser(DEFAULT_PATH))


def load(path=None, environ=None):
    """Read the JSON configuration if there is one; otherwise declare the environment as the configuration."""
    env = os.environ if environ is None else environ
    target = Path(path) if path is not None else config_path(env)
    if not target.is_file():
        return Configuration(data=None, path=None, environ=env)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfigError(f"{target}: not a readable JSON document ({error})") from None
    return Configuration(data=validate(raw, environ=env), path=target, environ=env)


def validate(raw, environ=None):
    """Validate, refuse host literals, then expand. Returns the expanded document; raises `ConfigError` otherwise."""
    env = os.environ if environ is None else environ
    if not isinstance(raw, dict):
        raise ConfigError("a configuration must be a JSON object")
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    _validate(raw, schema, "")
    _refuse_host_literals(raw, "")
    return _expand(raw, env, "")
