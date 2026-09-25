"""WP03: the input component's language model, chosen by configuration instead of by an import.

The interpreter is the one place in this product where a language model is consulted, and the owner's rule about it is
fixed: the model is shown the question and the DECLARED vocabulary, never the data, and whatever it returns is accepted
only if every value in it was already declared. That rule lives in `m5phet.interpret` and is the same for all of them.
What differs between the plugins below is only HOW the model is reached -- a subprocess, a local HTTP server, a remote
one -- and reaching it differently must not require editing this package:

    command             a program the operator already has, called as `COMMAND -z PROMPT`, stdout is the reply
    ollama              a local ollama server over HTTP, CPU only (`num_gpu: 0`), thinking off, reply bounded
    openai_compatible   any `POST {base_url}/chat/completions` endpoint, bearer token read from a NAMED variable

`openai_compatible` refuses to run unless the configuration says `"consent": "cloud_ok"`, because that plugin is the one
that can send the person's sentence and the engine's vocabulary off this machine. The owner's decision of 2026-09-24 is
that no cloud model is the default; consent is therefore explicit, per configuration file, and its absence is a refusal
by name rather than a silent fallback to another plugin.

A plugin is a class. It is constructed with the `interpreter` block of the configuration and the environment, and it
implements `_ask(text) -> str`, `available` and `identity()`; subclassing `m5phet.interpret.Interpreter` gives the last
two for free. External plugins register under the entry-point group below; the three shipped ones are registered the
same way and are looked up in the same order, so nothing here is privileged by being in this package.
"""

ENTRY_POINT_GROUP = "m5phet.interpreters"

DEFAULT_PLUGIN = "command"

#: the shipped plugins, by the name an operator writes in `interpreter.plugin`. They are declared as entry points in
#: `pyproject.toml` and found that way; this map is the fallback for an installation whose metadata is older than its
#: code, so a source tree is never silently without an interpreter.
BUILT_IN = {
    "command": ("m5phet.interpreters.command", "CommandInterpreter"),
    "ollama": ("m5phet.interpreters.ollama", "OllamaInterpreter"),
    "openai_compatible": ("m5phet.interpreters.openai_compatible", "OpenAICompatibleInterpreter"),
}


class InterpreterPluginError(ValueError):
    """A plugin that cannot be selected. Named, never replaced by another one that happens to be installed."""


def _from_entry_points():
    from importlib.metadata import entry_points
    found = {}
    try:
        for entry in entry_points(group=ENTRY_POINT_GROUP):
            found.setdefault(entry.name, entry)
    except Exception:                                                   # noqa: BLE001
        return {}
    return found


def names():
    """Every plugin this installation can select, shipped or external."""
    return sorted(set(_from_entry_points()) | set(BUILT_IN))


def load(name):
    """The class registered under `name`, or a refusal saying which names exist. Nothing is substituted."""
    if not isinstance(name, str) or not name.strip():
        raise InterpreterPluginError(f"an interpreter plugin is named by a string; got {name!r}")
    entry = _from_entry_points().get(name)
    if entry is not None:
        try:
            return entry.load()
        except Exception as error:                                      # noqa: BLE001
            raise InterpreterPluginError(f"interpreter plugin {name!r} is registered but could not be loaded: "
                                         f"{type(error).__name__}: {error}") from None
    if name in BUILT_IN:
        module_name, attribute = BUILT_IN[name]
        from importlib import import_module
        return getattr(import_module(module_name), attribute)
    raise InterpreterPluginError(f"no interpreter plugin named {name!r} is installed; this installation has "
                                 f"{names()}")
