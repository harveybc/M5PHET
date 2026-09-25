"""The `ollama` plugin: a model on this machine, reached over HTTP, with no network and no GPU.

This is `tools/interpreter_ollama.py` moved into the package, where it can be selected by configuration instead of by
wrapping it in `M5PHET_INTERPRETER_COMMAND`. It exists because the product must be able to read a person's sentence
with nothing leaving the machine: the owner's primary interpreter goes through a provider over the network, and when
that is unavailable -- or when an audit needs a run with no external dependency at all -- a local model answers.

Three settings are deliberate. `num_gpu: 0` keeps it off this host's GPUs, which are either the owner's or held for the
worker's external card; the coordinator's card is never used for reading a sentence. `think: false` (and the `</think>`
strip for a model that thinks anyway) keeps the reasoning out of the reply, because the reply must be the JSON object
the caller parses -- `qwen3:4b` was unusable for exactly this reason. `num_predict` bounds the reply, so a model that
starts writing an essay is cut rather than left to hold the request open.

What the model may CHOOSE is not decided here: `m5phet.interpret` shows it the declared vocabulary and refuses anything
outside it, for this plugin exactly as for the others.
"""

import json
import urllib.error
import urllib.request

from ..interpret import Interpreter

#: the loopback server ollama listens on by default. It is a literal here on purpose: this is code, not the operator's
#: configuration file, and 127.0.0.1 names no host but this machine. A configuration that needs another one writes
#: `"base_url": "$OLLAMA_HOST"` -- the file itself may not carry a host literal.
DEFAULT_BASE_URL = "http://127.0.0.1:11434"

DEFAULT_MAX_TOKENS = 512


class OllamaInterpreter(Interpreter):
    """A local ollama model, selected by `"plugin": "ollama"` with `"model": "llama3.2:3b"`."""

    plugin = "ollama"

    def __init__(self, settings=None, environ=None):
        settings = dict(settings or {})
        super().__init__(command=settings.get("command"), model=settings.get("model"), environ=environ)
        env = self.environ
        self.base_url = (settings.get("base_url") or env.get("OLLAMA_HOST") or DEFAULT_BASE_URL).rstrip("/")
        self.max_tokens = int(settings.get("max_tokens") or DEFAULT_MAX_TOKENS)
        if settings.get("timeout_seconds"):
            self.timeout = int(settings["timeout_seconds"])

    @property
    def available(self):
        """A model must be named. Whether the server answers is found out by asking it, and a server that does not is
        reported as the refusal it is -- not guessed at here with a second request per question."""
        return bool(self.model)

    def identity(self):
        return {"plugin": self.plugin, "command": None, "model": self.model or None, "available": self.available,
                "reports_confidence": self.reports_confidence,
                "endpoint": "local ollama server", "device": "CPU (num_gpu: 0)",
                "reading": "the model chooses among declared values; it cannot introduce one"}

    def body(self, text):
        """The request this plugin sends, as a dict, so a test can read it without a server."""
        return {"model": self.model, "stream": False, "think": False,
                "messages": [{"role": "user", "content": text}],
                "options": {"num_gpu": 0, "num_predict": self.max_tokens, "temperature": 0}}

    def _ask(self, text):
        if not self.model:
            raise ValueError("the ollama interpreter needs a model: set interpreter.model, e.g. \"llama3.2:3b\"")
        request = urllib.request.Request(f"{self.base_url}/api/chat", data=json.dumps(self.body(text)).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                answer = json.load(reply)
        except urllib.error.URLError as error:
            raise ValueError(f"the local ollama server did not answer ({error.reason}); is it running?") from None
        if not isinstance(answer, dict):
            raise ValueError("the ollama server returned something that is not an object")
        content = ((answer.get("message") or {}).get("content") or "")
        if "</think>" in content:                   # a model that thought anyway: only what follows is the reply
            content = content.split("</think>", 1)[1]
        return content.strip()
