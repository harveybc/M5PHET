"""The `openai_compatible` plugin: any `chat/completions` endpoint -- and only with the owner's written consent.

Every other plugin keeps the person's sentence on this machine. This one does not: the sentence and the engine's
declared vocabulary are sent to whatever `base_url` names. That is a decision about where the owner's words go, not a
performance setting, so it is not made by whoever edits the configuration last. The plugin refuses to run unless the
configuration states it in as many words:

    "interpreter": {"plugin": "openai_compatible", "consent": "cloud_ok",
                    "base_url": "$M5PHET_OPENAI_BASE_URL", "api_key_env": "M5PHET_OPENAI_API_KEY",
                    "model": "..."}

Without `"consent": "cloud_ok"` the plugin is unavailable and says why; asking it anyway raises rather than falls back
to another interpreter. The owner's standing decision of 2026-09-24 is that no ChatGPT and no Claude is the default;
this plugin exists so a paid model can be TRIED at the end of a comparison, named in the report, and never arrived at
by accident.

The key is never written in the configuration: `api_key_env` names the ENVIRONMENT VARIABLE that holds it, and an
unset variable is refused by name. The key is not put in `identity()`, so it cannot reach the browser or a receipt.
Temperature is 0 because the task is to choose among declared values, not to write.
"""

import json
import math
import re
import urllib.error
import urllib.request

from ..interpret import Interpreter

CONSENT = "cloud_ok"

DEFAULT_KEY_VARIABLE = "M5PHET_OPENAI_API_KEY"

DEFAULT_MAX_TOKENS = 512

NO_CONSENT = ("this interpreter would send the question and the declared vocabulary off this machine; it runs only "
              "when the configuration states interpreter.consent = \"cloud_ok\"")


class CloudConsentRequired(ValueError):
    """Asked to reach a remote model without the configuration's explicit consent. Never downgraded to a fallback."""


class OpenAICompatibleInterpreter(Interpreter):
    """A remote `POST {base_url}/chat/completions` model, selected by `"plugin": "openai_compatible"`."""

    plugin = "openai_compatible"

    def __init__(self, settings=None, environ=None):
        settings = dict(settings or {})
        super().__init__(command=settings.get("command"), model=settings.get("model"), environ=environ)
        self.base_url = (settings.get("base_url") or "").rstrip("/")
        self.api_key_env = settings.get("api_key_env") or DEFAULT_KEY_VARIABLE
        self.consent = settings.get("consent")
        self.max_tokens = int(settings.get("max_tokens") or DEFAULT_MAX_TOKENS)
        # an endpoint that does not return logprobs is configured with "logprobs": false, and then this plugin reports
        # no confidence either -- which the catalog prints as CONFIDENCE_NOT_REPORTED rather than as a passing gate
        self.logprobs = settings.get("logprobs", True) is not False
        if settings.get("timeout_seconds"):
            self.timeout = int(settings["timeout_seconds"])

    @property
    def consented(self):
        return self.consent == CONSENT

    @property
    def available(self):
        return bool(self.consented and self.base_url and self.model)

    def identity(self):
        """What reached the model, never how to reach it: no base URL, no key, no variable's value."""
        out = {"plugin": self.plugin, "command": None, "model": self.model or None, "available": self.available,
               "reports_confidence": self.reports_confidence,
               "consent": self.consent if self.consented else "NOT_GIVEN",
               "endpoint": "a configured OpenAI-compatible endpoint (off this machine)",
               "reading": "the model chooses among declared values; it cannot introduce one"}
        if not self.consented:
            out["why"] = NO_CONSENT
        elif not self.base_url or not self.model:
            out["why"] = "interpreter.base_url and interpreter.model are both required by this plugin"
        return out

    @property
    def reports_confidence(self):
        """This is the one shipped plugin that CAN say how sure the model was: `chat/completions` returns a logprob
        per token when asked for one. An endpoint configured with `"logprobs": false` reports none and says so."""
        return bool(self.logprobs)

    def body(self, text):
        """The request this plugin sends, as a dict, so a test can read it without a server."""
        out = {"model": self.model, "messages": [{"role": "user", "content": text}],
               "temperature": 0, "max_tokens": self.max_tokens, "stream": False}
        if self.logprobs:
            out["logprobs"] = True
        return out

    def propose_with_confidence(self, prompt, slots):
        """`(proposed, {field: probability})` -- the model's own probability for the tokens that spell each value.

        The number is not an opinion about the model and not a score this package computes: it is
        `exp(sum of the logprobs the endpoint returned for the tokens covering that value)`, which is the endpoint's
        own probability of having written those characters and nothing else. A field whose tokens cannot be located --
        an endpoint that returned no logprobs, a reply whose tokens do not concatenate to the text it also returned --
        is simply ABSENT from the mapping, because "not measured" and "measured low" are different facts and
        `m5phet.interpret` refuses on each by its own name."""
        raw, tokens = self._ask_detailed(self.instruction(prompt, slots))
        proposed = self.parse(raw)
        return proposed, confidences_from_logprobs(raw, tokens, proposed)

    def _ask(self, text):
        return self._ask_detailed(text)[0].strip()

    def _ask_detailed(self, text):
        if not self.consented:
            raise CloudConsentRequired(NO_CONSENT)
        if not self.base_url or not self.model:
            raise ValueError("interpreter.base_url and interpreter.model are both required by this plugin")
        key = self.environ.get(self.api_key_env)
        if not key:
            raise ValueError(f"environment variable ${self.api_key_env} is not set; it is where this plugin reads the "
                             f"bearer token, which is never written into a configuration file")
        request = urllib.request.Request(f"{self.base_url}/chat/completions",
                                         data=json.dumps(self.body(text)).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {key}"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                answer = json.load(reply)
        except urllib.error.URLError as error:
            raise ValueError(f"the configured endpoint did not answer ({error.reason})") from None
        choices = answer.get("choices") if isinstance(answer, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ValueError("the endpoint returned no choices")
        content = ((choices[0].get("message") or {}).get("content") or "") if isinstance(choices[0], dict) else ""
        tokens = ((choices[0].get("logprobs") or {}).get("content")) if isinstance(choices[0], dict) else None
        # NOT stripped: the tokens are offsets into exactly these characters, and trimming them here would move every
        # span by however much whitespace the model opened with. `_ask` strips, because its caller only reads words.
        return content, tokens if isinstance(tokens, list) else None


def confidences_from_logprobs(text, tokens, proposed):
    """`{field: probability}` for the fields whose value can be located in the tokens the endpoint returned.

    The reply's tokens concatenate to the reply. So the value of each field occupies a span of characters, that span
    covers a run of tokens, and the probability of the model having produced exactly those tokens is the product of
    their probabilities -- `exp(sum of logprobs)`. A token that straddles the boundary counts whole, which is the
    conservative direction: it can only lower the number, never raise it above what the endpoint reported.

    `None` -- not an empty mapping and never a 1.0 -- whenever the endpoint returned no logprobs. Nothing here
    estimates a confidence; it only reads one back."""
    if not tokens:
        return None
    spans = _value_spans(text, proposed)
    if not spans:
        return {}
    offsets, position = [], 0
    for entry in tokens:
        piece = entry.get("token") if isinstance(entry, dict) else None
        if not isinstance(piece, str):
            return None                             # not the shape this function reads; nothing is guessed from it
        offsets.append((position, position + len(piece), entry.get("logprob")))
        position += len(piece)
    if position < max(end for _start, end in spans.values()):
        return None                                 # the tokens do not cover the text they came with; measure nothing
    out = {}
    for field, (start, end) in spans.items():
        total, seen = 0.0, False
        for begin, stop, logprob in offsets:
            if stop > start and begin < end:
                if not isinstance(logprob, (int, float)):
                    seen = False
                    break
                total += float(logprob)
                seen = True
        if seen:
            out[field] = min(1.0, math.exp(total))
    return out


def _value_spans(text, proposed):
    """`{field: (start, end)}` -- where each field's VALUE sits in the reply, read with a JSON scanner, not a guess."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    document, base = match.group(0), match.start()
    decoder = json.JSONDecoder()
    spans = {}
    for field in proposed:
        key = re.search(r'"' + re.escape(str(field)) + r'"\s*:\s*', document)
        if not key:
            continue
        try:
            _value, end = decoder.raw_decode(document, key.end())
        except ValueError:
            continue
        spans[field] = (base + key.end(), base + end)
    return spans
