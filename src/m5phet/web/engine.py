"""Chat translates inputs; installed M5PHET providers remain the execution boundary."""
import csv
import hashlib
import io
import json
import math
import os
import shlex
import subprocess
from datetime import datetime, timezone

from m5phet.interpret import STATUS_OK, Interpreter, interpret
from m5phet.orchestrate import narrate, route
from m5phet.questions import catalog as question_catalog, run_task
from m5phet.runtime import Registry, request_digest, run


def strict_json(text):
    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def finite(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Non-finite JSON number")
        return parsed
    return json.loads(text, parse_constant=invalid, parse_float=finite, object_pairs_hook=pairs)


def parse_file(name, raw):
    text = raw.decode("utf-8-sig")
    if "\x00" in text:
        raise ValueError("Binary files are not supported")
    suffix = name.rsplit(".", 1)[-1].lower()
    if suffix == "json":
        return strict_json(text)
    if suffix == "csv":
        reader = csv.DictReader(io.StringIO(text))
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)) or any(not f.strip() for f in fields):
            raise ValueError("CSV requires unique, non-empty column names")
        rows = list(reader)
        if any(None in row or None in row.values() for row in rows):
            raise ValueError("CSV row width differs from its header")
        return rows
    if suffix in ("txt", "md"):
        return text
    raise ValueError("Supported attachments: .txt, .md, .json, .csv")


DEFAULT_CONFIG = {
    "input": "text", "provider": "laya_news", "family": "classification",
    "output_kind": "typed_questions", "state": "", "presentation": "structured",
    "context": "", "as_of": "", "asset": "EURUSD", "language": "en",
    "max_age_seconds": 900,
    "options": [["yes", "Yes"], ["no", "No"], ["unclear", "Insufficient information"]],
    "parameters": {},
}


def validate_config(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULT_CONFIG):
        raise ValueError("Unknown configuration fields")
    config = DEFAULT_CONFIG | value
    for key in ("provider", "family", "output_kind", "state", "context", "as_of", "asset", "language"):
        if not isinstance(config[key], str) or len(config[key]) > 16000:
            raise ValueError(f"Invalid {key}")
    if config["input"] not in ("text", "json", "csv", "typed_request"):
        raise ValueError("Unknown input reader")
    if config["presentation"] not in ("structured", "json"):
        raise ValueError("Unknown presentation")
    if type(config["max_age_seconds"]) is not int or not 0 <= config["max_age_seconds"] <= 31536000:
        raise ValueError("Invalid freshness limit")
    if not isinstance(config["parameters"], dict):
        raise ValueError("Parameters must be a JSON object")
    options = config["options"]
    if not isinstance(options, list) or len(options) > 12 or any(
        not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(s, str) for s in pair) for pair in options
    ):
        raise ValueError("Options must be ordered label/description pairs")
    return config


class Engine:
    def __init__(self, registry=None):
        self.registry = registry or Registry()
        self.discovery = self.registry.load_entry_points() if registry is None else {"registered": registry.names(), "refused": {}}
        self.remote = os.getenv("M5PHET_CHAT_LAYA_WORKER") if registry is None else None
        self.remote_command = os.getenv("M5PHET_CHAT_LAYA_COMMAND", "")
        self.remote_caps = None
        self.interpreter = Interpreter()
        if self.remote:
            self.remote_capabilities()

    def remote_capabilities(self):
        """The worker's declared capabilities, asked for again until they are known.

        They were once read at start-up only: an instance that started while another held the worker's lock kept the
        in-process fixture's state for the rest of its life and every classification it sent to the worker was refused
        as "not a state this provider holds" (verification of 2026-09-24, two instances started together). A failed
        describe is recorded and retried on the next need; a successful one is kept."""
        if self.remote and self.remote_caps is None:
            try:
                self.remote_caps = self._remote({"action": "describe"}).get("capabilities")
                self.discovery["refused"].pop("laya_worker", None)
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                self.discovery["refused"]["laya_worker"] = str(error)
        return self.remote_caps

    def _remote(self, command):
        if not self.remote_command or self.remote.startswith("-"):
            raise ValueError("Administrator worker command is not configured")
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=7", self.remote,
                shlex.join(shlex.split(self.remote_command))]
        done = subprocess.run(argv, input=json.dumps(command), text=True, capture_output=True, timeout=240)
        try:
            result = strict_json(done.stdout)
        except (ValueError, TypeError):
            raise ValueError("Worker transport failed; no valid JSON response") from None
        if done.returncode or result.get("transport_error"):
            raise ValueError(result.get("transport_error", "Worker returned failure"))
        return result

    def catalog(self):
        remote_caps = self.remote_capabilities()
        providers = [{"name": name, "capabilities": remote_caps if name == "laya_news" and remote_caps else self.registry.capabilities(name)}
                              for name in self.registry.names()]
        examples = []
        for name in self.registry.names():
            method = getattr(self.registry.get(name), "chat_examples", None)
            if callable(method):
                try:
                    examples.extend(method())
                except Exception as error:
                    self.discovery["refused"][name + ":examples"] = str(error)
        if "laya_news" in self.registry.names():
            examples.insert(0, {"title": "Noticia · Clasificación", "prompt": "Which economy is named in this news?",
                               "data": "The European Central Bank left its deposit facility rate unchanged. Incoming euro-area inflation data will guide its next decision.",
                               "config": {"input": "text", "provider": "laya_news", "family": "classification", "output_kind": "typed_questions",
                                          "options": [["euro_area", "Euro area"], ["united_states", "United States"], ["other", "Another economy"]]}})
        return {"providers": providers, "examples": examples, "discovery": self.discovery,
                "defaults": DEFAULT_CONFIG, "profile": "LOCAL_UNGOVERNED", "execution_authorized": False,
                "interpreter": self.interpreter.identity()}

    # --- the question envelope: one shape for every area ----------------------------------------------------------------
    def task_catalog(self):
        return question_catalog(self.registry)

    def propose_task(self, prompt, attachments):
        """A sentence and the SHAPE of the attachment become a proposed envelope. Nothing runs; the person sees it first."""
        data = [parse_file(item["name"], item["data"]) for item in attachments]
        payload = data[0] if len(data) == 1 else (data if data else None)
        return route(prompt, payload, self.registry, interpreter=self.interpreter)

    def execute_task(self, prompt, task, attachments, language="es"):
        """Run an envelope the person accepted (or wrote), then narrate its answers without touching a number."""
        data = [parse_file(item["name"], item["data"]) for item in attachments]
        payload = data[0] if len(data) == 1 else (data if data else None)
        if self.remote and isinstance(task, dict) and task.get("area") == "classification":
            # the real checkpoint lives on the private worker; the envelope goes there on the same contract and comes
            # back bound to the request it answered, exactly as the single-question path does
            response = self._remote({"action": "task", "task": task, "data": payload})
            from m5phet.questions import digest as task_digest, validate_task
            expected = task_digest(validate_task(task))
            if response.get("request_sha256") != expected or response.get("execution_authorized") is not False:
                raise ValueError("Worker result is not bound to this envelope or declares execution authority")
        else:
            response = run_task(task, self.registry, data=payload)
        narration = narrate(prompt, response, interpreter=self.interpreter, language=language)
        return {"task": task, "response": response, "narration": narration, "profile": "LOCAL_UNGOVERNED",
                "execution_authorized": False}

    def execute(self, prompt, config, attachments, *, dry_run=False):
        """Resolve a sentence into the typed request its selected engine will run, and run it.

        With `dry_run` the request is built exactly as it would be run -- words matched, interpreter consulted, adapter
        applied -- and returned unrun, so the person reviews what was understood before anything executes. That is the
        same review the envelope path offers; without it the sentence path would be the only door with no window."""
        config = validate_config(config)
        interpretation = None
        provider = self.registry.get(config["provider"])
        if provider is None:
            raise ValueError(f"Provider '{config['provider']}' is not installed; no fallback was used")
        remote_caps = self.remote_capabilities() if config["provider"] == "laya_news" else None
        if self.remote and config["provider"] == "laya_news" and not remote_caps:
            raise ValueError("The classification worker did not describe itself: "
                             + str(self.discovery["refused"].get("laya_worker", "no reason recorded")))
        caps = remote_caps if remote_caps else self.registry.capabilities(config["provider"])
        as_of = config["as_of"] or datetime.now(timezone.utc).isoformat()
        data = [parse_file(item["name"], item["data"]) for item in attachments]
        if config["input"] == "typed_request":
            request = data[0] if len(data) == 1 else strict_json(prompt)
            if not isinstance(request, dict) or request.get("operation") != "infer":
                raise ValueError("The typed chat route accepts explicit infer requests only")
            if request.get("provider_ref") != config["provider"]:
                raise ValueError("Request provider differs from the provider selected in this chat")
            if request.get("family") != config["family"] or request.get("output_kind") != config["output_kind"]:
                raise ValueError("Request output contract differs from this chat's selected contract")
        elif config["provider"] == "laya_news":
            from news_signal.provider import request_for
            from news_signal.question import ad_hoc_task
            if len(data) > 1:
                raise ValueError("This classifier consumes one text or news JSON per question, not a dataset batch")
            body = data[0] if data else config["context"]
            if isinstance(body, dict):
                if config["input"] != "json":
                    raise ValueError("Select the JSON reader for a structured news event")
                event = body
            elif isinstance(body, str) and body.strip():
                if config["input"] != "text":
                    raise ValueError("Select the text reader for text context")
                now = datetime.now(timezone.utc).isoformat()
                event = {"schema": "news_event.v1", "event_id": "manual:" + hashlib.sha256(body.encode()).hexdigest(),
                         "source": "MANUAL_CONTEXT", "asset": config["asset"], "language": config["language"],
                         "published_at": now, "received_at": now, "headline": "Manual context", "body": body}
                if config["as_of"]:
                    raise ValueError("Historical replay requires a JSON news event with its actual clocks")
                as_of = now
            else:
                raise ValueError("Attach a text/news JSON or enter context in this chat's settings")
            spec = {"name": "answer", "question": prompt, "options": config["options"], "schema": "choice"}
            task = ad_hoc_task(**spec)
            if config["family"] != "classification" or config["output_kind"] != "typed_questions":
                raise ValueError("The news adapter supports classification / typed_questions only")
            states = caps.get("known_states") or []
            state = config["state"] or (states[0] if len(states) == 1 else "")
            request = request_for(event, task_id=task["task_id"], state_ref=state,
                                  as_of=as_of, max_age_seconds=config["max_age_seconds"], question_spec=spec,
                                  request_id="chat:" + hashlib.sha256((prompt + as_of).encode()).hexdigest())
        else:
            builder = getattr(provider, "chat_request", None)
            if not callable(builder):
                raise ValueError(f"'{config['provider']}' has no question adapter yet; use an explicit typed request")
            context = data[0] if len(data) == 1 else data if data else config["context"]
            adapter_config = {key: config[key] for key in ("input", "provider", "family", "output_kind", "state", "parameters")}
            adapter_config["as_of"] = as_of
            # A provider that declares slots is telling us which parameters its engine needs and which values it can
            # accept. Ordinary phrasing is resolved against THAT vocabulary -- by the words first, and only then by the
            # configured interpreter, which may choose among those values and can introduce none.
            slots = getattr(provider, "chat_slots", None)
            if callable(slots):
                declared = slots()
                if declared:
                    resolution = interpret(prompt, declared, interpreter=self.interpreter)
                    if resolution["status"] != STATUS_OK:
                        raise ValueError(resolution["why"])
                    interpretation = resolution
                    request = builder(prompt, context, adapter_config, parameters=resolution["parameters"])
                else:
                    request = builder(prompt, context, adapter_config)
            else:
                request = builder(prompt, context, adapter_config)
            if not isinstance(request, dict) or request.get("operation") != "infer":
                raise ValueError("Question adapters must produce infer requests; chat does not authorize training")
            for field, selected in (("provider_ref", "provider"), ("family", "family"), ("output_kind", "output_kind")):
                if request.get(field) != config[selected]:
                    raise ValueError(f"Question adapter changed selected {field}")
        if dry_run:
            return {"request": request, "interpretation": interpretation, "config": config,
                    "profile": "LOCAL_UNGOVERNED", "execution_authorized": False, "ran": False,
                    "reading": "what the sentence was resolved into; nothing has run"}
        if self.remote and config["provider"] == "laya_news":
            result = self._remote({"action": "infer", "request": request})
            if result.get("request_sha256") != request_digest(request) or result.get("execution_authorized") is not False:
                raise ValueError("Worker result is not bound to this request or declares execution authority")
        else:
            result = run(request, self.registry)
        return {"request": request, "result": result, "profile": "LOCAL_UNGOVERNED",
                "backend": caps.get("backend", "declared_provider"), "execution_authorized": False,
                "interpretation": interpretation}
