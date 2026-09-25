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

from m5phet import config as configuration_module, datasets as dataset_catalog
from m5phet.interpret import ABSTENTION_STATUSES, STATUS_OK, build as build_interpreter, interpret
from m5phet.orchestrate import narrate, route
from m5phet.outputs import select as select_output
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
    def __init__(self, registry=None, configuration=None, environ=None):
        """Read the bindings before anything is constructed.

        Every provider owns its own environment variables and reads them when it is built, so the JSON configuration
        has to reach the environment BEFORE the registry loads the entry points -- otherwise a provider would be
        constructed from the env file while the catalog reported the JSON. `config.load()` returns the environment
        itself as a valid configuration when no file exists, so the operator's chat.env keeps working unchanged."""
        self.environ = os.environ if environ is None else environ
        self.configuration = configuration_module.load(environ=self.environ) if configuration is None else configuration
        self.config_source = self.configuration.source
        self.configuration.apply(self.environ)
        self.registry = registry or Registry()
        self.discovery = self.registry.load_entry_points() if registry is None else {"registered": registry.names(), "refused": {}}
        classification = self.configuration.core("classification")
        self.remote = classification.get("worker") or self.environ.get("M5PHET_CHAT_LAYA_WORKER") if registry is None else None
        self.remote_command = classification.get("command") or self.environ.get("M5PHET_CHAT_LAYA_COMMAND", "")
        self.remote_caps = None
        # WP03: which interpreter implementation reads a sentence is a configuration choice, not an import. The
        # `interpreter` block goes to the plugin unchanged; with no JSON file it is empty and the `command` plugin
        # reads M5PHET_INTERPRETER_COMMAND exactly as before.
        self.interpreter = build_interpreter(self.configuration.interpreter, environ=self.environ)
        # WP15: the catalog of what the data lake holds, read once. It carries descriptions and never rows, so a
        # sentence may NAME a dataset instead of attaching a file. An installation with no catalog gets an empty one
        # and every sentence is routed exactly as it was before.
        try:
            self.datasets = dataset_catalog.load_catalog(configuration=self.configuration, environ=self.environ)
        except dataset_catalog.CatalogError as error:
            self.datasets = {"schema": dataset_catalog.CATALOG_SCHEMA, "datasets": [], "aliases": {}}
            self.discovery["refused"]["dataset_catalog"] = str(error)
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
                # the interpreter's identity AND how often it was measured to read a sentence correctly -- or
                # NOT_MEASURED, which is what every M5PHET report said about it until 2026-09-25 without saying so
                "interpreter": {**self.interpreter.identity(),
                                "reliability": self.interpreter_reliability()},
                # the rule a language model's choices are held to here, published so that anything reading this
                # catalog -- the web, an MCP client, a Telegram skill -- can see it before it trusts an answer
                "abstention": self.abstention(),
                # which of the two configurations is in force: the JSON file, or the operator's environment
                "config_source": self.config_source,
                "config": {"areas": {area: {"provider": self.configuration.provider(area),
                                            "output": self.configuration.output(area)}
                                     for area in configuration_module.AREAS if self.configuration.provider(area)},
                           "surfaces": self.configuration.surfaces}}

    def interpreter_reliability(self):
        """`interpreter.reliability` as the catalog publishes it: a cited measurement of THIS interpreter, or
        `NOT_MEASURED`. Declared by `interpreter.reliability_report`; see `m5phet.interpret.declared_reliability`."""
        from m5phet import interpret as interpret_module
        return interpret_module.declared_reliability(self.configuration, self.environ, interpreter=self.interpreter)

    def abstention(self):
        """The declared rule, for the interpreter and for every area's chooser -- or `NOT_CONFIGURED`.

        Two consumers, one declaration. The `chooser` half is what `m5phet.decide` applies when Laya picks an area's
        configuration; the `interpreter` half is the same number applied to the interpreter's own choice of a
        parameter, and it carries whether the configured plugin can be held to it at all. A plugin that reports no
        confidence is published as `CONFIDENCE_NOT_REPORTED`: the rule exists, it cannot be applied here, and neither
        the catalog nor the sentence path pretends otherwise."""
        from m5phet import decide, interpret as interpret_module
        rule = decide.declared_rule(self.configuration, self.environ)
        identity = self.interpreter.identity()
        reports = bool(identity.get("reports_confidence"))
        interpreter_view = {"rule": rule,
                            "confidence": (interpret_module.CONFIDENCE_REPORTED if reports
                                           else interpret_module.CONFIDENCE_NOT_REPORTED),
                            "plugin": identity.get("plugin")}
        if not reports:
            interpreter_view["why"] = (f"the {identity.get('plugin')!r} interpreter returns text; it reports no "
                                       f"confidence for the values it chooses among the declared ones. With a "
                                       f"threshold declared, a sentence whose parameters only a model could settle is "
                                       f"refused as CONFIDENCE_NOT_REPORTED rather than passed as if the rule had run")
        return {"interpreter": interpreter_view,
                "areas": {area: rule for area in configuration_module.AREAS}}

    # --- the question envelope: one shape for every area ----------------------------------------------------------------
    def task_catalog(self):
        return question_catalog(self.registry, self.configuration)

    def propose_task(self, prompt, attachments):
        """A sentence and the SHAPE of the attachment become a proposed envelope. Nothing runs; the person sees it first.

        With nothing attached, the sentence may name a dataset of the data lake; the proposal then carries
        `dataset: {id, source, rows, columns, source_of_choice}` -- plus, when more than one dataset fitted the
        words, the `dataset_choice` decision record Laya made -- and the person reviews which data will be read and
        who chose it, before anything runs."""
        data = [parse_file(item["name"], item["data"]) for item in attachments]
        payload = data[0] if len(data) == 1 else (data if data else None)
        return route(prompt, payload, self.registry, interpreter=self.interpreter, datasets=self.datasets,
                     decider=self)

    def output(self, area):
        """The procedure configured for this area: `areas.<area>.output.plugin`, `default` when nothing is bound."""
        return select_output(area, self.configuration.output(area) if area else {})

    def output_headers(self):
        """What each area RETURNS, per its configured procedure: output kind, unit fields, statuses, refusal codes.

        This is the half of the output component that does not depend on the screen: an integrator reads it and knows
        the shape of an answer before asking anything."""
        return {area: {"plugin": self.output(area).name, **self.output(area).header(area)}
                for area in configuration_module.AREAS}

    def resolved_rows(self, prompt, task):
        """`(rows, resolution, governance)` for the dataset this envelope names, or `(None, resolution, None)`.

        The envelope carries the id the proposal resolved (`state.dataset`), so what runs reads what the person
        reviewed; a hand-written envelope may name one the same way, and a sentence with no envelope id is resolved
        from its own words. A governed resource is read THROUGH data-gov and `governance` is its receipt; if
        data-gov is not configured or refuses, the refusal is raised by name and is what the person is shown. There
        is no path from a refusal to a local read of the same bytes."""
        state = task.get("state") if isinstance(task, dict) else None
        subject = state if isinstance(state, dict) and state.get("dataset") else prompt
        # no decider at execution: the choice was made and recorded when the person reviewed the proposal, and its
        # id travels in `state.dataset`. A run is never the place to ask Laya again for a different dataset.
        resolution = dataset_catalog.resolve(subject, self.datasets, None)
        if resolution["status"] != dataset_catalog.OK:
            return None, resolution, None
        rows, governance = dataset_catalog.load_rows_with_receipt(resolution["dataset"])
        return rows, resolution, governance

    def execute_task(self, prompt, task, attachments, language=None):
        """Run an envelope the person accepted (or wrote), then narrate its answers without touching a number."""
        data = [parse_file(item["name"], item["data"]) for item in attachments]
        payload = data[0] if len(data) == 1 else (data if data else None)
        dataset = None
        governance = None
        if payload is None and (self.datasets or {}).get("datasets"):
            rows, resolution, governance = self.resolved_rows(prompt, task)
            if rows is not None:
                payload, dataset = rows, dataset_catalog.proposal_view(resolution)
                # how the inputs came to be what the engine was handed, recorded beside the answer
                receipt = dataset_catalog.rows_receipt(resolution["dataset"])
                receipt["rows_sha256"] = dataset_catalog.rows_sha256(rows)
                receipt["rows_handed_over"] = len(rows)
                dataset["rows_receipt"] = receipt
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
        # WP04: which procedure renders the answers is the area's configuration, not this method's business. The
        # plugin's text is checked against the answers exactly as a model's narration is, in `narrate`.
        area = task.get("area") if isinstance(task, dict) else None
        language = language or (self.configuration.output(area).get("language") if area else None) or "es"
        narration = narrate(prompt, response, interpreter=self.interpreter, language=language, area=area, task=task,
                            plugin=self.output(area))
        # WP15: a run whose rows came through data-gov IS a governed run, and says so beside its answers. Nothing
        # else in this method may set that profile: it is the receipt that makes it true, not an intention.
        profile = dataset_catalog.GOVERNED_PROFILE if governance else "LOCAL_UNGOVERNED"
        return {"task": task, "response": response, "narration": narration, "profile": profile,
                "dataset": dataset, "governance": governance, "execution_authorized": False}

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
                    resolution = interpret(prompt, declared, interpreter=self.interpreter,
                                           configuration=self.configuration, environ=self.environ)
                    if resolution["status"] in ABSTENTION_STATUSES:
                        # not an error: a refusal to CHOOSE. The person is shown which parameter went unresolved and
                        # what the declared values are, in the same review window the resolved request would have
                        # used, so the next step is theirs (name the value) and not a stack trace.
                        if dry_run:
                            return {"request": None, "interpretation": resolution, "config": config,
                                    "status": "REFUSED", "refusal": resolution["status"],
                                    "unresolved": resolution.get("unresolved") or [],
                                    "declared": resolution.get("declared") or {},
                                    "why": resolution["why"], "profile": "LOCAL_UNGOVERNED",
                                    "execution_authorized": False, "ran": False,
                                    "reading": "the interpreter did not choose; nothing was resolved and nothing ran"}
                        raise ValueError(f"{resolution['status']}: {resolution['why']}")
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
