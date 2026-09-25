"""An MCP server that exposes M5PHET as tools any language-model client can call.

The owner's design asks for exactly this: a client that speaks the Model Context Protocol reads the tool schema, learns how
an envelope is built, and calls `m5phet_execute_ml_task(area, state, questions)`. The model does the semantic mapping; the
engines compute; the contract stands between them unchanged.

What the server is, and is not. It speaks JSON-RPC 2.0 over stdio, the transport every MCP host supports, and it
implements the handshake and the tools methods. It is not a second runtime: every call goes through the same registry, the
same `run_task` and the same refusals as the web workbench, so a tool call cannot obtain anything the interface could not.
It has no filesystem tools, no shell and no network. It answers questions about fitted models and nothing else.

Run it with `python -m m5phet.mcp_server`. The tools:

    m5phet_catalog            -- the areas served and the question types each declares, with required and optional fields
    m5phet_execute_ml_task    -- run one envelope; every question comes back answered or refused by name
    m5phet_propose_task       -- turn a sentence and a data profile into a proposed envelope (requires the interpreter)
"""

import json
import sys

from .orchestrate import dataset_profile, route
from .questions import TaskError, catalog as question_catalog, run_task
from .runtime import Registry

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "m5phet", "version": "0.1.0"}

TOOLS = [
    {"name": "m5phet_catalog",
     "description": ("List the areas M5PHET serves in this installation and the question types each answers, with the "
                     "required and optional fields of every type. Call this before building an envelope."),
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "m5phet_execute_ml_task",
     "description": ("Run one typed envelope against the installed engines. Every question is answered on its own; a "
                     "question the engine cannot answer comes back REFUSED with a typed reason and never a number. No "
                     "answer authorizes execution of anything."),
     "inputSchema": {"type": "object",
                     "properties": {
                         "area": {"type": "string", "enum": ["classification", "forecasting", "causal", "rl", "unsupervised"]},
                         "state": {"type": "object", "description": "what is being asked about: dataset reference, fitted "
                                                                    "state, causal graph, observation, features"},
                         "questions": {"type": "object", "description": "name -> {type, ...fields}",
                                       "additionalProperties": {"type": "object"}},
                         "data": {"description": "the rows or text the questions are about; optional"},
                         "as_of": {"type": "string", "description": "ISO-8601 decision clock; optional"}},
                     "required": ["area", "state", "questions"], "additionalProperties": False}},
    {"name": "m5phet_propose_task",
     "description": ("Turn a sentence and the SHAPE of the attached data into a proposed envelope. The proposal is "
                     "validated against the catalog and the data's columns; it is returned for review and is not run."),
     "inputSchema": {"type": "object",
                     "properties": {"prompt": {"type": "string"}, "data": {"description": "rows or text; only its shape "
                                                                                          "is shown to the interpreter"}},
                     "required": ["prompt"], "additionalProperties": False}},
]


class Server:
    def __init__(self, registry=None, engine=None):
        self.registry = registry
        # Without an explicit registry the server is the workbench's engine: the same entry points AND the same route to
        # the private classification worker, bound by the envelope digest. Calling run_task on the local registry
        # alone answered classification from whatever backend this host declares, which on the coordinator is the
        # fixture -- an MCP client saw label `other` at 1.0 where the workbench says euro_area 0.9666 (WP11, 2026-09-24).
        self.engine = engine
        if self.registry is None:
            from .web.engine import Engine
            self.engine = engine or Engine()
            self.registry = self.engine.registry
            self.discovery = self.engine.discovery
        else:
            self.discovery = {"registered": self.registry.names(), "refused": {}}

    # --- JSON-RPC ---------------------------------------------------------------------------------------------------
    def handle(self, message):
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return self._error(None, -32600, "invalid request")
        method, params, ident = message.get("method"), message.get("params") or {}, message.get("id")
        if ident is None:
            return None                                         # a notification: nothing is returned for it
        try:
            if method == "initialize":
                result = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}}, "serverInfo": SERVER_INFO}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self.call(params.get("name"), params.get("arguments") or {})
            else:
                return self._error(ident, -32601, f"method not found: {method}")
        except TaskError as error:
            return self._error(ident, -32602, str(error))
        except Exception as error:                              # noqa: BLE001
            return self._error(ident, -32603, f"{type(error).__name__}: {error}")
        return {"jsonrpc": "2.0", "id": ident, "result": result}

    @staticmethod
    def _error(ident, code, message):
        return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}}

    def _execute(self, envelope, data):
        """Through the engine when there is one (worker route, digest binding, checked narration); else the runtime."""
        if self.engine is None:
            return run_task(envelope, self.registry, data=data)
        attachments = [] if data is None else [{"name": "data.json", "data": json.dumps(data).encode()}]
        detail = self.engine.execute_task("", envelope, attachments, language="en")
        return {**detail["response"], "narration": detail["narration"], "execution_authorized": False}

    # --- tools --------------------------------------------------------------------------------------------------------
    def call(self, name, arguments):
        if name == "m5phet_catalog":
            payload = {"areas": question_catalog(self.registry), "discovery": self.discovery,
                       "execution_authorized": False}
        elif name == "m5phet_execute_ml_task":
            envelope = {k: arguments[k] for k in ("area", "state", "questions") if k in arguments}
            if arguments.get("as_of"):
                envelope["as_of"] = arguments["as_of"]
            payload = self._execute(envelope, arguments.get("data"))
        elif name == "m5phet_propose_task":
            payload = route(arguments.get("prompt", ""), arguments.get("data"), self.registry)
            payload.pop("catalog", None)
            payload["profile"] = dataset_profile(arguments.get("data"))
        else:
            raise TaskError(f"unknown tool {name!r}")
        return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, default=str)}],
                "structuredContent": json.loads(json.dumps(payload, default=str)),
                "isError": False}


def serve(stdin=None, stdout=None, registry=None):
    """Read one JSON-RPC message per line, write one response per request. Notifications get no response."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    server = Server(registry)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            response = server._error(None, -32700, "parse error")
        else:
            response = server.handle(message)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            stdout.flush()


if __name__ == "__main__":
    serve()
