# Local chat workbench

## Scope and design before implementation

One local owner needs a browser interface to the existing M5PHET runtime, not a
second model registry. Conversations, per-chat settings, uploads, exact executed
requests and results stay in an application-local SQLite file. This is application
storage, not a replacement for the scientific warehouse or governed authority.
No training, broker execution, implicit cloud upload or model download.

Input readers: UTF-8 text, JSON and CSV. Processing: installed `m5phet.providers`
entry points. Output: the supported runtime output contract, rendered structurally
or as JSON. Native Laya questions reuse news-signal's request builder. Other engines
can initially consume explicitly typed requests; unsupported natural-language
routes refuse rather than claim to interpret a question. History is conversational
storage, not automatically injected into a stateless classifier's input.

The web extra stays optional. FastAPI serves a small self-hosted responsive client
and bounded HTTP API. No frontend CDN or telemetry. Default bind is loopback;
non-loopback operation requires an owner access token. One worker serializes model
calls, so manual requests cannot start a training sweep or concurrent model loads.
Uploaded names are labels, never filesystem paths. Single-owner access is explicit;
this is not a multi-tenant account system or an internet-hardened public deployment.

## Acceptance and negative-test matrix

| ID | Required behaviour | Test |
|---|---|---|
| W01 | Create, rename, switch, delete and recover chats with independent settings | API + SQLite restart |
| W02 | Upload/download bounded TXT/JSON/CSV per chat; reject foreign IDs and malformed/oversize files | API + parser tests |
| W03 | Question uses selected reader, provider and output contract; same native request and output | Recording provider + installed news fixture; real weights separately |
| W04 | Unsupported or absent provider refuses, never substitutes a fixture | Registry and HTTP refusal |
| W05 | Persist exact prompt/config/attachment digests/request/result and errors per attempt; reload without inference | restart, duplicate send, interrupted job |
| W06 | Text is inert, same-origin requests only, private local store, protected remote mode | traversal/XSS payload, Origin/Host/auth, size limits |
| W07 | Sidebar, composer attachment button, settings dialog, results and empty/error/loading states fit desktop/mobile | Playwright functional tests/screenshots |
| W08 | Core remains usable without web dependencies; existing runtime suite intact | full suite and wheel asset inspection |

Tests are specified before implementation. Independent real Laya equality is
retained evidence, not newly established by UI fixtures. Manual UI results are
LOCAL_UNGOVERNED; no backtest, calibration or financial benefit is implied.

## Running the workbench

```bash
python -m venv ~/.local/share/m5phet/chat-venv
~/.local/share/m5phet/chat-venv/bin/pip install '.[web]'
~/.local/share/m5phet/chat-venv/bin/m5phet-chat --port 8765
```

Open `http://127.0.0.1:8765`. The owner database and attachments live in
`~/.local/state/m5phet/chat/chat.sqlite3`; change with `--state-dir`. Attachments
are UTF-8 TXT/MD/JSON/CSV, up to 8 MiB each and 250 MiB total. JSON rejects
duplicate keys and nonfinite numbers; CSV rejects duplicate columns/ragged rows.
Uploaded model binaries cannot be loaded. Model states must be installed and
allowlisted by each provider's operator configuration. Chats can be renamed,
exported and deleted. Each turn retains settings and attachment digests; a queued
turn does not change when the chat settings change. Interrupted turns remain
INTERRUPTED and are not automatically repeated. No broker methods are exposed.

Provider wheels are independently installed into this environment through
`m5phet.providers`. The menu reflects only installed providers and declared
combinations. No provider is silently replaced. Native Laya questions use the
existing news-signal builder and runtime. Its English checkpoint is not a claim
of Spanish classification quality. A classification requires context/news data
and explicit ordered answer options; it is not a generative assistant.

Other provider adapters expose `chat_request(prompt,data,config)` and optional
`chat_examples()`. Their bounded supported language and model/input contracts
remain explicit. This release does not claim general natural-language reasoning
or free-form interpretation across all domains. The explicit typed-request mode
allows only inference, with the selected provider/family/output contract unchanged.

### Real Laya on a private worker

For local execution, configure the documented news-signal checkpoint, manifest,
device and UUID environment variables before starting the app. No weights are
downloaded on startup. For the existing private worker, deploy `worker.py` from
`src/m5phet/web/` and `tools/chat_laya_worker.sh` beside each other. Configure:

```bash
export M5PHET_CHAT_LAYA_WORKER='<your-ssh-host>'
export M5PHET_CHAT_LAYA_COMMAND='bash /absolute/path/chat_laya_worker.sh'
```

The existing worker SDK environment owns its pinned dependencies. The app uses
SSH authentication already configured by the operator; prompts cannot alter the
host, command or device. The wrapper requires exactly one external RTX5090 and
refuses busy/hot/low-memory execution. It takes a nonblocking chat lease, applies
a 175-second wall timeout and goes through the worker's M5PHET Registry. There
is no fallback to an internal GPU. Cold loading occurs per manual request; this
does not advertise the earlier resident-model hot latency. Worker responses must
match the exact request digest. A source hash/model identity is retained in the
response; it is not governed scientific acceptance or cross-device parity.

### Mobile and access boundary

The UI fits mobile browsers and the sidebar collapses. Loopback is the default,
not a public deployment. For a private-network bind, supply `M5PHET_CHAT_TOKEN`
(at least 24 characters), `--host` and explicit `--allowed-host` entries. Enter
the token in the login dialog; it becomes an HttpOnly SameSite cookie. Use TLS
or an authenticated private tunnel on untrusted networks. This is single-owner,
not multi-tenant. Do not expose an unauthenticated public reverse proxy.

### Verification and upstream UI dependencies

```bash
pip install '.[web,web-test]'
pytest tests/test_web.py tests/test_domain_outputs.py
playwright install chromium
python tools/check_chat_browser.py --out /tmp/chat-browser
```

`--infer` additionally invokes the configured real classification provider;
without it, screenshots and layout checks do not load weights. The UI is a small
self-hosted client, not an AdminLTE deployment; the requested sidebar/dialog/chat
layout does not require that full template distribution. Lucide 1.48.0 is vendored
with its license. References: [FastAPI uploads](https://fastapi.tiangolo.com/tutorial/request-files/)
and [Lucide vanilla usage](https://lucide.dev/guide/lucide). No CDN at runtime.
