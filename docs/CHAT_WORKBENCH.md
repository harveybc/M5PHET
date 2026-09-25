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

## Starting it with all five families

`tools/start_chat.sh` is the operator's declaration of where each engine and each
fitted state lives. Every engine stays in the environment where its dependencies
already are -- TensorFlow, Stable-Baselines3 and EconML never enter this one --
and a prompt cannot change any of it.

```bash
tools/start_chat.sh                       # http://127.0.0.1:8765
M5PHET_CHAT_LAYA_WORKER=<your-worker> \
M5PHET_CHAT_LAYA_COMMAND='bash /absolute/path/chat_laya_worker.sh' \
    tools/start_chat.sh                   # classification on the real checkpoint
```

Without the worker variables, classification uses the declared `NON_MODEL_FIXTURE`
and says so in every receipt. The five families and what each needs:

| Family | Provider | Engine runs in | Fitted state |
|---|---|---|---|
| Classification | `laya_news` | the private worker, or a local checkpoint | sealed Laya checkpoint |
| Forecasting | `predictor_forecast` | `M5PHET_FORECAST_PYTHON` (TensorFlow) | exported DEV bundle |
| Hierarchical regimes | `feature-eng-hierarchical-regimes` | this environment | fitted reference + assignments |
| Causal inference | `causal_inference` | this environment (inference only) | a study fitted beforehand |
| Policy | `trading_policy` | `M5PHET_POLICY_PYTHON` (Stable-Baselines3) | a retained policy checkpoint |

A study is fitted once, explicitly, before it can be served:

```bash
~/.local/share/m5phet/causal-fit-venv/bin/python -m causal_inference_provider prepare-demo
```

Every state above is DEVELOPMENT provenance. The workbench shows what these
engines answer; it establishes nothing about how well they answer.

## Asking in your own words

A provider that declares `chat_slots()` tells the workbench which parameters its
engine needs and, for each one, every value it actually has. Ordinary phrasing is
resolved against exactly that vocabulary: the words are matched first, and only
what remains goes to the configured interpreter, which is shown the question and
the allowed values and whose answer is accepted only if every value in it was
already declared.

```bash
M5PHET_INTERPRETER_COMMAND='hermes --ignore-user-config' \
M5PHET_INTERPRETER_MODEL='deepseek-v4-pro (OpenCode Go)' \
    tools/start_chat.sh
```

Without those variables the workbench still resolves whatever the words settle and
refuses the rest, naming the missing parameter and the values the model does have.
Each answer shows which parameters came from your words and which the interpreter
chose, with the interpreter's identity beside them.

What the interpreter cannot do, by construction: introduce a target the bundle
does not hold, a horizon the model was not trained for, or any value outside the
provider's declaration. A question naming an unsupported value is refused by name
rather than answered with the nearest supported one -- that would answer a
different question. An ambiguous question is refused with its candidates named.
The interpreter is never shown the uploaded data, only the question and the
vocabulary.

Measured against the real TensorFlow bundle: `forecast Global_active_power at 60
steps`, `predict household power one hour ahead` and `cuánta potencia habrá en la
próxima hora?` all reach the engine and return its recorded 0.5412255525588989;
`what will consumption look like shortly?` is completed by the interpreter and
returns the same; `at 90 steps` and `forecast Voltage` are refused.

## Which interpreter reads the sentence (`m5phet.interpreters`)

Reading a person's words is the one place a language model is consulted, and
**how** it is reached is a configuration choice, not an import. Three plugins
ship, registered under the entry-point group `m5phet.interpreters`; an external
distribution registers its own the same way and is found the same way.

| `interpreter.plugin` | How the model is reached | Settings it reads |
|---|---|---|
| `command` (default) | a program on this machine, called as `COMMAND -z PROMPT`; stdout is the reply | `command`, `model`, `timeout_seconds` |
| `ollama` | a local ollama server over HTTP, CPU only (`num_gpu: 0`), thinking off, reply bounded | `model`, `base_url`, `max_tokens`, `timeout_seconds` |
| `openai_compatible` | `POST {base_url}/chat/completions` with a bearer token | `base_url`, `model`, `api_key_env`, **`consent`**, `max_tokens` |

```json
{"schema": "m5phet.config.v1",
 "interpreter": {"plugin": "ollama", "model": "llama3.2:3b"}}
```

With **no** JSON file nothing changes: the `command` plugin reads
`M5PHET_INTERPRETER_COMMAND`, `_MODEL` and `_TIMEOUT` exactly as before, which is
how `tools/start_chat.sh` and `~/.config/m5phet/chat.env` keep working.

`tools/interpreter_ollama.py` is now a thin shim over the `ollama` plugin, kept
so the environment-only route (`M5PHET_INTERPRETER_COMMAND=".../interpreter_ollama.py
llama3.2:3b"`) still works; the request body lives in one place, so the two
routes cannot drift apart. The `ollama` plugin defaults to the loopback ollama
server, so leave `base_url` out for a local one — a **host literal** in the
configuration file is refused (WP02), and `"$OLLAMA_HOST"` is how another one is
named.

**`openai_compatible` refuses to run without written consent.** It is the only
plugin that can send the person's sentence and the engine's declared vocabulary
off this machine, so the configuration must say

```json
{"interpreter": {"plugin": "openai_compatible", "consent": "cloud_ok",
                 "base_url": "$M5PHET_OPENAI_BASE_URL", "model": "...",
                 "api_key_env": "M5PHET_OPENAI_API_KEY"}}
```

Without `"consent": "cloud_ok"` the interpreter is unavailable, says why, and
asking it anyway raises — it never falls back to another plugin. The token is
never written in the file: `api_key_env` names the *environment variable* that
holds it, an unset variable is refused by name, and neither the key nor the
endpoint appears in the interpreter's identity, so neither reaches the browser or
a receipt. The owner's standing decision of 2026-09-24 is that no cloud model is
the default; this plugin exists so a paid one can be tried at the end of a
comparison and named in the report, never arrived at by accident.

What no plugin can do, for any of the three transports: show the model the
uploaded data, or introduce a value the provider did not declare. Those rules
live in `m5phet.interpret` and are checked for all three in
`tests/test_interpreters.py`, against a fake command script and a fake HTTP
server on the loopback interface — no network, no key, no real model.

## The JSON configuration (`m5phet.config.v1`)

The environment is a valid configuration and stays one: without any file,
`~/.config/m5phet/chat.env` and `tools/start_chat.sh` declare where every engine
and every fitted state lives, exactly as before. What the environment cannot show
in one reviewable place is **which plugin serves which area, with which settings**.
That is what `~/.config/m5phet/m5phet.json` says. Copy the template and edit:

```bash
cp tools/m5phet.json.example ~/.config/m5phet/m5phet.json
M5PHET_CONFIG=/path/to/another.json tools/start_chat.sh   # a verification instance's own file
```

Precedence: **the JSON wins where it binds, the environment keeps everything it
does not.** `/api/catalog` reports which of the two is in force:

```bash
curl -s 127.0.0.1:8766/api/catalog | jq '.config_source, .config.areas'
```

`config_source` is `m5phet.json` when a file was read and `env` when there was
none. The `config` block shows the provider and output plugin bound per area and
the surfaces; the `core` settings are deliberately not published, so no path and
no host reaches the browser.

Each area binds its provider, its core settings and its output procedure:

| Area | `core` key | The variable the provider reads |
|---|---|---|
| `classification` | `worker`, `command`, `backend`, `checkpoint`, `manifest`, `device`, `gpu_uuid` | `M5PHET_CHAT_LAYA_WORKER`, `M5PHET_CHAT_LAYA_COMMAND`, `NEWS_SIGNAL_*` |
| `forecasting` | `bundle_dir`, `python` | `M5PHET_FORECAST_BUNDLE`, `M5PHET_FORECAST_PYTHON` |
| `unsupervised` | `reference_dir`, `state_path` | `FEATURE_ENG_REGIMES_DEMO_DIR`, `FEATURE_ENG_REGIMES_STATE_PATH` |
| `rl` | `bundle`, `python`, `gym_fx`, `sample` | `M5PHET_POLICY_BUNDLE`, `M5PHET_POLICY_PYTHON`, `M5PHET_GYM_FX`, `M5PHET_POLICY_SAMPLE` |
| `causal` | `studies_dir`, `state_refs` | `CAUSAL_INFERENCE_STATE_DIR`, `CAUSAL_INFERENCE_STATE_REFS` |

The mapping is explicit on purpose: each engine owns its own variables and this
package imports none of them, so the table above is the contract between the key
an operator writes and the variable a provider reads. `interpreter` binds
`M5PHET_INTERPRETER_COMMAND`, `_MODEL` and `_TIMEOUT`; `surfaces` declares the web
port, the API token file, whether MCP is offered and how Telegram is reached.

Three things the loader **refuses** instead of repairing, each naming the exact
path in the file (`src/m5phet/config.schema.json` is the validated schema):

* an **unknown key** — a configuration that ignores a misspelling gives you a
  machine running settings you believe you changed;
* an **unset `$NAME`** — refused *by name*, so you learn which variable to set
  instead of watching a provider refuse later for an unrelated-looking reason;
* a **host literal** — an IPv4 address, `user@host`, or a `.local` / `.lan` name.
  A host is never written into a file that gets committed or shared: write
  `"$M5PHET_CHAT_LAYA_WORKER"` and keep the value in your own environment. The
  check is on what the **file** says; expansion may of course yield a host.

`~` and `$NAME` / `${NAME}` are expanded on load. The example file carries no host
and no secret, and the workbench answers the same 7 examples, 12 sentences, 2
refusals and 11 envelope questions whether it is configured by the JSON or by the
environment.
