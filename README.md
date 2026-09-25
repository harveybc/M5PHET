<p align="center"><img src="docs/logo/m5phet.svg" width="120" alt="M5PHET"></p>

# M5PHET

System 1 Typed machine-learning framework. M5PHET gives five task families —
classification, forecasting, representation, reinforcement learning and causal
inference — one request/answer contract, so a workflow can ask a question in
ordinary words, see the typed request it resolved into, run it against an
already-fitted engine and get an answer that keeps its unit, its provenance and
its refusal reason. A language model may only *choose among values the engines
declare*; it never invents a target, a horizon, a study or a number. Providers
are thin adapters over engines that already exist
([news-signal](https://github.com/harveybc/news-signal),
[prediction_provider](https://github.com/harveybc/prediction_provider),
[feature-eng](https://github.com/harveybc/feature-eng),
[agent-multi](https://github.com/harveybc/agent-multi) +
[gym-fx](https://github.com/harveybc/gym-fx), EconML), reached through the
`m5phet.providers` entry-point group.

## Status

**Lifecycle: ACTIVE.** The runtime, the web workbench, the MCP server, the HTTP
API and all five providers run; 491 tests in this repository. Every fitted
state served today is `DEVELOPMENT`, no area carries a measured quality number
yet, and every answer is returned with `execution_authorized: false`.

> **Disclaimer:** inference only. This repository fits nothing, places no
> orders and authorizes no execution. A classification probability is not a
> calibrated belief, a forecast interval is not a causal interval, and a policy
> action is not an order.

## Run this with an AI agent

Paste this into Claude Code, Cursor, Codex, GitHub Copilot or any coding agent
with shell access:

> Read `AGENTS.md` in this repository and follow the **Agent quickstart**
> section end to end: create the environment, run the tests, start the
> workbench, ask one question of each family through the HTTP API, then tell me
> the exact URL where I can see the results and one question I should try first.

`AGENTS.md` is the [agents.md](https://agents.md) convention, read natively by
most coding agents. Its quickstart uses only CPU, inference-only commands.

## Architecture

```
  a sentence, a JSON envelope, an MCP tool call, a Telegram message
                              │
        ┌─────────────────────▼─────────────────────┐
        │ INPUT      readers (text/JSON/CSV) +      │   m5phet.interpreters
        │            interpreter: chooses only      │   (command | ollama |
        │            values the engines declare     │    openai_compatible)
        └─────────────────────┬─────────────────────┘
                              │  proposed envelope  ── the person reviews it
        ┌─────────────────────▼─────────────────────┐
        │ CONTRACT   m5phet.task.questions.v1       │   src/m5phet/questions.py
        │            {area, state, questions{}}     │   validated against the
        │            → {answers{}, answered,        │   catalog: types, fitted
        │               refused}                    │   values, combinations
        └─────────────────────┬─────────────────────┘
                              │
        ┌─────────────────────▼─────────────────────┐
        │ CORE       one bound provider per area    │   m5phet.providers
        │  classification  laya_news ───────────────┼──► Laya checkpoint (worker GPU)
        │  forecasting     predictor_forecast ──────┼──► TensorFlow bundles
        │  unsupervised    feature-eng-…-regimes ───┼──► fitted sklearn reference
        │  rl              trading_policy ──────────┼──► SB3 SAC + gym-fx observation
        │  causal          causal_inference ────────┼──► EconML studies, event studies
        └─────────────────────┬─────────────────────┘
                              │  typed answers, units, refusals by name
        ┌─────────────────────▼─────────────────────┐
        │ OUTPUT     header (kind, units, statuses) │   m5phet.outputs
        │            + procedure (default | telegram)│  every rendering and every
        │            checked: no figure, ratio,      │  narration is checked
        │            percent or claim the answers    │  against the answers
        │            do not carry                    │
        └───────────────────────────────────────────┘
```

Each component is a plugin chosen by JSON configuration, so an external
distribution can replace the interpreter, the output procedure or a whole area's
provider without touching this package.

## Prerequisites

- Python ≥ 3.10 (the contract library has no runtime dependencies).
- For the workbench and the API: `fastapi`, `httpx`, `python-multipart`.
- For answers from real engines: each provider installed in its own environment
  (TensorFlow for forecasting, Torch for classification, Stable-Baselines3 for
  the policy, EconML for causal). None is required to run the contract tests.

## Installation

```bash
git clone https://github.com/harveybc/M5PHET.git
cd M5PHET
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q                      # 491 passed, 1 skipped
```

Install whichever providers you want to answer with, into the same environment
(each pulls its own engine dependencies):

```bash
.venv/bin/python -m pip install /path/to/news-signal                       # classification
.venv/bin/python -m pip install /path/to/prediction_provider/forecast      # forecasting
.venv/bin/python -m pip install /path/to/feature-eng                       # representation
.venv/bin/python -m pip install /path/to/agent-multi/m5phet_policy         # rl
.venv/bin/python -m pip install /path/to/causal-inference-provider/provider # causal
```

## Smallest working local example

Ask one question without a workbench, without a network and without a model —
the fixture classification backend answers, and says it is a fixture:

```bash
cat > /tmp/task.json <<'JSON'
{"area": "classification",
 "state": {"asset": "EURUSD", "language": "en",
           "news": "The European Central Bank kept its deposit rate unchanged."},
 "questions": {"economy": {"type": "choice",
                           "instructions": "Which economy is named in this news?",
                           "options": [["euro_area", "Euro area"],
                                       ["united_states", "United States"],
                                       ["other", "Another economy"]]}}}
JSON

NEWS_SIGNAL_BACKEND=fixture .venv/bin/python - <<'PY'
import json
from m5phet.questions import run_task
from m5phet.runtime import Registry

registry = Registry()
registry.load_entry_points()
answer = run_task(json.load(open("/tmp/task.json")), registry)
print(json.dumps(answer["answers"]["economy"], indent=2)[:400])
print("answered", answer["answered"], "refused", answer["refused"],
      "execution_authorized", answer["execution_authorized"])
PY
```

The answer carries `label`, `uncalibrated_probabilities`, the exact
`instructions` it was scored with, `backend` and — because this run used the
fixture — `non_model_fixture: true` with a warning. Point
`NEWS_SIGNAL_CHECKPOINT` at a real checkpoint and the same call answers from
the model.

## The web workbench

```bash
.venv/bin/m5phet-chat --port 8765                  # then open http://127.0.0.1:8765
```

Write a sentence, attach a CSV (or name a dataset from your data lake), review
the typed request it resolved into — the chips say which parameter your own
words settled and which the interpreter chose — and run it. Configuration for
the engines lives outside the repository, in `~/.config/m5phet/chat.env` or
`~/.config/m5phet/m5phet.json` (`tools/m5phet.json.example`). Details and the
systemd unit: [docs/CHAT_WORKBENCH.md](docs/CHAT_WORKBENCH.md).

Questions that work today, one per family:

| Family | Ask |
|---|---|
| Classification | `¿De qué economía habla esta noticia?` (with a news text) |
| Forecasting | `predict household power one hour ahead` (with a window or a raw CSV) |
| Forecasting | `what is the direction_long probability at horizon 1?` |
| Representation | `describe el grupo de velas con cuerpo alto` |
| RL | `¿Qué acción propone la política para estas barras?` |
| Causal | `Report ATE of treatment on outcome, with its uncertainty.` |

## HTTP API

Every surface of the workbench is an API; programs authenticate with a bearer
token from a file. Full reference with one `curl` per endpoint:
[docs/API.md](docs/API.md); a stdlib client:
[`tools/api_client_example.py`](tools/api_client_example.py).

```bash
curl -s -H "Authorization: Bearer $(cat ~/.config/m5phet/api.token)" \
     http://127.0.0.1:8765/api/tasks/catalog | jq '.areas | keys'

curl -s -H "Authorization: Bearer $(cat ~/.config/m5phet/api.token)" \
     -H 'Content-Type: application/json' \
     -d '{"prompt":"pronostica la potencia","task":{...},"client_id":"run-1"}' \
     http://127.0.0.1:8765/api/chats/$CID/tasks/run
```

Re-sending the same `client_id` returns the same message; changing the envelope
under the same `client_id` is a `409`.

## MCP and chat clients

```bash
python -m m5phet.mcp_server        # stdio JSON-RPC 2.0, protocol 2025-06-18
```

Three tools — `m5phet_catalog`, `m5phet_propose_task`, `m5phet_execute_ml_task`
— over the same engine and the same refusals as the web. Registering it with an
agent (and reaching M5PHET from Telegram through it):
[docs/TELEGRAM.md](docs/TELEGRAM.md), skill in
[`tools/hermes_skill_m5phet/`](tools/hermes_skill_m5phet/).

## Configuration and plugins

```json
{"schema": "m5phet.config.v1",
 "interpreter": {"plugin": "command", "command": "hermes -m deepseek-v4-flash", "model": "deepseek-v4-flash"},
 "areas": {"forecasting": {"provider": "predictor_forecast",
                           "core": {"bundle_dir": "~/.local/state/m5phet/forecast-bundles"},
                           "output": {"plugin": "default", "language": "es"}}},
 "surfaces": {"api": {"token_file": "~/.config/m5phet/api.token"}}}
```

| Group | Ships | Replace it by |
|---|---|---|
| `m5phet.providers` | the five adapters, in their own repositories | registering your own provider for an area |
| `m5phet.interpreters` | `command`, `ollama`, `openai_compatible` | registering another; the cloud one refuses to run without `"consent": "cloud_ok"` |
| `m5phet.outputs` | `default`, `telegram` | registering another rendering; the figure guard applies to all of them |

Values may be `$VARIABLE` references; a literal hostname in the file is refused.

## Decisions: a model that chooses, never invents

`m5phet.decide` asks the classification engine to choose among *declared*
options — a preprocessor per feature, a clustering method, a causal estimator, a
dataset from the lake — and records the choice with its uncalibrated
probabilities, the digest of the state it saw and the checkpoint that answered.
A choice is a hypothesis: it becomes a label only when a closure-table row
measures the pipeline it led to
([docs/DECISIONS.md](docs/DECISIONS.md),
[docs/EVALUATION_STAGES.md](docs/EVALUATION_STAGES.md)).

## Tests and acceptance

```bash
.venv/bin/python -m pytest -q                                    # 491 passed, 1 skipped
python tools/verify_families.py  --base http://127.0.0.1:8766    # 9 examples, 14 sentences, 2 refusals
python tools/verify_envelopes.py --base http://127.0.0.1:8766    # 15 envelope questions
python tools/verify_outputs.py   --run /tmp/e.json               # every rendering checked
python tools/check_chat_browser.py --url http://127.0.0.1:8766 --infer   # Playwright
```

Run the harnesses against a verification instance with its own `--state-dir`,
never against the instance you use.

## Limitations

- No area has a measured quality number yet; the evaluation instrument and the
  stage-comparison table exist, the labelled corpora do not.
- Forecast intervals are refused (`NOT_ESTIMABLE`): no shipped bundle has a
  quantile head.
- Causal event studies over the economic calendar are `NOT_IDENTIFIED` while no
  calendar on disk carries a publication clock for the consensus.
- Parameter search (DOIN/DEAP) is not part of this framework: the envelope reads
  fitted states.
- The interpreter is nondeterministic and wording-sensitive, which is why every
  classification answer carries the exact instruction it was scored with.

## Related repositories

[news-signal](https://github.com/harveybc/news-signal) ·
[prediction_provider](https://github.com/harveybc/prediction_provider) ·
[feature-eng](https://github.com/harveybc/feature-eng) ·
[agent-multi](https://github.com/harveybc/agent-multi) ·
[gym-fx](https://github.com/harveybc/gym-fx) ·
[predictor](https://github.com/harveybc/predictor) ·
[doin-node](https://github.com/harveybc/doin-node)

## License

MIT; see [LICENSE](LICENSE). Provider code, weights and datasets keep their own
licenses. Independent project, not affiliated with TypeSafe, ConvAI or the Laya
directory.
