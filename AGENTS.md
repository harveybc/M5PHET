# AGENTS.md — M5PHET

Guidance for AI coding agents working in this repository. See
[agents.md](https://agents.md).

## Project overview

M5PHET is a typed machine-learning framework: one request/answer contract for
five task families (classification, forecasting, representation, reinforcement
learning, causal inference), a runtime that validates what a language model
proposes against what the engines declare, and plugin seams for the input
(readers + interpreter), the core (providers) and the output (header +
rendering procedure). It **serves** already-fitted engines; it does not train,
does not search parameters and authorizes nothing.

It is not: five new models, a best-model router, a trading system, or a
replacement for the engines it adapts. Training belongs to `predictor` and
`agent-multi`, feature construction to `feature-eng`, parameter search to
`doin-node`, execution to systems outside this fleet.

## Agent quickstart (install → run → show the user results)

Verified on 2026-09-25 with Python 3.12.

### 1. Environment and tests

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q          # expected: 491 passed, 1 skipped
```

No GPU, no network and no model download is needed for this step; the classifier
falls back to a declared fixture that marks itself `non_model_fixture: true`.

### 2. Smoke test: one typed question, no server

```bash
NEWS_SIGNAL_BACKEND=fixture .venv/bin/python - <<'PY'
from m5phet.questions import run_task, catalog
from m5phet.runtime import Registry
registry = Registry(); registry.load_entry_points()
print(sorted(catalog(registry)))                     # the five areas and their providers
task = {"area": "classification",
        "state": {"asset": "EURUSD", "language": "en", "news": "The ECB held rates."},
        "questions": {"economy": {"type": "choice",
                                  "instructions": "Which economy is named in this news?",
                                  "options": [["euro_area", "Euro area"], ["other", "Other"]]}}}
out = run_task(task, registry)
print(out["answers"]["economy"]["label"], out["execution_authorized"])
PY
```

### 3. The workbench, and one question per family

```bash
.venv/bin/m5phet-chat --port 8766 --state-dir /tmp/m5phet-agent-state
```

Open `http://127.0.0.1:8766`, click an example, send its sentence, review the
typed request shown before it runs, then run it. Through the API instead:
[docs/API.md](docs/API.md) has one `curl` per endpoint;
[`tools/api_client_example.py`](tools/api_client_example.py) drives the whole
flow from the command line.

The acceptance harnesses are the product's own proof; run them against an
instance with its own state directory, never the one a person is using:

```bash
python tools/verify_families.py  --base http://127.0.0.1:8766 --out /tmp/f.json
python tools/verify_envelopes.py --base http://127.0.0.1:8766 --out /tmp/e.json
python tools/verify_outputs.py   --run /tmp/e.json
```

### 4. What to tell the user

Where the workbench is (`http://127.0.0.1:8766`), which providers answered
(`/api/catalog` lists them with their backends and fitted states), and one
question to try first — `¿De qué economía habla esta noticia?` with a news text
attached, or `Report ATE of treatment on outcome, with its uncertainty.` if the
causal provider and a fitted study are installed.

## Build, test and lint commands

```bash
.venv/bin/python -m pip install -e '.[test]'   # dependencies + this package
.venv/bin/python -m pytest -q                  # 491 passed, 1 skipped
.venv/bin/python -m pytest tests/test_questions.py -q   # the envelope contract alone
python -m m5phet.mcp_server                    # the MCP surface, stdio JSON-RPC
```

There is no configured linter or type checker; do not claim the code is
lint-clean.

## Layout

| Path | Purpose |
|---|---|
| `src/m5phet/questions.py` | the envelope contract: areas, types, refusal codes, `run_task`, `catalog` |
| `src/m5phet/runtime.py` | the provider registry and the typed `run()` path |
| `src/m5phet/interpret.py`, `interpreters/` | slot matching by words, then a model that may only choose declared values |
| `src/m5phet/orchestrate.py` | routing, proposal validation, the narration guard |
| `src/m5phet/outputs/` | per-area output headers and rendering procedures |
| `src/m5phet/decide.py` | the classification engine as a chooser among declared options, with decision records |
| `src/m5phet/datasets.py` | the dataset catalog and the resolver (explicit id → words → chooser) |
| `src/m5phet/pipeline.py` | the pipeline spec: registries read from sibling repositories, never imported |
| `src/m5phet/web/` | the workbench: FastAPI app, engine, store, static UI |
| `src/m5phet/mcp_server.py` | three MCP tools over the same engine |
| `evaluation/` | protocol, freeze, scoring, reports, the stage comparison table |
| `tools/` | the acceptance harnesses, the API client, the config example, the Hermes skill |
| `docs/` | contracts, use cases, the work plan, the workbench and API references |

## Conventions and constraints

- **Refuse by name, never invent.** A question an engine cannot answer returns a
  typed refusal (`NOT_ESTIMABLE`, `STATE_REQUIRED`, `UNSUPPORTED_QUESTION_TYPE`,
  …) with the reason. No fallback provider, no substituted state, no number that
  an engine did not produce.
- **The model chooses, it does not author.** Everything a language model may say
  is a value some provider declared: `chat_slots()` for parameters,
  `chat_combinations()` for pairs that were actually fitted, the catalog for
  types. A data column is not a fitted target.
- **Every rendering is checked.** A narration or an output plugin may omit
  figures; it may not add one, scale one, express one as a percent of something
  that is not a probability, or claim a profit or an order.
- **No execution authority.** Every answer carries
  `execution_authorized: false`; an answer that claims otherwise is refused.
- **Inference only.** Nothing here fits, calibrates or searches. Those are
  explicit jobs in the engines' own repositories.
- **Configuration lives outside the checkout.** `~/.config/m5phet/chat.env` or
  `m5phet.json`; never write a hostname, token, key or account identifier into
  this repository, and refuse a config file that hardcodes one.
- **Provenance travels.** Answers carry the fitted state reference, the backend
  that produced them, and — for a declared fixture — a marker saying so.

## Do not touch

- **A running instance or its state directory.** Tests and harnesses use their
  own `--state-dir` and their own port.
- **Fitted states** under `~/.local/state/m5phet/` and the providers' own
  directories: they are artifacts of explicit fitting jobs, never regenerated by
  this repository.
- **Sibling repositories.** Changes to a provider belong in its own repository;
  M5PHET reads their declarations, it does not edit them.
- **Decision and outcome records.** They are content-addressed evidence; write
  new ones, never rewrite one.
