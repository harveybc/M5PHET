<p align="center"><img src="docs/logo/m5phet.svg" width="160" alt="M5PHET: a downward triangle with rounded corners enclosing a circle that holds a five-petal flower seen from above"></p>

# M5PHET

**Typed contracts for five machine-learning task families — classification, forecasting, representation,
reinforcement learning, causal inference — with a reference runtime in which a language model may only *choose
declared values* to write a request a person reviews, providers that wrap already-fitted engines answer it, and every
number keeps its unit, provenance and refusal reason.**

*Keywords: machine learning framework · typed ML contracts · natural-language ML orchestration · Laya decision
primitives · time-series forecasting · hierarchical regimes · causal inference · EconML · local projections · event
study · reinforcement learning · economic calendar · point-in-time data · DOIN · decentralized AI.*

---

## What this is, and what it is not

M5PHET is a framework, not a model and not a service. It says what a request to a machine-learning engine must
contain and what an answer must carry, so that a workflow can compose engines of different families without a number
silently changing its meaning: a tone score never becomes an expected return, a forecast interval never becomes a
causal confidence interval, a policy's action never becomes an order.

It is **not** five new models, **not** an automatic best-model router, **not** a trading system. Every fitted state it
serves today is `DEVELOPMENT`; no area has measured quality; nothing here authorizes anything.

## What runs today (2026-09-25)

| Piece | What it does | Where |
|---|---|---|
| **Envelope contract** `m5phet.task.questions.v1` | `{area, state, questions:{name:{type,…}}}` → named typed answers, each answered on its own or refused by name (`NOT_ESTIMABLE`, `STATE_REQUIRED`, `MALFORMED_QUESTION`, …). Answers claiming execution authority are refused. | `src/m5phet/questions.py` |
| **Five providers** | `laya_news` (classification: real Laya checkpoint on a worker GPU), `predictor_forecast` (TensorFlow bundles; raw rows or a standardized window), `feature-eng-hierarchical-regimes` (fitted sklearn reference), `trading_policy` (SB3 SAC through gym-fx's own observation builder), `causal_inference` (EconML studies fitted beforehand; ATE and CATE) | their own repositories, entry-point group `m5phet.providers` |
| **Interpreter** | Turns a sentence into an envelope by choosing among *declared* values only — the engines' vocabulary, never the data's; refuses what no engine has before any model is consulted | `src/m5phet/interpret.py`, `interpreters/` (`command`, `ollama`, `openai_compatible` — off unless consented) |
| **Orchestrator** | Routes, validates the proposal against the catalog (types, governed fields, fitted combinations, declared data requirement, attached columns), and guards every narration: no figure, ratio, percent or claim of profit or order that the answers do not carry | `src/m5phet/orchestrate.py`, `outputs/` (`default`, `telegram`) |
| **`m5phet.decide`** | Laya as the first layer of every area: a structured state and a `choice` among declared options → a recorded decision with uncalibrated probabilities. It chooses; explicit jobs fit; the closure table judges. | `src/m5phet/decide.py`, [docs/DECISIONS.md](docs/DECISIONS.md) |
| **Dataset resolver** | A sentence may name a dataset from the data foundation instead of attaching a file: explicit id → words → Laya among the candidates → refusal naming them. Descriptions in the catalog, never rows. | `src/m5phet/datasets.py` |
| **Web workbench** | Local single-owner chat (`m5phet-chat`): sentence or envelope, review before running, answers with units and refusals, provenance chips, token-guarded, reachable inside the owner's tailnet | `src/m5phet/web/`, [docs/CHAT_WORKBENCH.md](docs/CHAT_WORKBENCH.md) |
| **MCP server** | `python -m m5phet.mcp_server`: `m5phet_catalog`, `m5phet_propose_task`, `m5phet_execute_ml_task`, over the same engine and the same refusals; used from Hermes and Telegram | `src/m5phet/mcp_server.py`, [docs/TELEGRAM.md](docs/TELEGRAM.md) |
| **Evaluation instrument** | Protocol, freeze, scoring, report; the stage comparison table with the closure-table columns and `NOT_COMPARABLE`/`NO_NEW_MEASUREMENT` verdicts | `evaluation/`, [docs/EVALUATION_STAGES.md](docs/EVALUATION_STAGES.md) |
| **Configuration** | One JSON file binds, per area, the provider, its core settings, the interpreter and the output plugin; the env file still works | `src/m5phet/config.py`, `tools/m5phet.json.example` |

Acceptance, run on a verification instance after every change: 9/9 catalog examples answer, 14/14 sentences in
Spanish and English resolve and answer, 2/2 refusals are refused by name, 15/15 envelope questions answer or refuse as
expected, `execution_authorized` false everywhere — under two interpreters (DeepSeek through OpenCode Go, and a local
`llama3.2:3b` on CPU). Suite: 401 tests.

## Self-criticism: what is not here, said before anyone asks

- **No measured quality.** Not one area has a score on independent labels. The workbench shows what these engines
  *answer*; it establishes nothing about how *well*. The evaluation instrument exists; the labelled corpora do not.
- **Laya's choices are uncalibrated hypotheses.** In the first real decision (which transform suits a series) the
  probabilities were 0.35 / 0.32 / 0.32 — barely separated. That is a fact about a zero-shot checkpoint, and the only
  honest treatment is the one the plan prescribes: fit what it chose, fit the hand-made baseline, compare on the
  same held-out rows.
- **Forecast intervals are refused**, correctly, because no bundle has a quantile head. A person who asks for a
  range gets `NOT_ESTIMABLE`.
- **The causal question that matters — EUR/USD's transient response to calendar surprises — has no publication
  clock in any calendar on disk.** The event-study rows exist only under a *declared* assumption
  (`ASSUMED_SCHEDULED_PUBLICATION`), stamped on every row, and are not identified.
- **A narration guard is a filter, not a proof.** It refuses figures, ratios said in words, percents of
  non-probabilities and claims of profit or orders; it cannot see a false sentence that uses none of them.
- **Parameter search (DOIN/DEAP) is not in this framework.** The envelope reads fitted states; searching them is a
  separate product that has not started here.
- **The interpreter is nondeterministic and wording-sensitive.** The same news scored `euro_area 0.9666` in English
  and `0.9605` in Spanish; every classification answer therefore carries the exact instruction it was scored with.
- **Prospective data capture, MT5/Alpaca adapters and governed access through data-gov** are not built.

## The five families, as the code declares them

| Area | State | Question types | Refused by name when |
|---|---|---|---|
| classification | a news event or text | `choice{options, instructions?}` | fixture backend (`NON_MODEL_FIXTURE`), token budget exceeded |
| forecasting | `target_variable` | `point_forecast{horizon}`, `interval`, `anomaly_risk` | horizon or target not fitted; no distribution (`NOT_ESTIMABLE`); nothing attached |
| unsupervised | fitted reference (features absent = fitted) | `clustering{method}`, `cluster_description{target_metric}` | a metric the reference does not carry |
| rl | `policy_id` | `next_action`, `value_estimation` | too few rows, missing fitted columns, wrong vector length, contract mismatch |
| causal | `study` / `state_ref` | `ate`, `cate{subgroup}` | no effect modifier fitted; undeclared modifier; `NOT_IDENTIFIED` |

## Install and run

Python ≥ 3.10. The contract library has no runtime dependencies; the workbench needs `fastapi`, `httpx` and
`python-multipart`; each provider lives in its own environment (see the plan's map).

```bash
git clone https://github.com/harveybc/M5PHET.git && cd M5PHET
python3 -m venv .venv && .venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q                     # the framework's own tests
.venv/bin/m5phet-chat --port 8765                 # the workbench, providers from the entry points that are installed
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | .venv/bin/python -m m5phet.mcp_server
```

Operator configuration lives outside the repository (`~/.config/m5phet/chat.env` or `m5phet.json`); no hostname,
key or token is ever written here.

## Where to read next

- [docs/WORK_PLAN_2026_09_24.md](docs/WORK_PLAN_2026_09_24.md) — the executable plan: rules, the map of every
  repository and fitted state, the component model, required functionality per area and mode, and the ordered work
  packages with their proofs. Revision 2 states how Laya becomes the first layer of every area and how the doctoral
  comparison (hand baseline vs Laya-chosen vs searched representation) is measured.
- [docs/M5PHET_ANSWERS_TO_OPEN_QUESTIONS_2026_09_24.md](docs/M5PHET_ANSWERS_TO_OPEN_QUESTIONS_2026_09_24.md) — what
  the framework is and is not, answered against the code.
- [docs/INTERFACES.md](docs/INTERFACES.md), [docs/PROVIDERS.md](docs/PROVIDERS.md),
  [docs/REQUEST_EXAMPLES.md](docs/REQUEST_EXAMPLES.md) — the typed request/result contracts.
- [docs/USE_CASES.md](docs/USE_CASES.md), [docs/ECONOMIC_CALENDAR.md](docs/ECONOMIC_CALENDAR.md) — the first
  application domain and its point-in-time rules (publication ≠ receipt).
- [docs/INTEGRATION_AND_OPTIMIZATION.md](docs/INTEGRATION_AND_OPTIMIZATION.md) — local vs governed profiles, DOIN.
- Audits and their evidence live in the `predictor` repository under `docs/audits/`.

## License and attribution

MIT for this package; see [LICENSE](LICENSE). Provider code, weights and datasets retain their own licenses.
Independent project, not affiliated with TypeSafe, ConvAI or the Laya directory. Nothing here has been submitted
anywhere; [docs/SUBMISSION.md](docs/SUBMISSION.md) is a draft.
