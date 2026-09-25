<p align="center"><img src="docs/logo/m5phet.svg" width="160" alt="M5PHET: a downward triangle enclosing a circle that holds a five-petal flower"></p>

# M5PHET

**Typed contracts for five machine-learning task families — classification, forecasting, representation,
reinforcement learning, causal inference — with a reference runtime in which a language model may only *choose
declared values* to write a request a person reviews, providers that wrap already-fitted engines answer it, and every
number keeps its unit, provenance and refusal reason.**

*Keywords: machine learning framework · typed ML contracts · natural-language ML orchestration · Laya · decision
primitives · time-series forecasting · hierarchical regimes · causal inference · EconML · reinforcement learning ·
economic calendar · point-in-time data · DOIN · decentralized AI.*

The logo is the design: the **triangle** is the triad of life — decentralization, self-awareness, evolution; the
**circle** it encloses is the link that runs through everything and can be used in every stance; the **five-petal
flower** inside, seen from above, is the five disciplines.

## What runs today (2026-09-25)

- A local **web workbench** (`m5phet-chat`, port 8765): write a sentence, attach a dataset or name one from the data
  lake, review the typed envelope the sentence resolved into, run it, read the answers with their units and refusals.
  Five providers answer: `laya_news` (classification, real Laya checkpoint on a worker GPU), `predictor_forecast`
  (TensorFlow bundles), `feature-eng-hierarchical-regimes`, `trading_policy` (SB3 + gym-fx observation) and
  `causal_inference` (EconML studies fitted beforehand). See [docs/CHAT_WORKBENCH.md](docs/CHAT_WORKBENCH.md).
- The **envelope contract** `m5phet.task.questions.v1`: `{area, state, questions:{name:{type,…}}}` → named typed
  answers or refusals by name (`NOT_ESTIMABLE`, `STATE_REQUIRED`, …). No invented numbers; every answer carries
  `execution_authorized: false`.
- **Plugins by JSON configuration**: interpreters (`command`, `ollama`, `openai_compatible` — off unless consented),
  outputs (`default`, `telegram`), providers per area. See `tools/m5phet.json.example`.
- **`m5phet.decide`**: Laya as the first layer of every area — it chooses among *declared* options (a preprocessing,
  a clustering method, an estimator, a dataset) and the choice is recorded, then fitted and measured by explicit jobs.
  See [docs/DECISIONS.md](docs/DECISIONS.md).
- An **MCP server** (`python -m m5phet.mcp_server`) with three tools, reachable from Hermes and from Telegram
  ([docs/TELEGRAM.md](docs/TELEGRAM.md)); the evaluation instrument and the stage comparison table
  ([docs/EVALUATION_STAGES.md](docs/EVALUATION_STAGES.md)).
- What is **not** here yet, said plainly: measured quality for any area, forecast intervals (needs a quantile
  bundle), a calendar with a publication clock (the event study waits for it), parameter search (DOIN) inside the
  framework. The executable plan with locations, proofs and order is
  [docs/WORK_PLAN_2026_09_24.md](docs/WORK_PLAN_2026_09_24.md).

**Release status:** v0.1.0 shipped the classification contract; the runtime, workbench, envelope, plugins and decision
primitive above are on `master` since 2026-09-25, all states `DEVELOPMENT`. There is no demonstrated trading advantage.

## What you will be able to ask

These are five supported-by-design **task families**, not a claim that machine
learning has exactly five branches.

| Task family | Application question | Structured result | Engine boundary |
|---|---|---|---|
| Classification | Is this release relevant to EURUSD? What event type is it? How hawkish is this statement on a defined rubric? | Categories, binary probabilities or ordinal distributions, with abstention and calibration scope | Reuse Laya; optional Jev interoperability, not a new text classifier |
| Regression / forecasting | What are the price or return forecasts at 6 h and 72 h, and their uncertainty? | Target/horizon-indexed points, quantiles or predictive distributions with units and calibration evidence | Existing predictor models and matched forecasting engines |
| Representation / unsupervised learning | Which hierarchical market state describes the observable history? Is the current state unfamiliar? | Versioned embedding, cluster path, novelty score and missingness | Fitted temporal encoders and clustering engines, not text labels invented after a trade |
| Reinforcement learning | Given the market state, calendar and portfolio, what position should the policy target? | Proposed action, policy/value outputs when supported, validity and constraints | Existing agent-multi/gym-fx; execution remains in LTS |
| Causal inference | Under explicit assumptions, what is the effect of a release surprise on subsequent returns or volatility? | Identified estimand, effect and uncertainty, diagnostics; or NOT_IDENTIFIED | Specialized causal tools and a declared study design |

See [real-world use cases](docs/USE_CASES.md), [interface design](docs/INTERFACES.md)
and the [economic calendar contract](docs/ECONOMIC_CALENDAR.md).

## Why M5PHET instead of calling Laya directly?

**For standalone text decisions, call Laya or Jev directly.** Jev's documented
pattern is state plus typed questions; Laya offers a local implementation of
similar decision primitives. Those existing APIs are the starting point, not
something M5PHET needs to reinvent. See the [official Jev introduction](https://docs.typesafe.ai/introduction)
and [Laya source](https://github.com/NandhaKishorM/laya).

M5PHET adds value when a workflow combines heterogeneous models: a release
classifier, a market-state encoder, an uncertain forecast and a trading policy.
The framework's responsibility is to preserve their **input availability,
output meaning, fitted state, uncertainty and compatibility** across composition.
An ordinal tone score cannot silently become an expected return; a forecast
interval cannot silently become a causal confidence interval.

## Calendar-first application architecture

```text
economic releases + consensus vintages + prices + optional news
                    |
             point-in-time assembly
                    |
       +------------+------------------+
       |                               |
 typed event decisions        temporal market representation
 (Laya/news-signal)            (prices + calendar + observed releases)
       |                               |
       +---------------+---------------+
                       |
          multi-horizon forecast / RL policy
                       |
          existing risk and execution boundary
                 MT5 demo / Alpaca paper

causal studies: offline, using explicitly identified data and assumptions
DOIN: fit/search/evaluation across eligible providers, not an inference engine
```

The scheduled time of a release may be known in advance; its actual value is
not. Consensus, release values and revisions keep separate observation times.
Before release, use known schedule features. After receipt, use the observed
surprise. Geopolitical text is a later extension, not a reason to postpone the
structured calendar pipeline or to infer unpublished economic numbers from text.

## Interface and repository boundaries

Planned interface: `TaskRequest -> capability check -> bound provider -> TaskResult`.
Requests declare task, typed input, clock, output schema and model/calibration
references. Providers declare which operations and output types they support.
Inference never fits a model implicitly. Unsupported tasks return a typed refusal.

| Repository | Ownership |
|---|---|
| M5PHET | Task contracts, capability negotiation, provider interfaces and composition validation |
| [news-signal](https://github.com/harveybc/news-signal) | Laya adapter and news/event interpretation |
| [predictor](https://github.com/harveybc/predictor), [prediction_provider](https://github.com/harveybc/prediction_provider) | Forecast training and serving |
| [feature-extractor](https://github.com/harveybc/feature-extractor), [feature-eng](https://github.com/harveybc/feature-eng) | Learned representations and feature transformations |
| [agent-multi](https://github.com/harveybc/agent-multi), [gym-fx](https://github.com/harveybc/gym-fx) | Policies, training and trading environments |
| [data-gov](https://github.com/harveybc/data-gov), [data-lake](https://github.com/harveybc/data-lake), [data-warehouse](https://github.com/harveybc/data-warehouse) | Data delivery, storage and experiment accounting |
| [doin-core](https://github.com/harveybc/doin-core) | Distributed optimization |
| [lts](https://github.com/harveybc/lts), [trading-contracts](https://github.com/harveybc/trading-contracts) | Risk, intent validation and broker execution |

These are integration boundaries, not claims that all adapters already exist.
No classifier, forecast or causal estimate authorizes an order.

## Local use, governed campaigns and DOIN

The target architecture keeps local inference lightweight: no mandatory services
or distributed node. Example apps start locally with organized provenance and
metric records; optional embedded DuckDB provides analytical views without a
server. **data-gov + data-lake + data-warehouse** and **DOIN** are independent
opt-in integrations. Our already-governed campaigns keep their declared profile;
The five adapters and the workbench are implemented on this branch; see `docs/WORK_PLAN_2026_09_24.md` §1 and §3 for what runs and what is still missing.
Each provider declares supported parameters and task-specific evaluation; full
metrics remain traceable alongside DOIN's scalar objective. See the
[integration and optimization design](docs/INTEGRATION_AND_OPTIMIZATION.md).
These adapters are specified, not implemented in this release.

## Choosing engines

Use task-matched open-source implementations and reproduce their reference
protocols before making comparative claims. Compare predictive performance,
uncertainty, cost and deployment suitability under the same data contract.
There is no permanent best model for every task. Existing successful models must
not be replaced by toy implementations just to fit a common API.

[Provider decisions](docs/PROVIDERS.md) distinguish upstream capabilities,
integration candidates and evidence still required. Bayesian output heads,
ensembles and calibrated quantiles are forecasting options, not interchangeable
labels for guaranteed uncertainty. A provider must declare its actual method.

## Install and use what exists today

Python >=3.10; the current contract library has no runtime dependencies.

```bash
git clone https://github.com/harveybc/M5PHET.git
cd M5PHET
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

```python
from m5phet import make_classification_result

result = make_classification_result(
    {"tone": {"label": "positive",
              "uncalibrated_probabilities": {"positive": 0.75, "negative": 0.25}}},
    expected_labels={"tone": ["positive", "negative"]},
    input_sha256="a" * 64,  # illustrative; real callers compute these hashes
    model_sha256="b" * 64,
    task_sha256="c" * 64,
)
assert result["execution_authorized"] is False
```

This validates and snapshots already computed class results. It does not run a
model. For the implemented classifier application, use the
[news-signal fixture CLI](https://github.com/harveybc/news-signal#reproducible-fixture-demo).
The builder validates exact populations, finite probabilities, declared rounding,
argmax and identity shape; hashes do not authenticate a producer or calibrate it.
Binary, ordinal and hierarchical request adapters are not yet part of this API.

## Delivery plan and acceptance

Current implementation order: [one working Laya classification slice first,
with domain plugins in parallel](docs/CLASSIFICATION_FIRST_DELIVERY.md).
The first slice is EURUSD news relevance, with direct-SDK parity measured
separately from business accuracy. It is not yet a verified real-model release.

1. **Typed decisions:** preserve the working news-signal integration; add explicit
   binary/ordinal contracts and provider capability tests, reusing upstream SDKs.
2. **Economic calendar:** map the actual governed dataset, build vintage-aware
   as-of views and prospective receipt collection; test boundary and revision cases.
3. **Market representation and forecasting:** connect existing engines, expose
   hierarchical state and multi-horizon uncertainty; evaluate calendar ablations.
4. **RL and causal analysis:** integrate the same as-of features into trading
   environments and run separately identified causal studies. Neither is a shortcut
   around forecast evaluation or execution risk checks.

These are dependency-aware increments, not a global serial queue. Existing
experiments and independent tasks continue in parallel. Exact acceptance tests,
ownership and entry conditions are in [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## Testing and agent usage

Read [AGENTS.md](AGENTS.md), [DESIGN.md](docs/DESIGN.md) and
[PROJECT_METHOD_STATE.json](PROJECT_METHOD_STATE.json) before changes.
`tests/test_contract.py` validates the shipped classification contract.
`tests/test_product_spec.py` checks the design inventory's internal consistency,
not unimplemented model behavior. Future behavior tests are explicitly pending.
No weights, GPUs, broker requests or scientific outcomes are produced by these tests.

Agents must implement bounded increments against real consumers, preserve
availability and schema semantics, and run negative tests through actual adapters.
Never advertise a schema example as a running provider or a probability as profit.

## Submission, license and attribution

[Submission draft](docs/SUBMISSION.md) links M5PHET and its first application;
it distinguishes this framework design from released functionality. Not submitted.
MIT for this package; see [LICENSE](LICENSE). Provider code, weights and datasets
retain their own licenses. Independent project, not affiliated with TypeSafe,
ConvAI or the Laya directory. Existing GitHub attribution is not anonymity.
