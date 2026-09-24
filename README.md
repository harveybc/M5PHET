# M5PHET

**Typed machine-learning interfaces for applications that need decisions,
forecasts, market representations and policies, not generated prose.**

M5PHET is being built as a Python framework: supply text,
structured records or time series, specify a task and its output contract, and
use a suitable engine through a common interface. Its first application domain
is algorithmic trading, starting with news and **point-in-time economic calendar
data**. The interfaces are intended for other domains too.

**Release status:** v0.1.0 implements the classification result contract consumed
by [news-signal](https://github.com/harveybc/news-signal). The provider runtime,
calendar integration and other task contracts below are designed, not shipped.
There is no five-engine inference service or demonstrated trading advantage yet.

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
