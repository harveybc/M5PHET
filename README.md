# M5PHET

**M5PHET**: an open research program for five typed machine-learning
fronts, sharing reproducibility and distributed evaluation without requiring
one model to solve every problem. The name honours mentorship; the work and
its evidence take precedence over personal branding.

## Status: a foundation, not five completed engines

Version 0.1.0 implements a small, dependency-free **classification result
contract**. [news-signal](https://github.com/harveybc/news-signal) is its first
consumer: a local Laya adapter for shadow news analysis. The remaining four
fronts below are a research and implementation roadmap, not working backends.
There is no general inference service, broker, scheduler or universal learner
in this package. No financial performance has been demonstrated here.

## Five fronts

This is the program's operational grouping, not a universal taxonomy of ML.

| Front | Intended output and acceptance | Current implementation |
|---|---|---|
| Classification | Declared classes, probabilities, calibration scope and abstention; matched labelled evaluation | Uncalibrated result contract; Laya adapter in news-signal |
| Regression / forecasting | Variables and horizons with timestamps, units, point estimates or distributions; same-population baselines and interval coverage | Planned |
| Representation / unsupervised learning | Embeddings, clusters, novelty/OOD and missingness; downstream utility and stability tests | Planned |
| Reinforcement learning | Policy/actions/value estimates with environment and reward contracts; evaluated sequential behavior | Planned |
| Causal inference | Estimand, population, intervention, assumptions, identification, effect uncertainty and sensitivity | Planned |

Distributed optimization is **cross-cutting**, through the existing DOIN
ecosystem. It does not replace task-specific objectives or identification.
Classification confidence, forecast intervals and causal confidence intervals
are different mathematical objects, not interchangeable fields named confidence.

## Architecture and repository boundaries

```text
                 M5PHET task-specific contracts and evaluation protocols
                 / classification / forecasting / representation / RL / causal
data-gov -> task data -> specialized engine -> typed evidence -> application
                            ^
                  DOIN evaluates/searches eligible candidates

first application: news-signal -> future evaluated policy -> existing LTS risk
                                                     -> MT5 demo / Alpaca paper
```

- M5PHET owns the shared program and small task contracts, not existing engines.
- [news-signal](https://github.com/harveybc/news-signal) owns news interpretation
  and consumes the classification contract. Laya is one candidate engine.
- [predictor](https://github.com/harveybc/predictor) retains forecasting work;
  [agent-multi](https://github.com/harveybc/agent-multi) retains RL work.
- [doin-core](https://github.com/harveybc/doin-core) and related DOIN repositories
  retain distributed search; their generalized adapters remain to be evaluated.
- [data-gov](https://github.com/harveybc/data-gov) remains provenance/accounting
  authority; [trading-contracts](https://github.com/harveybc/trading-contracts)
  and [lts](https://github.com/harveybc/lts) retain trading intent/risk/execution.

No consumer should depend on a news-specific module to implement causal or RL
tasks. No classifier or optimization worker is authorized to place an order.

## Engine-selection policy

Use the strongest suitable open-source tools supported by a **matched task**,
not a permanent favourite or an unqualified SOTA label. Pin source/weights,
license and data provenance; reproduce the relevant reference; compare accuracy,
calibration, causal support, cost, resource use and deployment constraints.
Engine selection is per task/domain, with explicit evidence and limitations.
Do not claim current winners in the four unimplemented fronts.

Laya can be tested as a classifier, textual representation branch, or an
adapted head. A wrapper alone does not turn ordinal scores into regression,
create a calibrated joint forecast, or identify causal effects. Multi-horizon
direct heads and joint multi-output forecasting are separate proposed contrasts.

## Installation and working example

Python >=3.10; no runtime dependencies. Git source install:

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
    input_sha256="a" * 64,  # illustrative only; real callers compute hashes
    model_sha256="b" * 64,
    task_sha256="c" * 64,
)
assert result["execution_authorized"] is False
```

The builder checks exact task/label coverage, finite nonboolean probabilities,
declared rounding precision, argmax consistency and SHA256 shape. It snapshots
inputs and hashes the resulting envelope. It does not authenticate the producer,
recompute the model, certify calibration or establish data-gov acceptance.
Other result families are deliberately not accepted by this classification API.

## Testing and traceability

[Design and requirement matrix](docs/DESIGN.md), [method state](PROJECT_METHOD_STATE.json)
and `tests/test_contract.py` distinguish software validation from scientific
evidence. Tests include empty/missing/extra populations, duplicate labels,
NaN/inf/bool/oversized integers, wrong argmax, identity errors and rounding.
The downstream news-signal CLI verifies the package is actually consumed.
No model downloads, GPU jobs, broker calls or financial conclusions in this suite.

## Using with an agent

Read [AGENTS.md](AGENTS.md), the method state and design first. Choose a bounded
task family, define its output semantics and negative tests, then implement its
adapter. Reuse existing engines and validators. Keep unsupported tasks explicit;
do not return fabricated confidence, blanket VERIFIED states or successful stubs.
Parallel work must not interrupt experiments, bypass provenance or mutate brokers.

## Roadmap and collaboration

1. Exercise the real Laya checkpoint and prospective news-signal application.
2. Specify each next family's contract with an actual consumer and independent
   oracle, including temporal support and distinct uncertainty semantics.
3. Reproduce task-matched open-source references before engine comparisons.
4. Add bounded DOIN adapters with task-specific fitness and reproducible evidence.
5. Evaluate transfer/fusion and multi-task components as experiments, not defaults.

The [combined Laya submission](docs/SUBMISSION.md) links the general proposal
and its concrete application. The broader program is a collaboration proposal;
it is not claimed as a completed Laya feature set. No form has been submitted.

## License and attribution

MIT for this package, see [LICENSE](LICENSE). Engines, weights and datasets retain
their own licenses. Project-facing attribution uses M5PHET contributors. Hosting
under an existing GitHub account and preserving Git history are not anonymity.
Independent project, not affiliated with ConvAI or the Laya community directory.
