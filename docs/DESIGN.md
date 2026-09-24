# Product design and implementation boundary

## Discovery and scope

The owner clarified the intended product: a typed machine-learning framework,
not a general research program. Applications submit text/structured/numerical
state and task/output schemas, and consume task-specific structured results.
Five task families share interfaces, not one universal model. Start from existing
Jev/Laya decision use cases and extend the interface to actual forecasting,
hierarchical representation, trading policy and causal-analysis consumers.

Product design: [USE_CASES.md](USE_CASES.md), [INTERFACES.md](INTERFACES.md),
[ECONOMIC_CALENDAR.md](ECONOMIC_CALENDAR.md), [PROVIDERS.md](PROVIDERS.md) and
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). The economic calendar is the
first structured multimodal input, not an optional future text-only feature.
`specs/use_cases.json` is the machine-readable design inventory, not an engine
registry. [REQUEST_EXAMPLES.md](REQUEST_EXAMPLES.md) contains design-only requests.

## Requirements and acceptance tests, declared before implementation

| ID | Requirement | Evidence |
|---|---|---|
| M1 | Typed classification outputs match the complete declared question/label population | missing, extra, duplicate, unknown and argmax tests |
| M2 | Probabilities finite, not boolean, bounded, sum under declared precision | negative numeric tests; rounded SDK distribution test |
| M3 | Input, task and model identity mandatory | malformed SHA256 tests |
| M4 | No authority/calibration inference; consumer cannot mutate result through original input | literal false authorization, uncalibrated status, copy/hash test |
| M5 | Real downstream dependency | news-signal installed with pinned M5PHET revision; CLI receipt test |

Current implementation: pure standard-library contract builder; no brokers, networks,
model weights, training or inference in this package. Task engines live outside
this module. A content hash is not an authenticated governance certificate.
The first consumer passes explicit question labels and SDK probability precision.
The envelope records uncertainty semantics rather than inventing calibration.

Regression, representation, RL and causal adapters need their own requirement,
negative-test and reference-oracle designs before implementation. No stub may
claim a working engine. A classifier probability cannot become a causal interval
or a predictive distribution merely by changing an output type string.

## Design revision 2 verification

The P01-P09 acceptance designs precede provider implementation. Specification
tests check task coverage, consumers, required outcomes, refusal paths, evaluation
plans, dependency closure and explicit implementation status. They do not count
as passing calendar, model, causal or broker acceptance tests. M1-M5 remain the
only implemented contract scope; this revision does not change runtime behavior.
