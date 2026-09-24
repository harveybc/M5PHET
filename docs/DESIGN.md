# Contract-first foundation

## Discovery and scope

The owner approved M5PHET as the general five-front research program, with
news-signal as its first consumer. Reuse DOIN for distributed optimization,
data-gov for authenticated provenance and existing task/execution repositories.
This repository must not duplicate these systems or make Laya the universal engine.

## Requirements and acceptance tests, declared before implementation

| ID | Requirement | Evidence |
|---|---|---|
| M1 | Typed classification outputs match the complete declared question/label population | missing, extra, duplicate, unknown and argmax tests |
| M2 | Probabilities finite, not boolean, bounded, sum under declared precision | negative numeric tests; rounded SDK distribution test |
| M3 | Input, task and model identity mandatory | malformed SHA256 tests |
| M4 | No authority/calibration inference; consumer cannot mutate result through original input | literal false authorization, uncalibrated status, copy/hash test |
| M5 | Real downstream dependency | news-signal installed with pinned M5PHET revision; CLI receipt test |

Architecture: pure standard-library contract builder; no brokers, networks,
model weights, training or inference in this package. Task engines live outside
this module. A content hash is not an authenticated governance certificate.
The first consumer passes explicit question labels and SDK probability precision.
The envelope records uncertainty semantics rather than inventing calibration.

Regression, representation, RL and causal adapters need their own requirement,
negative-test and reference-oracle designs before implementation. No stub may
claim a working engine. A classifier probability cannot become a causal interval
or a predictive distribution merely by changing an output type string.
