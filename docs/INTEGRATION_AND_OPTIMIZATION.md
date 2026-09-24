# Local use, governed campaigns and DOIN

Design, 2026-09-24. Adapters below are specified, NOT IMPLEMENTED by this release.
Reuse existing services and optimization interfaces; do not embed another server,
database, scheduler or distributed runtime in M5PHET.

## Explicit deployment profiles

**Local:** Python inputs/local files, provider and local result/artifact sink.
No governance service, network or DOIN node required for basic use. Retain task,
input, model and configuration identities. Label results LOCAL_UNGOVERNED, never
manufacture accepted delivery receipts. Engines remain usable without DOIN.

**Governed (opt-in, including example apps):** our governed deployments use data-gov, data-lake and
data-warehouse, optionally executed/optimized through DOIN. The task schemas and
metric meanings remain the same. A governed run cannot silently fall back to
local access when authorization or reporting fails.

Governance and DOIN are independent options: a local experiment may use a local
optimizer without either, a governed run may use no DOIN, and DOIN integration
does not implicitly enable our storage services. A published example has a local
quickstart first and an opt-in governed counterpart using the same task code.
The existing internal governed campaigns retain their own declared requirements.

## Organized local evidence and optional embedded OLAP

Proposed example-app output layout, not yet implemented by the contract library:

```text
runs/<run-id>/manifest.json       # effective task/config, code/model/data identities
runs/<run-id>/attempts.jsonl      # attempts, events, failures and measured costs
runs/<run-id>/metrics.jsonl       # typed metric records with units and population
runs/<run-id>/artifacts.json      # references, hashes, retention and availability
```

Capture input snapshot/receipt provenance and transformations without copying
licensed raw inputs into every run. Each metric identifies run/candidate/attempt,
task, split, horizon/target, metric definition/version, unit/scale, aggregation,
population size and status. Missing/undefined metrics are explicit, not zeros.
Maintain atomic manifests, durable records and unique event identities; recover
interrupted writes and distinguish retry from a new attempt. Never store secrets.

Offer an optional DuckDB embedded analytics reader over these versioned local
records: views by model, dataset, task, horizon, metric and cost. No HTTP service,
PostgreSQL, Metabase or distributed node is required. Core inference must not
import DuckDB unless analytics is selected. The analytical file/views are a
rebuildable projection, not a second authoritative accounting system. Reuse
applicable existing adapters; do not bring predictor's production cube into a
standalone package or create an alternative warehouse host.

The governed adapter maps the same result records to existing terminals and
warehouse reporting. Local traceability is valuable but is not authenticated
governance. Importing historical local results later must preserve that scope;
it cannot retroactively create campaign-before-work or delivery authorization.
Test profile parity using controlled identical inputs, not fabricated receipts.

## Ownership

| Component | Responsibility |
|---|---|
| M5PHET | Typed task/provider contracts, supported search parameters and integration adapters |
| data-gov | Authorization, campaign identity, deliveries, terminal accounting/reconciliation |
| data-lake/provider | Versioned inputs, calendar vintages and supported derived datasets/artifacts |
| data-warehouse/provider | Comparable experiment metrics, populations, parameters, cost, status and artifact references |
| Existing DOIN runtime | Candidate search and independent evaluation through domain plugins |

Use the installed providers and existing reporting APIs, not direct database
credentials. Check provider capabilities before promising artifact upload;
materialization/registration must use a supported path. Derived representations
retain encoder, training population, transform and availability lineage.
Proposed analytical fields require tested additive migrations, not a new cube.

Do not require a synchronous warehouse write per trading tick. Serving uses an
authorized versioned local bundle and bounded durable telemetry. Define queue,
outage, stale-input and bundle-expiry policies without weakening LTS risk checks.
Retention follows existing disk budgets and independent deletion gates, not an
unlimited tensor archive or a claim that metrics answer every future question.
No credentials, licensed raw data or full weights in public DOIN messages.

## Common optimization contract

Each provider declares a typed parameter space, including an empty space where
the engine exposes no supported tunables. Specify bounds/choices, conditional
dependencies, invalid combinations and the compatible provider revision.

Each task seals data/splits/clocks/targets/metric definitions, seeds, calibration
and fit protocol, resource limits, objective direction and scalar computation,
feasibility constraints, retry/checkpoint rules and measured cost. Candidate and
evaluation identities bind code, fitted state, numerical scope and input population.
Never let a candidate select its validation data, even if an upstream API accepts
`data=None`. Keep final evaluation outside search; use suitable nested validation.

| Family | Example tunables, where supported | Fixed protections |
|---|---|---|
| Classification | Adaptation settings, explicit rubric contrasts, calibration/abstention thresholds | Label meaning, held-out examples and reporting metric |
| Forecasting | Context, architecture, heads, LR, MAE/Huber and delta, Adam/AdamW and decay, uncertainty method | Task/horizon population, target scale, availability and final holdout |
| Representation | Encoder size, embedding dimension, hierarchy/clustering parameters | Train-only fit and downstream evaluation population |
| RL | Policy, optimizer, discount/exploration, approved reward shaping | Execution costs, solvency/risk rules and final trading objective |
| Causal inference | Estimator/nuisance-model settings and justified cross-fitting | Estimand, treatment/outcome and identification assumptions |

Never optimize causal effect magnitude or significance. Use a justified estimator
criterion or nuisance validation; known-effect synthetic tests are not knowledge
of true effects in market data. Refuse nonidentified tasks regardless of score.
Joint provider/branch searches require explicit experimental authorization, not
silently expanding a controlled comparison. Small forecasting gains remain
measured at full precision and require downstream business validation.

## Actual DOIN compatibility

Inspected `doin-core/src/doin_core/plugins/base.py` and its plugin groups:

- `doin.optimization`: `OptimizationPlugin.configure`, `optimize` and
  `get_domain_metadata` receive the domain objective/search specification.
- `doin.inference`: `InferencePlugin.evaluate(parameters, data) -> float` verifies
  candidate performance. It is NOT the structured prediction API. Bind the scalar
  to the complete evaluation evidence; keep full metrics in the result store.
- `doin.synthetic_data`: deterministic `SyntheticDataPlugin.generate(seed)` and
  `generate_with_hash`. The interface documents zero consensus weight without
  synthetic verification. Local optimization success is not consensus acceptance.

The existing scalar interface is not a vector-valued Pareto API. Start with an
explicit scalar plus constraints, or separate objective-specific domains. Any
multiobjective protocol extension needs its own design. Never compare raw MAE,
Sharpe and classification scores as if they shared a unit or objective.
Use the actual plugin loader and runtime in tests; inheritance alone is not
compatibility. Synthetic verification does not replace real-domain evaluation.

## Acceptance tests to implement

| ID | Required behavior |
|---|---|
| INT01 | Local install/run with no services; result remains explicitly ungoverned |
| INT02 | Fixed input/model gives equivalent local/governed outputs within declared numerical scope; authority labels remain distinct |
| INT03 | Denied/missing delivery refuses before work, without local fallback |
| INT04 | Reporting outage persists outbox; restart/retry produces one accepted effect and complete reconciliation |
| INT05 | Parameter/model/data/calibration change changes identity and cannot reuse incompatible cache |
| INT06 | Invalid conditional parameter or resource overrun cannot become a best candidate |
| INT07 | Installed DOIN plugin proposes and independently evaluates a real bounded candidate; scalar direction/value matches stored metrics |
| INT08 | Seeded synthetic generation repeats its hash; missing generator gives no claimed consensus acceptance |
| INT09 | Every claimed family exercises a real provider and refusal path, not a classifier standing in for all five |
| INT10 | Search cannot change final holdout, causal estimand or execution constraints |
| INT11 | Interrupted/retried local writes preserve unique attempts; embedded OLAP rebuild has identical metric values, units and populations |
| INT12 | The same example runs offline without optional packages, and in opted-in governance mode; historical local import never gains retrospective authority |

First delivery: local classification with durable records and optional OLAP,
then its governed counterpart and one bounded
DOIN candidate with an actually supported tunable. Extend the common adapter to
the remaining families with their own objectives and consumer tests. Satoshi
implements; ongoing experiments stay independent. Tests above are pending, not
evidence of completed integration or permission for production migrations.
