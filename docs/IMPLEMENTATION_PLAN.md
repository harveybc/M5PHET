# Product implementation plan

This revision designs a framework, not five new model implementations. Existing
classification code remains unchanged. All tests below except M1-M5 are
**specified, not executed**. `test_product_spec.py` checks the documentation
inventory only and cannot change that status.

## Requirements and acceptance traceability

| Requirement | Use case | Required structural and behavioral tests | State |
|---|---|---|---|
| P01 Typed decisions | UC-01 | Exact labels/primitive/rubric; upstream response parity; unknown/partial/unsupported cases; ordinal mean independently recomputed | Existing choice-result contract only; primitive adapters pending |
| P02 Provider lifecycle | All | Reject unsupported output BEFORE model invocation; fit/calibrate explicit; duplicate entry point and wrong model/state refused | Specified |
| P03 Calendar availability | UC-02 | CAL01-CAL12 in calendar contract, against real as-of entrypoint plus governed sample | Specified |
| P04 Hierarchical representation | UC-03 | New-point assignment without refit; cluster-version mismatch; future perturbation; collapsed clustering; no hierarchy-probability invention | Specified |
| P05 Uncertain forecasts | UC-04 | Target/horizon/axis permutation, train-only scale, full vs prefix, checkpoint reload, independent MAE/quantile coverage; point-only provider refuses joint samples | Specified |
| P06 Trading policies | UC-05 | Pending/rejected/partial orders, staleness, action units and masks, weekly release; real LTS paper/demo integration | Specified |
| P07 Causal effects | UC-06 | Known-effect and confounded fixtures; missing graph/estimand, unsupported identification and poor support refuse; interval method declared | Specified |
| P08 Composition | All | Foreign model/calibration, stale clocks, lost masks and unit/scale mismatches; supported round trip across actual producers/consumers | Specified |
| P09 Evaluation | UC-03..06 | Paired populations, untouched holdout, matched reference, negative controls, measured costs and domain revalidation | Specified |

## Delivery increments and parallelism

**I1: decision interface and calendar foundation (independent lanes).**
M5PHET owns capability/output specs and tests, news-signal wraps only supported
Laya primitives. In parallel, data owner/feature-eng maps the economic dataset,
builds a vintage-aware adapter and exercises CAL tests. Fit/evaluate no new
financial models until the consumed data contract is established. Unknown source
vintages constrain only the affected calendar experiment, not ongoing SOTA work.

**I2: representation and forecast adapters.** Reuse actual feature-extractor and
predictor entrypoints. Implement hierarchy/new-point semantics and forecast
uncertainty separately. Begin with existing fitted checkpoints where compatible;
do not train replacement toy models to demonstrate JSON. Wire price-only and
calendar-augmented variants to the same consumer. Expose Bayesian-head method
and samples only after inspection and measured calibration.

**I3: policy observations and causal study.** Calendar-enabled observations may
enter RL directly after I1; they do not wait for a forecasting winner. Reuse the
actual policy and LTS risk route. The causal study proceeds independently once
its data and identification prerequisites exist. Its conclusions may motivate a
new held-out policy contrast, never retrospective selection on the live reserve.

**I4: deployment and optimization.** Choose providers using domain evidence,
latency and supported output types. DOIN searches declared bounded configurations;
it does not promote invalid tasks by obtaining a high score. Existing broker
promotion gates, retention limits, resource admission and experiment budgets stay.

## Concrete first handoff deliverables

1. Map the existing economic dataset to UC-02: governed resource identity,
   columns/units, timezones, schedule/consensus/actual/previous revisions, rights,
   availability evidence and missing fields. No credentials or private records
   in the public mapping. Name archive-only limitations precisely.
2. Write failing CAL and provider-capability tests before implementing those paths.
   Include fixture arithmetic for surprise and prefix-invariant as-of selection.
3. Deliver the smallest real adapters with independent numeric tests, then a
   governed bounded sample and forecast/RL observation wiring. Report unsupported
   paths rather than simulated successful results.
4. Seal the calendar ablation described in USE_CASES, using the existing financial
   reference/data prerequisites and compute budget. Report model/reference/naive,
   metric space, intervals with temporal support, cost and operational status.

No form submission, capital authorization, GPU job or broker mutation is implied
by publishing this design. Independent admitted experiments remain in flight.
