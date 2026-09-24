# Framework interface design

Status: SPECIFIED, NOT_IMPLEMENTED except the existing classification result
builder. Names below are design contracts, not importable APIs or compatible
versions of the shipped `m5phet.classification.v1` envelope.

## Request and result

| Request field | Required meaning |
|---|---|
| schema_version, request_id, task_id | Versioned interface and immutable task specification |
| operation | Explicit fit, calibrate, infer, evaluate or identify; no implicit training |
| family, output_kind | Specific supported task and output semantics |
| state | Typed text, records or ordered tensor references, never implicit stringification of a time series |
| input_schema | Names, types, shape/axes, units, masks, label order and roles |
| as_of, temporal_contract | Decision clock and field availability; no generic timestamp-as-proof |
| output_schema | Target/horizon/label population, uncertainty method and requested scale |
| provider_ref, fitted_state_ref | Pinned adapter, model, transforms and dependency environment |
| calibration_ref | Bound task/model/population/period when calibration is required |
| execution_constraints | Bounded resources, latency and allowed output modes, not broker permission |
| data_refs | Governed resource and delivery identities; schemas do not grant data access |

Results bind the complete request, input population, provider and fitted state.
They include a per-output status, typed payload, declared uncertainty,
availability/support, transforms, latency/cost and evidence references.
Do not overload one VERIFIED boolean: schema validity, replay, calibration,
governance acceptance and application eligibility are separate facts.

Proposed statuses: OK, ABSTAINED, UNSUPPORTED_TASK, INPUT_UNAVAILABLE,
INVALID_INPUT, MODEL_NOT_FITTED, CALIBRATION_UNAVAILABLE, NOT_IDENTIFIED and
RESOURCE_EXCEEDED. A refused numerical result carries no invented score.
Multi-question batching declares whether partial responses are allowed; omitted
questions never count as success. Execution authorization remains outside ML.

## Provider capabilities and lifecycle

Proposed Python entry-point group: `m5phet.providers`. This group is NOT registered
in the current release. External distributions own their backend dependencies;
the contract package must not require every ML framework in one environment.
Use process/service boundaries when sibling packages have conflicting namespaces.

1. `capabilities()` declares operations, input schemas, output kinds, uncertainty
   methods, fit requirements and resource limits. Reject duplicate registrations.
2. `validate(request)` rejects unsupported semantics BEFORE model loading or work.
3. `fit(...)` and `calibrate(...)` are explicit governed jobs, returning immutable
   state references with train/calibration population and clocks.
4. `infer(request, state_ref)` cannot modify fitted state. It checks capability
   and identity, invokes the real engine and validates the complete response.
5. `evaluate(...)` uses an independent scorer and matched population. Causal
   `identify(...)` precedes effect estimation, not a generic prediction shortcut.

Cache identity covers task/input/clock, provider code and model, transformation,
calibration and numerical scope. A schema hash alone is insufficient. No silent
fallback between providers or CPU/GPU paths in a sealed experiment. Resource
admission uses existing orchestration; this project does not build a scheduler.

## Composition rules

- A consumer names the producer output and expected schema version, columns,
  axes, units, target transformation and missingness behavior.
- A shared as-of view precedes all branches; every contributing field is known
  by the decision cutoff. Derived output availability includes processing delay.
- Scaling/encoders/hierarchies/calibrators fit only on their declared historical
  populations. Stacking on model-generated features requires out-of-fold training
  outputs unless the task explicitly studies in-sample representations.
- No inference-only graph edge can fit, refit or optimize an upstream model.
- Bayesian samples, cluster distances, ordinal levels and policy logits keep
  their own meanings. Conversion needs an explicit evaluated adapter.
- Forecast quantiles must preserve target/horizon keys and monotonicity; do not
  silently sort away a quantile-crossing model error.
- RL action shape alone is insufficient: quantity units, observation age,
  portfolio and pending orders must match the execution contract.

## Decisions and rejected alternatives

**Thin adapters over existing engines**, not a universal model: numerical time
series and causal identification are not text-classification tasks.

**Typed task payloads**, not arbitrary requested JSON: output shape cannot confer
mathematical meaning or make an unsupported task executable.

**Explicit provider selection first**, not an unmeasured automatic best-model
router: task suitability and comparative evidence must exist before routing.

**Calendar data transform before another neural layer**: numeric source values
and temporal eligibility need no neural guess. A learnable event encoder is a
later measured representation option, not required for ingestion correctness.

**Offline causal analysis separate from per-tick inference**: a fitted,
identified heterogeneous-effect model may eventually serve estimates, but the
identification study cannot be replaced by one per-event classifier call.
