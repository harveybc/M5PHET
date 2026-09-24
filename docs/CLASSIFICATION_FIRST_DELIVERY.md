# Classification first, external domain engines

Owner-approved direction, 2026-09-24. Satoshi owns implementation; Musashi
reviews behavioral evidence. This is the executable delivery order, not a claim
that its providers already exist. It supplements P01-P09 and INT01-INT12.

## Present evidence and immediate priority

The reviewed runtime at `d9ffc4d` implements provider discovery and contracts,
but has no real Laya provider. `news-signal@d9b22e4` has a pinned SDK backend
and uses the classification result builder, not the Registry dispatch path.
The owner reports real weights materialized on one worker and SDK setup on
another. That is setup progress, not a successful inference or parity result.
Do not discard or duplicate that installation. Refresh its actual status first.

**First working business slice: English news relevance to EURUSD.** Use the
existing `relevance` choice: `related`, `unrelated`, `unclear`. Interpret relevance
as direct economic relevance to EUR or USD, not a prediction of price direction.
Keep the rubric explicit about central-bank, inflation/employment and currency
events. A headline that merely names an asset is not automatically relevant.
Retain event/tone as existing optional outputs, not prerequisites for this slice.
No new asset universe, return classifier, trading threshold or model training.

Input: a versioned news record, asset, source, event/revision identity, language,
publication time, receipt time, decision cutoff and declared text fields.
Output: one task-bound label, unchanged SDK probabilities with their precision,
uncalibrated status, optional abstention reason, input/model/task identities,
measured processing availability, latency and provenance. Never infer a trade.

## Implement and demonstrate, not just document

1. Register the real provider through the EXISTING `m5phet.providers` group in
   the `news-signal` distribution; reuse its LayaBackend. Refactor the backend's
   hard-coded question set only as needed to select the versioned task. Preserve
   current callers. No second encoder, tokenizer, registry or question compiler.
2. Expose the actual path: installed entry point -> Registry -> typed request ->
   provider -> pinned Laya Agent -> validated result -> durable news receipt.
   Show the runnable CLI/API invocation and a result from real weights.
3. Provide a direct-SDK comparison command which DOES NOT call the wrapper. Use
   identical text serialization, question/rubric/option order, tokenizer, weights,
   temperature settings, dtype, device and token budgets. Compare answers,
   probability maps and order by input identity. Ignore only explicitly named
   timing/provenance envelope fields. No wrapper-only normalization or rounding.
4. Under the same deterministic device/runtime, require exact equality of the
   SDK's exposed decision fields. Diagnose mismatches; do not change tolerance
   after seeing them. Separate process reload parity from wrapper parity and
   from any cross-device numerical experiment. The wrapper must not improve or
   degrade the classifier by changing inputs implicitly.
5. Report cold/warm latency, wrapper overhead, peak RAM/VRAM, actual device UUID,
   processed count and refusals. Use the existing 600 CPU s / 900 wall s pilot
   allocation after installation; external 5090 preferred, fresh admission, no
   preemption/duplicate work or silent fallback. Persistent service comes after
   the bounded pilot with existing admission/lease controls.
6. Deliver receipt -> classifier -> persisted shadow result -> replay after
   restart. Local recorded news is a valid integration input but is not a live
   feed. Attach an entitled feed when available; a missing feed must not stop
   the local end-to-end slice. No MT5/Alpaca order in this classifier milestone.

## First acceptance tests, before implementation

| ID | Test through installed public path | Acceptance |
|---|---|---|
| CL01 | Real SDK direct versus real provider, same inputs/settings | Decision fields equal; no fixtures presented as a model |
| CL02 | Relevant, unrelated, ambiguous, negated, duplicate/revised and instruction-like news | Complete expected population; text never executes instructions; failures retained |
| CL03 | Wrong language, overlong text including task/options, future receipt, absent weights, invalid output | Explicit refusal before inappropriate work; no silent truncation/provider/device fallback |
| CL04 | Restart, batch order/permutation, single versus batch | Identities preserved, paired comparisons reported, no result attached to another news item |
| CL05 | Installed provider absent/broken or unsupported task | Other providers remain usable; request names its failed provider |
| CL06 | Real persisted shadow result, process restart, replay | No duplicate actionable event, revisions linked, no broker call |
| CL07 | Labeled business corpus frozen before scoring | Confusion matrix, macro-F1, relevant-class precision/recall, coverage and class counts; direct SDK and wrapper on same rows |

CL01 establishes integration, NOT business accuracy. CL07 uses independently
reviewed labels and a sealed evaluation split, not Laya's own answers as truth.
If only author-written examples exist, label the result a smoke test. An absent
labeled corpus means business quality is unmeasured, not that the adapter must
wait to become runnable. Keep a small categorized regression bank; size and class
coverage must be explicit, not presented as a power calculation. Probabilities
are not calibrated merely because they agree with the SDK. No automatic
discarding of news or trading promotion until domain quality is reviewed.

## Where reuse ends

Inspected pinned SDK `1e28ac20c0896b1c37a744cd11f740eb98f8b178`:
`Agent.system_one` delegates to `predict_batch`; `build_sequence` combines the
question, options and state before encoding; `DecisionModel.forward` obtains
encoder states and scores candidate markers. Thus text features are conditioned
on the question. They are not one task-independent temporal representation.

Reuse the SDK unchanged for supported classification, including its tokenizer
and heads. Share M5PHET's validation, availability, provider lifecycle and
evidence contracts across families. Numerical windows stay numerical; causal
identification stays an explicit study. Replacing a neural head is a separate,
trained and evaluated model variant, not how a general engine plugin works.

Sources: [pinned Agent](https://github.com/NandhaKishorM/laya/blob/1e28ac20c0896b1c37a744cd11f740eb98f8b178/laya/agent.py),
[pinned sequence/model code](https://github.com/NandhaKishorM/laya/blob/1e28ac20c0896b1c37a744cd11f740eb98f8b178/laya/common.py).
Inspect installed resolved code before upgrading. Do not move a running pilot to
upstream main. Any tokenizer compatibility rewrite must be recorded as a derived
snapshot before its final seal; never silently invalidate the checkpoint manifest.

## Repository ownership, based on inspected code

Provider names below are proposed entry points, NOT shipped integrations.

| Family | Engine / external provider owner | First concrete adapter |
|---|---|---|
| Classification | `news-signal`, pinned Laya SDK | `laya_news`: relevance slice above |
| Forecasting | `predictor`; serving through `prediction_provider` | `predictor_forecast`: existing fitted model/config/transforms, exact target x horizon outputs |
| Unsupervised market hierarchy | `feature-eng`; optional encoders from `feature-extractor` | `market_hierarchy`: train-only fitted hierarchy and explicit new-point assignment |
| Causal inference | repair `causal-inference`, reuse EconML where assumptions apply | `calendar_effects`: event-study/identified effect curves with uncertainty and support |
| RL | `agent-multi` policies and `gym-fx` environment; LTS execution | `trading_policy`: existing fitted policy -> proposed action, not broker order |

Each owner registers an external provider, which delegates to its existing
native plugins. Preserve native entry points/configs. DOIN remains transversal:
candidate/evaluator round trips over these providers, never the RL engine.
Do not place TensorFlow, PyTorch, EconML and every sibling `app` package in one
mandatory environment. Prefer namespaced adapter modules; use existing isolated
process boundaries for incompatible top-level packages. Optional extras remain
optional; no import-time downloads/fits or mandatory governance stack.

### Forecasting

Bind the effective config, selected checkpoint, preprocessing/target inverse,
window/horizon units and feature order. First infer without retraining and prove
native predictor-versus-provider parity. Inventory branching, multi-output,
direct/recursive horizons and Bayesian options from real plugins; declare only
tested combinations. Point forecasts, marginal quantiles and joint samples need
different schemas. Bayesian uncertainty is not claimed calibrated without data.
No missing target/horizon cell may be silently omitted. Keep the earlier runtime
multi-horizon/population repair orders active.

### Hierarchical regimes

Found `feature-eng/regime_analysis.py`: standardization, PCA, Ward linkage,
cluster cuts and KNN assignment. `app/regime_detector.py` contains threshold and
GMM-derived labels; these are not the same fitted hierarchical model.
The analysis script fits scaler/PCA on the whole frame and reports forward
returns; it is NOT an accepted online pipeline. Extract reusable behavior without
importing/running its top-level analysis. Fit scaler/PCA/tree/assignment on train
only, save them, and infer without refit. Return versioned cluster paths and
distances/novelty with their actual meaning, not invented probabilities.
Test unseen points, label permutations, hierarchy consistency, restart and future
perturbations. Compare quality/stability and downstream utility separately.

### Economic-calendar effects and causal repair

The inspected `causal-inference` checkout has a README describing causal plugins
but `setup.py` and `app/plugin_loader.py` still register `rl_optimizer.*`; its
entry point is an RL template. There IS relevant research code:
`causal_regime_analysis.py` and local `causal_regime_analysis_v2.py` use EconML.
The latter plus `nfp_event_response_poc.py` and other files are UNTRACKED local
work at inspection. Preserve and inventory them with digests; no reset, blanket
commit or overwrite. Do not claim they belong to the HEAD revision.

Repair a narrow installable, namespaced causal package and its real entry point
in this repo, retaining the old exploratory history. Use clean isolated build,
import and known-effect tests before the M5PHET provider. Do not port a whole
broken RL template or create another repo with the same responsibility.

Initial task: response of EURUSD returns/volatility over declared elapsed-time
horizons to observed release surprises, conditional on pre-event market state,
and then interacting/overlapping releases. Acquire actual, consensus, previous,
schedule and revisions with field-level receipt/vintage semantics. Surprise is
actual minus the pre-release consensus, with train-fitted units/scale. Missing
consensus stays missing. Known schedule and post-release actual are separate
features; model serving uses only available fields.

Never reuse the old NFP script's price reaction as the surprise or choose the
timezone by maximum market response. It lacks actual/consensus data and cannot
identify this question. Inspect real release timestamps including DST rather
than a fixed first-Friday/UTC rule. Exclude outcome/post-treatment information
from controls. Separate descriptive response curves from identified effects.
Declare treatment, counterfactual, estimand, confounders, overlap/positivity and
identification assumptions; use temporal/event-group cross-fitting and purging.
Intervals for an average effect are not averages of individual CI endpoints.
Use EconML's existing estimation interfaces, not a home-made causal algorithm.
If identification fails, return NOT_IDENTIFIED while preserving descriptive work.
No assumed additive composition across event types: model/test interactions and
overlapping events explicitly. CAL01-CAL12 and synthetic known-effect/null/
confounded/missing-vintage tests precede a real held-out study.

Reference: [EconML CausalForestDML API](https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html).

### RL

`agent-multi/setup.py` registers PPO/DQN/SAC and `gym_fx_env`; use that engine
boundary, not DOIN or the old `rl-optimizer` template. Start with the existing
policy selected by its real effective config, not a new algorithm contest.
Expose observation/action schema, state/reset semantics, normalization, masks,
policy identity and recurrent state where applicable. Native/provider outputs
must agree under the same deterministic or recorded stochastic mode. Verify
pending/rejected/partial orders, stale observations and weekly releases in the
existing environment/risk tests. A policy output does not authorize execution.
Demo/paper deployment remains separate from model integration and real capital.

## Parallel implementation and definition of done

- Satoshi integrates, with available Hermes/subagents on disjoint worktrees:
  real classification; forecast/hierarchy; causal/calendar; RL/optional DOIN.
  Check ongoing jobs and leases before dispatch. Never duplicate a running pilot.
- The first delivery is the working classification slice, not another framework
  specification or five stubs. Finish other ready CPU paths while SDK/GPU jobs run.
- Each provider uses the existing capability tuples, rejects unsupported tasks
  before loading, declares bounded optimizer parameters and evaluator semantics,
  and supplies native parity plus domain-specific negative tests. DOIN can vary
  parameters, never dataset identity, reserve labels or evidence acceptance.
- Keep local durable evidence and optional DuckDB; governed and DOIN profiles
  are independently opt-in. No silent downgrade for governed requests.
- Do not wait for unrelated doctoral closure, causal data or broker credentials
  to deliver classification. Continue ready tasks without asking at every step.
  An external dependency blocks its operation only, not the whole framework.
- Return a working command, real input/output example, direct SDK parity counts,
  business-quality scope, measurements, commits and next runnable task for each
  lane. Installation, fixture tests, real inference, accuracy and trading utility
  are distinct statuses. Unsupported capabilities stay explicit.

This order authorizes the existing bounded integration work, not unbudgeted
training sweeps, external project submission or real-capital trading.
