# Product use cases

Design revision 2. These are application contracts, not completed experiments.
Actors: application developer, model developer, experiment evaluator and trading
operator. Their shared need is composable machine-readable results whose meaning
does not change between training, replay and service use.

## UC-01: typed decisions over text and records

**User:** a news or economic-event consumer. **Decision:** route an event, select
its category, assess relevance or score a defined ordinal attribute.

Input: text/structured state, asset context and named questions. Requested outputs
may be single-choice, several independently evaluated binary predicates, ordinal
rubrics or a declared hierarchical label tree. Independent predicates do not form
a joint probability distribution. A hierarchy requires explicit parent/child
semantics; several flat predictions are not automatically coherent.

Normal path: use the upstream Laya SDK, preserve its primitive and label meaning,
validate the complete population, then emit a task-specific envelope. Optional
Jev adapter is a hosted interoperability choice, not an open-source engine claim.
For a standalone application with no cross-task contract, recommend the upstream
SDK directly. Existing news-signal implements three choice questions only.

Alternate: unknown/abstain under a declared decision policy; measure coverage and
error among non-abstained cases. Refuse unsupported primitive, missing class,
truncation without policy, contradictory hierarchy or unavailable state.

Evaluation: independently annotated chronological examples; macro-F1, classwise
precision/recall, log loss/Brier where applicable, calibration and selective risk,
latency and cost. Ordinal scoring needs ordinal-specific error, not dollar MAE.
Reference: same questions and examples through the pinned upstream SDK. Never
promote vendor confidence to locally validated calibration.

## UC-02: economic calendar as point-in-time temporal input

**User:** feature builder and live observation service. **Decision:** what facts
could the model have used at this decision time?

Input: existing economic dataset, field definitions, release/consensus/schedule
vintages, acquisition timestamps and source rights; prices use their own
availability contract. Numeric published values come from structured records,
not classifier guesses. See [ECONOMIC_CALENDAR.md](ECONOMIC_CALENDAR.md).

Output: event identity, known schedule, country/currency/event family, time to
known release, time since observed release, available consensus/actual/previous,
revision flags, surprise and missingness, with field-level source lineage.
It is a data transform shared by the five task families, not a sixth ML model.

Alternate: schedule-only features where actual/consensus is unavailable. Refuse
an as-of reconstruction requiring unknown historical availability; archive-only
studies remain separate. No fake receipt timestamp from the present-day download.

Evaluation: hand-derived vintage fixtures, boundary/DST/revision tests and future
perturbations. Success means correct information sets, not improved trading.

## UC-03: hierarchical market-state representation

**User:** modular forecasting or RL model. **Decision:** which learned states and
features describe the history, and is this observation outside fitted support?

Input: causal price/volume/volatility/liquidity features where available, calendar
features and optional news outputs; masks, scaling and rolling-window identity.
Fit an encoder and hierarchy on training data only. Compare raw features and
learned representations using the same downstream protocol. Candidate hierarchy
may separate broad volatility/activity states and finer local patterns; these
are hypotheses, not prescribed clusters or profit labels.

Output: embedding, fitted hierarchy version, node path and declared assignment
method, distances or memberships with their true semantics, novelty/OOD and masks.
Inductive assignment for new observations must be specified and evaluated: a
batch clustering library does not automatically provide an online predictor.
Never refit on the test batch or use a future-smoothed state in live replay.

Alternate: unknown state instead of forced assignment. Refuse incompatible
encoder/hierarchy versions. Cluster IDs are model-version scoped, not eternal
economic regimes; align across refits using train/calibration data only.

Evaluation: perturbation stability, occupancy/collapse, drift, runtime and
downstream forecasting/policy utility on held-out time blocks. A silhouette score
alone does not justify routing a trader. Soft memberships are not necessarily
probabilities. Model switching requires a separate out-of-sample routing study.

## UC-04: uncertain multi-horizon forecasting

**User:** forecasting application or policy feature consumer. **Decision:** what
is forecast for each target/horizon, and what uncertainty is actually supported?

Input: named numeric windows, as-of calendar/representation branches, known-future
covariates only where genuinely known, target transform and desired horizons.
Keep the existing financial 6 h/72 h plan's elapsed-time and closure rules; these
are application requirements, not defaults for every dataset.

Output: target and horizon keys, origin/target times, point statistic (mean or
median), scale/units, quantiles, predictive samples/distribution when supported,
and method/calibration/population references. Direct per-horizon heads and joint
multi-output heads are separate provider capabilities; marginal quantiles do not
define a joint path distribution.

Candidates include the existing predictor Bayesian-head lineage, only after
checking its implementation, likelihood and inference sampling, plus matched
ensemble/quantile alternatives. Bayesian naming alone does not demonstrate
interval coverage or separate epistemic from aleatoric uncertainty.

Alternate: return point-only if the request explicitly permits it; otherwise
refuse missing calibration or unsupported uncertainty, never invent bounds.
Reject swapped targets, reordered horizons, stale transforms and unavailable
covariates. Preserve inverse-transform semantics, especially for nonlinear maps.

Evaluation: MAE/MSE in the exact reference space; MAE_z with train-only scale;
same-row persistence/seasonal references and skill; pinball/CRPS where defined,
coverage AND width by horizon/regime, latency and memory. Keep loss separate from
reporting metrics. Retain financial MAE/Huber x Adam/AdamW tests and investigate
small paired gains rather than rounding them away. No global model winner here.

## UC-05: calendar-aware trading policy

**User:** agent-multi training and the existing LTS paper/demo route. **Decision:**
which target position to propose given observable state and current execution?

Input: prices/representations, available event information, optional forecasts,
cash, positions, pending orders, action mask, costs and model release identity.
The policy may consume the representation directly; a forecast is not mandatory.
Use the actual gym-fx environment and existing policy implementations, not a
classifier renamed to buy/sell or a separate toy broker.

Output: action with quantity/units, policy ID, observation ID, decision and expiry
times, optional distribution/value estimate with method. Risk evaluation and
execution are separate. Invalid/stale proposals produce a configured safe
disposition through LTS, not a simulated successful order.

Evaluation: chronological weekly retraining/release, execution delay, spread,
fees, funding where applicable, partial fills, rejections, restart and position
reconciliation. Compare existing policy, price-only, calendar-augmented and
no-trade references on identical opportunities. Report net return, drawdown,
turnover, cost, exposure, Sharpe methodology and temporal support. Forecast MAE
does not prove profitability. Initial external use is MT5 demo/Alpaca paper only.

## UC-06: effects of economic releases

**User:** model researcher deciding whether an event feature or policy interaction
has defensible causal support. **Question:** effect of a defined release surprise
on a defined subsequent return/volatility outcome in a declared population.

Input: vintage-correct releases, units, pre-release consensus, outcomes with exact
windows, pre-treatment covariates, concurrent events, treatment definition, causal
graph and identification assumptions. An unexpected release is not automatically
randomized. A predictive event study is not automatically causal inference.

Output: estimand and identification status, estimator, effect units, uncertainty
method, overlap/support, sensitivity and falsification results; NOT_IDENTIFIED
when the design cannot identify the requested effect. Do not invent a scalar
effect in that branch. CATE needs its own support; individual counterfactuals
require stronger structural assumptions than average-effect estimation.

Evaluation: synthetic known-effect and confounded controls, pre-trend/placebo
checks where justified, dependence-aware uncertainty and sensitivity analyses.
Refutation tests cannot prove absence of hidden confounding. Causal findings enter
a later held-out feature/policy evaluation, never an automatic live strategy.

## First business experiment using the calendar

After source mapping and temporal tests, compare price-only against price plus
known schedule, then price plus schedule plus observed release/consensus surprise.
Keep output task, architecture, sample origins, training schedule and evaluation
budget paired; report any capacity change. Distinguish added information from
added parameters with a predeclared capacity-matched control.

Apply this to a matched forecasting reference and the actual trading policy as
separate outcomes. Report event vs non-event performance, horizon, missingness,
cost and week. Add hierarchical state only as a separate ablation; do not change
encoder, policy and input information at once. Calendar-aware does not mean
causally identified. Geopolitical inputs follow only after source/label/receipt
contracts are available. No new training was run to write this design.
