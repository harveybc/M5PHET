# Provider choices and primary references

Checked 2026-09-24 UTC. This is a capability-oriented shortlist, not a benchmark
ranking or a claim that these packages have been integrated. Pin a source revision,
weights, license and environment when implementing each adapter. Hosted Jev is
optional interoperability; open-source Laya is the initial local decision option.

| Task | Starting integration | Upstream/reference | Decision still requiring evidence |
|---|---|---|---|
| Typed classification | news-signal's pinned Laya adapter; add binary/ordinal outputs without coercion | [Jev primitives](https://docs.typesafe.ai/introduction), [ordinal Score semantics](https://docs.typesafe.ai/primitives/score), [Laya implementation](https://github.com/NandhaKishorM/laya) | Real checkpoint tests, task/language calibration, abstention, primitive and hierarchy coverage |
| Forecasting | Existing predictor + prediction_provider; reproduce task-matched reference | [NeuralForecast model interfaces](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html), [sktime forecasting interface](https://www.sktime.net/en/stable/examples/01_forecasting.html) | Exact target/split/scale/horizon parity and uncertainty method; no automatic replacement of current reference |
| Hierarchical state | feature-extractor/feature-eng encoders with an explicit hierarchy and new-point assignment | [scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) | Stable online mapping, train-only fit, predictive utility and versioned state semantics |
| RL | Existing agent-multi + gym-fx policy/environment contracts | [Stable-Baselines3 evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) | Observation/action/reward parity, weekly releases, actual costs and execution behavior |
| Causal effects | Candidate DoWhy identification/refutation with appropriate estimator, optionally EconML | [DoWhy estimation and EconML integration](https://www.pywhy.org/dowhy/v0.11/user_guide/causal_tasks/estimating_causal_effects/effect_estimation_with_estimators.html) | Economic identification, overlap, temporal dependence and sensitivity before causal claims |

Jev's useful design precedent is a state plus explicit questions returning typed
answers. Ordinal levels are rubric positions, not arbitrary continuous financial
targets. Laya documents similar primitives; its reported performance and
calibration are upstream claims, not local financial evidence. The pinned
news-signal adapter supports only its currently tested questions.

Forecasting, representations and RL use numerical arrays with explicit axes and
fitted state rather than serializing all values as a text prompt. Causal tools
help express and estimate a causal question; they cannot supply missing economic
assumptions. Choosing a provider and certifying an application are different jobs.
