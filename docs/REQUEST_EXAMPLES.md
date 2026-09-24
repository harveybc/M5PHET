# Proposed task requests

Design examples, NOT runnable SDK calls or implemented JSON Schema validators.
References below are symbolic; production requests must resolve them to accepted
immutable resources. No predictions or measured scores are fabricated here.
Common field semantics are in [INTERFACES.md](INTERFACES.md).

## Classification request

```json
{
  "schema_version": "m5phet.task.draft2",
  "request_id": "release-triage-example",
  "task_id": "asset-release-relevance-v1",
  "operation": "infer",
  "family": "classification",
  "output_kind": "typed_questions",
  "state": {"asset": "EURUSD", "text": "<received release statement>"},
  "questions": {
    "relevant": {"kind": "binary", "proposition": "The statement concerns the named asset's economic exposure"},
    "event": {"kind": "choice", "labels": ["inflation", "policy", "employment", "other", "unclear"]},
    "tone": {"kind": "ordinal", "levels": ["dovish", "neutral", "hawkish"]}
  },
  "provider_ref": "<pinned-laya-adapter>",
  "fitted_state_ref": "<checkpoint-identity>",
  "as_of": "<decision-time-UTC>",
  "temporal_contract": "<receipt-and-processing-contract>",
  "input_schema": "<asset-text-schema>",
  "output_schema": "<binary-choice-ordinal-schema>",
  "data_refs": ["<governed-statement-delivery>"],
  "execution_constraints": {"mode": "SHADOW", "partial_results": false}
}
```

Provider maps these to upstream primitives without losing their distinctions.
The shipped news-signal adapter supports choice only; the above full request is
pending. An unsupported primitive refuses rather than silently becoming choice.

## Forecast request

```json
{
  "schema_version": "m5phet.task.draft2",
  "request_id": "calendar-forecast-example",
  "task_id": "eurusd-multihorizon-v1",
  "operation": "infer",
  "family": "regression_forecasting",
  "output_kind": "marginal_quantiles",
  "state": {
    "price_window_ref": "<ordered-as-of-window>",
    "calendar_view_ref": "<vintage-correct-features>",
    "market_state_ref": "<optional-versioned-embedding>"
  },
  "input_schema": "<named-columns-axes-units-masks>",
  "as_of": "<decision-time-UTC>",
  "temporal_contract": "<elapsed-horizon-and-feature-availability-contract>",
  "output_schema": {
    "targets": ["log_return"],
    "horizons": ["PT6H", "PT72H"],
    "point_statistic": "median",
    "quantiles": [0.1, 0.5, 0.9],
    "joint_paths_required": false,
    "scale": "original_log_return",
    "transform_ref": "<train-only-target-transform>"
  },
  "provider_ref": "<pinned-predictor-adapter>",
  "fitted_state_ref": "<released-model>",
  "calibration_ref": "<disjoint-calibration-population-and-method>",
  "data_refs": ["<governed-price-delivery>", "<governed-calendar-delivery>"],
  "execution_constraints": {"mode": "SHADOW", "partial_results": false}
}
```

This illustrates one requested task, not a choice of target or quantile levels
for an already sealed experiment. The task contract fixes those values before
measurement. Point forecasts and quantiles are keyed by origin, target and
horizon; evaluation reports the declared normalized reference scale separately
from application units. Unsupported calibration cannot be replaced with model
confidence. The provider must refuse missing required references.

## Other typed payloads

- Representation request: input window/schema + encoder/hierarchy reference +
  requested embedding/path/OOD outputs. Result includes the model-version-scoped
  node path and assignment method, not an unqualified regime label.
- RL request: observation, cash/position/pending orders, action schema and policy
  release. Result is a time-bounded action proposal, not a broker receipt.
- Causal request: treatment/outcome/population, graph/assumptions, estimand and
  analysis-data reference. Result may be NOT_IDENTIFIED; schema validity is not
  evidence that an effect exists or is identifiable.
