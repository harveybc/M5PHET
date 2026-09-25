# WP11 evidence — Hermes reaching M5PHET through MCP, 2026-09-24

Coordinator, CPU only (CUDA_VISIBLE_DEVICES="" forced by tools/mcp_hermes.sh). No Telegram involved.
Every command below ran under crispdm-run -m 4G -t 600 -n wp11.

## 1. Hermes' own MCP client lists the three tools
See hermes_mcp_list_test.txt in this directory.

## 2. One-shot round trip: the catalog, asked in words
$ hermes -z "Use the m5phet MCP tools: call m5phet_catalog and reply with exactly the list of areas it returns
            and, for each area, the provider name and the question types it declares. Do not add anything the
            tool did not return. Do not run any other tool."
(exit 0)

Areas:

classification — provider: laya_news — question types: choice
forecasting — provider: predictor_forecast — question types: point_forecast, interval, anomaly_risk
causal — provider: causal_inference — question types: ate, cate
rl — provider: trading_policy — question types: next_action, value_estimation
unsupervised — provider: feature-eng-hierarchical-regimes — question types: clustering, cluster_description

NOTE: --ignore-user-config must NOT be used here. It ignores ~/.hermes/config.yaml, which is where the
mcp_servers.m5phet registration lives, so the MCP tools would not exist in that invocation.

## 3. The skill stops for review instead of executing
$ hermes -s m5phet -z "Report the ATE of treatment on outcome, with its uncertainty. Follow the m5phet skill exactly."
(exit 0)

The interpreter confirmed the canonical envelope for this installation. Per the m5phet skill I must show it and wait for your go-ahead before running it.

Envelope:

```json
{
  "area": "causal",
  "state": {"study": "ATE of treatment on outcome"},
  "questions": {"ate_effect": {"type": "ate"}}
}
```

Reply "ok" (or "dale", "run it", "corre", "sí, ejecuta") and I'll execute it once.

## 4. The same question with the envelope already approved: the engine answers
$ hermes -s m5phet -z "Follow the m5phet skill. I have already reviewed and approved this envelope, so treat
                       this message as the ok: {causal envelope with ate + cate baseline == 1}. Run it and
                       reply under the skill's reply rules."
(exit 0)

efecto: ATE = 2.0293820021534623 synthetic outcome units, 95% interval [1.931308938896731, 2.1274550654101936], development: true. Assumptions: consistency, constant_effect, iid_sampling, no_interference, nuisance_models_correct, positivity, pre_treatment_adjustment, sufficient_adjustment.

jovenes: REFUSED NOT_ESTIMABLE — the retained study estimates a constant average effect and was not fitted with an effect modifier, so a conditional effect under 'baseline == 1' cannot be read from it; answering this would need a study fitted explicitly with that modifier declared, which this engine does not offer.

execution_authorized: false — this is not an instruction to act

## 5. The same numbers straight from the MCP server (no language model in the path)
m5phet_execute_ml_task, area causal, questions {efecto: ate, jovenes: cate condition "baseline == 1"}:
  efecto  OK       effect_size 2.0293820021534623  unit "synthetic outcome units"
                   confidence_interval [1.931308938896731, 2.1274550654101936]  confidence_level 0.95
                   assumptions: consistency, constant_effect, iid_sampling, no_interference,
                   nuisance_models_correct, positivity, pre_treatment_adjustment, sufficient_adjustment
                   not_carried: p_value
  jovenes REFUSED  NOT_ESTIMABLE (no effect modifier in the retained study)
Hermes reproduced these digits exactly; nothing was rounded and no number was added.

## 6. Other areas through the same MCP surface (same day, same server)
  rl         next_action OK  action [-0.019998908042907715] "target position fraction of the policy s own
             action scale"; value_estimation OK expected_return 0.1352158784866333 (all-zero observation
             vector of 2724 features — a different input from the market-bars example)
  unsupervised  clustering OK and cluster_description OK on the 8 stored OHLC rows
  forecasting   point_forecast REFUSED PROVIDER_ERROR when no windowed input is attached
                ("data must contain exactly columns, values, scale and scaler_digest");
                interval REFUSED NOT_ESTIMABLE (no quantile or ensemble head in the bundle)
  classification OK but from the backend this host declares locally; label "other" at 1.0, NOT the
                workbench s euro_area 0.9666. The MCP server has no worker route (web/engine.py has it),
                and the answer carries no NON_MODEL_FIXTURE marker. See docs/TELEGRAM.md §3.
