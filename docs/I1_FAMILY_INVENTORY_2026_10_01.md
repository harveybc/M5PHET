# Family inventory — front I, 2026-10-01

Base: `a16317a` (descends from master `8a6d1a3`; master alone is 18 commits behind the verified line). Counts of
`def test_` are static greps, not runs. Nothing below was executed on the coordinator.

"Real" = a provider that loads a fitted engine through the `m5phet.providers` entry point. "E2E" = an automated test that
goes envelope -> real provider -> typed answer. "Persistence" = that answer read back from the store by a new process.

| Family | Provider (entry point) | Provider repo, tests (static) | Real provider | E2E through M5PHET | Persistence of its answer |
|---|---|---|---|---|---|
| forecast | `predictor_forecast` | prediction_provider `forecast/tests`, 112 | yes, TF bundles (own interpreter) | `tools/verify_families.py`, `verify_envelopes.py` against a live instance (not pytest) | `tools/verify_restart.py` (AP01), forecast case only |
| regimes | `feature-eng-hierarchical-regimes` | feature-eng `tests/test_*regime*`, 458 in the repo | yes, sklearn reference | same harnesses (live instance) | none before this branch; `tests/test_family_e2e.py` adds it |
| ATE / causal | `causal_inference` | causal-inference `provider/tests`, 168 | yes, EconML studies fitted beforehand | harnesses (live instance) | none |
| policy / RL | `trading_policy` | agent-multi `m5phet_policy/tests`, 100 | yes, SB3 SAC (own interpreter) | harnesses (live instance) | none |
| classification | `laya_news` | news-signal `tests`, 185 | real only with the Laya weights on a GPU worker; otherwise a declared `NON_MODEL_FIXTURE` | harnesses; `tests/test_classification_backend.py` | none for the real path (GPU, out of front I's resources) |

M5PHET's own suite (`python -m pytest -q`, 701 `def test_` at this base) drives the web layer with fake providers, except
`tests/test_web.py` (news_signal, importorskip). So no pytest in M5PHET proved a real provider's answer was stored
and returned until `tests/test_family_e2e.py`.

Plan items: every WP in the 2026-09-24 plan has code except WP32 (point-in-time capture; lives in `financial-data`, a
sibling) and the GPU stages (WP18 step 7, WP06 stages 3-4), which are not front I's to run.
