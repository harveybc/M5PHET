# RB04 — the preserved RR05 work integrated, the envelope's three holes closed, and the corpus measured

Satoshi, successor technical lead, 2026-09-28 (America/Bogota).
Base: M5PHET master `8a6d1a3`. Branch: `satoshi/rb04-m5phet-integration-20260928`.
Orders: RB04 of `SATOSHI_RESUME_AND_BENCHMARK_ORDERS_2026_09_28.md`, under the
work plan of 2026-09-24 (revisions 1-3) and its standing rules §0.

Nothing here authorizes execution, an order, or a claim about a market. No
service the owner is running was started, stopped or restarted. The owner's
workbench on port 8765 was never driven, and no conversation was overwritten.
The deployment below is **prepared and proposed, not applied**.

---

## 1. The thirteen preserved files, and how they were integrated

The worktree `m5phet-rr05-20260926` sat on master `8a6d1a3` with thirteen
uncommitted paths and no commit, left by an agent cut off mid-task by a rate
limit. They were read, understood, and committed **byte-identical** as the
first commit of this branch — `c713fa2` — before any merge or edit could touch
them. Each of the twelve tracked files was compared with `cmp` against the
source tree and the evidence directory with `diff -rq`; all thirteen matched.
The source worktree was not modified, staged, stashed, reset or cleaned.

What they contained, and what each one is for:

| Path | What it adds |
|---|---|
| `src/m5phet/decide.py` | `diagnose_chooser`: what a set of RETAINED decision records says about the chooser that wrote them |
| `src/m5phet/quality.py` | `reference_collapse` + `QUALITY_REFERENCE_COLLAPSED`: a collapsed regimes reference publishes a refusal, not a stability |
| `src/m5phet/orchestrate.py` | `complete_under_answer` / `completable_types`: the under-answered second ask, recovered from declared types only — **see §4.4, which is where this one stops being good news** |
| `evaluation/src/m5phet_evaluation/scoring.py`, `report.py`, `__init__.py` | `NO_DECISION_TAKEN`: an all-zero action series is the flat baseline under another name |
| `tools/measure_route.py` | `--only` with `corpus_scope`, so a subset can never be quoted as the router's reliability |
| `tests/test_chooser_diagnosis.py`, `tests/test_route_completion.py` | 6 + 15 tests, counterexamples beside the positive paths |
| `tests/test_quality.py`, `evaluation/tests/test_report.py`, `evaluation/tests/test_scoring.py` | the assertions those three changes required, including one deliberately **corrected** assertion |
| `docs/evidence/RR05_2026_09_26/` | six evidence JSONs |

Three of them are substantive findings and not refactorings, and each one makes
a claim *narrower* than the claim it replaces. The fourth, the completion pass,
is the one thing in the preserved work that the full corpus contradicts, and
§4.4 says so at length rather than burying it:

- **The chooser is not ignoring the state.** The finding this answers said the
  checkpoint used as a chooser "abstained on 28 of 28 corpora and ranked the
  same option first on all 28 distinct state digests", which was read as the
  model ignoring what it was shown. The retained records say something
  narrower, and I re-read it from the retained evidence rather than repeating
  it: all 28 probability vectors **differ** across 28 distinct state digests,
  so the state IS read; the argmax is `agglomerative` on all 28; the top
  probability runs `0.3171`–`0.4360`, mean `0.372975`, against a chance level
  of `0.25` over four options; and the cited threshold is `0.80`, which **no**
  record reaches. The verdict is `ARGMAX_INVARIANT_BELOW_THRESHOLD` and not
  `STATE_INDEPENDENT` — every abstention is the measured rule holding, not a
  failure of the model. No accuracy appears in the diagnosis and none can: the
  records carry no correct answer.
- **The regimes area carries no quality number.** The served reference put all
  10 080 scored rows in ONE cluster, so silhouette and Davies-Bouldin do not
  exist for it by arithmetic, and what survived in `values` was a *stability* —
  the agreement between two assignments, which is reproducibility and never a
  quality. Until this change the area published `MEASURED` with that number
  while the same report said both indices were `INDEX_NOT_DEFINED`. The
  preserved work **corrected its own earlier test** to assert the refusal, and
  kept the stability under the name `reproducibility_only`, which cannot be
  read as a quality.
- **A stage that proposed nothing is not a stage that proposed the baseline.**
  The 2026-09-25 WP21 `laya_first_layer` report was numerically identical to
  its `flat` report because the chooser abstained on all 256 rows and abstention
  falls back to flat — and nothing in either document said so. A reader of the
  three-stage table would have seen the Laya stage "match the baseline". It is
  the baseline under another name, and it is now flagged in the metric set, the
  report flags, the statements and the headline.

**The merge with `44130ae`.** Musashi's `musashi/route-values-20260927` was
written from the same base `8a6d1a3`, and both it and the preserved work touch
`tools/measure_route.py`. It is a merge (`d4fb481`, two parents) and not a
replacement, because the two changes are in different halves of that file and
both are needed. Git auto-merged it; I then verified by inspection that both
halves survived — his `value_problems` v2 (every explicit governed-value
occurrence, across aliases, no truncation of fractional values, no boolean
accepted as an integer, conflicts carrying their paths) and `"outcome":
outcome`, beside the preserved `measure(..., only=None)` and `corpus_scope`.

One thing his work claimed that it did not achieve is corrected in `9477997`
and reported in §3.

---

## 2. The three authorized items

Written top-down: the behavioural tests in `tests/test_route_validation.py`
came first and were confirmed red, then the implementation. Nine cases were
**probed** against the integrated tree before a line of either was written —
five of them turned out to be holes the envelope let through, a sixth was the
completion pass never offering `interval` at all, and three were controls that
already behaved. So every counterexample below records a defect that was really
there rather than a hypothetical, and the rows marked *accepted* are the
measured ones.

| Probe | Envelope before | Envelope now |
|---|---|---|
| `horizon: True` where `1` is fitted | **accepted** | refused |
| `horizon: 60.5`, `inf`, `nan` | already refused | still refused, now by an explicit finiteness rule rather than by a membership test that happened to fail |
| `horizon: 60.0` where `60` is fitted | accepted | accepted (unchanged on purpose) |
| question field `target_variable: "NOPE"` | **accepted** | refused, naming the fitted targets |
| `target` and `target_variable` disagreeing | **accepted** | refused as a contradiction |
| `interval` with `confidence_level: 0.99`, nothing fitted there | **accepted** | refused, naming the fitted levels |
| `interval` with `confidence_level: 95` | **accepted** | refused: not a two-sided level |
| `completable_types` offering `interval` | **never** | offered where levels are declared |
| `output.plugin: "carrier_pigeon"` | n/a (no such field) | refused by name, listing what is installed |

**Numeric validation.** `orchestrate.admits` replaces plain `value in allowed`.
`True == 1` in Python, so a slot whose fitted horizon was 1 admitted `True` and
a boolean reached an engine as a horizon — the same defect Musashi repaired in
the measurement scorer, still live in the product validator. A boolean is now
admitted only by a slot that declares booleans; nothing non-finite is admitted
at all; a finite number is admitted by an equal number of either type, so
`60.0` IS the fitted `60`; everything else must match type and value, so the
string `"60"` is still not the integer `60`. That last one is deliberately
*stricter* than the scorer, which accepts an exact integer string in order to
score what a model wrote: `verdict_of` reads `INVALID_PROPOSAL` before it reads
any value, so a refused envelope is never scored correct and the two cannot
disagree.

**Alias validation.** A governed field is now governed under every spelling it
has, in questions as well as in the state, and two spellings of one field
holding different values are refused even when both values are admissible —
which one the engine would read is not a thing a validator should leave to the
engine. `GOVERNED_ALIASES` is asserted against the measurement tool's own
`FIELD_SPELLINGS` by a test, because a spelling one of them governs and the
other does not is a hole that reopens without anybody noticing.

**Interval requests.** `confidence_level` is now a governed value read from the
forecast provider's **own** `capabilities()[…]["fitted_confidence_levels"]` —
the symmetric pairs of quantiles a bundle really has, a field the installed
provider already publishes. The live verification workbench publishes
`forecasting.confidence_levels: [0.9, 0.95]` and `[]` for the other four areas.
Nothing is derived: a 0.95 interval is not a widened 0.90 one, and its two
bounds are two quantiles somebody fitted or two numbers somebody invented.

Rule 2 of the completion pass is widened by exactly one thing and no more: a
required field whose values the **area declares** may be chosen among them, in
the interpreter's constrained mode — the mode whose reliability was measured
(0.9434 when it chose) rather than the free-text envelope mode. An area that
fitted no level declares an empty list, so `interval` stays unofferable there.
The counterexample is asserted.

**Output selection.** An envelope may now carry `output: {"plugin": "<name>"}`.
It is validated against what is installed, refused by name listing the
alternatives, and **never substituted** — a request naming a surface this
installation does not have is told so rather than quietly answered by
`default`. `narrate` honours it ahead of the area's configuration, and the
area's header is unchanged by it, because a header is a property of the area
and its provider and not of the screen the answer lands on.

It also **did not work** when I first claimed it did, and the way that was
caught is worth recording. `narrate`'s precedence is explicit argument, then
the envelope, then the configuration — and `Engine.execute_task` passed the
configured plugin as the explicit argument on every run, so the configuration
won every time and a request that selected a surface was silently rendered by
the other one. My unit test of `narrate` passed. The feature did nothing
through the API, which is the only path a request actually takes. The web test
added in `7189e66` is red before the fix and green after; the configured
procedure is now a fallback used only when the envelope selects nothing, which
is its own counterexample test. That is the difference between testing a
function and testing a product, and it is the second time in this work that a
test passing at the wrong altitude hid a real defect — §3 is the first, §4.4
the third and largest.

**A correction I made to my own work, recorded rather than hidden.** An earlier
draft refused the WHOLE envelope when a well-formed confidence level was not
fitted, and added an envelope-wide required-field check. Both violate this
repository's own stated rule, in `m5phet/questions.py`: *"a request asking for a
point forecast and an interval gets the point forecast from an engine that has
one and an explicit refusal for the interval from an engine that does not. The
alternative — refusing the whole request, or worse, inventing the interval —
would hide which half the engine can actually do."* It surfaced as a real
regression in `tests/test_web_tasks.py`. The rule is now split: a **malformed**
level is an envelope problem; a level outside a **declared** vocabulary is an
envelope problem, for the same reason an unfitted horizon is; and a well-formed
level where the area declares none is left to the provider's per-question
refusal while the other half is answered. The completion pass keeps its own
completeness check instead, so an addition whose required field nobody chose is
dropped rather than sent half-formed. Both behaviours are now tested, including
the run-through that asserts the point forecast is answered and the interval
refused `NOT_ESTIMABLE` in the same response.

---

## 3. A defect in the merged work, found by using it

`44130ae` set out to retain "the complete endpoint outcome" so a later
rescoring is possible, and retained `one_run`'s return value — which was
already a hand-listed subset of the response. Everything it did not name was
still discarded, and the field it did not name **by name** was `completion`:
the record the preserved RR05 pass writes saying whether a sentence's second
ask was recovered, declined, or never offered. A measurement of the completion
pass that discards the completion record measures the pass with its own
evidence thrown away.

It was invisible to the test written to cover it, because that test
monkeypatches `one_run` away and asserts the report keeps whatever `one_run`
returned. This is how it was found instead: the first measured run of the
integrated build returned `completion: null` for *"pronostica la potencia y
dame un rango"*, which read as the pass never running. It had run.

Fixed in `9477997`. The response is kept whole under `response`, minus exactly
two fields — `catalog`, the same large object on every run and already
published once at the top of the report, and `profile`, which describes the
attachment rather than the routing — with a counterexample asserting the
exclusion is a list of two and does not grow, and another asserting that a
field the endpoint grows tomorrow reaches the report without anybody editing
the tool. New tests drive the real `one_run` against a fake transport.

With the record kept, that sentence reads:
`offered: ["point_forecast"], choice: "none", outcome: NOTHING_FURTHER_ASKED`.
The pass was reached, it offered the missing half, and **the model said the
sentence asks nothing further.** That is a measurement of the model, not a
broken pass, and it is now visible instead of silent.

---

## 4. The full-corpus measurement

### 4.1 What could and could not be rescored

**The historical measurement cannot be rescored, and I verified that rather
than repeating it.** `docs/evidence/ROUTE_RELIABILITY_2026_09_25/route_reliability_n5.json`
— `sha256 aac62e58…da4aa8fa`, the digest Musashi published — holds 19 sentences
and 95 runs of which **0 retain a complete outcome**; the RR05 subset report
(`sha256 3d58926c…914ed08f`) holds 4 sentences and 20 runs, also **0**. Its
81/95 therefore stands under its original scorer and no replacement rate has
been invented for it.

What could be done, and was, is a **new full-corpus measurement under the
corrected scorer, with every run's complete response retained** so the next
person can rescore what I could not. It is **VERSIONED**: the `PROTOCOL` text
that fingerprints checkpoints changed twice today (once with Musashi's scorer
v2, once with the retention of §3), so no checkpoint was resumed across either
change — the reports say `sentences_resumed_from_checkpoint: 0`.

### 4.2 The build that was measured

Port 8766, its own state directory, `CUDA_VISIBLE_DEVICES=""`, the staging
build of §6, interpreter `command · deepseek-v4-flash (OpenCode Go)` through
the owner's `hermes` — the same interpreter the 2026-09-25 baseline used.
Module digests of the measured build:

| module | sha256 |
|---|---|
| `orchestrate.py` | `0f7d4c79fe2a0faf083ee7a75bcfb48d4267d4a705b34dde8e0a66f677b8f0e5` |
| `questions.py` | `3a7ae5b026a8bd30c17573bce16e7c71268dadc6cf11c21181925d5b52ae6bf9` |
| `quality.py` | `3949eea3105b207ad4d2cd2b8615728106a3fe9217d693012e272de3d7ab2b94` |
| `decide.py` | `3a076f1b2dc9423ef3c8b2329de951f843f5da9437d0c826ccd4941f970f2d4a` |

That `orchestrate.py` is commit `454d0d2` plus nothing; it differs from this
branch's tip in exactly two lines, both of them an unused parameter dropped
from a private function's signature (`60c795f`), which is why that cleanup was
committed on its own.

### 4.3 The measurement

19 sentences, 5 runs each, 95 runs, `corpus_scope.scoped: false`,
`selected: 19 of 19`. **Not a focused run and not presented as one.**

| verdict | 2026-09-25, old scorer | 2026-09-28, completion **on** |
|---|---|---|
| CORRECT | 81 | **79** |
| WRONG_TYPE | 12 | 12 |
| INVALID_PROPOSAL | 1 | 2 |
| REFUSED | 1 | 2 |
| WRONG_AREA | 0 | 0 |
| WRONG_VALUE | 0 | 0 |
| reliability | 0.8526 | 0.8316 |

**Read that table as a diagnostic and not as a like-for-like rate.** The scorer
changed between the two columns, the router is a language model and its runs
are stochastic, and the left column cannot be re-derived under the right
column's rules. What the per-sentence breakdown supports is an account of
*which* sentences moved and why — and for that, every run's response is
retained.

| sentence | 2026-09-25 | completion **on** |
|---|---|---|
| Which economy is named in this news? | C:5 | C:4 REFUSED:1 |
| ¿De qué economía habla esta noticia? | C:5 | C:5 |
| de que economia habla y con que tono | C:5 | C:4 REFUSED:1 |
| predict household power one hour ahead | C:5 | C:5 |
| ¿cuánta potencia habrá en la próxima hora? | C:5 | C:5 |
| what is the direction_long probability at horizon 1? | C:5 | C:5 |
| ¿cuál es la probabilidad de direction_long a horizonte 1? | C:4 INVALID:1 | C:5 |
| **pronostica la potencia y dame un rango** | C:1 WRONG_TYPE:4 | **C:3 INVALID:2** |
| **describe el grupo de velas con cuerpo alto** | C:5 | **WRONG_TYPE:5** |
| **describe the cluster with a large body** | C:5 | **WRONG_TYPE:5** |
| **assign hierarchical regimes to these rows** | C:2 WRONG_TYPE:3 | **C:5** |
| asigna los regímenes jerárquicos a estas filas | C:3 WRONG_TYPE:2 | C:3 WRONG_TYPE:2 |
| segmenta estas filas y describe el cluster alto | C:5 | C:5 |
| Report ATE of treatment on outcome, with its uncertainty. | C:5 | C:5 |
| ¿Cuál es el ATE of treatment on outcome y su incertidumbre? | C:5 | C:5 |
| **cual fue el efecto del tratamiento y en jovenes** | C:1 REFUSED:1 WRONG_TYPE:3 | **C:5** |
| What action does …anchor_v1 propose? | C:5 | C:5 |
| ¿Qué acción propone la política para estas barras? | C:5 | C:5 |
| que accion propone y que retorno espera | C:5 | C:5 |

### 4.4 The finding: the completion pass over-answers, deterministically

**Every one of the twelve WRONG_TYPE runs is the completion pass adding
`clustering`** — `outcome: ADDED, added: clustering, choice: "clustering"` on
all twelve. It is visible at all only because §3 stopped the completion record
being discarded.

The pass repairs and breaks, and the two are not symmetric:

- **repaired** — *"cual fue el efecto del tratamiento y en jovenes"* 1→5,
  *"assign hierarchical regimes to these rows"* 2→5, and on the very sentence
  it was designed for, *"pronostica la potencia y dame un rango"*, it added
  `point_forecast` to the model's `interval`-only envelope on 3 of 5 runs,
  taking that sentence from 1 correct to 3. **+9 runs.**
- **broken** — *"describe el grupo de velas con cuerpo alto"* and *"describe
  the cluster with a large body"*, each **5/5 → 0/5**, deterministically, on
  both languages of the same request. **−10 runs.**

The model, asked whether "describe the cluster with a large body" *also* asks
for `clustering`, says yes on all ten runs. Describing a cluster does imply
that rows were assigned, so this is not a plumbing fault: it is the mirror
image of the under-answer the pass exists to repair. And the trade is worse
than the aggregate −1 suggests, because the two losses are deterministic on
sentences that were deterministically correct, while the gains are partial.

**This is exactly the rule the order insists on, met rather than dodged.** The
pass was built and validated on a 4-sentence subset — a subset whose own report
says `"a SUBSET of the declared corpus … not this router's reliability"` — and
its 15 focused tests all pass. The full corpus is what found the over-answer.
A focused run would have reported a success.

So the pass now ships **off**, behind `M5PHET_ROUTE_COMPLETION`, and the
default is this measurement rather than a preference (`1a0df15`). The
measured regression is retained as a counterexample test in both directions.

### 4.5 The confidence rule catching a real model error

The 2 INVALID_PROPOSAL runs are **not** a regression; they are the new
confidence-level rule firing on a genuine fault. On runs 3 and 5 of *"pronostica
la potencia y dame un rango"* the model wrote:

```
"confidence_level": 95        # runs 3 and 5 — refused
"confidence_level": 0.95      # runs 1, 2 and 4 — a fitted level, accepted
```

`95` is a percentage somebody typed where a two-sided level belongs. Before
today it **passed validation** and reached the engine. It is now refused before
the run, naming what the area fitted: `[0.9, 0.95]`. Those 2 runs were not
correct under the old scorer either — they were part of that sentence's
WRONG_TYPE:4 — so nothing correct was lost by refusing them; what changed is
that the person is now told the request was misread instead of receiving a
refusal that blames the engine.

The 2 REFUSED runs are the **interpreter itself** declining two classification
sentences, on the grounds that `choice` needs an `options` field the message
does not provide. That is the model's own reasoning, it is stochastic (the
baseline had one such refusal), and it is an abstention rather than a wrong
answer.

### 4.6 The paired completion-off measurement

The `ON` column above cannot settle whether the pass is worth having, because
it is compared against a run under a different scorer. So the corpus was
measured a **second** time, same day, same interpreter, same scorer, same
harness, the same 19 sentences and 5 runs each — with the pass **off**. That is
the comparison the decision needs, and it is a like-for-like one.

| verdict | completion **on** | completion **off** |
|---|---|---|
| CORRECT | 79 | **85** |
| WRONG_TYPE | 12 | 9 |
| INVALID_PROPOSAL | 2 | 0 |
| REFUSED | 2 | 1 |
| WRONG_AREA | 0 | 0 |
| WRONG_VALUE | 0 | 0 |
| reliability | 0.8316 | **0.8947** |
| reliability when it proposed | 0.8495 | 0.9043 |
| sentences always correct | 13 | 14 |
| sentences never correct | 2 | **0** |

**The pass costs six runs.** And the sharpest number in this document is the
precision of its additions: over the 95 runs it made **15 additions, of which
3 were right and 12 were wrong.**

| what it added | runs | how those runs ended |
|---|---|---|
| `point_forecast` | 3 | CORRECT — the under-answer it was built to repair, repaired |
| `clustering` | 12 | WRONG_TYPE — every one |

Precision **0.20**. It is right exactly where it was designed and measured, on
*"pronostica la potencia y dame un rango"*, and wrong everywhere else it fires.
`sentences_never_correct` tells the same story from the other end: **2** with
the pass on, **0** with it off — the pass is the only thing on this corpus that
takes a sentence from working to never working.

Per sentence, the two runs of today:

| sentence | completion **on** | completion **off** |
|---|---|---|
| Which economy is named in this news? | C:4 REFUSED:1 | C:5 |
| de que economia habla y con que tono | C:4 REFUSED:1 | C:4 REFUSED:1 |
| pronostica la potencia y dame un rango | C:3 INVALID:2 | C:2 WRONG_TYPE:3 |
| describe el grupo de velas con cuerpo alto | WRONG_TYPE:5 | **C:5** |
| describe the cluster with a large body | WRONG_TYPE:5 | **C:5** |
| assign hierarchical regimes to these rows | C:5 | C:3 WRONG_TYPE:2 |
| asigna los regímenes jerárquicos a estas filas | C:3 WRONG_TYPE:2 | C:2 WRONG_TYPE:3 |
| cual fue el efecto del tratamiento y en jovenes | C:5 | C:4 WRONG_TYPE:1 |
| the other 11 sentences | C:5 each | C:5 each |

Two honest qualifications, both of which cut against the tidiness of the story:

- **The pass does help four sentences.** With it on, *"assign hierarchical
  regimes"* goes 3→5, *"cual fue el efecto…"* 4→5, *"asigna los regímenes"* 2→3
  and the interval sentence 2→3. Its idea is sound; its firing rule is not. The
  finding is not "the pass is worthless", it is "at a precision of 0.20 it must
  not be the default".
- **The 2 INVALID_PROPOSAL runs did not recur.** With the pass off, the model
  wrote `confidence_level: 0.95` on all five runs of that sentence and the new
  rule refused nothing. So `confidence_level: 95` is a **stochastic** fault of
  the router that the new rule catches when it happens — not a rule that
  systematically refuses valid envelopes. One run in ten of that sentence, on
  today's evidence, and previously it passed straight through.

The whole-corpus state of the completion record, with the pass on, for anyone
who wants to see where it does and does not fire: `NOTHING_FURTHER_ASKED` 47,
`NO_COMPLETABLE_TYPE_DECLARED` 28, `ADDED` 15, `ADDITION_DID_NOT_VALIDATE` 1,
and 4 runs where the envelope never validated so the pass was never reached.
With it off, `COMPLETION_NOT_ENABLED` on all 94 runs that produced an envelope.

**Both reports are committed with every run's complete endpoint response**, so
the next person can rescore what I measured — which is the thing I could not do
to the 2026-09-25 report.

| report | sha256 |
|---|---|
| `route_reliability_full_n5_completion_on.json` | `c512ad051ac784ee0b0a62b2520ec4e0fa20bb341720700895b68f23ea8f7f9e` |
| `route_reliability_full_n5_completion_off.json` | `3e7ae31f226ce8b2be94cdf81967b0e43e264a890c2661a3e5ccce1bafbf85ee` |

The off-run's build is commit `1a0df15`'s `orchestrate.py`
(`sha256 0b2b5bab5eb14c54d2558351f07ad1259538c09249010c999452c868224070ba`) and
the same `questions.py` as the on-run. Its `web/engine.py` predates `7189e66`,
which does not touch routing: the measurement only ever calls
`POST /api/tasks/propose`, which builds an envelope and runs nothing.

---

## 5. A manually usable chat path per family, and what actually answered

Driven through the verification workbench's own HTTP API — `POST /api/login`,
`GET /api/tasks/catalog`, `GET /api/catalog` — exactly as the browser does.
Every family keeps a ready example carrying its dataset, its prompt, its
provider, its fitted state and its evidence scope in its own title.

| Family | Provider | What answers | Task / question types | Dataset in the kept example | Evidence scope, in its own words | Quality the answer carries |
|---|---|---|---|---|---|---|
| classification | `laya_news` | **real provider** — `backend: laya`, `weights_present: true`, `device: cuda:0`, checkpoint `laya-checkpoint:bd12df88…` | `choice` | the news text of *"Noticia · Clasificación"*, prompt *"Which economy is named in this news?"* | the example's own title; the quality block names `INDEPENDENT_LABELS` from the economic-calendar country field | **MEASURED**: macro-F1 `0.3778`, n `450`, ECE `0.1315`, Brier `0.6473`, declared `UNCALIBRATED`, naive on the same rows: majority class `0.1667`, keyword baseline `0.1760` |
| forecasting | `predictor_forecast` | **real provider** — `backend: tensorflow_saved_model_cpu_subprocess`, fitted bundles including a quantile head | `point_forecast`, `interval`, `anomaly_risk`; `confidence_levels [0.9, 0.95]` | *"DEVELOPMENT: retained household-power model, 60-minute forecast"*, prompt *"forecast Global_active_power at 60 steps"* | "DEVELOPMENT"; the reading states the unit: *"a level in kW (original scale)"* | **NOT_MEASURED** — no held-out report is bound in this configuration |
| unsupervised | `feature-eng-hierarchical-regimes` | **real provider** — a reference fitted offline, assignment only | `clustering`, `cluster_description` | *"DEVELOPMENT: 8 historical OHLC rows, no market-performance claim"*, prompt *"Assign hierarchical regimes"* | the title refuses a market claim outright | **NOT_MEASURED** — no report bound; the collapse refusal of §1 is implemented and tested but not exercised here |
| causal | `causal_inference` | **real provider** — `backend: EconML LinearDML / explicit offline fitted study` | `ate`, `cate`, `counterfactual_path`, `impulse_response`, `sensitivity` | *"SYNTHETIC/DEVELOPMENT: effect modifier baseline, known subgroup effects…"*, prompt *"Report wp20-spec-modifier-v1, with its uncertainty."* | "SYNTHETIC/DEVELOPMENT", with the known subgroup effects stated in the title | **abstention** — `REFUSED: causal_accuracy`, named by the evaluation package |
| rl | `trading_policy` | **real provider** — `backend: stable_baselines3_cpu_subprocess`, fitted policy `policy:4579e6f0…` | `next_action`, `value_estimation` | *"DEVELOPMENT: an all-zero observation vector; its action is this input's, not the bars example's"* | the reading explains that the same policy answers the market-data example differently because the input differs | **abstention** — `REFUSED: policy_profitability` |

Read that table as three kinds of answer and not one: **five real providers**,
**no fixture** on this configuration, **one measured quality**, two areas with
nothing measured and **two named abstentions**. The rl example's zeros are an
all-zero *observation*, which is an input; the preserved `NO_DECISION_TAKEN` of
§1 is about an all-zero *action series*, which is an output. The two must not be
read as the same thing.

**Fixture, real provider and abstention are three different things and are
reported as three.** `chat.env` sets `NEWS_SIGNAL_BACKEND="fixture"`, which
would make classification the declared `NON_MODEL_FIXTURE`; it also sets
`M5PHET_CHAT_LAYA_WORKER`, which overrides it to the private worker with real
weights. The live provider declares `backend: "laya"`,
`weights_present: true` and a checkpoint digest, so on this configuration
classification is answered by a **real provider**, not a fixture. Nothing
disguises itself in either direction. `execution_authorized` is `false` on
every answer.

---

## 6. The pinned build, prepared and NOT applied

The owner's verified fact is confirmed independently. The chat service
(`m5phet-chat`, port 8765, running throughout this work) executes a
**non-editable** pip copy in `$HOME/.local/share/m5phet/chat-venv`, whose
`direct_url.json` names the worktree `m5phet-chat` at `f532ee5` on
`satoshi/wp22-dml-20260925`. That is **not** master `8a6d1a3`: `f532ee5` is
three commits ahead of master on its own line and does not contain master's tip
commit. A restart would not make it master; it would re-execute `f532ee5`.

Prepared under `$HOME/.local/share/m5phet/staging-rb04-20260928`:

| Artifact | What it is |
|---|---|
| `venv/` | a copy of the service's venv — the owner's exact provider set — with `m5phet` and `m5phet-evaluation` reinstalled **non-editable** from this branch's tip. `direct_url.json` names this worktree, and `orchestrate.py` / `questions.py` / `web/engine.py` in it are byte-identical to the tip (`e94754c3…`, `3a7ae5b0…`, `3b905283…`). |
| `backup-20260928/chat.sqlite3` | the owner's conversations, taken with sqlite's **online backup** so a live writer cannot tear it. `PRAGMA integrity_check` = `ok`. `sha256 0cbc2668…8d74005` |
| `backup-20260928/installed-m5phet-packages.tar.gz` | the exact installed bytes a deployment would replace. `sha256 199ae1ae…9ef0a4d` |
| `rollback.sh` | restores those bytes, refusing to run if the archive is not the one it was written for. It deliberately does **not** restore the conversation database: a rollback of code must never roll back conversations, which are newer than the backup by every minute the service has run. |
| `read-only-smoke.sh` | starts the staging build on its own port and its own `--state-dir`, logs in, and prints each area's provider, question types, fitted confidence levels and quality status. It never touches 8765 or the owner's state. **Run against the pinned build and verified**, output in §5. |
| `run-verification-workbench.sh` | the launcher used for every measurement in this document (port 8766, own state directory, `CUDA_VISIBLE_DEVICES=""`). |

**The proposed deployment, for the owner to apply or refuse:**

```bash
# 1. read-only proof on a separate port, changing nothing
$HOME/.local/share/m5phet/staging-rb04-20260928/read-only-smoke.sh

# 2. re-take the conversation backup, because it will be older than the last message
sqlite3 "$HOME/.local/state/m5phet/chat/chat.sqlite3" \
        ".backup '$HOME/.local/share/m5phet/staging-rb04-20260928/backup-20260928/chat-at-deploy.sqlite3'"

# 3. install this branch over the service's venv (code only; state is untouched)
#    RB04 is the worktree of satoshi/rb04-m5phet-integration-20260928; set it to your own checkout path
RB04="$HOME/Documents/GitHub/.worktrees/m5phet-rb04-20260928"
"$HOME/.local/share/m5phet/chat-venv/bin/python" -m pip install -q --no-deps --force-reinstall \
    "$RB04" "$RB04/evaluation"

# 4. the OWNER restarts, when no session is in progress
systemctl --user restart m5phet-chat

# rollback, if anything is wrong
$HOME/.local/share/m5phet/staging-rb04-20260928/rollback.sh   # then the owner restarts again
```

Step 4 is the owner's and is not mine to take: the service has been serving
throughout, and restarting it in the middle of somebody's session is one of the
things this order forbids. Steps 1-3 change no running process, but step 3
rewrites the venv the live service would re-execute on its next restart, so it
is proposed and not applied either.

---

## 7. Resources

Every child ran through `$HOME/.local/bin/crispdm-run` at the revision already
deployed; nothing was reinstalled and no declared cap was shrunk to evade a
refusal. All of it ran on the coordinator, which is admissible here because
none of it is a fit or a model import: the heaviest thing measured is a
pure-Python test suite, and the corpus measurement's own cost is remote calls.

| Job | Cap | Whole-cgroup peak |
|---|---|---|
| test suites (`rb04-*`, many) | 2 GiB | ≤ 17 MiB per suite |
| verification workbench, completion on (`rb04-verify-workbench`) | 3 GiB | 516 MiB |
| verification workbench, completion off (`rb04-verify-wb-off`) | 3 GiB | 423 MiB |
| full-corpus measurement, completion on (`rb04-fullcorpus`) | 2 GiB | 16 MiB |
| full-corpus measurement, completion off (`rb04-corpus-off`) | 2 GiB | 16 MiB |
| staging venv copy and installs (`rb04-stage-*`) | 2 GiB | small |

Coordinator: 22 GiB available at the start and 22 GiB at the end, memory PSI
`avg10 = 0.00` throughout, no swap used, nothing killed. The preferred external
5090 and the secondary worker were **not** used and were not disturbed, and the
owner's chat service on 8765 kept serving from start to finish.

The measurement's real cost is not memory but **190 calls to the owner's
`hermes` interpreter** — 95 per full-corpus run, two runs — each one a remote
language-model call on the owner's credentials, at a median 10-18 s. That is
the same interpreter and the same 95-call shape as the 2026-09-25 baseline, and
it is the whole budget this work spent outside this machine.

Two verification workbenches were started and stopped, both mine, both on port
8766 with their own state directory, and the second was started only because
the completion switch is read at start-up. Stopping mine to restart it is not
the forbidden act: the owner's service was never signalled.

One flaw of my own, found and fixed here rather than left: the first version of
`read-only-smoke.sh` used a bare `trap … EXIT`, which does not fire when the
wrapper is killed by a wall limit, and it leaked a listener on its port. It now
runs its server under its own `timeout` and traps `EXIT INT TERM HUP`;
re-verified, and the port is free afterwards.

**No GPU work occurred and there is therefore no device UUID to record.** The
workbench runs with `CUDA_VISIBLE_DEVICES=""` by design, and
`POST /api/tasks/propose` builds an envelope and **runs nothing** — no engine
was reached, no bundle loaded, no allocation made. Anything in this document
that describes what an engine answers is read from a provider's *declaration*
or from a *retained* record, never from a fit performed here.

---

## 8. What is NOT done, refused, or not measured

- **The historical 95-run report cannot be rescored, and I verified it rather
  than repeating the claim.** `route_reliability_n5.json`
  (`sha256 aac62e58…da4aa8fa`, matching the digest Musashi published) holds 19
  sentences and 95 runs, of which **0 retain a complete outcome**; the RR05
  subset report (`sha256 3d58926c…914ed08f`) holds 4 sentences and 20 runs,
  also **0**. No replacement reliability has been fabricated for either, and
  the historical 81/95 stands under its original scorer.
- **`NO_NEW_MODEL_MEASUREMENT` for every engine.** Nothing here measures a
  forecast, a policy, a study or a checkpoint. The one number this document
  contributes is a property of the **router**, and the router is a language
  model writing an envelope.
- **Focused tests are not a measurement of full-corpus quality**, and none is
  presented as one. The 63 new validation tests, and the 149 collected across
  the five files this work most affects, are tests; §4 is the measurement, and
  §4.4 is the case in point: 15 focused tests passed while the pass they cover
  was breaking two sentences of the corpus.
- **The regimes collapse refusal is not exercised in this configuration.**
  `chat.env` binds no unsupervised quality report, so the live catalog says
  `NOT_MEASURED` for that area rather than `QUALITY_REFERENCE_COLLAPSED`. The
  refusal is implemented and tested against the real served report's own
  numbers; it will speak when a report is bound.
- **Forecasting quality is `NOT_MEASURED`** on the live catalog, and causal and
  rl are `REFUSED` by name. Only classification carries a measured quality
  block, and it is the real Laya provider's.
- The completion pass **does not repair** the measured under-answers by itself;
  §4 says what it did. It makes them visible, which is a smaller and true
  claim.
- No closure table is offered, because no model error was measured here:
  `NO_NEW_MEASUREMENT` in the owner's standing sense.

---

Satoshi, successor technical lead.
