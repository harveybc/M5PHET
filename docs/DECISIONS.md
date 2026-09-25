# Decisions — Laya as a configuration chooser (`m5phet.decide`)

WP17 of the 2026-09-24 work plan, revision 2. This page is the contract for the decision primitive that WP18–WP21
build on.

## What a decision is

A **decision** is one recorded answer to this question and nothing wider:

> Given this *state* — a structured, textual description written by the framework — and these *declared options*,
> which one does Laya choose, and with what uncalibrated probabilities?

Three parts, all of them explicit:

1. **The state.** `decision_state(kind, payload)` renders a mapping as deterministic text: keys sorted at every
   level, lists in the caller's order, floats at six declared decimals, integers left alone. The same payload always
   yields the same text and therefore the same `state_sha256`, whatever order its keys were built in. A state is a
   *description* — a feature's metric sheet, a dataset profile, a problem statement. It carries **no rows**: a list
   longer than 128 entries is refused as `ROWS_IN_STATE`.
2. **The options.** Always **declared by the repository that will have to execute the choice**: registered plugin
   names, declared parameter grids, fitted references. `decide` never invents one, and never lets the answer invent
   one either — a label outside the declared set is refused as `CHOICE_OUTSIDE_OPTIONS`.
3. **The answer.** Laya's label plus its own uncalibrated probabilities, copied verbatim: no rounding, no rescaling,
   no renormalising. `probability_decimals` is the number the *answer* declared, not one chosen here.

`ask(engine_or_registry, state_text, questions, *, as_of=None, kind, record_dir=None)` builds the classification
envelope

```json
{"area": "classification",
 "state": {"news": "<state text>", "asset": "M5PHET", "language": "en"},
 "questions": {"<name>": {"type": "choice", "options": [["<key>", "<label>"], …], "instructions": "…"}}}
```

and runs it. Given an `Engine` (`m5phet.web.engine`) the envelope takes the **worker route exactly as the workbench
does** — the real checkpoint on the private worker's GPU. Given a bare `Registry` it goes through `run_task` to
whichever provider declares the classification area.

`as_of` stamps the record; it is deliberately **not** sent in the envelope. The classification provider refuses to
replay a text state at a historical clock (`HISTORICAL_REPLAY_NEEDS_EVENT`) because a state text carries no clocks of
its own. A decision is made now, about a description.

## What a decision is not

- **Not a fit.** Nothing is trained, nothing is estimated. The choice is then fitted and measured by the explicit
  jobs the plan already has (WP06 stages 3–5, WP13).
- **Not a measurement.** A probability of 0.81 is the model's uncalibrated head output for this exact wording of
  this exact question. It is not an accuracy, not a confidence in the sense of correctness, and not evidence that the
  chosen configuration performs better. The closure table is the judge; a decision is only a hypothesis.
- **Not an authorization.** Every record carries `execution_authorized: false`, and `record()` refuses to write one
  that says otherwise. No order, no broker mutation, no run is authorized by a decision.
- **Not a choice Laya could make freely.** Laya never sees an option nobody declared, never sees rows, never fits
  anything.
- **Not something a fixture may produce.** An answer whose `backend` is not `laya`, or which carries
  `non_model_fixture`, is refused as `NON_MODEL_FIXTURE` and **nothing is written**. If the worker is unreachable,
  `decide` refuses (`PROVIDER_ERROR`) and the package reports `NOT_RUN`; the fixture is never substituted, and
  neither is the interpreter.

## Refusals, by name

Checked **before** anything is asked, each under its own question's name, the other questions still asked:

| Code | When |
|---|---|
| `STATE_REQUIRED` | the state text is empty |
| `MALFORMED_OPTIONS` | fewer than two options, a duplicate key, or a pair that is not `[key, label]` |
| `INSTRUCTIONS_REQUIRED` | a choice question with no instruction text |
| `MALFORMED_QUESTION` | the question carries a field a decision question does not take, or a type other than `choice` |

Checked on the answer:

| Code | When |
|---|---|
| `NON_MODEL_FIXTURE` | `backend != "laya"`, or the answer carries `non_model_fixture` |
| `CHOICE_OUTSIDE_OPTIONS` | the chosen label, or a probability's key, is not in the declared option set |
| `OPTIONS_MISMATCH` | the answer echoes an option set or an instruction other than the one asked |
| `NOT_A_DECISION` | the answer carries no label or no uncalibrated probabilities |
| `PROVIDER_ERROR` | the engine could not be reached, or returned no answer for the question |

The provider's own refusals are passed through **verbatim**, so their names survive this layer — `TOKEN_BUDGET_EXCEEDED`
(raised by news-signal as a `MALFORMED_QUESTION` whose reason names it) among them. `decide` never rewrites a reason
it did not produce.

## The record

`m5phet.decision.v1`, exactly these fields:

| Field | Meaning |
|---|---|
| `schema` | `m5phet.decision.v1` |
| `kind` | what is being decided (`feature_profile`, `clustering_method`, `estimator`, …) |
| `state_sha256` | sha256 of the state text **as it was shown to the model** |
| `question` | the question's name in the envelope |
| `options` | the declared option set, in the order it was offered |
| `chosen` | the key Laya chose; always one of `options` |
| `probabilities` | the answer's `uncalibrated_probabilities`, verbatim |
| `probability_decimals` | the decimals the answer declared |
| `checkpoint` | the answer's `state_ref` — which fitted checkpoint answered |
| `backend` | always `laya`; a record is never written for anything else |
| `as_of` | the clock the decision was stamped at |
| `execution_authorized` | always `false` |

`record(decision, dir)` writes it as **content-addressed** JSON: canonical serialisation (sorted keys, compact
separators), file name `<sha256 of those bytes>.json`. Writing the same decision twice is the same file; a different
decision is a different name, so a record cannot be silently revised. `load(path)` reads it back, re-validates it and
verifies that the content still hashes to the name it is filed under — a tampered record raises `DecisionError`
rather than loading.

The WP17 proof, one real decision made on 2026-09-25 against the real checkpoint on the worker's GPU (state = the
`Global_active_power` stationarity and autocorrelation block of `wp06_household_design_candidates.json`), written to
`9e173e4069e567b8e72cf01cf5ca04abbf9b3292eae6eaa74401f24bb6f2cc2c.json`:

```json
{"as_of":"2026-09-25T05:39:28.171818+00:00","backend":"laya",
 "checkpoint":"laya-checkpoint:bd12df887789924672d1c848319caf9866a172183e6959beab369d9e20aaaa89",
 "chosen":"log_return","execution_authorized":false,"kind":"feature_profile",
 "options":[["level","keep the level"],["diff","first difference"],["log_return","log return"]],
 "probabilities":{"diff":0.3249,"level":0.3219,"log_return":0.3532},"probability_decimals":4,
 "question":"transform","schema":"m5phet.decision.v1",
 "state_sha256":"21012333edfe2ea14ac29725777c1dc32342fb0cf1b8a0712f114356ccbbd638"}
```

That record proves the **plumbing** — a declared option set reaches the real checkpoint and its choice comes back
bound to a state digest, recorded and reloadable. It says nothing whatever about whether the log return is the right
transform for this series. Read the numbers as they are: 0.3532 / 0.3249 / 0.3219 over three options is close to
uniform, so this zero-shot checkpoint barely separated them. Whether such a choice is worth anything is exactly the
question WP18's closure table exists to answer, and it is still open.

## How WP18–WP21 consume it

Each package declares its own option sets — **never** `decide` — and writes one decision record per choice beside the
artifact that carries it. The records' digests go into the spec the fit then runs, so a fitted pipeline names the
decisions it came from.

- **WP18 (forecasting, the doctoral pipeline).** State = a feature's metric sheet (feature-eng). One decision per
  feature for the preprocessor (options = the `preprocessor.plugins` entry points that actually exist) →
  `preprocessing_plan.json`; one for the grouping cut among the declared `k` → `groups.json`; one per group for the
  extractor (options = what `feature-extractor` registers) → `extractors.json`; one for the core over the fused
  branches (options = the `predictor.plugins` that accept several input branches, tested first) → `core.json`. All of
  them plus the WP06 representation spec make `m5phet.pipeline.v1`, which is fitted and scored against
  `baseline_hand` and `searched` on the same sealed holdout.
- **WP19 (unsupervised).** Two decisions: the clustering method, then the parameter point within its declared grid.
  An explicit fit job writes a new fitted reference; `regime_accuracy` stays refused and the table carries the
  internal indices under their own names.
- **WP20 (causal).** Decisions for the estimator, the nuisance models, and one per column for its role
  (`treatment`/`outcome`/`confounder`/`modifier`/`exclude`). A study whose identification could not be stated is
  `NOT_IDENTIFIED` and is never fitted. The records sit beside the study manifest.
- **WP21 (RL).** One decision per decision point over a rendered window, options
  `[["long", …], ["flat", …], ["short", …]]`, replayed as a candidate policy in the gym-fx simulation and scored by
  the existing evaluation protocol against `flat` and the fitted SAC on the same rows. Nothing here sends an order.

In every one of them the decision is the hypothesis and the closure table is the verdict.
