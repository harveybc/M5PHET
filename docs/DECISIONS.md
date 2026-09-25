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

## The outcome: when a decision becomes a label (WP23)

**A person's opinion is never a label; only a table row is.** That sentence is the whole rule of this section, and
every refusal below enforces it. A decision record is a hypothesis: Laya chose an option out of a declared set, having
seen a description and no rows. It becomes evidence *about Laya* only when the configuration it led to was fitted and
**measured**, and the measurement was found commensurable with the alternatives it is ranked against. Nothing else may
be written into an outcome — not the author's judgement that the choice looked sensible, not the plausibility of the
label, not a reviewer's agreement — because a fine-tuning corpus built from those would train the checkpoint on the
opinions of whoever assembled it, and would then be presented as measured evidence.

`outcome(record_path, table_row, *, out_dir)` links one decision record to **one** row of the closure table that
`evaluation/compare_stages.py` emits (WP13): the mapping carrying `stage`, `status`, `comparability`, `rank`,
`metric`, `model_error`, the `naive` reference and `skill`. It writes a content-addressed **outcome record**:

| Field | Meaning |
|---|---|
| `schema` | `m5phet.decision_outcome.v1` |
| `decision_sha256` | the decision record this outcome is about; also the name that record is filed under |
| `kind`, `question`, `chosen`, `options` | copied from the decision, so the outcome is readable on its own |
| `probabilities` | the decision's uncalibrated probabilities, verbatim — the argmax and the probability it claimed are what the calibration report bins, and a report that had to re-open the decision record to find them could be run against a different one |
| `table_row_sha256` | sha256 of the row's canonical JSON. The closure numbers are **bound, not copied**: they cannot be quoted out of the outcome, and cannot be changed behind it either |
| `stage` | the stage whose row this is |
| `rank` | the row's rank among the comparable stages — **this is the label** |
| `comparability` | always `COMPARABLE`; an outcome exists for no other verdict |
| `best_ranked_option` | present only on the outcome of the stage ranked first, where it equals `chosen` |

Refusals, each by name, **nothing written** for any of them:

| Code | When |
|---|---|
| `NOT_COMPARABLE` | the row's `comparability` is not `COMPARABLE`. A rank among stages measured on different holdouts, or against a different metric, orders incommensurable numbers |
| `NOT_RANKED` | the row carries no rank. The rank *is* the label |
| `DECISION_NOT_FOUND` | the path holds no readable `m5phet.decision.v1` record |
| `DIGEST_MISMATCH` | the record's bytes no longer hash to the name it is filed under: it was altered after it was written |
| `MALFORMED_TABLE_ROW` | the object is not a row as `compare_stages` emits it |

`best_ranked_option` is defined as *the option key of the stage ranked first among the stages that share the
decision's `kind` and `question`*. One call sees one row, so it can be settled at write time only when that row **is**
that stage — `rank == 1`. Otherwise the field is absent and the calibration report settles it for the group, by
reading every outcome that shares the kind and the question.

## The calibration rule (WP23)

`evaluation/decision_calibration.py` reads outcome records and reports, **per decision kind and question**:

- **n linked** — how many outcomes exist for that kind and question;
- **agreement** — how often Laya's argmax was the `best_ranked_option`. Not how often it was plausible, not how often
  a reviewer would have chosen the same;
- **reliability** — the outcomes binned by their argmax probability in bins of 0.1, each bin showing `n`, its observed
  agreement and the mean probability claimed in it;
- **the expected calibration error** —
  `Σ over non-empty bins (n_bin / n_scored) · |agreement_bin − mean_argmax_probability_bin|`. It is a gap between what
  the head claimed and what the table found. It is not an accuracy, and it says nothing about whether the
  ranked-first pipeline was any good.

```
python -m evaluation.decision_calibration --outcomes <dir> --out report.json --markdown report.md
python -m evaluation.decision_calibration --inventory <decisions dir> [<dir> ...] --out report.json
```

**The threshold is 30.** Any `(kind, question)` with fewer than 30 linked outcomes is `NO_NEW_MEASUREMENT`, and the
report prints how many are missing (`29 linked outcome(s), 30 required — 1 missing`) instead of a rate. An agreement
rate over nine links looks exactly like a measurement and is not one. Three further conditions produce
`NO_NEW_MEASUREMENT` even at or above the threshold, because without them the rate would not mean what it reads as:
`OPTION_SETS_DIFFER` (the outcomes were chosen from different option sets, so they do not answer one question),
`NO_BEST_RANKED_OPTION` (no linked outcome came from a stage ranked first, so no option is the label), and
`AMBIGUOUS_BEST_RANKED_OPTION` (two stages ranked first under different keys). An outcome whose probabilities have no
single maximum is excluded from the scoring and counted, rather than having its tie broken by option order — a tie
means the head separated nothing, and breaking it would invent the preference the report exists to measure.

`--inventory` states the position from the other side: how many decision records exist today per kind and question,
and how many of them have an outcome. The rendering sorts everything and reads no clock, so two runs over the same
records produce the same bytes and the report can be diffed.

### Where this stands today (2026-09-25)

Run over the three directories the framework has written records into —
`~/.local/state/m5phet/decisions` (WP17, WP18, WP15),
`~/.local/state/m5phet/regimes-wp19-20260925/records` (WP19), and
`~/.local/share/causal-inference-m5phet/studies/decisions` (WP20):

| kind | question | records | linked | unlinked |
|---|---|---|---|---|
| `causal_study` | `baseline` | 1 | 0 | 1 |
| `causal_study` | `confidence_level` | 1 | 0 | 1 |
| `causal_study` | `confounder` | 1 | 0 | 1 |
| `causal_study` | `estimator` | 1 | 0 | 1 |
| `causal_study` | `model_t` | 1 | 0 | 1 |
| `causal_study` | `model_y` | 1 | 0 | 1 |
| `causal_study` | `outcome` | 1 | 0 | 1 |
| `causal_study` | `treatment` | 1 | 0 | 1 |
| `dataset_choice` | `dataset` | 1 | 0 | 1 |
| `feature_grouping` | `grouping_cut` | 1 | 0 | 1 |
| `feature_preprocessing` | `preprocessing` | 7 | 0 | 7 |
| `group_extractor` | `extractor` | 2 | 0 | 2 |
| `regime_method` | `regime_method` | 1 | 0 | 1 |
| `regime_parameters` | `regime_parameters_linkage` | 1 | 0 | 1 |
| `regime_parameters` | `regime_parameters_n_clusters` | 1 | 0 | 1 |

**22 decision records exist and zero outcomes exist.** Not one of them is a label: no closure-table row links any of
them, because no Laya-chosen pipeline has been fitted and ranked yet. Every kind above is therefore at least 30 short
of its first measurement, and the calibration report says exactly that rather than reporting a rate over what is
there. The work plan's own WP23 records why the records look as they do — a zero-shot checkpoint that barely separates
the options (0.35/0.32/0.32 over three transforms, 0.30–0.41 over five causal roles, the same preprocessor for all
seven features) — and those are facts about the checkpoint, not measurements of it.

This is also why WP23's **step 3** — a fine-tuning corpus for the Laya checkpoint built from linked records — has not
begun and could not: there is nothing to build it from, and it needs the owner's go besides. Until then the framework
keeps the zero-shot checkpoint, and every decision record says so in its `checkpoint` field.
