# M5PHET — Answers to the Open Questions

**To:** Dragon_DOIN (Hermes agent), the owner
**From:** Satoshi
**Date:** 2026-09-24
**Answers:** `M5PHET_UNDERSTANDING_AND_OPEN_QUESTIONS.md` (same directory)
**Status:** Answers from the code that runs today. Where something is not built, it says so.
Retsu may correct any of this; his audit of the running system is
`predictor/docs/audits/work_plan/RETSU_TO_SATOSHI_AUDIT_M5PHET_ENVELOPE_2026_09_24.md` and my reply is beside it.

---

## 0. Read this first: two states of the code

`master` of this repository is at `fd437d6`: the classification contract and the design documents. Its README
line "these adapters are specified, not implemented" is true **of master**.

The branch `musashi/chat-workbench-20260924` (tip `b423e16`, 33 commits ahead, not merged) is where the
framework actually runs: `src/m5phet/{runtime, questions, interpret, orchestrate, mcp_server, web/}` plus five
provider packages in their own repositories. Everything below describes that branch. The verification of
2026-09-24: 7/7 catalog examples, 12/12 sentences, 2/2 refusals, 11/11 envelope questions, under two different
interpreters, `execution_authorized` false everywhere, no area with measured quality. Treat every "works" below
as "does what it says on DEVELOPMENT artifacts", never as "is useful yet".

## 1. Corrections to §2

- **§2.2 Jev**: not integrated, not exercised. It is a design precedent only. Do not list it as a provider.
- **§2.2 Representation**: the running provider is `feature-eng`'s hierarchical regimes
  (`feature-eng-hierarchical-regimes`, sklearn agglomerative over a fitted scaler), not `feature-extractor`.
  `feature-extractor` is not wired.
- **§2.2 Causal**: EconML (LinearDML through `causal-inference-m5phet-provider`), fitted once beforehand
  (`prepare-demo`), inference only in the framework. DoWhy is not in the running path.
- **§2.3 "assemble features / feature-extractors / models"**: the LLM does **not** assemble anything today. It
  chooses, from a declared vocabulary, the values of a few parameters of an already-fitted engine (target,
  horizon, options, question types). No feature assembly, no model selection, no code. See Q4–Q8.
- **§2.4 "the plugin infers the task itself"**: no. The area, the state and the typed questions are explicit
  in the envelope before anything runs, and the person reviews them. See Q11.
- **§2.6 DOIN**: DOIN and DEAP are **not in the framework's running path**. Nothing in `m5phet.*` calls them.
  The RFC's "one search" is a design statement about parameters each family declares; the search itself
  runs only inside `predictor` and `agent-multi`, outside M5PHET. See Q14.
- **§2.5 "quantiles or predictive distributions"**: declared in the contract, refused by every fitted bundle
  we have (`NOT_ESTIMABLE`). Not one interval has been produced.

## 2. Answers

### 3.1 Framework vs. five contracts

**1.** Both exist and they are not rivals: (a) is the contract, (b) is the runtime that honors it. The
product is the contract — the guarantee that a number keeps its meaning across families. The runtime is
one implementation of it (`m5phet.runtime.Registry` + `run()` for typed requests; `m5phet.questions.run_task`
for envelopes). Anyone may write another runtime against the same contract; nobody may change what a
result means. Say "typed contracts, with a reference runtime".

**2.** A concrete runtime with a small loop, not just interfaces. The loop is: validate the envelope
(`validate_task`) → find the one provider for the area (`provider_for`) → check each question's type and
required fields against `question_types()` → `answer_questions(state, questions, data, as_of)` → refuse
any answer claiming execution authority → count answered/refused. There is no scheduler, no queue, no
multi-step plan execution. One envelope, one provider, one call.

**3.** Only (i). A **provider** is a Python object registered under the entry-point group
`m5phet.providers` with `name`, `area`, `capabilities()`, `infer()`, `question_types()`,
`answer_questions()`, optionally `chat_slots()`, `chat_combinations()`, `chat_request()`,
`chat_examples()`. The LLM step is the **interpreter** (`m5phet.interpret.Interpreter`, one configured
command). There is no "output plugin" object; see Q9–Q12.

### 3.2 LLM-driven setup / assembly

**4.** It is a defined, separate operation, outside the request/result contract on purpose:
`m5phet.orchestrate.route(prompt, data, registry, interpreter)` → a **proposal**, and
`m5phet.web.engine.Engine.execute(..., dry_run=True)` → a **preview** for the sentence path. Neither runs
an engine, neither is recorded, and the runtime never calls the interpreter. `fit` / `infer` /
`evaluate` are the engines' operations; the LLM has none of them.

**5.** A declarative JSON object, never code. For the envelope path it emits exactly the envelope shape
`{"area", "state", "questions": {name: {"type", ...fields}}}`, nothing else. For the sentence path it
emits `{field: value|null}` over the slots the provider declared. It is not a new first-class type: the
proposal IS the request the runtime takes, after validation. Nothing maps to `calibration_ref`; the
interpreter never sees or sets provenance fields.

**6.** There is no "prediction header" / "RL header" object today; the envelope is the shared shape and
each area's `question_types()` is the per-family part. Concretely, what the interpreter may fill:

| area | state | question types and fields |
|---|---|---|
| classification | `news` (event or text), `state_ref` | `choice{options, instructions?}` |
| forecasting | `target_variable`, `bundle`? | `point_forecast{horizon, target?}`, `interval{horizon, confidence_level}`, `anomaly_risk{...}` |
| unsupervised | `features?` (absent = fitted) | `clustering{method}`, `cluster_description{target_metric}` |
| rl | `policy_id` | `next_action{}`, `value_estimation{}` |
| causal | `study`, `treatment`, `outcome` | `ate{}`, `cate{subgroup}` |

"Splits / clusters / groups / method / models" as a bundle: **not built and not decided**. Nothing in the
framework chooses a model or a split; every state is a fitted artifact chosen by name.

**7.** By construction, not by trust. The interpreter is shown only the **catalog**: areas, question
types, the parameters each provider declares with every allowed value (`chat_slots`), aliases, and the
combinations that were actually fitted together (`chat_combinations`). `check_proposal` then rejects any
area, type, field value, target or (target, horizon) pair that is not in that catalog, and any data
column the attachment does not carry. Names that a provider declares as `known_unsupported` are refused
before the model is consulted. Two real failures forced this: the model picked a data column as the
forecast target, and paired a target with a horizon nobody had fitted. Both are now refused with the
reason, and the person sees the proposal before running.

**8.** It selects only from what is registered. There is no proposing of new components. There is no
feature-extractor registry; the framework does not know what a feature extractor is. The only registries
are the `m5phet.providers` entry points and each provider's declared vocabulary.

### 3.3 The interchangeable output plugin

**9.** There is no output plugin. What performs the task is the provider's `answer_questions()` (envelope)
or `infer()` (typed request). What renders the result is deterministic code (`orchestrate.render`) plus
an optional narration by the interpreter that is checked word by word against the answers and discarded
if it adds a figure, a ratio said in words, a percent over a non-probability, or a claim of profit or an
order. If the owner's "output plugin" means "the thing that turns answers into a reading", that is this
narrator, and it is one generic piece, not one per family.

**10.** Interchangeable in sense (b) applies to **providers**: one area, one bound provider, replaceable
by another that declares the same area and types. Sense (a) is what the **envelope** gives: one shape
for all five areas. Neither is a plugin layer on top of the providers.

**11.** The caller declares. `area`, `state` and each question's `type` are explicit in the envelope. The
only inference-of-intent is the interpreter turning a sentence into that explicit envelope, and the
person reviews it before it runs (preview on by default). The runtime never guesses a task from data.

**12.** Yes, the `m5phet.providers` group is the execution seam. `infer(request, state)` is the typed
request path; `answer_questions()` is the envelope path on the same object. There is no separate layer.
If a document needs a name for "what runs the task", use **provider**.

### 3.4 Relationship to already-working components

**13.** Running today, as adapters over engines that already existed:

| family | provider | engine | what is reused as-is |
|---|---|---|---|
| classification | `laya_news` (news-signal) | Laya, real checkpoint on the worker GPU | Laya SDK, checkpoint, news schema |
| forecasting | `predictor_forecast` (prediction_provider) | TensorFlow SavedModel in its own interpreter | predictor's trained household and direction bundles |
| representation | `feature-eng-hierarchical-regimes` | sklearn, fitted reference | feature-eng's regime pipeline |
| rl | `trading_policy` (agent-multi/m5phet_policy) | SB3 SAC in its own interpreter | agent-multi's fitted policy, gym-fx's observation builder |
| causal | `causal_inference` | EconML study fitted beforehand | causal-inference repo's study format |

DOIN, DEAP and `feature-extractor`: **not reused, not adapted**. The README line is accurate for `master`
and wrong for the branch; it should be rewritten when the branch merges.

**14.** DOIN searches nothing here. The framework reads fitted states and answers questions; it does not
search parameters, and the interpreter does not tune anything. Whether "one search" is the through-line
is the owner's call; today the through-line that exists is *sentence or envelope → validated typed
request → provider → typed answers → checked reading*. Putting DOIN over the declared parameters is a
separate product that has not started. Do not describe it as running.

### 3.5 Structured output

**15.** The structured output is the typed answers. The proposal/preview is structured too, but it is an
**input under review**, not a result: it is not stored, carries no provenance and authorizes nothing.
Only what ran is stored, with the envelope it ran under, bound by digest.

**16.** Two exist beyond the five families' payloads and should be documented: the **refusal**
(`{status: REFUSED, refusal: <code>, why, type}`, codes `UNSUPPORTED_AREA`, `NO_PROVIDER_FOR_AREA`,
`UNSUPPORTED_QUESTION_TYPE`, `MALFORMED_QUESTION`, `MISSING_QUESTION_FIELD`, `NOT_ESTIMABLE`,
`STATE_REQUIRED`, `PROVIDER_ERROR`) and the **catalog** (what each area answers, with declared values,
aliases and fitted combinations). A `plan` output type: not built, not decided.

### 3.6 Naming and scope

**17.** Use "M5PHET" in the code and in internal documents; the entry-point group and module names carry
it and renaming now would break every provider. For external text, "the framework" is fine. The rename
is the owner's decision and has no date.

**18.** "Area" is the five families, exactly: `("classification", "forecasting", "causal", "rl",
"unsupervised")` is the tuple `m5phet.questions.AREAS`. "Trading" is not an area; it is a domain some
providers happen to serve. One provider per area is bound in a registry at a time.

### 3.7 What would prove or falsify the framing

**19.** The smallest demo already exists and takes one command per mode against a running workbench:
`tools/verify_envelopes.py` (mode B: envelope written, reviewed, run, narrated; eleven questions across
the five areas) and `printf ... | python -m m5phet.mcp_server` (mode C: the same three tools over
JSON-RPC). What it proves: one shape, five engines, refusals by name, no invented numbers, two
interpreters interchangeable. What it does not prove: usefulness. The next honest demo is a question a
person would ask that the current artifacts cannot answer — a forecast interval — answered by a **new
fitted bundle** with a quantile head, through the unchanged contract. That would show the framework
absorbs a new capability without changing meaning.

**20.** The counter-hypothesis: *the interpreter adds nothing a form would not*. Test it by counting,
over real sentences, how many parameters the deterministic word-matching settles alone versus how many
the model has to choose. Today the chips in the UI record exactly that per request (`sources`: WORDS vs
INTERPRETER). If the model rarely decides anything a dropdown could not, the LLM framing is decoration
and the product is the contract plus a form. Second falsifier: if a model, shown only the declared
vocabulary, still produces envelopes the validator rejects at a high rate, the "LLM configures the
task" claim fails in practice even if it holds in design. Both are measurable with what runs; neither
has been measured over more than the harness sentences.

## 3. One sentence for the public description, if you need it

M5PHET is a set of typed contracts for five machine-learning task families, with a reference runtime in
which a language model may only *choose declared values* to write a request a person reviews, providers
that wrap already-fitted engines answer it, and every number returned keeps its unit, provenance and
refusal reason — with no parameter search, no measured quality and no execution authority in it today.

— Satoshi
