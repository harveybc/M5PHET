# The router's own reliability — first measurement, 2026-09-25

`m5phet.orchestrate.route` is the path where the language model does **not** choose among values a provider
declared: it writes a whole envelope — the area, the state, the named typed questions — and `check_proposal` then
refuses an area nobody serves, a question type no provider declares, a governed value the engine does not have and a
column the data lacks. That refusal bounds what a wrong envelope can do. It says nothing about how often the model
writes one, and until this run nobody had measured it. `tools/measure_interpreter.py`'s report said so in as many
words: *"it says nothing about the router … that path is unmeasured."*

`tools/measure_route.py` measures it. `route_reliability_n5.json` is the run this file reports
(sha256 `aac62e58938eca2d33314f95a6f8a546827379588e20ea060450456eda4aa8fa`).

## Protocol

Corpus: 19 sentences — the 14 of `tools/verify_families.py`'s `PROSE` list plus the 5 prompts of
`tools/verify_envelopes.py`'s envelopes. Each is written against one shipped example, so its correct envelope is
known: the **area** from the harness the sentence belongs to, the **question types** from the envelope that harness
runs, the **governed values** from the engines' own declared vocabularies. No expectation comes from a model's
answer, which would make the measurement circular. A governed field the sentence does not name is not scored — the
engine settles it, and a model that leaves it out has not misread anything.

**N = 5 routings per sentence**, 95 runs, each through the running workbench's proposal endpoint
(`POST /api/chats/{id}/tasks/propose`), which builds the envelope and runs nothing. N is part of the protocol and is
published in the report; a rate over three routings and a rate over five are two different measurements.

Interpreter measured: `command` plugin, `deepseek-v4-flash (OpenCode Go)`, `reports_confidence: false` — the same
interpreter `docs/evidence/INTERPRETER_RELIABILITY_2026_09_25/` measured, on a verification instance of this branch
(port 8784, its own state directory), 2026-09-25.

A miss is not one thing, so six verdicts are kept apart:

| verdict | what it is |
|---|---|
| `CORRECT` | it validates AND the area, the set of question types and every governed value are the expected ones |
| `WRONG_AREA` | it proposed a different engine. Checked first: which engine was asked is the primary fact |
| `INVALID_PROPOSAL` | `check_proposal` refused it; the reason is counted by kind |
| `WRONG_TYPE` | it validates, and asks question types other than the ones the sentence asks for |
| `WRONG_VALUE` | it validates and asks the right types, and a governed value is wrong or absent |
| `REFUSED` | nothing was proposed: no interpreter, no JSON, malformed JSON, or the model declaring no served area fits |

## Result

| | sentences | runs | correct | rate |
|---|---|---|---|---|
| the router, writing whole envelopes | 19 | 95 | 81 | **0.8526** |
| of the runs that produced a proposal at all | 19 | 94 | 81 | 0.8617 |

| verdict | runs | |
|---|---|---|
| `CORRECT` | 81 | |
| `WRONG_TYPE` | 12 | it asked for different questions than the sentence asks for |
| `INVALID_PROPOSAL` | 1 | `MALFORMED_ENVELOPE`: an envelope with no `state` mapping, refused by `check_proposal` |
| `REFUSED` | 1 | the `hermes` subprocess failed on one call — an infrastructure failure, not a reading |
| **`WRONG_AREA`** | **0** | it never named a different engine |
| **`WRONG_VALUE`** | **0** | it never named a target, a horizon, a study, a policy or a metric the engine does not have |

14 of 19 sentences were correct in all five routings; 15 of 19 were stable (identical resolution every time); none was
never correct.

### The finding that matters more than the rate

**Every miss on this corpus is about the SHAPE of the envelope, never about which engine answers or which value it
answers about.** `WRONG_AREA` and `WRONG_VALUE` are both zero over 95 routings; the 12 `WRONG_TYPE` runs are all the
same mistake in two forms:

* four sentences that ask for **two** things get an envelope with **one** — "pronostica la potencia y dame un rango"
  proposed only the `interval` (4 of 5 runs), and "cual fue el efecto del tratamiento y en jovenes" only the `cate`
  (3 of 5);
* "assign hierarchical regimes to these rows" / "asigna los regímenes jerárquicos a estas filas" proposed a
  `cluster_description` where the sentence asks only for the assignment (`clustering`) — 5 of 10 runs across the two
  languages.

That is a real limitation and it is an under-answer, not an invention: the person reviewing the proposal sees an
envelope with one question where they asked for two, and the missing question is visible before anything runs. It is
also exactly the case the review window exists for.

### Per sentence, five routings each

| area | sentence | correct | verdicts | stable |
|---|---|---|---|---|
| classification | Which economy is named in this news? | 5/5 | CORRECT:5 | yes |
| classification | ¿De qué economía habla esta noticia? | 5/5 | CORRECT:5 | yes |
| classification | de que economia habla y con que tono | 5/5 | CORRECT:5 | yes |
| forecasting | predict household power one hour ahead | 5/5 | CORRECT:5 | yes |
| forecasting | ¿cuánta potencia habrá en la próxima hora? | 5/5 | CORRECT:5 | yes |
| forecasting | what is the direction_long probability at horizon 1? | 5/5 | CORRECT:5 | yes |
| forecasting | ¿cuál es la probabilidad de direction_long a horizonte 1? | 4/5 | CORRECT:4 INVALID_PROPOSAL:1 | yes |
| forecasting | pronostica la potencia y dame un rango | 1/5 | CORRECT:1 WRONG_TYPE:4 | no |
| unsupervised | describe el grupo de velas con cuerpo alto | 5/5 | CORRECT:5 | yes |
| unsupervised | describe the cluster with a large body | 5/5 | CORRECT:5 | yes |
| unsupervised | assign hierarchical regimes to these rows | 2/5 | CORRECT:2 WRONG_TYPE:3 | no |
| unsupervised | asigna los regímenes jerárquicos a estas filas | 3/5 | CORRECT:3 WRONG_TYPE:2 | no |
| unsupervised | segmenta estas filas y describe el cluster alto | 5/5 | CORRECT:5 | yes |
| causal | Report ATE of treatment on outcome, with its uncertainty. | 5/5 | CORRECT:5 | yes |
| causal | ¿Cuál es el ATE of treatment on outcome y su incertidumbre? | 5/5 | CORRECT:5 | yes |
| causal | cual fue el efecto del tratamiento y en jovenes | 1/5 | CORRECT:1 REFUSED:1 WRONG_TYPE:3 | no |
| rl | What action does eth_4h_sac_current_stack_anchor_v1 propose? | 5/5 | CORRECT:5 | yes |
| rl | ¿Qué acción propone la política para estas barras? | 5/5 | CORRECT:5 | yes |
| rl | que accion propone y que retorno espera | 5/5 | CORRECT:5 | yes |

## Beside the interpreter's number — and why they are not one number

`docs/evidence/INTERPRETER_RELIABILITY_2026_09_25/` measured the same model on the other path: **0.8333** strict
(50/60) and **0.9434** of the runs where it chose a declared value.

| | corpus | runs | strict rate | rate when it acted |
|---|---|---|---|---|
| interpreter — choosing one declared value | 12 sentences | 60 | 0.8333 | 0.9434 (excludes 7 `DECLINED`) |
| router — writing a whole envelope | 19 sentences | 95 | 0.8526 | 0.8617 (excludes 1 `REFUSED`) |

The strict rates are close, with the router a little higher. **That is not evidence that the router is the better
path, and it must not be quoted as such.** Three reasons, each sufficient on its own:

* they are different jobs. One picks a value from a list the provider declared; the other writes an area, a state and
  a set of typed questions from nothing;
* they are different corpora — 12 sentences against 19 — and different denominators;
* the two conditional rates are conditioned on different things. The interpreter's excludes `DECLINED` (7 runs where
  it returned null); the router's excludes `REFUSED` (1 run where a subprocess failed). Putting 0.9434 and 0.8617 in
  one sentence as though one were smaller than the other compares two different populations.

What can be said without qualification is that the two failure **kinds** differ. The interpreter's misses are
`WRONG_VALUE` — the wrong declared metric among nine. The router's misses are all envelope shape, with zero wrong
areas and zero wrong values. On this corpus the router does not send a question to the wrong engine and does not name
a value an engine does not have.

## The gate, and what it does not cover

`route` reports **no confidence at all**, with every shipped interpreter plugin — including `openai_compatible`,
whose endpoint can return logprobs, because `route` calls `_ask` and `_ask` discards them. There is no per-value
choice to attach a probability to when the model is writing a whole envelope. So a declared abstention threshold is
**not applied on this path**, and `/api/catalog` says so by name:

```json
{"abstention": {"paths": {
   "decide":    {"covered": true,  "confidence": "CONFIDENCE_REPORTED"},
   "interpret": {"covered": false, "confidence": "CONFIDENCE_NOT_REPORTED"},
   "route":     {"covered": false, "confidence": "CONFIDENCE_NOT_REPORTED", "reliability": {...}}}}}
```

What *does* guard this path is `check_proposal`, and one run of this measurement shows it working: an envelope with
no `state` mapping was refused (`MALFORMED_ENVELOPE`) and did not run.

## What this does NOT say

* it is not a closure table. There is no naive baseline for "writing an envelope" and no literature value matched to
  this protocol, so the owner's closure table is `NOT_COMPARABLE` here and is not produced;
* n is small: 19 sentences, 95 runs, one model, one day, one instance. A first measurement, not a characterisation;
* it says nothing about sentences nobody wrote. The corpus is the framework's own examples;
* it says nothing about a router asked to route over a data profile it has never seen. Every run here had the
  shipped example's data attached;
* the one `REFUSED` was an `hermes` subprocess failure, not a refusal to route. It is counted in the strict rate,
  because a person typing that sentence got no envelope, and excluded from the conditional rate, which is what that
  second rate is for.

## Where the number lives

`interpreter.route_reliability_report` in `m5phet.json` (or `M5PHET_ROUTE_RELIABILITY_REPORT`) names this file, and
`/api/catalog`'s `route.reliability` then publishes the rate, both rates apart, the verdicts, the protocol, the
counts, N per sentence and the report's digest. With nothing declared it says `NOT_MEASURED`. A report measured on a
different plugin or model is refused (`ROUTE_MEASURED_ON_ANOTHER_INTERPRETER`) rather than published beside this
interpreter's identity.

## One error in this artifact, recorded rather than edited

The stored report's `corpus` field reads *"18 sentences"*. The corpus is **19**, as `summary.sentences`, the
per-sentence table and `len(CASES)` all say; the string was a hand-written count and was wrong. The artifact is
published exactly as the run wrote it — editing evidence after the fact is worse than carrying a wrong string — and
`tools/measure_route.py` now computes that count from `CASES`, so it cannot disagree with the corpus again.

A long measurement should also not be lost to one kill: `tools/measure_route.py --checkpoint DIR` now writes each
sentence's five routings as they complete and resumes from them, reusing a stored sentence only when the case, the
instance, N and the protocol are identical. This run predates that flag and was made in one sitting.
