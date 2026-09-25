# The interpreter's own reliability — first measurement, 2026-09-25

Every quality number M5PHET had written was about an engine. WP09 scored the classification checkpoint on 450
independently labelled rows; the closure tables score forecasts against naive baselines on held-out rows. The
**interpreter** — the language model that reads a person's sentence and picks, among the values a provider declared,
which target, which horizon, which study, which policy an engine is to answer about — had never been scored. It had
been argued about ("it can only choose among declared values, so the worst case is a refusal"), which is a statement
about the blast radius of its mistakes and not about how often it makes one.

`tools/measure_interpreter.py` measures it. `interpreter_reliability_n5.json` is the run this file reports.

## Protocol

Corpus: the 14 sentences of `tools/verify_families.py`'s `PROSE` list — the framework's own ordinary-words corpus,
two languages, five areas. The correct resolution of each is declared in `tools/measure_interpreter.py`'s `EXPECTED`,
taken from the engines' declared vocabularies and the shipped example each sentence is written against, never from a
model's answer. N = 5 resolutions per sentence. Nothing is fitted and no engine runs.

Two paths, measured apart because they answer different questions.

**A. The product path** — each sentence through the running workbench's review window
(`POST /api/chats/{id}/preview`). This is what a person typing gets.

**B. The interpreter path** — the same sentence and the same declared vocabulary asked of the configured interpreter
alone (`Interpreter.propose`), with no deterministic pass in front of it, and validated exactly as `interpret()`
validates: a value outside the declared list is not kept.

## What was measured

Interpreter: `command` plugin, `deepseek-v4-flash (OpenCode Go)`, `reports_confidence: false`.
Measured 2026-09-25 against a verification instance of this branch.

| | sentences | runs | correct | rate |
|---|---|---|---|---|
| A. product path (deterministic pass first) | 12 | 60 | 60 | 1.0000 |
| B. interpreter alone | 12 | 60 | 50 | **0.8333** |

Two of the fourteen sentences (`laya_news`) resolve no parameters at all — that provider declares no slots, the
sentence *is* the question put to the checkpoint — so they are `NO_SLOTS` in both paths and enter neither count.

### A. The finding that matters more than the rate

**On this corpus the interpreter is never consulted for any scored field.** All 12 parameter-bearing sentences
resolve entirely through the deterministic pass — the providers' declared values and aliases — in all 60 runs
(`sources: QUESTION_TEXT`). The harness reports them as `DETERMINISTIC` and leaves them out of the interpreter's own
rate, because a sentence the words settle is not evidence about a model.

So `verify_families.py`'s standing `prose 14/14` is a true statement about the product and **not** a statement about
the interpreter, which it has never exercised. Anyone reading that line as evidence that the language model reads
ordinary words correctly has been reading it wrong, this branch included until it was measured.

(The interpreter *is* consulted on three sentences for the optional `bundle` field, which no sentence names and no
expectation scores. That is recorded in the JSON and counted nowhere.)

### B. The interpreter alone

50 of 60, **0.8333**. A miss is not one thing, so the breakdown is reported and the aggregate never quoted alone:

| verdict | runs | what it is |
|---|---|---|
| `CORRECT` | 50 | the expected declared value |
| `WRONG_VALUE` | 3 | it chose a declared value and chose the wrong one — the only outcome that is a misreading |
| `DECLINED` | 7 | it returned `null`: "the request does not name one" |
| `OUTSIDE_DECLARED` | 0 | a value nobody declared. `interpret()` refuses these; it proposed none |
| `INTERPRETER_FAILED` | 0 | no call failed |

Of the 53 runs where it did pick a declared value, 50 were right: **0.9434** (`reliability_when_it_chose`).

Per sentence, 5 runs each:

| area | sentence | correct | verdicts |
|---|---|---|---|
| forecasting | predict household power one hour ahead | 5/5 | CORRECT:5 |
| forecasting | ¿cuánta potencia habrá en la próxima hora? | 5/5 | CORRECT:5 |
| forecasting | what is the direction_long probability at horizon 1? | 5/5 | CORRECT:5 |
| forecasting | ¿cuál es la probabilidad de direction_long a horizonte 1? | 5/5 | CORRECT:5 |
| unsupervised | describe el grupo de velas con cuerpo alto | 3/5 | CORRECT:3 WRONG_VALUE:2 |
| unsupervised | describe the cluster with a large body | 3/5 | CORRECT:3 DECLINED:1 WRONG_VALUE:1 |
| unsupervised | assign hierarchical regimes to these rows | 3/5 | CORRECT:3 DECLINED:2 |
| unsupervised | asigna los regímenes jerárquicos a estas filas | 5/5 | CORRECT:5 |
| causal | Report ATE of treatment on outcome, with its uncertainty. | 5/5 | CORRECT:5 |
| causal | ¿Cuál es el ATE of treatment on outcome y su incertidumbre? | 5/5 | CORRECT:5 |
| rl | What action does eth_4h_sac_current_stack_anchor_v1 propose? | 5/5 | CORRECT:5 |
| rl | ¿Qué acción propone la política para estas barras? | 1/5 | CORRECT:1 DECLINED:4 |

8 of 12 sentences were stable across the 5 runs (identical resolution every time); 4 were not. None was never
correct.

The three `WRONG_VALUE` runs are all one area and one kind of question: "the cluster with a large body" against nine
declared `target_metric` values, where the model sometimes picked a neighbouring metric. The seven `DECLINED` are
two sentences that genuinely do not name the value (the Spanish RL sentence names no policy; "assign hierarchical
regimes" names no metric) — the request still has to resolve, because the engine has one policy and a default
metric, but a model returning `null` there is declining rather than misreading.

## What this does NOT say

* it is not a closure table. There is no naive baseline for "reading a sentence" and no literature value matched to
  this protocol, so the owner's closure table is `NOT_COMPARABLE` here and is not produced;
* n is small: 12 sentences, 60 runs, one model, one day. It is a first measurement, not a characterisation;
* it says nothing about sentences nobody wrote. The corpus is the framework's own examples, which are the sentences
  the framework was built to answer;
* it says nothing about the router (`orchestrate.route`), where the same model writes a whole envelope rather than
  choosing a declared value. That path was unmeasured when this was written; it was measured the same day —
  `docs/evidence/ROUTE_RELIABILITY_2026_09_25/` — and the two rates are about two different jobs and are not
  comparable as one number.

## Where the number lives

`interpreter.reliability_report` in `m5phet.json` (or `M5PHET_INTERPRETER_RELIABILITY_REPORT`) names this file, and
`/api/catalog`'s `interpreter.reliability` then publishes the rate, both rates apart, the protocol, the counts and
the report's digest. With nothing declared it says `NOT_MEASURED`. A report measured on a different plugin or model
is refused (`RELIABILITY_MEASURED_ON_ANOTHER_INTERPRETER`) rather than published beside this interpreter's identity.
