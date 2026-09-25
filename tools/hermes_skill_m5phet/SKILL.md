---
name: m5phet
description: "Answer questions about fitted ML models (classification, forecasting, causal, RL, regimes) through the m5phet MCP tools: catalog, envelope, review, run."
version: 1.0.0
license: MIT
metadata:
  hermes:
    tags: [m5phet, MCP, ml, forecasting, causal, classification, rl, clustering, telegram]
    related_skills: [native-mcp]
---

# M5PHET — questions to fitted models, through three MCP tools

The `m5phet` MCP server answers **named typed questions about models that were fitted beforehand**. It has no
filesystem, no shell, no network of its own, and it never trains anything. Its three tools:

| Tool | What it does |
|---|---|
| `m5phet_catalog` | the areas this installation serves, each area's provider, and every question type with its required and optional fields |
| `m5phet_propose_task` | turns a sentence (and the *shape* of attached data) into a proposed envelope; returns it unrun |
| `m5phet_execute_ml_task` | runs one envelope: `{area, state, questions}` plus optional `data` and `as_of` |

## The procedure — never skip a step

1. **Call `m5phet_catalog` first, every time.** What is installed changes between hosts and between days. Never
   build an envelope from memory or from the examples in this file.
2. **Write the envelope** `{"area": …, "state": {…}, "questions": {"<your name>": {"type": …, …}}}` using **only**
   areas, question types, field names and parameter values the catalog just declared. You may choose among declared
   values; you may not invent one. If the person asks for something no declared type covers, say exactly that and
   stop — do not substitute a neighbouring question.
3. **Show the envelope to the person as JSON and wait.** Do not call `m5phet_execute_ml_task` until they reply
   `ok` (or an unmistakable equivalent: "ok", "dale", "run it", "corre", "sí, ejecuta"). Anything else — a new
   question, a correction, silence — is not consent: revise and show it again. In a one-shot invocation with no
   chance to answer, stop at the envelope.
4. **Run it once** with `m5phet_execute_ml_task`. Same question twice is the same envelope; do not resend it because
   the answer displeased anyone.
5. **Reply as plain text** under the rules below.

## The reply rules

- Plain text, **at most 4000 characters**. No markdown tables, no code fences: this is read in a chat window.
- **One line per answered question**, named as the person named it, with the value, its unit and its scale exactly as
  the tool returned them. Copy digits; never round, re-scale, average, convert or "tidy" them.
- **Never add a number.** If a figure is not in the tool's output, it does not go in the reply — no percentages, no
  differences, no "about", no comparison with yesterday. This is the one rule with no exception.
- **A refusal is an answer.** Print the question name, the word `REFUSED`, the refusal code
  (`NOT_ESTIMABLE`, `STATE_REQUIRED`, `MISSING_COLUMNS`, `TOO_FEW_ROWS`, `PROVIDER_ERROR`, …) and the engine's own
  reason, shortened but not reworded. Never fill a refused question with a number from another question, another
  horizon or your own head.
- Carry the qualifiers the answer carries: `UNCALIBRATED` on classification probabilities, `development: true` on a
  development state, `assumptions` on a causal effect, "the policy's own action scale" on an RL action.
- Say which input the answer belongs to when the person sent data: two inputs to the same model are two answers, not
  one number.
- **Never say what to do.** An RL action and a critic value are a score, not an order; a forecast is not advice.
- **Always end with exactly this line:**

```
execution_authorized: false — this is not an instruction to act
```

## What each area needs in `state` and `data`

Read this from the catalog, but expect this shape:

- **classification** — `state`: `{"asset": …, "language": …}`; `data`: the news text or the news JSON event.
  Each `choice` question carries its own `instructions` and its `options` as `[[value, label], …]`. The
  probabilities belong to that exact wording: a reworded question is a different input, and the answer says so.
- **forecasting** — `state`: `{"target_variable": <declared target>}`; `questions` name a **declared** `horizon`.
  `data` is required and must be the windowed input the bundle was fitted on (`columns`, `values`, `scale`,
  `scaler_digest`). Without it the question comes back `REFUSED PROVIDER_ERROR` — report that refusal, do not guess a
  number. `interval` and `anomaly_risk` are refused `NOT_ESTIMABLE` while the bundle has no quantile head.
- **causal** — `state`: `{"causal_graph": {"treatment": …, "outcome": …, "confounders": [...]}}`; attach **no data**:
  the engine only reads a study fitted beforehand, and rows would mean "fit", which it refuses.
- **rl** — `state`: `{"policy_id": <declared policy>}`; `data`: the observation vector of the fitted size, or the
  bars the observation builder accepts. Too few rows or missing columns are refusals, not approximations.
- **unsupervised** — `state`: `{}` (or a declared `state_ref`); `data`: `{"rows": [...]}`. `cluster_description`
  needs a `target_metric` expression over the supplied columns.

## Five worked examples (shapes only)

> The numbers below were recorded on 2026-09-24 from these exact inputs. **They are illustrations of shape.**
> Never repeat a number from this file as if it were an answer: every reply's numbers come from that reply's own
> tool output.

**1. Classification** — "¿De qué economía habla esta noticia y con qué tono?"

```json
{"area": "classification",
 "state": {"asset": "EURUSD", "language": "en"},
 "data": "The European Central Bank left its deposit facility rate unchanged. Incoming euro-area inflation data will guide its next decision.",
 "questions": {"economia": {"type": "choice", "instructions": "Which economy is named in this news?",
                            "options": [["euro_area", "Euro area"], ["united_states", "United States"], ["other", "Another economy"]]},
               "tono": {"type": "choice", "instructions": "What tone does the news take?",
                        "options": [["hawkish", "tighter policy"], ["dovish", "easier policy"], ["neutral", "neither"]]}}}
```

Answers carry `label`, `uncalibrated_probabilities` and `calibration: UNCALIBRATED` (the workbench recorded
`euro_area 0.9666` for this instruction, `0.9605` for the Spanish wording — a different input, and the answer says
so). Print the label, the probabilities as returned, and the word `UNCALIBRATED`.

**2. Forecasting** — "Pronostica la potencia a 60 pasos y dame un rango."

```json
{"area": "forecasting",
 "state": {"target_variable": "Global_active_power"},
 "data": {"columns": ["…"], "values": [[…]], "scale": "…", "scaler_digest": "…"},
 "questions": {"prediccion": {"type": "point_forecast", "horizon": 60},
               "rango": {"type": "interval", "horizon": 60, "confidence_level": 0.95}}}
```

`prediccion` returns a point with its unit (`0.5412255525588989 kW` was recorded for the household example);
`rango` returns `REFUSED NOT_ESTIMABLE — the bundle emits a point estimate and no predictive distribution`. Print
both lines. A horizon the catalog does not declare (90) is refused too.

**3. Causal** — "Report the ATE of treatment on outcome, with its uncertainty."

```json
{"area": "causal",
 "state": {"causal_graph": {"treatment": "treatment", "outcome": "outcome", "confounders": ["baseline"]}},
 "questions": {"efecto": {"type": "ate"},
               "jovenes": {"type": "cate", "condition": "baseline == 1"}}}
```

`efecto` returns `estimand`, `effect_size`, `unit`, `confidence_interval`, `confidence_level` and the list of
`assumptions` (recorded: ATE `2.0293820021534623` synthetic outcome units, 95% interval
`[1.931308938896731, 2.1274550654101936]`). Print the effect, its interval, and name the assumptions. **There is no
p-value in the output: never write one.** `jovenes` returns `REFUSED NOT_ESTIMABLE` — the study was fitted without an
effect modifier.

**4. RL** — "¿Qué acción propone la política y qué retorno espera?"

```json
{"area": "rl",
 "state": {"policy_id": "eth_4h_sac_current_stack_anchor_v1"},
 "data": [0.0, 0.0, "… the observation vector of the fitted size …"],
 "questions": {"accion": {"type": "next_action"}, "retorno": {"type": "value_estimation"}}}
```

`accion` returns an action in the policy's own scale with its `unit` (recorded for the all-zero vector:
`-0.019998908042907715`; the market-bars example gives a different action, because it is a different input).
`retorno` is a critic's estimate under the training reward, not money. Say that a score is not an order.

**5. Unsupervised / regimes** — "Segmenta estas filas y describe el cluster con cuerpo positivo."

```json
{"area": "unsupervised",
 "state": {},
 "data": {"rows": [{"row_id": "…", "body_pipettes": 371.0, "range_pipettes": 555.0}]},
 "questions": {"segmentacion": {"type": "clustering", "method": "auto"},
               "perfil": {"type": "cluster_description", "target_metric": "body_pipettes > 0"}}}
```

`segmentacion` assigns the supplied rows under the fitted reference; `perfil` describes the cluster satisfying the
declared metric. Report the assignment and the description, and that the reference is a fitted state, not a claim
about markets.

## Cautions that apply on this installation

- The engines are declared in the operator's environment file (`~/.config/m5phet/chat.env`), read by the server's
  launcher. You never need to know a host or a path: if an engine is not reachable, the tool refuses by name and you
  print the refusal.
- A state marked `development: true` is a development artifact. Say so in the line that carries its numbers, and
  never present it as a measured quality result.
- The **classification** answers you get through MCP come from whichever backend this host declares. The real
  checkpoint on the private worker is reached by the web workbench's own route, so a probability here may differ from
  the workbench's for the same text. If the person is comparing the two, say plainly that the MCP surface does not
  report which backend answered, and let them check the workbench.
- `m5phet_propose_task` needs the configured interpreter to be reachable. If it fails, write the envelope yourself
  from the catalog and show it — that path needs no interpreter.
- Two identical runs with a different input are two answers. Never merge them into one number.
