# AP01 — the family that cannot be served is no longer offered, and an answer survives the process

**Author:** Satoshi, successor technical lead
**Date:** 2026-09-29
**Repository / branch / tip:** M5PHET · `satoshi/m5phet-next-acceptance-20260929` · base `b6cbc5d`
(the CB05 delivery, itself based on RB04 `d17f377`)
**Worktree:** `$HOME/Documents/GitHub/.worktrees/m5phet-next-acceptance`
**Shape:** §6 of `docs/WORK_PLAN_2026_09_24.md`, at the end of this document; everything before it is the reasoning
and the evidence the §6 block cites.

> **Nothing in this round measured a model.** `NO_NEW_MODEL_MEASUREMENT`: no forecast, no policy, no study, no
> checkpoint and no classifier was scored. One real engine *answered*, and that is a different claim from a
> measurement. `NO_NEW_MEASUREMENT` in the owner's standing sense, so **no closure table is offered**.

---

## 1. Which acceptance case, and why it was the next one

The plan's revision 3 closed WP26–WP31 and left three user-facing items. Two of them are not independently
executable today: **WP28** (the observed-clock event study) needs a consensus source with a publication instant that
no feed on this machine carries — the round of 2026-09-26 records that as one of the three things outside the
successor's reach — and **WP09** (classification quality on an independent labelled corpus) waits on the owner's
annotator. **WP32** (start the point-in-time capture) is independently executable, and it is the one item whose
acceptance is *entirely* persistence; but no provider answers in it, so it could not be delivered "end to end with a
real provider", and it is left where the plan has it.

What was left, and what this round took, is the hole the **most recent delivery named against itself**. CB05 §9,
under *what is NOT done*:

> the typed-question catalog (`GET /api/tasks/catalog`) still LISTS the classification area when the declared backend
> is unserved. Every run of it refuses by name and nothing answers, but the listing itself is not yet gated, and I am
> recording that rather than implying otherwise.

That is the next independently executable user-facing acceptance case, for four reasons:

1. **It is user-facing in the strictest sense.** It is not about what an answer says; it is about what the product
   *offers* before anyone asks. A person, a router, or an MCP client reading the catalog was being shown a family the
   installation could not serve.
2. **It needs nobody.** No GPU, no new data, no owner decision, no other lane's runner, no corpus.
3. **It is the last step of a chain CB05 closed everywhere else.** CB05 made every *answering* door pass the backend
   contract. The *offering* door was the one left open, and a promise that cannot be kept is the same
   misrepresentation one step earlier.
4. **Its acceptance is exactly persistence and restart.** An offer resolved at start-up must still be right after a
   restart, and the answers a person already has must still be theirs. That is the part this lane exists for.

The acceptance case, in one sentence:

> **A family the product cannot serve is never offered, on any door; and an answer a real provider gave is still that
> answer — with the receipt of what answered — after the process comes back.**

---

## 2. The probe first: the hole was real, and a restart did not heal it

Before any code changed, the staged build of `b6cbc5d` was put on **my own port** (`8771`) with **my own state
directory**, configured with a bound classification worker whose declared checkpoint is **not** the pinned one. The
transport is a **local fake** (`docs/evidence/AP01_RESTART_2026_09_29/fake-ssh-describe.py`): it answers `describe`
and nothing else, no packet leaves this machine, and no real worker was contacted.

`docs/evidence/AP01_RESTART_2026_09_29/before-gate.json`, taken `2026-09-29T17:06:05-0500`:

| door | what it said |
|---|---|
| `GET /api/catalog` | `classification_backend: {"status": "CLASSIFICATION_CHECKPOINT_MISMATCH", "validated": false, "checkpoint_pinned": true, ...}` — the pin refused by name, which is the pin working |
| `GET /api/tasks/catalog` | `classification: {"provider": "laya_news", "question_types": ["choice"]}` — offered |

And after the process was stopped and a new one started on the same state directory, **both doors said exactly the
same thing again**. The hole was not a transient of start-up ordering; it was the steady state.

Two things that were *already* right, and are recorded so nobody repairs them twice:

- **the checkpoint pin refuses by name** rather than answering — information, not a fault;
- **the conversation survived the restart byte for byte.** The refused answer's message digest before and after was
  the same `cf7cef97a882e188…`, the attachment was still there, and the chat was still listed.

---

## 3. What was built

Four changes. The first is the acceptance case; the other three are defects the acceptance case's own proof exposed,
each one measured before it was fixed.

### 3.1 The offer is gated on what can actually be served

`questions.catalog(registry, configuration, unserved=...)` gained one parameter: `{area: {code, why, mode}}` for the
areas whose declared engine cannot be served. Such an area is published with its **declared provider named** —
hiding it would make the configuration unreadable from outside — and with **nothing on offer**: no question types, no
parameters, no aliases, no combinations, no confidence levels, `quality: NOT_MEASURED`, and its refusal's own code on
its face.

It is a parameter and not a lookup on purpose: `questions` must not acquire an opinion about anybody's backend. The
caller that resolved the contract passes the finding in, and there is exactly one such caller —
`web.engine.Engine.unserved_areas()`, which reads the **effective** backend block (measured from the path that would
answer, never inferred from the configuration).

Three doors now offer the same thing:

| door | before | after |
|---|---|---|
| `GET /api/tasks/catalog` | offered `choice` | `question_types: []`, `unavailable: {code, why, mode}` |
| `m5phet_catalog` (MCP, with an engine) | offered `choice` | the same gated catalog |
| `POST /api/chats/{cid}/tasks/propose` (the router) | proposed classification envelopes | the area reaches the model with nothing on offer; a hand-written proposal naming it is refused `CLASSIFICATION_CHECKPOINT_MISMATCH: …` by `check_proposal` |

The standalone MCP server (`python -m m5phet.mcp_server`, no engine) has no backend contract to consult in its own
process; it publishes the ungated catalog and `run_task` refuses what it cannot answer. That is stated in the code
and here rather than implied away.

### 3.2 A request the restart interrupted can be sent again

`Store.recover()` already named an in-flight request `INTERRUPTED` at every start-up. What it could not do is let the
person act on it: `Store.begin` found the prior record, saw a matching request fingerprint, and handed it straight
back **running nothing**. The caller received `202 Accepted` for work that would never begin and polled a message
that would never finish; the only escape was to invent a new request id, which is a *different* request.

Now an `INTERRUPTED` prior of the same identity is **run, once**, from the snapshot it was accepted with — so what
runs is the request the person sent and not whatever the chat's configuration has become since. A request that
*finished* is still a replay (same id, same answer, nothing re-run), and a request id reused for **different** input
is still refused, which is the rule that makes the rest safe.

### 3.3 A shutdown is not an engine's refusal

Measured while proving the restart: stopping the workbench killed the forecast engine's subprocess with it, and the
person was then shown

> `La pregunta de pronóstico puntual (tipo point_forecast) fue rechazada (REFUSED) con el código PROVIDER_ERROR. El
> motivo dado: RuntimeError: native CPU forecast process refused:`

— the shutdown wearing the engine's name. A reader would conclude the forecaster had declined their question. It had
not; it had been killed. This is the same class of mistake as a fixture wearing a model's name: the receipt names the
wrong cause.

Three parts, and the third is why the first two were not enough:

1. `main()` now notices the stop **in its own signal handler**, before uvicorn begins the graceful shutdown, and
   records which requests were in flight at that instant. A flag set in the lifespan's shutdown is too late: the
   signal reaches the whole process group, so the engines die immediately while uvicorn is still finishing
   responses, and the run thread usually records its outcome first.
2. A run that ends in an exception while the server is stopping is recorded `INTERRUPTED`, with the engine's own
   words kept in `error_at_shutdown` — evidence is never discarded, it is just not presented as the answer.
3. **The app's `except` branch is not where a stopped engine arrives.** `run_task` converts a provider exception into
   a *typed* refusal, `PROVIDER_ERROR`, so the run "succeeded" as far as the app was concerned. The rule therefore
   also applies to a completed run that **answered nothing and whose every refusal is `PROVIDER_ERROR`**, and only
   then. A typed refusal an engine *reached* — `NOT_ESTIMABLE`, `STATE_REQUIRED`, `NOT_IDENTIFIED` — stays that
   refusal, shutdown or no shutdown, because the engine considered the question and declined it by name, and that is
   the person's answer. There is a test in each direction.

### 3.4 A model's measurement does not travel with a fixture's answer

Measured on the served instance, whose classification backend is the **declared `NON_MODEL_FIXTURE`**: the fixture
answered `other` with `euro_area 0.0`, and the answer carried

> `quality: {"status": "MEASURED", "values": {"macro_f1": 0.37775954555995367, …}, "n": {"scored_rows": 450}}`

— a real checkpoint's retained record, published beside an answer **no model gave**, with nothing in the block saying
the answering path has no weights. CB05 closed the mirror image of this (a worker answered, and the locally installed
provider's record would have been published beside it, which now raises). This is the same mistake in the other
direction.

The rule added is narrow, because a wider one would hide real measurements:

- a path that declares real weights keeps its record, unchanged;
- a path that declares **no** weights, or names itself the `fixture` backend, has the record **withheld**: the block
  becomes `NOT_MEASURED` with the reason `QUALITY_RECORD_IS_NOT_OF_THE_ANSWERING_PATH`, keeping the record's
  **identifiers** — corpus id, seal, protocol digest, n — so a reader can go and find it, and **not** its values;
- a provider that declares neither a backend nor a weights flag is not contradicted: this rule does not guess what
  answered, it only refuses to let a model's number travel with a declared non-model's answer.

Confirmed live, on the served instance, in `docs/evidence/AP01_RESTART_2026_09_29/verify_envelopes.json`: the
classification entry's quality is now

```json
{"status": "NOT_MEASURED",
 "why": "the path that answers declares no model weights (backend 'fixture'), and the retained quality record was
         measured on model weights, so it says nothing about these answers; it is named here and not quoted",
 "quality_record_withheld": {"reason": "QUALITY_RECORD_IS_NOT_OF_THE_ANSWERING_PATH",
                             "corpus_id": "7e4789e5…", "corpus_seal": "31257d47…",
                             "protocol_digest": "b9aefb3c…", "n": {"scored_rows": 450}}}
```

— the record findable, its values not standing beside a fixture's answer. Both standing harnesses stayed green
through the change (12/12 · 5/5 · 14/14 · 2/2 and 15/15).

One limitation, stated because it is the honest shape of the gap: the record (`news_signal.quality.v1`) carries **no
checkpoint of its own**, so nothing here can check a record *against* the checkpoint that served it. That is a gap in
the record's schema, not a licence to publish it beside anything.

---

## 4. The end-to-end result: input, output, persistence, restart

`tools/verify_restart.py` (new, committed) starts a workbench it owns, drives one conversation through it, stops it
with `SIGTERM` to **the process group it created itself**, starts a **new** process on the **same** state directory
with a **new** session, and compares. It never signals a process it did not start.

Evidence: `docs/evidence/AP01_RESTART_2026_09_29/after-real.json`, taken `2026-09-29T17:31:29-0500`, port `8771`.

| property | proof |
|---|---|
| **input** | the shipped household example attached exactly as the `+` button attaches it — `sha256 8148ba924eafd90fe3e47959f2af0bee2826339c39a3db1903a047d5ea4a13fd` — and an envelope naming the fitted bundle the catalog declares |
| **output** | `OK` from a **real provider**: `predictor_forecast`, `point_forecast` horizon 60, **`0.5412255525588989 kW`**, `scale: original`, `state_ref e1-household-r0-s1:15bd0451…`, `execution_authorized: false` |
| **persistence** | the same answer read back out of the store: message digest `c87b964337baa8ff240e7eb0375cacbb497f0649711a005c3dce7b7573abc4cf`, detail digest `a34fd7fc618b1229…` |
| **restart** | process stopped (`SIGTERM`, `returncode -15`), a new one started on the same state directory: the chat is listed, the attachment is present, and the message and detail digests are **identical** — `c87b9643…` / `a34fd7fc…` |
| **restart, the hard half** | a second request, submitted and still `RUNNING` when the stop arrived, came back named **`INTERRUPTED`** — *"Server stopped while this request was running; no answer was produced. Send the same request again to run it."* — and **re-sending it ran it to `OK` with the same value**, `0.5412255525588989 kW` |
| **the offer** | `classification` is served here (validated `fixture`), so it is offered — the gate does not fire where it should not |

And the gated case, `after-gate.json`, taken `2026-09-29T17:32:14-0500`, port `8772`, same four properties on the
same conversation plus the gate closed:

```json
"classification": {"provider": "laya_news", "question_types": [],
                   "unavailable": {"code": "CLASSIFICATION_CHECKPOINT_MISMATCH",
                                   "why": "M5PHET_CLASSIFICATION_EXPECT_CHECKPOINT pins '…aaaa' and the
                                           remote_worker path serves '…bbbb'.",
                                   "mode": "remote_worker"}}
```

— before the restart and, identically, after it. The forecast in that same instance answered `0.5412255525588989 kW`
(message digest `7396aac840b3fcfc…`, unchanged across the restart): one area being unserved does not take the others
down, which is this repository's own stated rule that every question is answered separately.

Both runs' verdicts, machine-checked, all `true`:

```json
{"input_attached": true, "output_answered": true, "persisted_identically": true,
 "conversation_survived_restart": true, "receipt_survived_restart": true,
 "interrupted_named": true, "interrupted_request_reran": true, "offer_matches_servability": true, "all": true}
```

---

## 5. Fixture, real provider and abstention — three outcomes, and they are reported as three

| family | what answered in the proven configuration | which kind of outcome |
|---|---|---|
| forecasting | `predictor_forecast`, a fitted TensorFlow bundle in its own interpreter, `0.5412255525588989 kW` | **real provider** |
| classification | the **declared `NON_MODEL_FIXTURE`** in this process (`NEWS_SIGNAL_BACKEND=fixture`, `weights_present: false`, `device: cpu`), and since §3.4 it publishes **no** measured quality | **fixture** |
| classification, gated case | a bound worker whose declared checkpoint is not the pinned one, over a **local transport fake** | **abstention** — `CLASSIFICATION_CHECKPOINT_MISMATCH`, refused by name, and now not offered either |
| causal | `causal_inference` over a study fitted beforehand (in the standing harnesses) | **real provider**, quality **`REFUSED: causal_accuracy`** |
| rl | `trading_policy`, SB3 in its own interpreter (in the standing harnesses) | **real provider**, quality **`REFUSED: policy_profitability`** |
| unsupervised | `feature-eng-hierarchical-regimes`, assignment under a fitted reference (in the standing harnesses) | **real provider**, quality **`NOT_MEASURED`** |

**A named refusal is not a completed family**, and that sentence governs this document as it governs CB05's §5. Of
the five areas: **one** carries a measured quality and it is a **retained record**, not a measurement taken today;
**two** carry `NOT_MEASURED`; **two refuse by name**. Nothing here promotes a refusal into a delivery, and nothing
here presents a fixture's answer as a model's.

**The real Laya weights were not contacted.** They live on the worker host that this round's dispatch declares
**ineligible** (≈3 GiB free of 14, with 4.93 GiB unreclaimable slab, read minutes before the round began). So the
classification family in every instance I ran answered either from the **declared fixture** or from a **local
transport fake**, both labelled as such in every report, and **no statement in this document is a measurement of any
checkpoint**.

---

## 6. What is NOT done, refused, or not measured

- **`NO_NEW_MODEL_MEASUREMENT`.** Nothing here measured a forecast, a policy, a study, a checkpoint or a classifier.
  One real engine answered one envelope; an answer is not an accuracy.
- **`NO_NEW_MEASUREMENT`** in the owner's standing sense, so **no closure table is offered**: no model error, no
  paired naive on the same rows, no literature value. This round scored no task.
- **The router was not re-measured, and the development corpus was not touched.** The 19 prompts × 5 repeats stand
  exactly where RB04 and CB05 left them (85/95 with the completion pass off, 79/95 with it on, one scorer, one day).
  That corpus measures the **router** and is not a classification benchmark under any reading.
- **The frozen held-apart set was not opened.** `docs/evidence/ROUTE_GENERALIZATION_HELDOUT_2026_09_28/`
  `heldout_paraphrases.json` remains `FROZEN_NOT_MEASURED`; no run of it exists, nothing in this round read it, and
  nothing in this round was tuned, selected or fixed against it. Its pinned digest is unchanged.
- **The tests were written after the probe, not before it.** The plan asks for tests first. What came first here was
  the read-only probe of §2, which is what established that the hole was real; the 19 new tests were written against
  the measured behaviour and then the fixes. I record the order rather than claiming the plan's.
- **Focused tests are not a measurement of full-corpus quality.** The 19 new tests are tests. §4 is the measurement,
  and it measures the product's behaviour, not any model's accuracy.
- **The standalone MCP server is not gated** (§3.1): without an engine it has no backend contract in its process.
- **`news_signal.quality.v1` carries no checkpoint**, so §3.4 can refuse to publish a model's record beside a
  non-model's answer but cannot bind a record to the checkpoint that served it.
- **The deployment is PREPARED, not applied.** The owner's service on `8765` served throughout, was never signalled,
  and its venv was never written to. Adopting this build inherits CB05's step 0: `~/.config/m5phet/chat.env` is the
  contradiction this build refuses by name, and **his file was not edited by me**.
- **Host names were checked structurally, not by substring search.** Every string value of every committed
  artifact was walked as parsed JSON and compared with the operator's worker binding, because that binding's value
  collides with unrelated identifiers and a raw `grep` produces false positives. No string value in any file of this
  round carries it; the worker is `worker.invalid` throughout, a reserved name belonging to nobody.
- **No conversation store was overwritten or migrated.** Every instance in this round used a state directory created
  for it under `$HOME/.local/share/m5phet/staging-ap-20260929/`.

---

## 7. Resources

Everything heavy went through `$HOME/.local/bin/crispdm-run` at the revision already deployed; nothing was
reinstalled. Coordinator only; `CUDA_VISIBLE_DEVICES=""` in every instance; **no GPU was used and no other host was
touched**.

| job | cap | what it was |
|---|---|---|
| `m5phet-ap-install`, `…-reinstall*` | `-m 3G -t 10m` | non-editable `pip install --no-deps` of this branch into my own staged venv copy |
| `m5phet-ap-suite*` | `-m 5G/6G -t 20m` | the M5PHET suite |
| `m5phet-ap-eval2` | `-m 3G -t 15m` | the `m5phet_evaluation` suite |
| `m5phet-ap-before`, `…-real*`, `…-gate`, `…-final` | `-m 6G -t 25m/30m` | the restart acceptance, including the forecast engine's own subprocess inside the same scope |
| `m5phet-ap-harness*` | `-m 8G -t 40m` | the two standing harnesses, with the five providers' interpreters |

**One admission refusal was met and obeyed.** With the harness scope holding 8G, a 3G evaluation-suite request was
refused: *"the observed aggregate budget would be 15.32G against the crispdm-batch.slice ceiling 14.00G (in use
3.12G, unrealised reservations 9.20G) — REFUSED, nothing was started"*. **The declared cap was not shrunk to get
past it.** The job was re-run after the harness released its reservation, and it passed 42/42.

The external RTX 5090 host was **not** used and **not** contacted: this round's dispatch declares it ineligible, and
nothing here needed a GPU. The secondary worker was not used either — this work is CPU-only and belongs where its
state directories are.

---

## 8. The report, in the §6 shape

```
AP01 — the offer gated on servability, and one conversation proven across a restart
repo/branch/tip: M5PHET · satoshi/m5phet-next-acceptance-20260929 · base b6cbc5d (CB05), tip = this commit
files:
  src/m5phet/questions.py            `unserved` parameter + `unserved_area`: an area that offers nothing and names its refusal
  src/m5phet/web/engine.py           `unserved_areas()` read from the EFFECTIVE backend; task_catalog and propose_task gated
  src/m5phet/orchestrate.py          route passes it through; check_proposal refuses an unserved area by its contract's name
  src/m5phet/mcp_server.py           m5phet_catalog offers the gated catalog when it has an engine
  src/m5phet/web/store.py            recover()'s one sentence; interrupt_stopped(); begin() RE-RUNS an INTERRUPTED request
  src/m5phet/web/app.py              the stop noticed at signal time; a stopped engine is INTERRUPTED, not a refusal
  src/m5phet/quality.py              a measured record is withheld from a path that declares no model weights
  tests/test_unserved_offer.py       (new) 19 gate + quality-withholding tests
  tests/test_restart_recovery.py     (new) 11 restart, replay and shutdown-attribution tests
  tools/verify_restart.py            (new) the four-property acceptance harness: input, output, persistence, restart
  docs/evidence/AP01_RESTART_2026_09_29/  the probe before the fix and both runs after it, with SHA256SUMS
  docs/SATOSHI_M5PHET_NEXT_ACCEPTANCE_2026_09_29.md  (this)
suites (after reinstall): M5PHET 830 passed / 1 skipped · m5phet_evaluation 42 passed / 0
  (news-signal, prediction_provider and m5phet_policy were NOT re-run: nothing in those repositories was
   touched, and their 2026-09-28 numbers stand as RB04 reported them)
acceptance:
  verify_families  {"examples_answering": 12, "examples": 12, "families_answering": 5, "families": 5,
                    "examples_that_resolve": 12, "prose_answered": 14, "prose": 14, "prose_declared": 14,
                    "refusals_correct": 2, "refusals": 2, "any_execution_authorized": false}
  verify_envelopes {"questions_as_expected": 15, "questions": 15, "any_execution_authorized": false}
  verify_restart (real provider, port 8771)  all eight properties true; 0.5412255525588989 kW before and after the
                    restart, message digest c87b9643… unchanged, INTERRUPTED request re-sent and answered OK
  verify_restart (gated case, port 8772)     all eight true; classification question_types [] with
                    CLASSIFICATION_CHECKPOINT_MISMATCH on its face, before AND after the restart
  browser: NOT RUN (no interface change; the static assets are untouched)
what is NOT done / refused / not measured: §6 above, in full. In one line: NO_NEW_MODEL_MEASUREMENT and
  NO_NEW_MEASUREMENT, the router untouched, the held-apart set unopened, the real Laya weights not contacted
  because their host is ineligible, the deployment prepared and not applied, and the owner's service never signalled.
```

---

Satoshi, successor technical lead.
