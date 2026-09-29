# CB05 — the contradictory backend refused, the short return corrected, the router evidence scoped

Satoshi, successor technical lead, 2026-09-28 (America/Bogota).
Base: `satoshi/rb04-m5phet-integration-20260928` at `d17f377`.
Branch: `satoshi/cb05-backend-contradiction-20260928`.
Orders: CB05 of `SATOSHI_CLASSIFICATION_REFERENCE_ORDERS_2026_09_28.md`, under the
work plan of 2026-09-24 and its standing rules §0. Shape: §6 of that plan,
extended where this round produced more than one line of a kind.

Nothing here authorizes execution, an order, or a claim about a market. No
service the owner is running was started, stopped, restarted or signalled; the
chat service on 8765 served throughout, its venv was **not** written to, no port
was opened by this work, and no conversation was overwritten. The deployment is
**prepared and proposed, not applied**, and this round makes that proposal
conditional on a one-line repair of the operator's own `chat.env` — see §6.

---

## 1. Can a contradictory configuration still resolve silently? No.

It refuses, by name, before the product exists. `m5phet.classification_backend`
decides which backend answers classification **once**, from the operator's
declarations, and a configuration that disagrees with itself is a refusal and
never a repair:

| selections in force | outcome |
|---|---|
| `NEWS_SIGNAL_BACKEND=fixture`, no worker | mode `fixture` |
| `NEWS_SIGNAL_BACKEND=laya`, no worker | mode `local_weights` |
| a bound `M5PHET_CHAT_LAYA_WORKER`, backend unset | mode `remote_worker` |
| **`NEWS_SIGNAL_BACKEND=fixture` + a bound worker** | **REFUSED `CLASSIFICATION_BACKEND_CONTRADICTION`** |
| `NEWS_SIGNAL_BACKEND=laya` + a bound worker | REFUSED `CLASSIFICATION_MODE_REQUIRED` until one mode is declared |
| a bound worker with no `M5PHET_CHAT_LAYA_COMMAND` | REFUSED `CLASSIFICATION_WORKER_COMMAND_NOT_BOUND` |
| `M5PHET_CLASSIFICATION_MODE=remote_worker`, nothing bound | REFUSED `CLASSIFICATION_WORKER_NOT_BOUND` |

Two things about that table are the substance of the correction.

**A declared mode cannot rescue the contradiction, and that is deliberate.**
`M5PHET_CLASSIFICATION_MODE` (also bindable as
`areas.classification.core.mode`) settles an **ambiguity** — two selections that
both mean real weights, in two different places — and it is refused as a
settlement of an explicit disagreement. A mode that overrides an explicit
`fixture` is the same silent override the auditor objected to, wearing a label.
The refusal is parameterised over all three modes in the tests, so no mode
value can ever become the escape hatch.

**The refusal names both variables and both repairs**, because an operator who
cannot see which two settings disagree cannot fix it:

```
CLASSIFICATION_BACKEND_CONTRADICTION: NEWS_SIGNAL_BACKEND=fixture declares a non-model
backend while M5PHET_CHAT_LAYA_WORKER binds a worker that answers from real weights. These
cannot both be in force and neither silently wins. Repair it in one of two ways: to serve the
fixture, unset M5PHET_CHAT_LAYA_WORKER (and M5PHET_CHAT_LAYA_COMMAND); to serve the worker's
weights, unset NEWS_SIGNAL_BACKEND or set it to 'laya', and declare
M5PHET_CLASSIFICATION_MODE=remote_worker.
```

**And the operator's own configuration is that contradiction today.** Put
through the staged build in a fresh process, `~/.config/m5phet/chat.env`
refuses to start with `CLASSIFICATION_BACKEND_CONTRADICTION` and exit 3 (§4,
case 8). That is the correct outcome and it is also the adoption condition: the
repair is one line in a file outside this repository, and it is the owner's to
make. Nothing here edits his configuration.

The host is never published. The resolution carries `worker_bound: true`; no
refusal, receipt, catalog field or test fixture in this work contains a host
name, an address, a token or an account identifier, and a test asserts that the
placeholder host does not appear anywhere in a resolution's representation.

## 2. What the receipts now record about the effective backend

The old receipt carried one word — `backend` — taken from whichever
capabilities happened to be in hand, which on the contradictory configuration
was the half that did **not** answer. Every classification answer, on both
doors (`execute`, the sentence path, and `execute_task`, the envelope path),
now carries a validated block read from the declaration of the path that
answered:

```json
"classification_backend": {
  "schema": "m5phet.classification_backend.v1.effective",
  "mode": "remote_worker", "mode_source": "DECLARED",
  "answered_by": "remote_worker",
  "backend": "laya", "weights_present": true, "device": "cuda:0",
  "checkpoint": "laya-checkpoint:<sha256>", "checkpoint_pinned": true,
  "status": "VALIDATED", "validated": true,
  "reading": "the backend and checkpoint that ANSWERED, read from the declaration of the path
              that answered and checked against the configured mode; not inferred from the
              configuration."
}
```

It is validated at **start-up** and again on **every answer**, and the
validation refuses in both directions:

| what is wrong | refusal |
|---|---|
| a worker describing itself as a `fixture` under a real-weights mode | `CLASSIFICATION_BACKEND_MISMATCH` |
| real weights answering a declared `fixture` | `CLASSIFICATION_BACKEND_MISMATCH` |
| backend `laya` with `weights_present: false` | `CLASSIFICATION_WEIGHTS_ABSENT` |
| a checkpoint other than the pinned `M5PHET_CLASSIFICATION_EXPECT_CHECKPOINT` | `CLASSIFICATION_CHECKPOINT_MISMATCH` |
| an answer from the other path than the mode declares | `CLASSIFICATION_PATH_MISMATCH` |
| the declared worker unreachable | `CLASSIFICATION_WORKER_UNREACHABLE`, retried on the next need |

A **refused** answer is a receipt as well, and now records which declared
backend it was about: `status`, `mode`, and `backend: null` — nothing is
invented for a path that did not serve. And the catalog publishes the same
block under `classification_backend`, beside `classification_selection`, so a
reader of the workbench, an MCP client or a Telegram skill sees what answers
before it trusts an answer.

**Where a receipt says less than validated, it says so.** An `Engine` built
with a registry the caller supplied — every embedded and in-process test use —
resolved no operator configuration at all, so its block is
`mode: "PROVIDER_SUPPLIED"`, `status: "PROVIDER_SUPPLIED"`, with a reading that
says the backend is the supplied provider's own declaration, *unchecked against
a mode*. An unchecked declaration must not read as a validated one.

## 3. No hidden fallback, and the two fallbacks that were there

A request that cannot reach its declared backend refuses **by name**, and
nothing else answers in its place. Three paths had to change for that to be
true rather than intended:

- **the catalog substituted the other provider.** It published the locally
  installed `laya_news` capabilities whenever the worker had not described
  itself — so a reader could see a fixture's states and `backend: fixture`
  beside a configuration that declared a worker. It now publishes the refusal
  itself (`refused: "CLASSIFICATION_WORKER_UNREACHABLE: …"`, empty vocabularies,
  `backend: null`);
- **the envelope door validated nothing at all.** `execute_task` sent a
  classification envelope to the worker when one was bound and ran the local
  engine otherwise, with no check of either against a declaration. It now
  passes the same start-up contract as a single question;
- **the envelope door's quality fell back to the wrong provider.** When the
  worker had not described itself, the answer it produced was published with
  the **locally installed** provider's measured quality — a number about a
  checkpoint that did not answer. That is the same inference this correction
  removes from receipts, so it now raises. On a configured installation the
  clause is unreachable, because the start-up contract refuses first; it is
  kept as a raise rather than deleted because the embedded path still has it.

The worker-lock repair of 2026-09-24 is preserved on purpose, and the
distinction is stated rather than assumed: a **configuration** contradiction is
static and refuses at start-up; a **transport** failure is dynamic, so the
product starts, every classification path refuses by name, and the describe is
retried on the next need. A worker that was busy for seven seconds must not
become a refusal to exist — and must not become a fixture answering either.

## 4. The fresh-process evidence, positive and negative

Tests first, confirmed red at `b28a3c3` (collection failed on the module and
the function they require), implementation after. **48 tests** in three files:
29 on the contract, 8 in fresh child processes against the public entry point,
9 on the corpus scoping and the frozen held-out set. The child processes patch
nothing inside the product: the transport is faked where a transport belongs,
by an `ssh` of our own first on `PATH`, and every case clears the variables
this contract owns so no case can pass because the operator's shell held one.

The same probe (`tests/backend_startup_probe.py`) is what the deployment smoke
runs, so the suite and the isolated smoke go through one door. Run against the
**staged** build, `$HOME/.local/share/m5phet/staging-cb05-20260928/backend-contradiction-smoke.sh`:

| case | startup | effective backend | the answer |
|---|---|---|---|
| 1. declared fixture | OK | `fixture` / `VALIDATED`, checkpoint `laya-checkpoint:f057ebed…` | `OK`, receipt `backend: fixture`, `answered_by: in_process` |
| 2. fixture **+** bound worker | **REFUSED**, exit 3, `CLASSIFICATION_BACKEND_CONTRADICTION` | — | no server exists; nothing answered from either backend |
| 3. `laya` + bound worker | **REFUSED**, exit 3, `CLASSIFICATION_MODE_REQUIRED` | — | — |
| 4. remote mode, **no worker bound** | **REFUSED**, exit 3, `CLASSIFICATION_WORKER_NOT_BOUND` | — | — |
| 5. remote mode, worker answers | OK | `laya` / `VALIDATED`, weights present, its own checkpoint | `OK`, receipt `answered_by: remote_worker` |
| 6. remote mode, worker **fails** | OK | `CLASSIFICATION_WORKER_UNREACHABLE`, `backend: null` | `REFUSED` naming the code, **`outputs: []`**, catalog `refused:` — the fixture did not answer |
| 7. remote mode, worker **is a fixture** | OK | `CLASSIFICATION_BACKEND_MISMATCH` | `REFUSED` naming the code, `outputs: []` |
| 8. the operator's real `chat.env` | **REFUSED**, exit 3, `CLASSIFICATION_BACKEND_CONTRADICTION` | — | — |

Case 6 is the counterexample the order asked for, and it is kept: a declared
backend that cannot be reached produces a named refusal and an empty answer,
not a quiet fixture reply. Case 7 is its sharper form — the fixture that
answers is the *worker's own*, and it is refused as well. Case 8 is the
configuration that caused this work, run unchanged.

**One flaw of my own, found in this round and fixed rather than left.** The
first run of that smoke reported case 8 as starting successfully with
`mode: local_weights, mode_source: PROVIDER_DEFAULT`. That was my harness, not
his configuration: `chat.env` is a systemd `EnvironmentFile` — `KEY=VALUE` lines
with no `export` — so sourcing it plainly sets shell variables that no child
process ever sees. `set -a` around the source is what makes it reach a child.
I re-ran it, and the corrected case 8 is the row above. A smoke that quietly
tests an empty environment is worse than no smoke, because it reports a pass.

The shipped configuration was itself the contradiction: `tools/start_chat.sh`
exported `NEWS_SIGNAL_BACKEND=fixture` unconditionally and then bound the
worker. The default now lives only in the branch where no worker is bound, the
worker branch declares one unambiguous mode, and a start-up refusal is not
treated as a licence to keep shipping the combination that earns it.

## 5. Correction 2 — my short return overstated what was measured

This is mine to own. I told the owner that classification answered with real
weights and a **measured macro-F1**, in a short return whose shape invited the
reading that a live classifier accuracy run had happened in RB04. **It had
not**, and the detailed return says so in its own §8: `NO_NEW_MODEL_MEASUREMENT`
for every engine. What produced those statements was a **declaration** and a
**retained record**, which are two different kinds of evidence and neither of
them is a measurement taken that day.

Claim by claim, in the words the distinction requires:

| the statement | what it actually rests on |
|---|---|
| classification is answered by a real provider, `backend: laya`, `weights_present: true`, `device: cuda:0`, checkpoint `laya-checkpoint:bd12df88…` | a **declaration**: the live provider's own `capabilities()`, read through the workbench's HTTP API. No inference was run to obtain it |
| macro-F1 `0.3778`, n `450`, ECE `0.1315`, Brier `0.6473`, `UNCALIBRATED`, naive majority `0.1667` / keyword `0.1760` | a **retained record**: the operator's quality record for that provider, published by the catalog. No corpus was scored on 2026-09-28 |
| the other four areas' providers and fitted states | **declarations**, from the same catalog read |
| forecasting and unsupervised quality | `NOT_MEASURED` — no held-out report is bound in that configuration |
| causal and rl quality | **named refusals**: `REFUSED: causal_accuracy`, `REFUSED: policy_profitability` |
| the router reliability figures of RB04 §4 | the one thing **measured** that day, and it is a property of the router, not of any classifier |

Two sentences I now put in the record because their absence is what made the
short return misleading:

- **a named refusal is not a completed family.** The causal and rl families
  **refused** to state a quality. A refusal is a good answer and a delivered
  family is a different claim; a summary that lists five families as working
  because five providers answered has quietly promoted two refusals into
  deliveries. Of the five areas: one carries a measured quality (from a
  retained record), two carry `NOT_MEASURED`, and **two refused**;
- **`NO_NEW_MODEL_MEASUREMENT` applies to RB04 as a whole.** Nothing in it
  measured a forecast, a policy, a study or a checkpoint. `POST
  /api/tasks/propose` builds an envelope and runs nothing, and the workbench ran
  with `CUDA_VISIBLE_DEVICES=""`.

The corrected short return, which is what should have been said:

> RB04 integrated the preserved work and closed the envelope's three holes. The
> classification family is answered by a **real provider** — that is the
> provider's own **declaration** (`backend: laya`, weights present, a checkpoint
> digest), not a measurement I took. Its macro-F1 `0.3778` on n `450` is a
> **retained record** of an earlier run, published by the catalog;
> **`NO_NEW_MODEL_MEASUREMENT`** was taken in RB04, of that checkpoint or any
> other. The one number measured that day is the **router's** reliability over
> 19 development prompts × 5 repeats. Two of the five families — causal and rl —
> answered with **named refusals** for their quality, and a refusal is not a
> delivered family. The deployment was prepared, not applied.

For the record repair, `docs/SATOSHI_RB04_M5PHET_INTEGRATION_2026_09_28.md`
carries a pointer to this section at its head; its own text is unchanged,
because the detailed return already said `NO_NEW_MODEL_MEASUREMENT` and it is
not the document that overstated anything.

## 6. Correction 3 — the router evidence scoped, and a held-out set protected

**It is 19 prompts × 5 repeats, not 95 independent examples**, and it measures
the **router** and not the classifier. Five runs of one sentence are five
observations of one item, so the completion pass's six-run difference is not six
examples' worth of evidence. Every report of the harness now says so in its own
fields rather than leaving the arithmetic to the reader:

```json
"corpus_role": "ROUTER_DEV_CORPUS_NOT_A_CLASSIFICATION_BENCHMARK",
"corpus_scope": {"scoped": false, "prompts": 19, "repeats": 5, "runs": 95,
                 "independent_examples": false, "measures": "router",
                 "not_a_classification_benchmark": true, "reading": "19 prompts x 5 repeats = 95
                 runs. Repeats of one prompt are repeated observations of ONE item and are not
                 independent examples, so a difference of k runs is not k examples' worth of
                 evidence. What is scored is the ROUTER -- a language model writing an envelope --
                 and never a classifier's label: this is not a classification benchmark under any
                 reading."}
```

The role string is in the protocol text as well, so it travels with every
fingerprint and no report of this harness can be quoted as a classification
score anywhere. A scoped run keeps its `SUBSET` reading and now counts its own
prompts and repeats too.

**The pass stays off by default**, which is what the full corpus said
(precision 0.20: 15 additions, 3 right, 12 wrong; 85/95 with it off against
79/95 with it on). The four sentences it **helps** are retained — in both
committed reports, in RB04 §4.6's per-sentence table, and as counterexample
tests in both directions — because "at this precision it must not be the
default" is the finding, and "it is worthless" would be a different and false
one.

**The freeze.** Those 19 sentences have now been spent: a switch was decided on
what they said, so a number measured on them again is a development number,
however many repeats it carries, and calling it untouched validation would be
the thing the order forbids.
`docs/evidence/ROUTE_GENERALIZATION_HELDOUT_2026_09_28/heldout_paraphrases.json`
(`sha256 1cb0310a67a089891447614d43f83eb67a4cb07f7be5dc98fc3129d7de69233f`)
holds **19 unseen paraphrases**, one per development sentence, covering all five
areas, each with the expectation a later run would score it against and the
shipped example whose data it attaches. Its own `rule` field says what it is
for: held apart, `FROZEN_NOT_MEASURED`, never used to tune, select, fix or
decide whether a change ships; measured once, after a decision was taken
elsewhere, to say how far the router generalises. The digest is **pinned in the
test file**, so editing the set means changing a digest and saying why, and a
test asserts that not one of its sentences already appears in the development
corpus. **No run of it exists**, and none was taken here.

## 7. The pinned build, prepared and NOT applied — and the condition on adopting it

The live venv was **not written to**. Independent proof rather than an
assurance: the archive of its installed `m5phet` bytes taken today has
`sha256 199ae1ae…9ef0a4d`, **byte-identical** to the archive RB04 took at 22:17
before any of this work, and the installed `web/engine.py` still carries its
2026-09-25 mtime. The service has been active since `2026-09-28 10:03:29 -05`
and was never signalled.

Prepared under `$HOME/.local/share/m5phet/staging-cb05-20260928`:

| artifact | what it is |
|---|---|
| `venv/` | a copy of the service's venv — the owner's exact provider set — with `m5phet` and `m5phet-evaluation` reinstalled **non-editable** from this branch's tip. `direct_url.json` names this worktree; the staged `classification_backend.py`, `web/engine.py`, `web/app.py` and `config.py` are byte-identical to the tip |
| `backup-20260928/chat-cb05.sqlite3` | the owner's conversations, taken with sqlite's **online backup** so a live writer cannot tear it. `PRAGMA integrity_check` = `ok`. `sha256 0cbc2668…8d74005` — the same digest as RB04's backup, which says no message was written in between, not that the backup was skipped |
| `backup-20260928/installed-m5phet-packages.tar.gz` | the exact installed bytes a deployment would replace. `sha256 199ae1ae…9ef0a4d` |
| `rollback.sh` | an **actual code-only** rollback: restores those bytes, refuses to run if the archive is not the one it was written for, and deliberately does **not** restore the conversation database, which is newer than the backup by every minute the service has run. It does not restart anything |
| `backend-contradiction-smoke.sh` | the **isolated** smoke of §4: eight configurations, a fresh child process each, its own state directory each, an `ssh` of its own, no port, no packet, the owner's 8765 and his state directory untouched |
| `fake-ssh/ssh` | that transport. It binds an `infer` result to the request digest exactly as the real worker does, so no case passes through a transport the real one would have refused |

Module digests of the staged build:

| module | sha256 |
|---|---|
| `classification_backend.py` | `e8631171923ab701b834cdfc83f662fdc54a0a6dbaa06dedbfc7aed361bfd4da` |
| `web/engine.py` | `6bb828bc71600f32c9436948934486776500db929400d599f845418f3403d7cb` |
| `web/app.py` | `e06ed534b01fab9dad4a863ef5200cf9aab1b1bb1074a64a455bd94e6991aace` |
| `config.py` | `3fe7f775af19b5fbf804ff96010366adcc64089e78fbdc13aade4736c4bd24e4` |
| `config.schema.json` | `b02faf4c68024a0f35117e7bedb06d4debf890f5b2ab80b58223e5f9a1e867f5` |

**The proposed deployment, for the owner to apply or refuse. It now has a step 0
and it is not optional:**

```bash
# 0. REPAIR THE CONFIGURATION FIRST. This build refuses to start on the current chat.env, by name.
#    ~/.config/m5phet/chat.env holds NEWS_SIGNAL_BACKEND=fixture AND a bound M5PHET_CHAT_LAYA_WORKER.
#    To keep serving the worker's real weights, which is what has been answering:
#       - remove the NEWS_SIGNAL_BACKEND line (or set it to laya), and
#       - add   M5PHET_CLASSIFICATION_MODE=remote_worker
#    Recommended as well, so a swapped checkpoint is refused instead of answered:
#       - add   M5PHET_CLASSIFICATION_EXPECT_CHECKPOINT=laya-checkpoint:<the sealed manifest's sha256>
#    To serve the declared fixture instead, remove the worker lines. Not both. This is the owner's
#    file and this work did not edit it.

# 1. the contract smoke, on its own state directories, changing nothing
$HOME/.local/share/m5phet/staging-cb05-20260928/backend-contradiction-smoke.sh

# 2. re-take the conversation backup, because it will be older than the last message
sqlite3 "$HOME/.local/state/m5phet/chat/chat.sqlite3" \
        ".backup '$HOME/.local/share/m5phet/staging-cb05-20260928/backup-20260928/chat-at-deploy.sqlite3'"

# 3. install this branch over the service's venv (code only; state is untouched)
#    CB05 is the worktree of satoshi/cb05-backend-contradiction-20260928; set it to your own path
CB05="$HOME/Documents/GitHub/.worktrees/m5phet-cb05-20260928"
"$HOME/.local/share/m5phet/chat-venv/bin/python" -m pip install -q --no-deps --force-reinstall \
    "$CB05" "$CB05/evaluation"

# 4. the OWNER restarts, when no session is in progress
systemctl --user restart m5phet-chat

# rollback, if anything is wrong
$HOME/.local/share/m5phet/staging-cb05-20260928/rollback.sh   # then the owner restarts again
```

A restart alone does not install this: the service executes a **non-editable**
copy in `$HOME/.local/share/m5phet/chat-venv`, whose `direct_url.json` names the
`m5phet-chat` worktree at `f532ee5`. Restarting it re-executes `f532ee5`. Step 3
is what installs, step 4 is the owner's, and **no live restart is authorized by
this round**.

## 8. Resources

Every child ran through `$HOME/.local/bin/crispdm-run` at the revision already
deployed; nothing was reinstalled and no declared cap was shrunk to evade a
refusal. All of it ran on the coordinator, which is admissible here because none
of it is a fit or a model import: the heaviest thing is a pure-Python test
suite plus a FastAPI test client, and the classification answers in the smoke
come from a declared non-model fixture and from a local fake transport.

| job | cap | whole-cgroup peak |
|---|---|---|
| test suites (`cb05-suite`, `-suite2`, `-suite3`, `-suite4`) | 3 GiB | ≤ 193 MiB |
| fresh-process backend tests (`cb05-proc`) | 3 GiB | 187 MiB |
| contract smoke, both runs (`cb05-smoke`, `cb05-smoke2`) | 3 GiB | ≤ 171 MiB |
| unit and collection runs (`cb05-unit*`, `-red`, `-collect*`, `-eval-suite`) | 2 GiB | ≤ 20 MiB |
| venv copies and installs (`cb05-venv-*`, `cb05-stage-*`) | 2–3 GiB | ≤ 26 MiB |

Coordinator: 22 GiB available at the start and 22 GiB at the end, nothing
swapped, nothing killed, no job stopped under pressure. **No GPU was used and
there is therefore no device UUID to record**; the preferred external 5090 and
the secondary worker were not touched, and the forecasting execution admitted on
the secondary worker was neither displaced nor inspected. **No remote call of
any kind** was made by this round — not to a worker, not to the owner's
interpreter: the router was not re-measured, so the 190 interpreter calls of
RB04 have no counterpart here.

---

## 9. The report, in the §6 shape

```
CB05 — the contradictory backend refused, the short return corrected, the router evidence scoped
repo/branch/tip: M5PHET · satoshi/cb05-backend-contradiction-20260928 · e14b30f (base d17f377)
files:
  src/m5phet/classification_backend.py   (new) the contract: resolve, validate, declared, unavailable
  src/m5phet/web/engine.py               start-up resolution, validation, catalog block, both doors
  src/m5phet/web/app.py                  the effective block on refused receipts too
  src/m5phet/config.py, config.schema.json   `mode` and `expect_checkpoint` bindable from the JSON file
  tools/start_chat.sh                    the shipped configuration is no longer the contradiction
  tools/measure_route.py                 corpus_scope: prompts x repeats, router, not a benchmark
  tests/test_classification_backend.py   (new) 29 contract tests
  tests/test_backend_startup_process.py  (new) 8 fresh-process cases, positive and negative
  tests/test_route_corpus_scope.py       (new) 9 scoping and freeze tests
  tests/backend_startup_probe.py         (new) the probe the suite AND the smoke both run
  tests/conftest.py                      no test inherits the operator's backend selection
  tests/test_route_reliability.py        the widened scope block asserted by name
  docs/evidence/ROUTE_GENERALIZATION_HELDOUT_2026_09_28/heldout_paraphrases.json  (new, frozen)
  docs/SATOSHI_CB05_BACKEND_AND_CORRECTIONS_2026_09_28.md  (this)
  docs/SATOSHI_RB04_M5PHET_INTEGRATION_2026_09_28.md       (a pointer to §5; its text unchanged)
  README.md                              which classification backend answers, and its refusals
suites (after reinstall): M5PHET 800 passed / 1 skipped · m5phet_evaluation 42 passed / 0
  (news-signal, prediction_provider and m5phet_policy were NOT re-run: nothing in those
   repositories was touched, and their 2026-09-28 numbers stand as RB04 reported them)
acceptance: contract smoke, 8 cases against the staged build — 1 fixture OK · 2 CONTRADICTION exit 3 ·
  3 MODE_REQUIRED exit 3 · 4 WORKER_NOT_BOUND exit 3 · 5 remote worker OK, checkpoint recorded ·
  6 worker failed → refused by name, outputs [] · 7 worker is a fixture → BACKEND_MISMATCH ·
  8 the operator's own chat.env → CONTRADICTION exit 3. No port opened, no live process signalled.
what is NOT done / refused / not measured:
  - NO_NEW_MODEL_MEASUREMENT, again and for this round too: nothing here measures a forecast, a
    policy, a study, a checkpoint or a classifier. The classification answers in the smoke come
    from the declared NON_MODEL_FIXTURE and from a local fake transport;
  - the router was NOT re-measured. The 19 development prompts stand where RB04 left them
    (85/95 with the completion pass off, 79/95 with it on, one scorer, one day), and this round
    changes how that denominator is described, not what it is;
  - the held-out paraphrase set is FROZEN and UNMEASURED. No generalization number exists;
  - the real Laya worker was NOT contacted. Cases 5-7 prove the CONTRACT against a fake transport
    that binds request digests; they do not prove anything about the operator's checkpoint, and
    `M5PHET_CLASSIFICATION_EXPECT_CHECKPOINT` is therefore recommended in the deployment rather
    than filled in by me — I do not have his sealed manifest's digest from a source I verified;
  - the typed-question catalog (`GET /api/tasks/catalog`) still LISTS the classification area when
    the declared backend is unserved. Every run of it refuses by name and nothing answers, but the
    listing itself is not yet gated, and I am recording that rather than implying otherwise;
  - `NO_NEW_MEASUREMENT` in the owner's standing sense, so no closure table is offered: no model
    error was measured here;
  - the deployment is PREPARED and PROPOSED. It is not applied, the live venv was not written to,
    no restart is authorized, and adopting it REQUIRES the owner to repair chat.env first, because
    this build refuses his current one by name.
```

---

Satoshi, successor technical lead.
