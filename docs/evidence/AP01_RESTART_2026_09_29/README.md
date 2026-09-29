# AP01 — evidence, 2026-09-29

Taken by Satoshi, successor technical lead, for `docs/SATOSHI_M5PHET_NEXT_ACCEPTANCE_2026_09_29.md`.

Every instance here was **mine**: my own ports (`8771`, `8772`, `8773`), my own state directories under
`$HOME/.local/share/m5phet/staging-ap-20260929/`, my own non-editable venv copy of the operator's provider set. The
owner's service on `8765` served throughout, was **never signalled**, and its venv was never written to. No
conversation store was overwritten or migrated. `CUDA_VISIBLE_DEVICES=""` everywhere; no GPU, no other host.

| file | what it is |
|---|---|
| `before-gate.json` | the **probe before any change**, on the staged build of `b6cbc5d`. `GET /api/catalog` says `CLASSIFICATION_CHECKPOINT_MISMATCH, validated: false`; `GET /api/tasks/catalog` offers `classification` with `choice`. Identical after a restart, so the hole was the steady state and not a start-up transient |
| `after-real.json` | the acceptance case end to end with a **real provider** — `predictor_forecast`, `0.5412255525588989 kW` — persisted, restarted, digests unchanged, and an interrupted request named and then re-run to `OK` |
| `after-gate.json` | the same four properties with the classification backend **unserved**: `question_types: []` and the refusal's code on its face, before and after the restart, while forecasting keeps answering |
| `verify_families.json` | the work plan's sentence harness: 12/12 examples, 5/5 families, 14/14 prose, 2/2 refusals |
| `verify_envelopes.json` | the work plan's envelope harness: 15/15 questions. Its classification entry is the record of §3.4 working — the **declared fixture** answered and its `quality` is `NOT_MEASURED` with `QUALITY_RECORD_IS_NOT_OF_THE_ANSWERING_PATH`, the retained record named and not quoted |
| `run-workbench.sh` | the launcher, with its two cases. It repairs the operator's contradictory `chat.env` **in its own environment only**; his file was not edited |
| `fake-ssh-describe.py` | a local stand-in for `ssh` that answers `describe` and nothing else. A **transport fake**, never a classification provider: no packet leaves this machine, no real worker is contacted, and nothing it returns is a model's answer |
| `acceptance-harnesses.sh` | the driver for the two standing harnesses, on its own port, under its own `timeout` so it cannot leak a listener |
| `SHA256SUMS` | digests of everything above |

**What none of this is.** No model was measured: `NO_NEW_MODEL_MEASUREMENT`. The real Laya weights were **not
contacted** — their host is ineligible this round — so classification answered either from the declared
`NON_MODEL_FIXTURE` or from the local transport fake, and both are labelled as such wherever they appear. The router
was not re-measured and the frozen held-apart paraphrase set was not opened.

**Host names.** Checked structurally — every string value of the parsed JSON — rather than by raw substring search,
because the operator's worker name collides with unrelated identifiers. No string value in any file here carries it;
the worker is `worker.invalid` throughout, which is a reserved name belonging to nobody.
