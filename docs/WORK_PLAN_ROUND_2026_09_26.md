# Work plan — round of 2026-09-26

**Author:** Satoshi, successor technical lead
**Date:** 2026-09-26
**Scope:** five items that were prepared, measured or already written, and were waiting on a review that is not coming.

This is an index of a single round. Each item's reasoning and evidence live inside that item's own
return document and evidence directory, not here. Nothing in this file is a measurement.

---

## 1. Why this round exists

Five pieces of work had been finished or frozen and then left unrecorded, each waiting for a countersignature from
a reviewer who has not returned. Waiting had stopped protecting anything: the numbers were already measured, the
plans were already frozen, and in one case a live defect stayed live. The owner instructed on 2026-09-26 that the
successor technical lead decide these himself and stop holding them.

Two things this does **not** change:

- **Nothing is signed for the absent reviewer.** Where a review was missing, the record says a review is missing and
  carries mine instead. No document in this round bears another person's name.
- **No standard of evidence moves.** Every figure is cited from where it was measured, on the rows it was measured on,
  with the naive reference on those same rows. Superseded evidence is never carried forward silently. No record
  claims completeness over a gap; a gap is printed on the face of the record. No real capital, no broker mutation, no
  `execution_authorized: true`. Every GPU run pinned by device UUID and every heavy run memory-capped through the
  guard.

## 2. The five items

| # | Item | What was already true | What this round adds |
|---|---|---|---|
| 1 | RP49–RP56 and RP57–RP64 dispositions | The ranges' work is published; the two reviews were never written, and four modules read that absence as a block | A real adversarial audit of both ranges, recomputing identity from artifacts, ruling on each of the four modules by name |
| 2 | M4 CONFIRMATION screen | Prepared and frozen: 3024 units, 21 eligible slots, 16 contrasts, M2 excluded, census pinned. Never executed | The correction ruling recorded **before** execution, then the screen run on its own frozen terms, verdict published whichever way it falls |
| 3 | E1 record | A partial seal exists with five named gaps | The measured portion recorded completely, with every remaining gap in the record's header |
| 4 | Q2_CONTEXT fits | Prepared, never run | Run bounded on CPU, each with its closure table |
| 5 | Calendar resources in data-gov | Five resources feed the event-study work ungoverned; their clock was measured and is not UTC | Registered with their absences declared as catalog facts, so a future study is refused by the catalog |

## 3. The one ordering rule this round depends on

For item 2 the correction over the 16 contrasts changes from Bonferroni to Holm, which the order's own clause C34.7
permits when declared before execution and not less conservative per-family. Holm controls the same family-wise error
rate and is uniformly at least as powerful, so it qualifies. **The value of that ruling is entirely in its
timestamp**: it is committed to the evidence directory before any fit runs, so it cannot have been chosen after
seeing a result. If the commit order is ever found reversed, the screen is void and must be re-prepared.

## 4. What remains outside my reach

Three things still need the owner, because they are money or accounts and no ruling of mine can substitute:

- a point-in-time consensus feed with an observed publication instant (without it the event studies stay
  `NOT_IDENTIFIED`, and item 5 records exactly that);
- a scheduled refresh of the actuals archive;
- confirmed demo accounts, if execution canaries are ever to run — and they do not run before those exist.

## 5. Where each item reports

Each item publishes its own return in the `docs/audits/work_plan/` directory of the repository it changed, with its
evidence beside it, and leads with its own verdict rather than with a summary. This file is the index only; if it
disagrees with a return, the return governs.

— Satoshi
