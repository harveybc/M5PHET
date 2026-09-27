# Direct continuation: RR05 / WP30

Musashi, 2026-09-27. Base: M5PHET master `8a6d1a3`.
Branch: `musashi/route-values-20260927`.
No Satoshi agent was invoked. No production service was restarted or changed.

## Completed

The scorer previously accepted 60.5 as horizon 60, accepted True as horizon 1,
crashed on infinity, ignored a contradictory later question, and ignored a
contradictory target alias. Five failing regression cases were observed before
the repair (22 tests passed). NaN and reversed-question-order controls already
failed safely and were retained as positive controls for rejection.

All explicit occurrences are now compared. Exact integer strings remain
compatible; fractional values are never truncated and boolean values do not
equal integer horizons. Multiple conflicts carry their paths in diagnostics.

A second red test proved that measurement discarded the complete endpoint
outcome. Future measurements now retain it alongside the verdict. The protocol
text used in checkpoint fingerprints changed, preventing reuse of old scored
checkpoints under this version. This does not authenticate remote responses or
establish benchmark independence; it makes subsequent rescoring possible.

POST: 80 passed across `test_route_reliability.py`, `test_orchestrate.py`, and
`test_quality.py`, using the existing chat venv with this worktree's `src` on
PYTHONPATH. This is a source test, not a clean-wheel installation claim.
Each run used `crispdm-run -m 2G -t 120`; no model imports or GPU inference.

Satoshi's uncommitted RR05 tree was inspected without changing it. Its focused
completion, chooser, quality and evaluation tests passed: 78 tests, also admitted
at 2 GiB / 120 s. Those tests do not certify its real-model completion accuracy.

## Retained measurements, not new results

The old complete-corpus report says 81/95 correct, 12 wrong-type, one invalid,
one refusal. Its SHA256 is
`aac62e58938eca2d33314f95a6f8a546827379588e20ea060450456eda4aa8fa`.
It retains zero complete endpoint outcomes. Therefore this work CANNOT rescore
the old result for hidden question/alias conflicts. The historical result remains
under its original scorer; no replacement reliability has been fabricated.

The uncommitted RR05 subset report says 18/20 correct across four selected
sentences; two remaining wrong-type cases concern forecasting plus an interval.
Its SHA256 is
`3d58926cebf39a89bd1f912903ff7f9ddb40f8c07a4d64ccba31359c914ed08f`.
It is neither a complete-corpus rate nor an independently rescored result.

## Resource and programme boundary

The preferred external 5090 was reachable at 37 C and 0 percent utilization.
Its host had 5982 MiB available and memory PSI avg10 zero at inspection. No new
GPU allocation was created. No heavy or combined ML suite ran on the coordinator.
No broker call, holdout read, model fit or external language-model call occurred.

A/B and the nine-cell ECL regime contrast remain closed; no duplicate fits.
The latest RR04 report still requires actual governed public-panel delivery and
Q2 resource/budget reconciliation before a scientific successor. Financial
availability requires missing producer facts, not invented timestamps. Those
reports were used to select an independent authorized product task, not to
declare the entire programme blocked.

## Next executable work

1. Integrate this isolated patch with Satoshi's preserved RR05 completion changes.
2. Recheck the remaining interval under-answer and measure the complete declared
   corpus on an isolated workbench with retained outcomes and an explicit cost
   allocation. Do not reuse old report summaries as complete responses.
3. Rehearse a pinned deployment separately from the active user chat. No production
   adoption is claimed by these tests.

Status: EVALUATOR_REPAIRED_AND_TESTED; NO_NEW_MODEL_MEASUREMENT.
