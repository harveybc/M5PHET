# Economic calendar: point-in-time contract

Status: design for the existing economic dataset; its actual resource ID,
columns, license and vintage coverage have NOT been verified in this revision.
Map real columns before implementing a provider. Do not invent a resource or
reinterpret today's revised download as historical point-in-time data.

## Field-level observations

Each observation must identify event/series, reference period, country/currency,
field (schedule, consensus, first actual, previous, revision), value/unit,
source, vintage/version and immutable source bytes. Record source publication
time where known, actual system receipt time, source sequence, and the field's
availability basis. Unknown clocks are null with reason, not zero-latency values.
Store the scheduled release instant separately from these clocks.

Schedule changes and consensus updates are observations too. Preserve cancellation,
postponement and revisions; never overwrite the earlier vintage. A source saying
"previous" may mean a revised previous value: it is usable only when that version
was known. Missing consensus is not zero surprise.

For live replay use an observed availability cutoff incorporating source
publication when evidenced, system receipt and processing readiness. Order events
at equal timestamps by a measured sequence; otherwise apply a conservative
declared exclusion rule. Unknown publication time does not invalidate a trusted
prospective receipt, but unknown historical receipt cannot be fabricated.
Historical source-publication simulations and measured-receipt replays have
different scopes and must never silently share an eligibility label.
Contradictory publication/receipt clocks require a documented clock correction
or refusal, not silently choosing the more convenient timestamp.

## As-of assembly

For each decision and field, select the latest eligible vintage known by the
cutoff, not the final value of that field. A future scheduled release can be a
known feature if its schedule was observed earlier. Actual release values cannot.
An outcome used as a supervised label may be future data, but it must remain
separate from the observation consumed by the model.

Features and their eligibility:

| Feature | Before release | After receipt |
|---|---|---|
| Known event family, currency, schedule version | Yes, if observed | Yes |
| Time until scheduled release | From known schedule, with cancellation state | Not a substitute for measured release time |
| Consensus | Latest eligible estimate; snapshot before actual becomes known | Frozen pre-release version for surprise |
| Actual, revised previous value | No, unless separately published and already received | Only the observed version |
| Surprise | Missing, never zero-filled as if measured | Actual minus eligible pre-release consensus in compatible units |
| Standardized surprise | Missing | Divide by a scale fitted on prior eligible training releases, with support and zero-scale policy |
| Time since receipt, revision/missingness flags | Known where applicable | Known |

Do not set the surprise sign according to a guessed asset impact. A positive
inflation surprise and a positive GDP surprise need not mean the same trade.
Keep units/series identities and let evaluated task-specific components consume
them. Time-to-event and time-since-event channels may use a declared deterministic
encoding or learned encoder; that choice is an experiment, not an ingestion fact.

Freeze expectation before the evidenced first public release, not merely before
our delayed receipt of the actual: a consensus update after public release can
already incorporate the outcome. If that boundary cannot be established, a
locally observed difference is not certified as an economic expectation surprise
and must not enter a causal surprise study under that name.

## Temporal acceptance cases (not yet implemented)

| ID | Counterexample / required outcome |
|---|---|
| CAL01 | Schedule observed yesterday for tomorrow is available; tomorrow's actual is absent |
| CAL02 | Release published now but received later: no actual or surprise before receipt |
| CAL03 | Later consensus update cannot change the frozen pre-release surprise |
| CAL04 | Late revision changes later as-of views, leaves every earlier view unchanged |
| CAL05 | DST ambiguity, mixed units and incomparable periods refuse before tensors |
| CAL06 | Missing consensus or zero historical residual scale gives explicit missing/refusal policy, never inf/zero invention |
| CAL07 | Future perturbation cannot change earlier features; scalers remain train-only |
| CAL08 | Duplicate/reordered arrivals and restart reproduce the same vintage identity and result |
| CAL09 | Unknown historical availability refuses PIT use while retaining archive metadata |
| CAL10 | Actual and revision at equal clock values respect observed sequence or conservative exclusion |
| CAL11 | Simultaneous events remain separate, not one silently selected row; cancellation and schedule updates respected |
| CAL12 | Model computation completes after decision deadline: result is stale, not backdated |

## Ownership and first increment

Store raw and normalized vintages under the existing data-lake provider and
data-gov contract. Feature generation belongs in feature-eng; M5PHET defines
input compatibility and result semantics. news-signal handles textual statements
only; it does not scrape or hallucinate structured economic values.

First increment inventories the actual economic dataset, produces a column/unit/
clock map and implements CAL01-CAL12 on tiny deterministic fixtures plus a bounded
governed sample. If historical vintages are missing, start prospective capture
and use schedule-only information where proven; do not block unrelated training.
Provider rights, calendars and timezone rules must be checked against the selected
source, not inferred from a familiar column name.
