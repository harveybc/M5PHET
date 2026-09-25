# Comparing doctoral stages (WP13)

`evaluation/compare_stages.py` turns N evaluation reports — one per doctoral stage: baseline representation, designed
representation, searched representation, calendar-augmented — into **one closure table per area**, with the columns the
owner requires of every closure: model error with its metric and scale, the naive reference **on the same rows**, the
skill, the literature value with its source, and a comparability verdict.

```
python -m evaluation.compare_stages \
    --report 1_baseline=A.json --report 2_designed=B.json --report 3_searched=C.json \
    --out table.json --markdown table.md
```

Run it from the checkout root. `--report` takes `PATH` or `STAGE=PATH` and is repeated once per stage **and area**; a
stage that reports two areas passes two files under the same stage name. Without `--markdown` the table goes to stdout.

## What the generator is allowed to do

It reads. Every number in the table comes out of a report written by `m5phet_evaluation.build_report`
(`m5phet-evaluation-report/1`). The generator derives exactly three things, and says so in the header of every table it
writes:

1. **skill**, and only where the report does not already carry it. The definition is the package's own,
   `1 - model_error / naive_error` on the same rows (`m5phet_evaluation.scoring.score_forecast`). When the scorer
   computed it, the table quotes the report field (`skill source` reads `report:metric_sets[0].values.skill_mae`)
   instead of recomputing a second definition beside it;
2. **the comparability verdicts**;
3. **the rank**, among comparable measured stages only.

No reported number is transformed. There is no path by which an accuracy becomes an error rate, or a percentage becomes
a ratio, on the way into a cell.

## The reading per area

| area | metric read (preference order) | orientation | naive reference | skill |
|---|---|---|---|---|
| `forecast` | `mae`, then `rmse` | error (lower is better) | the metric set's `baseline`, which the scorer computed on the same rows | the report's `skill_mae` / `skill_rmse`, else `1 - model/naive` |
| `classification` | `macro_f1`, then `accuracy` | score (higher is better) | `majority_class` on the same scored rows | `NOT_DEFINED`, with the reason |
| `regimes` | — | — | — | `NO_NEW_MEASUREMENT`: `regime_accuracy` is refused |
| `causal` | — | — | — | `NO_NEW_MEASUREMENT`: `causal_accuracy` is refused |
| `policy` | — | — | — | `NO_NEW_MEASUREMENT`: `policy_profitability` is refused |

Two of these deserve their sentence.

**Classification has no skill column here.** The package defines skill against an *error* baseline. Macro-F1 and
accuracy are scores; `1 - model/naive` on a score is not a skill, and turning a score into an error to obtain one would
be the generator inventing a number. So the cell says `NOT_DEFINED` and names the reason. A skill for this area is
obtained by declaring an error metric in the protocol, not by transforming the table.

**Three areas refuse quality outright**, and the table quotes the package's own reason from
`m5phet_evaluation.scoring.REFUSED_METRICS` rather than paraphrasing it: an unsupervised assignment has no ground
truth, a causal estimate has no held-out counterfactual, and a proposed action has no realised return. Those rows are
`NO_NEW_MEASUREMENT`. They are not blanks, because a blank cell in a comparison table reads as a zero or as agreement.

## Comparability

A stage is `COMPARABLE` only against the reference stage (the first measured stage in name order) and only when all of
these match, checked in this order, the first difference winning:

1. **holdout** — `corpus_seal` and `sealed_row_count`. The seal covers the rows *and* their labels, so a relabelled,
   trimmed or re-sealed corpus is a different holdout even when the row count is unchanged;
2. **metric** — the metric key actually read, so an MAE is never ranked against an RMSE;
3. **target** and **horizon** — the report annotations. When neither report carries them *and the seals matched*, they
   are bound by the seal (the same sealed labels are the same target at the same horizon on the same rows) and the
   table says so under the area. When one report carries them and the other does not, that is a difference, because it
   cannot be checked;
4. **naive reference** — the baseline's name, since a skill is a ratio against one declared baseline.

Otherwise the verdict is `NOT_COMPARABLE: <field> differs (...)`, naming the field, and the stage is left `NOT_RANKED`.
A stage with no measurement in an area is `NO_NEW_MEASUREMENT` with its reason — either the refusal above, or "no
report for area X in stage Y", or a report that carries none of the area's declared metrics.

## Annotations a report may carry

`build_report` writes none of these; a producer that knows them adds them, and `compare_stages.annotate()` is the one
writer for both fixtures and real producers, so a fixture cannot drift from the real key names. Each is looked for at
the top level of the report first, then inside any metric set's `values`.

| key | used for |
|---|---|
| `stage` | the stage name, when it is not given on the command line and is not the file name |
| `target` | the comparability check, and the `target` column |
| `horizon` | the comparability check, and the `horizon` column |
| `scale` | the `scale` column beside the metric |
| `literature` | `{"value": …, "metric": …, "scale": …, "source": …}` — value and source are shown; anything absent is `NOT_CARRIED` |

`NOT_CARRIED` is a statement about the report, not about the world: it says this report does not carry the number, not
that no literature value exists.

## Determinism

Areas are sorted by family, stages by name, numbers rendered with **6 decimals, fixed**, and no clock is read — the
generator writes no timestamp of its own and quotes the reports' own `generated_at`. Two runs over the same reports
produce byte-identical Markdown and JSON, which is what makes the table reviewable in a diff.

## Worked example — FIXTURES, not a measurement

**No evaluation report has been produced on this machine yet.** Nothing under `~/.local/state/m5phet`, nothing in this
repository and nothing in the audit evidence carries the `m5phet-evaluation-report/1` schema, so there is no real
doctoral stage to tabulate today. The table below is `NON_MODEL_FIXTURE`: it was generated by the generator, from
reports written by `build_report`, over six invented rows whose errors are exactly 0.50, 0.25, 0.20 and 0.15 kWh
against a last-value naive of 1.00. **None of these numbers describes any M5PHET model.** It is here to show the
shape of the table and to be replaced by the WP06 household-forecast stages the day they are scored.

```markdown
# Doctoral comparison across stages

Generated by `evaluation/compare_stages.py` (`m5phet-evaluation-stage-comparison/1`) from 5 evaluation report(s) (`m5phet-evaluation-report/1`).

Every number below is read from a report. The generator computes exactly three things: the skill column where the report carries none, the comparability verdicts, and the rank. Numbers are rendered with 6 decimals, fixed. Areas are sorted by family, stages by name, and no clock is read, so two runs over the same reports produce the same bytes.

Stages: `1_baseline_representation`, `2_designed_representation`, `3_searched_representation`, `4_calendar_augmented`.

## Area: forecast

| stage | status | metric | scale | target | horizon | model error | naive reference | naive error | skill | skill source | literature value | literature source | comparability | rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1_baseline_representation | MEASURED | mae | kWh (mean absolute error in kWh) | global_active_power | h+1 | 0.500000 | last_value | 1.000000 | 0.500000 | report:metric_sets[0].values.skill_mae | NOT_CARRIED | NOT_CARRIED | COMPARABLE | 4 |
| 2_designed_representation | MEASURED | mae | kWh (mean absolute error in kWh) | global_active_power | h+1 | 0.250000 | last_value | 1.000000 | 0.750000 | report:metric_sets[0].values.skill_mae | NOT_CARRIED | NOT_CARRIED | COMPARABLE | 3 |
| 3_searched_representation | MEASURED | mae | kWh (mean absolute error in kWh) | global_active_power | h+1 | 0.200000 | last_value | 1.000000 | 0.800000 | report:metric_sets[0].values.skill_mae | NOT_CARRIED | NOT_CARRIED | COMPARABLE | 2 |
| 4_calendar_augmented | MEASURED | mae | kWh (mean absolute error in kWh) | global_active_power | h+1 | 0.150000 | last_value | 1.000000 | 0.850000 | report:metric_sets[0].values.skill_mae | 0.310000 | PLACEHOLDER (fixture): no literature value has been read into a report yet | COMPARABLE | 1 |

Comparability is judged against the reference stage `1_baseline_representation` (the first measured stage in name order).

Conditions (no number above can be quoted without them):

- **1_baseline_representation** — 6 of 6 sealed rows scored (declared minimum 3) · labels REALISED_OUTCOME from household-power-holdout-2026-09 · protocol ab27aabb7f6e · seal 6670a26fd6c5 sealed 2026-09-24T00:00:00Z · report written 2026-09-24T00:00:00Z · flags: none · file `1_baseline_representation.json`
- **2_designed_representation** — 6 of 6 sealed rows scored (declared minimum 3) · labels REALISED_OUTCOME from household-power-holdout-2026-09 · protocol ab27aabb7f6e · seal 6670a26fd6c5 sealed 2026-09-24T00:00:00Z · report written 2026-09-24T00:00:00Z · flags: none · file `2_designed_representation.json`
- **3_searched_representation** — 6 of 6 sealed rows scored (declared minimum 3) · labels REALISED_OUTCOME from household-power-holdout-2026-09 · protocol ab27aabb7f6e · seal 6670a26fd6c5 sealed 2026-09-24T00:00:00Z · report written 2026-09-24T00:00:00Z · flags: none · file `3_searched_representation.json`
- **4_calendar_augmented** — 6 of 6 sealed rows scored (declared minimum 3) · labels REALISED_OUTCOME from household-power-holdout-2026-09 · protocol ab27aabb7f6e · seal 6670a26fd6c5 sealed 2026-09-24T00:00:00Z · report written 2026-09-24T00:00:00Z · flags: none · file `4_calendar_augmented.json`

## Area: regimes

**regime_accuracy is refused by this package.** an unsupervised assignment has no ground truth: cluster identities are arbitrary and no row carries a correct regime, so agreement between two assignments is stability, never correctness. If independently produced regime labels exist, declare them in the protocol and score them as a classification

| stage | status | metric | scale | target | horizon | model error | naive reference | naive error | skill | skill source | literature value | literature source | comparability | rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1_baseline_representation | NO_NEW_MEASUREMENT | NOT_CARRIED | NOT_CARRIED | NOT_CARRIED | NOT_CARRIED | NO_NEW_MEASUREMENT | NO_NEW_MEASUREMENT | NO_NEW_MEASUREMENT | NO_NEW_MEASUREMENT | NOT_DEFINED: no measurement | NOT_CARRIED | NOT_CARRIED | NO_NEW_MEASUREMENT | NOT_RANKED |
| 2_designed_representation | NO_NEW_MEASUREMENT | NOT_CARRIED | NOT_CARRIED | NOT_CARRIED | NOT_CARRIED | NO_NEW_MEASUREMENT | NO_NEW_MEASUREMENT | NO_NEW_MEASUREMENT | NO_NEW_MEASUREMENT | NOT_DEFINED: no measurement | NOT_CARRIED | NOT_CARRIED | NO_NEW_MEASUREMENT | NOT_RANKED |
| 3_searched_representation | NO_NEW_MEASUREMENT | NOT_CARRIED | NOT_CARRIED | NOT_CARRIED | NOT_CARRIED | NO_NEW_MEASUREMENT | NO_NEW_MEASUREMENT | NO_NEW_MEASUREMENT | NO_NEW_MEASUREMENT | NOT_DEFINED: no measurement | NOT_CARRIED | NOT_CARRIED | NO_NEW_MEASUREMENT | NOT_RANKED |
| 4_calendar_augmented | NO_NEW_MEASUREMENT | regime_accuracy (refused) | NOT_CARRIED | global_active_power | NOT_CARRIED | NO_NEW_MEASUREMENT | NO_NEW_MEASUREMENT | NO_NEW_MEASUREMENT | NO_NEW_MEASUREMENT | NOT_DEFINED: regime_accuracy is refused by name | NOT_CARRIED | NOT_CARRIED | NO_NEW_MEASUREMENT | NOT_RANKED |

No stage carries a measurement in this area, so there is no reference stage and every row is NO_NEW_MEASUREMENT.

Conditions (no number above can be quoted without them):

- **1_baseline_representation** — NO_NEW_MEASUREMENT: no report for area 'regimes' in stage '1_baseline_representation'
- **2_designed_representation** — NO_NEW_MEASUREMENT: no report for area 'regimes' in stage '2_designed_representation'
- **3_searched_representation** — NO_NEW_MEASUREMENT: no report for area 'regimes' in stage '3_searched_representation'
- **4_calendar_augmented** — 6 of 6 sealed rows scored (declared minimum 3) · labels AUTHOR_WRITTEN_SMOKE from NOT_CARRIED · protocol 82284f813d72 · seal f7ddf0ae10b2 sealed 2026-09-24T00:00:00Z · report written 2026-09-24T00:00:00Z · flags: AUTHOR_WRITTEN_SMOKE · file `4_calendar_augmented_regimes.json` · NO_NEW_MEASUREMENT: regime_accuracy is refused by this package: an unsupervised assignment has no ground truth: cluster identities are arbitrary and no row carries a correct regime, so agreement between two assignments is stability, never correctness. If independently produced regime labels exist, declare them in the protocol and score them as a classification. The report carries what this area can report; none of it is a quality claim.
```

And the case the table exists for — the same two stages after the corpus was re-sealed with one realised value edited.
The second stage looks better (skill 0.642857 against 0.500000) and is refused a rank:

```markdown
| 2_designed_representation | MEASURED | mae | NOT_CARRIED | NOT_CARRIED | NOT_CARRIED | 0.416667 | last_value |
1.166667 | 0.642857 | report:metric_sets[0].values.skill_mae | NOT_CARRIED | NOT_CARRIED | NOT_COMPARABLE: holdout
differs (corpus_seal 187ebf943b43 over 6 sealed rows vs 6670a26fd6c5 over 6 in stage '1_baseline_representation') —
the two stages were not measured on the same rows, and the seal covers the rows and their labels | NOT_RANKED |
```

(one row, wrapped here for width; the generator emits it on a single line.)

## Tests

`tests/test_compare_stages.py`, run with the rest of the suite:

```
crispdm-run -m 4G -t 600 -n wp13 -- ~/.local/share/m5phet/chat-venv/bin/python -m pytest tests evaluation -q -p no:cacheprovider
```

The fixtures there are built by `build_report`, never typed, so a change to the report format breaks the tests rather
than leaving the generator reading fields that no longer exist.
