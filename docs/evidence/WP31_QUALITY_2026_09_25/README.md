# Every answer carries its area's measured quality — WP31, 2026-09-25

The bank had come to know things about itself that a person reading an answer could not see. WP09 scored the
classification checkpoint on 450 independently labelled rows. WP06/WP07 scored a forecast bundle against a naive
baseline on 9 824 sealed rows and measured its interval's coverage. WP19 computed internal indices for a fitted
regime reference. All of it lived in reports, and an answer said nothing about any of it.

Now every area's answer and every area's catalog entry carries a `quality` block, and the `default` and `telegram`
output procedures render exactly one line of it.

## Where each area's number comes from — and where it does not

| area | source | on the verification instance of 2026-09-25 |
|---|---|---|
| classification | the provider's own `capabilities()['quality']` (`news_signal.quality.v1`, named by `NEWS_SIGNAL_QUALITY`) | **MEASURED**: macro-F1 0.37775954555995367, ECE 0.13151177777777778 (UNCALIBRATED), skill 0.2533114546719444, 450 rows, labels INDEPENDENT_LABELS |
| forecasting | an `m5phet-evaluation-report/1` document, accepted only when a configured bundle's manifest names its digest | **MEASURED**: MAE 0.526293524060822 kW, skill 0.12185848183739156 vs `last_value`, interval coverage 0.9259975570032574 at nominal 0.95, 9 824 sealed rows, labels REALISED_OUTCOME — measured on `quantile-household-95-20260925` |
| unsupervised | an `m5phet-evaluation-report/1` document of family `regimes` | **NOT_MEASURED**: the WP19 report measured a reference this instance does not serve, and a measurement of another fitted state is not published as this one's |
| causal | `m5phet_evaluation.scoring.REFUSED_METRICS['causal_accuracy']` | **REFUSED**, with the package's own reason |
| rl | `m5phet_evaluation.scoring.REFUSED_METRICS['policy_profitability']` | **REFUSED**, with the package's own reason |

`forecast_quantile_hand_95_report.json` in this directory is the forecast report, byte-for-byte, sha256
`2a982800e24ce037f035cfbd4b3beddb86f2956f566499204353e144d5fe7d48` — the digest
`~/.local/state/m5phet/forecast-bundles-20260924/quantile-household-95/manifest.json` carries in
`provenance.evaluation.report_sha256`. That binding is what makes the number this engine's rather than a neighbour's.

## Four rules, each of which refuses something

**Nothing is computed here.** Every figure is read from a record somebody else measured. If this code computed one,
the number would have no protocol.

**A number that cannot name its protocol is not published.** A report is accepted only with its version, protocol
digest, corpus seal, counts and label provenance; anything else is `QUALITY_REPORT_UNREADABLE`.

**A measurement of another thing is refused, never borrowed.** A report of another family is
`QUALITY_MEASURED_ON_ANOTHER_FAMILY`. A forecast report no configured bundle references is
`QUALITY_MEASURED_ON_ANOTHER_STATE` — an area may serve three bundles and have a measurement of one, so the block
names the state that was scored (`measured_on`) and the reader is never told the other two were measured.

**Nothing known is said out loud.** `NOT_MEASURED` is the default. Where a bundle records only *where* its numbers
live, the block stays `NOT_MEASURED` and carries the pointer (protocol, seal, report digest) — a pointer is not a
number.

## The guard, and the one place it bites

A quality number is a figure, so the narration guard applies to it. It passes because the **answer carries** the
quality: `response_view` includes the block, so the figures in the line are figures the answers carry. Remove the
block and the same line is refused — `tests/test_quality.py` pins exactly that, and pins that a fabricated MAE is
discarded like any other invention.

One real collision, worth recording rather than smoothing over. `policy_profitability`'s reason ends *"profit belongs
to an execution record from the system that actually traded"*. The guard refuses the words profit, loss and order in
any sentence that does not negate them, and that last clause does not — so the guard refuses the evaluation package's
own wording, correctly by its own rule. The rule was **not** relaxed and the reason was **not** reworded: the line
names the refusal and says the reason travels with the answer in `quality.why`, where it sits verbatim. The causal
reason passes the guard and is rendered in full.

## Rendered, on the run of 2026-09-25 (`tools/verify_outputs.py`, 16/16 faithful)

```
prediccion (point_forecast): values=[0.5412255525588989], unit=kW, scale=original, ...
rango (interval): values=[[0.03167128562927246, 2.82295298576355]], confidence_level=0.95, ...
riesgo: not answered -- no configured bundle that serves 'Global_active_power' can: ...
quality (forecasting): MAE 0.526293524060822 [kW (mean absolute error in kW, 60 steps of 60 s ahead)]; interval
  coverage 0.9259975570032574 at nominal 0.95; skill 0.12185848183739156 vs last_value — 9824 scored rows, measured
  on quantile-household-95-20260925, labels REALISED_OUTCOME, protocol d0ebd9a4bc..., seal 33820b552ddf...
2 answered, 1 refused; nothing here is an instruction to act.
```

(The quality line is one line; it is wrapped here to fit the page.)

```
quality (unsupervised): NOT_MEASURED — no evaluation report is declared for this area, so nothing is known here
  about how well it answers
quality (rl): policy_profitability — this quantity is not measured here and no answer states one; its reason cannot
  be put in a line the narration guard admits, and travels verbatim with this answer in quality.why
quality (causal): causal_accuracy — a causal estimate has no held-out truth: the counterfactual outcome of a row is
  never observed, so there is no row against which the estimate can be scored right or wrong. ...
```

## What this does NOT say

* it is not a new measurement. Nothing was scored for WP31; every number here was measured by WP09, WP06/WP07 or
  WP19 and is quoted from its own report with its own conditions. `NO_NEW_MEASUREMENT`;
* interval coverage is stated only where an interval was returned, and it is a measurement of that interval on those
  9 824 rows — not a guarantee for other rows, and not a confirmation of the nominal level;
* the forecasting number is about **one** of the three configured bundles. An answer from the other two states its
  area's quality as measured on `quantile-household-95-20260925`, which is why the state is named in the line;
* the classification macro-F1 is one task on one corpus (which economy a calendar line names) and is not a general
  claim about that checkpoint;
* nothing here calibrates anything. The classification probabilities remain `UNCALIBRATED`; the ECE says by how much.
