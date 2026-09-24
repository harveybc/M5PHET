# One submission, two linked repositories

Directory form: https://laya-ai.com/submit?type=integration
One source-URL field: use the runnable consumer there; include both repositories
in Description. Do not concatenate two URLs into the URL field. No submission sent.

| Field | Value |
|---|---|
| Type | Integration |
| Project / source URL | https://github.com/harveybc/news-signal |
| Name (optional) | Leave blank |
| Email (optional) | Leave blank, or provide a project contact chosen by the owner |
| Website | Leave the hidden anti-spam field blank |

## Description

M5PHET and news-signal are two complementary open-source repositories:

Typed machine-learning framework: https://github.com/harveybc/M5PHET
Concrete Laya integration: https://github.com/harveybc/news-signal

M5PHET is building a common task interface: text, structured records or time
series plus an explicit task/output schema become validated structured results
through specialized providers. Reuse Laya for typed decisions rather than train
another text classifier. Other designed use cases are hierarchical market-state
representations, multi-horizon forecasts with uncertainty, trading policies and
identified causal effects of economic releases. A point-in-time economic calendar
pipeline supplies schedule, consensus, actuals, revisions and surprise to the
temporal models. Providers reuse existing open-source engines and the DOIN
optimization ecosystem instead of treating every task as text classification.
Currently shipped: the classification result contract, consumed by news-signal.
The common provider runtime and the other use cases are specified, not implemented.

news-signal consumes that contract and wraps local Laya for asset-specific news
relevance, event and financial-tone classification. It includes a pinned SDK
adapter, timestamp and response validation, provenance hashes, an offline demo
and tests. Results are shadow-only. Real-weight financial performance and
calibration remain unmeasured; MT5 demo and Alpaca paper integration are planned.
We welcome review of the integration and provider interfaces. The repositories
include application use cases, input/output contracts and acceptance-test designs.
No universal-model or profitability claim is made.

## Privacy

Do not include the owner's personal biography in this submission. Blank contact
fields avoid additional disclosure; existing repository URLs and Git attribution
remain public. A project name is not a promise of anonymity.
