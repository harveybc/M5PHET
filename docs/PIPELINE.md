# The pipeline Laya configures (WP18)

`m5phet.pipeline` turns the owner's WP18 design into four recorded decisions and one spec:

| Step | Function | One decision per | Options declared by |
|---|---|---|---|
| 2 | `choose_preprocessing(engine, feature_metrics, catalog)` | feature | `preprocessor.plugins` in **predictor** and **preprocessor** |
| 3b | `confirm_grouping(engine, groups_document)` | the cut `k` | the cuts the grouping job wrote |
| 4 | `choose_extractors(engine, groups_cut, catalog)` | group | `feature_extractor.encoders` in **feature-extractor** |
| 5 | `choose_core(engine, groups_cut, catalog)` | the fused core | `predictor.plugins`, **restricted to the probed multi-branch ones** |
| 6 | `build_pipeline_spec(...)` / `validate_pipeline(...)` | — | — |

Steps 1 and 3 are feature-eng's (`feature_eng_m5phet.metrics`, `feature_eng_m5phet.grouping`); step 7 — fit, score,
closure table — is predictor's and had not run when this page was written.

Everything here is `m5phet.decide`'s definition of a decision: a **state** (a description, never rows), a **declared
option set**, an answer with **uncalibrated probabilities**, written as a content-addressed record. A choice is a
hypothesis. Nothing on this page is a measurement of anything.

## Where the option lists come from

They are **read from the repositories**, never typed. `catalog_preprocessors()`, `catalog_extractors()` and
`catalog_cores()` parse the entry points declared in each checkout's `setup.py` or `pyproject.toml`. They do **not**
import those packages: TensorFlow and the rest are not installed beside M5PHET, and an option list must not depend on
whether an import happens to succeed.

- A **label** is the plugin's own docstring first line, read from its source with `ast` (the class the entry point
  names first, then its module). When the module cannot be read the label is the **key**. Nothing invents a
  description.
- A label longer than `LABEL_MAX_CHARS` (60) is shortened and the record says `label_shortened: true`. That limit is
  not cosmetic: the provider keeps at most 48 tokens of each option and **refuses** the whole question rather than
  truncate one (`TOKEN_BUDGET_EXCEEDED`).
- A checkout that cannot be found contributes **nothing** and lands in `problems` as `REGISTRY_NOT_READ`. An empty
  option list is an honest answer.
- Two registries declaring the **same key** in the same shared group are refused as `DUPLICATE_PLUGIN_KEY`: a choice
  between them could not be executed unambiguously.
- Checkouts are located by walking up from the package and from the working directory, or by
  `M5PHET_PREDICTOR_REPO`, `M5PHET_PREPROCESSOR_REPO`, `M5PHET_FEATURE_EXTRACTOR_REPO`. No path is written into the
  repository.

What that produced on 2026-09-25 (the labels are the repositories' own words, quoted as found):

| Role | Key | Label | From |
|---|---|---|---|
| preprocessing | `default_preprocessor` | Default Preprocessor Plugin | predictor |
| preprocessing | `stl_preprocessor` | *(no docstring found — the key is the label)* | predictor |
| preprocessing | `default_plugin` | Plugin to preprocess the dataset for feature extraction with external… | preprocessor |
| preprocessing | `normalizer` | `1.60139 1.16481` | preprocessor |
| preprocessing | `unbiaser` | Unbiaser Plugin to apply unbiasing methods to the dataset. | preprocessor |
| preprocessing | `trimmer` | *(no docstring found)* | preprocessor |
| preprocessing | `feature_selector` | Feature Selector Plugin to perform feature selection on the dataset… | preprocessor |
| preprocessing | `cleaner` | *(no docstring found)* | preprocessor |
| extractor | `default`, `ann`, `transformer`, `lstm`, `vae` | "An encoder plugin using a convolutional neural network (CNN)…" | feature-extractor |
| extractor | `rnn`, `cnn_signed` | *(module missing — the key is the label)* | feature-extractor |
| extractor | `cnn` | A CNN-based encoder plugin for feature extraction using Keras. | feature-extractor |
| extractor | `vae_small` | Plugin to define and manage a per-step inference network (encoder-like component) | feature-extractor |

Two things in that table are worth reading twice, because they change how the step-2 and step-4 choices should be
read, and neither is something this module may fix by inventing better words:

1. `normalizer`'s class docstring first line in **preprocessor** is `1.60139 1.16481` — a leftover number line. The
   chooser is shown that, because that is what the repository declares.
2. Five of feature-extractor's nine encoders carry the **same copy-pasted docstring**, which describes a CNN
   regardless of whether the plugin is the ANN, the LSTM, the transformer or the VAE. A chooser reading labels cannot
   tell those five apart. Fixing those docstrings in feature-extractor would change what the chooser sees; that work
   belongs to that repository and was not done here.

**The "grouped extractor" WP18 points at is not in feature-extractor.** `grep -rin "group" feature-extractor/app/*.py`
finds only entry-point plumbing, and no branch of that repository carries one. The only implementation of the idea in
the owner's code is `agent-multi/agent_plugins/grouped_features_extractor.py` (with `feature_fusion_plugins/
gated_fusion.py` and `cross_family_attention.py`), the RL observation extractor: it takes one `(B, T, F)` tensor,
selects channels per branch and fuses the per-branch vectors. It is registered in agent-multi's own groups, not in
`feature_extractor.encoders`, so it is **not** offered as an option here. It is the closest existing design to the
fusing core step 5 is missing.

## Step 5: which cores accept several input branches

WP18 forbids assuming. The probe is `predictor/tests/test_wp18_branch_capability.py` (branch
`satoshi/rp132-rp134-20260923`): for every `predictor.plugins` entry point it tries to build a model from two branches
in three shapes (a list of shapes, a tuple of shapes, a mapping of branch name to shape, each with matching arrays),
then a **single-branch control** that tells "refuses branches" apart from "could not be built here at all". A positive
control proves the two-input detector itself works. Run on the CPU
(`CUDA_VISIBLE_DEVICES=""`, `crispdm-run -m 6G -t 900 -n wp18-probe`) with `~/anaconda3/bin/python` — Python 3.12.7,
TensorFlow 2.18.0, tensorflow-probability 0.25.0. (`~/anaconda3/envs/predictor` does not exist on this host; the
conda **base** environment is the one that carries predictor's TensorFlow stack.)

Result, 2026-09-25: **no declared core accepts several input branches.** 26 of 28 entry points are
`SINGLE_BRANCH_ONLY`, 2 are `NOT_PROBED`, 0 are `MULTI_BRANCH`.

| Verdict | Entry points | Why |
|---|---|---|
| `SINGLE_BRANCH_ONLY` (26) | `ann`, `default_predictor`, `lstm`, `mimo`, `n_beats`, `tcn`, `tft`, `transformer`, and every `binary_*` and `direction_*` | each unpacks `input_shape` as one `(window, channels)` pair; the two-branch attempts raise `ValueError: Invalid dtype: tuple` (or `TypeError: can't multiply sequence by non-int` in the N-BEATS family) while the one-tensor control builds a one-input model |
| `NOT_PROBED` (2) | `cnn` | its `build_model` exits (`SystemExit: 1`) when TensorFlow sees no GPU, so neither attempt nor control could run on the CPU; its capability is **unknown**, not single-branch |
| `NOT_PROBED` (2) | `base` | `setup.py` points it at `predictor_plugin.predictor_plugin_base` — a package name that does not exist (the others are `predictor_plugins`); it cannot be imported at all |

The full table, with each plugin's exact reason, is `src/m5phet/branch_capability.json`
(`m5phet.branch_capability.v1`), which `catalog_cores()` reads. A plugin is offered **only** with a `MULTI_BRANCH`
verdict: an unknown capability is not a declared one.

**So step 5's option list is empty**, no decision is asked, and the spec carries
`core: NOT_AVAILABLE_MULTI_BRANCH`. The plan recorded with it — **not part of WP18 steps 2–6** — is: add a fusing core
to predictor as a new `predictor.plugins` entry point (one `Input` per group branch, a declared fusion such as
agent-multi's gated fusion, then the existing multi-horizon Bayesian heads), and re-run the probe. Until then a
WP18 pipeline cannot be fitted end to end, and step 7's closure table cannot have a `laya_chosen` row.

## The states

| Decision | State | Why it is shaped that way |
|---|---|---|
| preprocessing | `feature_eng_m5phet.metrics.decision_payload(sheet, feature)` rendered by `decide.decision_state` | the feature's own measured sheet: stationarity, ACF peaks and decay, missing fraction, distribution, scale, cross-correlation to the target at the declared lags |
| grouping cut | `cuts_state_payload(groups_document)` | every cut in one state, with feature **aliases** (`f1`…) and a legend written once; the three summary numbers of each cut ride in its option label instead of being repeated |
| extractor | `group_state_payload(cut, group)` | that group's members, tightness, dominant stationarity verdict, shared ACF peaks and the grouping job's own summary line |
| core | `core_state_payload(cut, extractor_plan)` | how many branches, of what, with the extractor already chosen for each |

The compaction in the grouping state is not cosmetic either: the first attempt sent all five cuts with full names and
numbers (594 tokens against 333 of room) and the provider refused the whole question as `TOKEN_BUDGET_EXCEEDED` —
by name, as designed, rather than silently truncating it.

## The spec, and what `validate_pipeline` refuses

`m5phet.pipeline.v1` carries the WP06 representation (validated through
`feature_eng_m5phet.representation.validate_spec`), the preprocessing plan, the chosen cut, the extractors, the core,
the declared catalogs and every decision digest. `validate_pipeline(spec, catalogs=…, record_dir=…)` refuses by name:

| Refusal | When |
|---|---|
| `WRONG_SCHEMA`, `MISSING_KEY`, `UNKNOWN_KEY` | the document is not a `m5phet.pipeline.v1` |
| the representation's own refusals | passed through from feature-eng |
| `FEATURE_WITHOUT_PREPROCESSING` | a feature of the pipeline that no decision covers |
| `GROUP_WITHOUT_EXTRACTOR` | a group of the chosen cut that no decision covers |
| `UNKNOWN_PREPROCESSOR` / `UNKNOWN_EXTRACTOR` / `UNKNOWN_CORE` | a plugin key outside the declared lists |
| `DECISION_NOT_ON_DISK` | a digest with no record, or one that does not read back |
| `DECISION_DOES_NOT_MATCH` | a record that chose something other than what the spec claims |
| `EXECUTION_AUTHORIZED` | a spec that claims authority; a pipeline spec authorizes nothing |

With `catalogs=` the keys are checked against the registries **as they are now**; without it, against the lists the
spec itself carries — which is what a later reader of an archived spec has.

## Running it

```bash
set -a; source ~/.config/m5phet/chat.env; set +a
crispdm-run -m 4G -t 900 -n wp18-laya -- python tools/wp18_pipeline.py \
    --metrics feature_metrics.json --groups groups.json --candidates candidates.json \
    --out pipeline_spec.json
```

Records go to `~/.local/state/m5phet/decisions` by default. Each decision takes 10–20 s on the real checkpoint, and
there are `features + 1 + k (+ 1)` of them.

## The household run of 2026-09-25 (uncalibrated choices, not measurements)

Artifacts: the WP18 step 1 metric sheet and step 3 groups document of the household power slice (50 400 rows,
target `Global_active_power`), and the WP06 candidate `short_memory`. Ten decisions against the real Laya checkpoint
`laya-checkpoint:bd12df88…` on the private worker's GPU, written to `~/.local/state/m5phet/decisions`.

**Step 2 — preprocessing, one decision per feature.** All seven features chose `default_plugin` (preprocessor's
"Plugin to preprocess the dataset for feature extraction…"), with the top probability between 0.335
(`Global_reactive_power`) and 0.493 (`Voltage`); `unbiaser` was second everywhere (0.18–0.29). The chooser did not
separate the features: seven different metric sheets produced the same label, so the profile is doing little work
here. That is a fact about the answer, not a defect this module may paper over.

**Step 3b — the cut.** `k=2` at 0.372, over `k=3` 0.213, `k=6` 0.163, `k=4` 0.138, `k=5` 0.114 — against the
deterministic recommendation `k=3` (the highest silhouette). The chooser is free to differ from it, and it did.

**Step 4 — the extractor per group.** `g1` (`Global_active_power, Global_intensity, Global_reactive_power,
Sub_metering_1, Sub_metering_3, Voltage`) → `rnn` 0.297 over `lstm` 0.222; `g2` (`Sub_metering_2`) → `lstm` 0.324
over `default` 0.192.

> `rnn` is a **declared but broken** entry point: feature-extractor's `setup.py` registers
> `app.plugins.encoder_plugin_rnn` and that module does not exist in the checkout (its README lists `rnn` and
> `cnn_signed` as broken for exactly this reason). The catalog now reports it as `MODULE_NOT_FOUND` and keeps it as
> an option, because the repository declares it. The consequence is concrete: the `g1` branch of this spec cannot be
> built until feature-extractor ships that module or removes the entry point.

**Step 5 — the core.** Nothing was asked: the probe found no `predictor.plugins` plugin that accepts several input
branches, so the option list was empty and the spec carries `core: NOT_AVAILABLE_MULTI_BRANCH` with the plan above.

**Step 6 — the spec.** `m5phet.pipeline.v1` with representation `short_memory`
(`representation_id e74ec65c0372…`), 7 preprocessing decisions, 1 grouping decision, 2 extractor decisions, no core;
it validates against the live catalogs and against the records on disk.

None of this was fitted, scored or compared. WP18 step 7 — `baseline_hand` / `laya_chosen` / `searched` on the same
sealed holdout, with the owner's closure table — has **not** run, and it cannot run for `laya_chosen` while the core
is `NOT_AVAILABLE_MULTI_BRANCH`.
