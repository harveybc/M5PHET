# The pipeline Laya configures (WP18)

`m5phet.pipeline` turns the owner's WP18 design into recorded choices and one spec:

| Step | Function | One decision per | Options declared by |
|---|---|---|---|
| 2 | `choose_preprocessing(engine, feature_metrics, catalog)` | feature | `preprocessor.plugins` in **predictor** and **preprocessor** |
| 3b | `confirm_grouping(engine, groups_document)` | the cut `k` | the cuts the grouping job wrote |
| 4 | `choose_extractors(engine, groups_cut, catalog)` | group | `feature_extractor.encoders` in **feature-extractor** |
| 5 | `choose_core(engine, groups_cut, catalog)` | the fused core, **when there is more than one candidate** | `predictor.plugins`, restricted to the probed multi-branch ones |
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
  names first, then its module). When the module has no docstring the label is the **key**. Nothing invents a
  description.
- A label longer than `LABEL_MAX_CHARS` (60) is shortened and the record says `label_shortened: true`. That limit is
  not cosmetic: the provider keeps at most 48 tokens of each option and **refuses** the whole question rather than
  truncate one (`TOKEN_BUDGET_EXCEEDED`).
- Every declared plugin lands in exactly one of three buckets, and none is silently dropped:
  **`options`** (offered to the chooser), **`not_offerable`** (declared, but its module is not in the checkout — a
  choice that landed on it could not be executed, so it is never put in front of the chooser; also reported as
  `MODULE_NOT_FOUND` in `problems`), **`excluded`** (readable, filtered out by the role's own rule — for cores, the
  branch-capability verdict).
- A checkout that cannot be found contributes **nothing** and lands in `problems` as `REGISTRY_NOT_READ`. An empty
  option list is an honest answer.
- Two registries declaring the **same key** in the same shared group are refused as `DUPLICATE_PLUGIN_KEY` — whether
  or not the modules exist — because a choice between them could not be executed unambiguously.
- Checkouts are located by walking up from the package and from the working directory, or by
  `M5PHET_PREDICTOR_REPO`, `M5PHET_PREPROCESSOR_REPO`, `M5PHET_FEATURE_EXTRACTOR_REPO`. No path is written into the
  repository.

What that produced on 2026-09-25 after WP25's registry hygiene (preprocessor and feature-extractor on
`satoshi/wp25-registry-hygiene-20260925`), quoted exactly as the repositories now declare them:

| Role | Key | Label | From |
|---|---|---|---|
| preprocessing | `default_preprocessor` | Default Preprocessor Plugin | predictor |
| preprocessing | `stl_preprocessor` | *(no docstring — the key is the label)* | predictor |
| preprocessing | `default_plugin` | Trims, splits into D1-D6 and normalizes the dataset. | preprocessor |
| preprocessing | `normalizer` | Normalizes the dataset columns by min-max or z-score. | preprocessor |
| preprocessing | `unbiaser` | Unbiaser Plugin to apply unbiasing methods to the dataset. | preprocessor |
| preprocessing | `trimmer` | Removes the listed columns and rows from the dataset. | preprocessor |
| preprocessing | `feature_selector` | Selects features from the dataset by the chosen method. | preprocessor |
| preprocessing | `cleaner` | Cleans the dataset: missing rows and values, or outliers. | preprocessor |
| extractor | `default` / `ann` | Encoder of per-channel Dense branches over the window. | feature-extractor |
| extractor | `rnn` | Encoder of two recurrent layers (SimpleRNN or GRU). | feature-extractor |
| extractor | `transformer` | Encoder: positional encoding, attention, strided Conv1D. | feature-extractor |
| extractor | `lstm` | Encoder: positional encoding, attention, two BiLSTM layers. | feature-extractor |
| extractor | `cnn` | Encoder of two strided Conv1D layers over the input window. | feature-extractor |
| extractor | `vae` | Encoder of two strided Conv1D layers, with no sampling step. | feature-extractor |
| extractor | `vae_small` | Per-step CVAE inference network: mean and log-variance. | feature-extractor |
| core | `fused_branches` | Multi-branch core: one encoder per feature group, fused. | predictor (WP24) |

Before WP25 those labels were worse in ways that mattered to a chooser reading them: five of nine encoders shared one
copy-pasted docstring describing a CNN whatever the plugin was, `rnn` and `cnn_signed` were registered with no module
at all, and preprocessor's `normalizer` docstring first line was the leftover text `1.60139 1.16481`. `cnn_signed` is
gone from the registry, `rnn` has a module, and each label is now one truthful sentence. `base` in predictor's
`predictor.plugins` is still `not_offerable`: it names the package `predictor_plugin`, which does not exist (the
others are `predictor_plugins`).

**The "grouped extractor" WP18 points at is not in feature-extractor.** The only implementation of that idea in the
owner's code is `agent-multi/agent_plugins/grouped_features_extractor.py` (with `feature_fusion_plugins/
gated_fusion.py` and `cross_family_attention.py`), the RL observation extractor: it takes one `(B, T, F)` tensor,
selects channels per branch and fuses the per-branch vectors. It is registered in agent-multi's own groups, not in
`feature_extractor.encoders`, so it is **not** an option here. It is the closest earlier design to WP24's fusing core.

## Step 5: the core over the fused branches

WP18 forbids assuming which cores accept several input branches. The probe is
`predictor/tests/test_wp18_branch_capability.py`: for every `predictor.plugins` entry point it tries to build a model
from two branches in three shapes (a list of shapes, a tuple of shapes, a mapping of branch name to shape, each with
matching arrays), then a **single-branch control** that tells "refuses branches" apart from "could not be built here
at all". A positive control proves the two-input detector itself works. CPU only, under `crispdm-run`.

Its table is `src/m5phet/branch_capability.json` (`m5phet.branch_capability.v1`), which `catalog_cores()` reads. A
plugin is offered **only** with a `MULTI_BRANCH` verdict: an unknown capability is not a declared one.

- **First probe (before WP24):** 26 of 28 entry points `SINGLE_BRANCH_ONLY` — each unpacks `input_shape` as one
  `(window, channels)` pair — 2 `NOT_PROBED` (`cnn` exits with `SystemExit: 1` when TensorFlow sees no GPU; `base`
  cannot be imported at all), **0 `MULTI_BRANCH`**. That is why the WP18 spec could carry
  `core: NOT_AVAILABLE_MULTI_BRANCH`, and why WP24 exists.
- **After WP24:** `predictor_plugins.fused_branches` is declared and probed `MULTI_BRANCH` ("built a model with 2
  inputs from a list_of_shapes of two branches"). It is now the **only** multi-branch core.

`choose_core` therefore has three cases, and each says plainly how the core got there, because only one of them is a
decision:

| Candidates | What happens | `chosen_by` | `decision` |
|---|---|---|---|
| none | nothing is asked; the spec carries `core: NOT_AVAILABLE_MULTI_BRANCH` and the plan to add one | `NOT_AVAILABLE` | `null` |
| exactly one | nothing is asked | `ONLY_CANDIDATE` | `null` |
| two or more | one decision, asked and recorded like the others | `LAYA_DECISION` | the record's digest |

The single-candidate case is not a workaround for `decide`'s refusal — `decide` is right to refuse an option list of
one (`MALFORMED_OPTIONS`: one option is not a choice, it is an instruction). Sending one option and recording the
answer would manufacture a decision out of a foregone conclusion, with probabilities that mean nothing. The spec says
`ONLY_CANDIDATE` and names no decision, because there is none. `validate_pipeline` enforces the same thing from the
other side: `CHOSEN_BY_MISMATCH` if a sole candidate is claimed while several cores are declared, or if a sole
candidate still names a decision.

### The per-branch encoder

`fused_branches` implements its **own** inline encoders; feature-extractor's `feature_extractor.encoders` is a
different namespace, written by different people. Whether one repository's `ann` — per-channel Dense branches over a
window — IS the core's `dense` family is a judgement about two implementations, and no program may make it. So it is
**declared by a person, in the repository that has to execute it**: `predictor_plugins.fused_branches` carries
`EXTRACTOR_FAMILIES`, one entry per feature-extractor key with the reason as a comment, beside the `ENCODERS` it maps
onto. `extractor_families()` reads that table exactly as `inline_encoders()` reads `ENCODERS` — parsed with `ast`,
never imported — and `map_encoders()` uses it:

| Branch outcome | When | What the record carries |
|---|---|---|
| `MAPPED` | the table sends the chosen extractor to a family the core implements | `encoder`, and `mapped_by: "EXTRACTOR_FAMILIES declared in predictor_plugins.fused_branches"` |
| `NOT_MAPPED` | the table declares `None` for that key | that this is a declaration, not a gap to be filled |
| `NOT_MAPPED` | the table does not carry the key | the key, and the keys it does carry |
| `NOT_MAPPED` | the table names a family the core does not implement | both names |

With no table at all (an older core), the only mapping left is identity — the extractor key IS an inline encoder —
and the record says that is what happened. Nothing is ever mapped by resemblance, and `encoder_mapping` carries the
whole table it used, so a reader of the spec can check the judgement instead of taking it.

## The states

| Decision | State | Why it is shaped that way |
|---|---|---|
| preprocessing | `feature_eng_m5phet.metrics.decision_payload(sheet, feature)` rendered by `decide.decision_state` | the feature's own measured sheet: stationarity, ACF peaks and decay, missing fraction, distribution, scale, cross-correlation to the target at the declared lags |
| grouping cut | `cuts_state_payload(groups_document)` | every cut in one state, with feature **aliases** (`f1`…) and a legend written once; the three summary numbers of each cut ride in its option label instead of being repeated |
| extractor | `group_state_payload(cut, group)` | that group's members, tightness, dominant stationarity verdict, shared ACF peaks and the grouping job's own summary line |
| core | `core_state_payload(cut, extractor_plan)` | how many branches, of what, with the extractor already chosen for each |

The compaction in the grouping state is not cosmetic: the first attempt sent all five cuts with full names and
numbers (594 tokens against 333 of room) and the provider refused the whole question as `TOKEN_BUDGET_EXCEEDED` — by
name, as designed, rather than silently truncating it.

## Replaying recorded decisions

`load_records(dir)` indexes the decision records by `(kind, question, state_sha256)`, and every chooser takes
`replay=`. With it **nothing is asked**: the record made on that exact state answers, and a state with no record is
refused as `NO_RECORD_TO_REPLAY` rather than quietly asked again. This is how an artifact is rebuilt without
spending the checkpoint, and it is bound to the state, not to a file name.

The registries move under records. Replay is loud about it instead of pretending:

- `CHOICE_NO_LONGER_DECLARED` — the recorded choice is not in today's option list; it cannot be carried into a spec.
- `options_changed` on the entry, and `options_changed_since_decision` in the spec — the choice stands, it was made
  among **those** candidates, and both lists stay visible (`added`, `removed`, `labels_differ`).

## The spec, and what `validate_pipeline` refuses

`m5phet.pipeline.v1` carries the WP06 representation (validated through
`feature_eng_m5phet.representation.validate_spec`), the preprocessing plan, the chosen cut, the extractors, the core
with its `chosen_by` and `encoder_mapping`, the declared catalogs and every decision digest.
`validate_pipeline(spec, catalogs=…, record_dir=…)` refuses by name:

| Refusal | When |
|---|---|
| `WRONG_SCHEMA`, `MISSING_KEY`, `UNKNOWN_KEY` | the document is not a `m5phet.pipeline.v1` |
| the representation's own refusals | passed through from feature-eng |
| `FEATURE_WITHOUT_PREPROCESSING` | a feature of the pipeline that no decision covers |
| `GROUP_WITHOUT_EXTRACTOR` | a group of the chosen cut that no decision covers |
| `UNKNOWN_PREPROCESSOR` / `UNKNOWN_EXTRACTOR` / `UNKNOWN_CORE` | a plugin key outside the declared lists |
| `CHOSEN_BY_MISMATCH` | the core's `chosen_by` contradicts the declared list or the decision it names |
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
    --out pipeline_spec.json            # add --replay to rebuild from the records, asking nothing
```

`--replay --reask extractors` asks one step again and replays the rest. That is the move when a registry changed
under one step's records: the choice is then made over the option list that exists now, and every other decision
stays the one that was recorded.

Records go to `~/.local/state/m5phet/decisions` by default. Each decision takes 10–20 s on the real checkpoint, and
there are `features + 1 + k` of them, plus one for the core when more than one core qualifies.

## The household run of 2026-09-25 (uncalibrated choices, not measurements)

Artifacts: the WP18 step 1 metric sheet and step 3 groups document of the household power slice (50 400 rows,
target `Global_active_power`), and the WP06 candidate `short_memory`. Ten decisions against the real Laya checkpoint
`laya-checkpoint:bd12df88…` on the private worker's GPU, written to `~/.local/state/m5phet/decisions`. The spec was
later rebuilt from those same records with `--replay` after WP24 and WP25 landed — no new decisions were asked.

**Step 2 — preprocessing, one decision per feature.** All seven features chose `default_plugin`, with the top
probability between 0.335 (`Global_reactive_power`) and 0.493 (`Voltage`); `unbiaser` was second everywhere
(0.18–0.29). The chooser did not separate the features: seven different metric sheets produced the same label, so the
profile is doing little work here. That is a fact about the answer, not a defect this module may paper over.

**Step 3b — the cut.** `k=2` at 0.372, over `k=3` 0.213, `k=6` 0.163, `k=4` 0.138, `k=5` 0.114 — against the
deterministic recommendation `k=3` (the highest silhouette). The chooser is free to differ from it, and it did.

**Step 4 — the extractor per group.** First asked over the pre-WP25 registry: `g1` → `rnn` 0.297, `g2` → `lstm`
0.324. WP25 then changed that option set (`cnn_signed` removed, every label rewritten), so step 4 was **asked again**
over the registry as it now is — `--replay --reask extractors`, two real decisions, everything else replayed:

| Group | Members | Chosen | Uncalibrated probabilities |
|---|---|---|---|
| `g1` | `Global_active_power, Global_intensity, Global_reactive_power, Sub_metering_1, Sub_metering_3, Voltage` | `default` | default 0.1816, lstm 0.1767, ann 0.1606, cnn 0.1565, vae 0.0985, rnn 0.0951, vae_small 0.0820, transformer 0.0490 |
| `g2` | `Sub_metering_2` | `ann` | ann 0.1759, default 0.1571, cnn 0.1558, lstm 0.1441, vae 0.1252, vae_small 0.1109, rnn 0.0909, transformer 0.0400 |

Both distributions are much flatter than the first pass (top mass 0.18 against 0.30) and both landed on the
Dense-branch encoder. The labels changed, so the question changed; that is what the numbers say and nothing more.

**Step 5 — the core.** `fused_branches`, `chosen_by: ONLY_CANDIDATE`, `decision: null`. Its encoder mapping is
`MAPPED` on both branches, by the core's own declaration: `g1`'s `default` → `dense` and `g2`'s `ann` → `dense`, each
`mapped_by: "EXTRACTOR_FAMILIES declared in predictor_plugins.fused_branches"` ("flattened window through a dense
stack"). Before that table existed both branches were `NOT_MAPPED`, twice over — first `rnn`/`lstm` against a core
with no `rnn` encoder, then `default`/`ann` against a core that had no way to know they are its `dense` family. The
mapping came from a person writing it down, which is the only place it could have come from.

**Step 6 — the spec.** `m5phet.pipeline.v1` with representation `short_memory`
(`representation_id e74ec65c0372…`), 7 preprocessing decisions, 1 grouping decision, 2 extractor decisions, a core
that was the only candidate; it validates against the live catalogs and against the records on disk.

None of this was fitted, scored or compared. WP18 step 7 — `baseline_hand` / `laya_chosen` / `searched` on the same
sealed holdout, with the owner's closure table — has **not** run. It runs on the 5090 only with **the owner's
admission**, and that is where WP18 steps 2–6 stop. The spec is now complete enough to be fitted: a representation, a
preprocessor per feature, a cut, an extractor per group, a core, and every branch mapped to an encoder that core
implements. Whether any of it is any good is the closure table's answer, and the closure table does not exist yet.
