"""WP18: the forecasting pipeline Laya configures, one declared choice at a time.

The owner's design (work plan 2026-09-24, revision 2, WP18) is a pipeline in the order the data flows: a metric sheet
per feature, a preprocessing choice per feature, a grouping of the features, an extractor per group, a core over the
fused branches. Every one of those choices is made the way `m5phet.decide` defines a decision -- a state, a DECLARED
option set, a recorded answer with uncalibrated probabilities -- and the whole thing is then written as one
`m5phet.pipeline.v1` spec that a fit job can run and a closure table can judge.

Two properties carry the whole module.

**The options are read from the repositories, never typed from memory.** `catalog_preprocessors`,
`catalog_extractors` and `catalog_cores` parse the entry points the repositories DECLARE -- predictor's and
preprocessor's `preprocessor.plugins`, feature-extractor's `feature_extractor.encoders`, predictor's
`predictor.plugins` -- out of their `setup.py` / `pyproject.toml` text. They do not import those packages: their
dependencies (TensorFlow among them) are not installed beside M5PHET, and an option list must not depend on whether an
import happens to succeed. A label is the plugin's OWN docstring first line when the module can be read, and the key
itself when it cannot. A registry that cannot be read contributes NOTHING and says so in `problems`: an empty list is
an honest answer, an invented plugin is not.

**A choice is a hypothesis, never a measurement.** Nothing here fits anything or claims a number. The probabilities
are Laya's uncalibrated head outputs for one wording of one question, copied verbatim by `decide`. Whether the chosen
pipeline is any good is decided by WP18 step 7 (the fit) and the closure table, which are not in this module and had
not run when it was written.

The core is the one step that can come back empty. WP18 says the core options are only those `predictor.plugins` that
accept SEVERAL INPUT BRANCHES, tested first. That test is `predictor/tests/test_wp18_branch_capability.py`; its
verdict table is `branch_capability.json`, which `catalog_cores` reads. When no plugin qualifies, the option list is
empty, no decision is asked, and the spec carries `core: NOT_AVAILABLE_MULTI_BRANCH` with the plan to add one.
"""

import ast
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from m5phet import decide

PIPELINE_SCHEMA = "m5phet.pipeline.v1"
CATALOG_SCHEMA = "m5phet.plugin_catalog.v1"
PREPROCESSING_PLAN_SCHEMA = "m5phet.preprocessing_plan.v1"
GROUPING_CHOICE_SCHEMA = "m5phet.grouping_choice.v1"
EXTRACTOR_PLAN_SCHEMA = "m5phet.extractor_plan.v1"
CORE_CHOICE_SCHEMA = "m5phet.core_choice.v1"
BRANCH_CAPABILITY_SCHEMA = "m5phet.branch_capability.v1"

#: what the spec carries as its core when no declared plugin accepts several input branches
NOT_AVAILABLE_MULTI_BRANCH = "NOT_AVAILABLE_MULTI_BRANCH"

#: how a core came to be in the spec. A decision is asked only when there is something to decide.
LAYA_DECISION, ONLY_CANDIDATE, NOT_AVAILABLE = "LAYA_DECISION", "ONLY_CANDIDATE", "NOT_AVAILABLE"
CHOSEN_BY = (LAYA_DECISION, ONLY_CANDIDATE, NOT_AVAILABLE)

#: the per-branch encoder of a group is carried into the spec only when the chosen extractor IS one of the fusing
#: core's own inline encoders; a name that is not one is never translated into one
MAPPED, NOT_MAPPED = "MAPPED", "NOT_MAPPED"
#: the verdict a core must carry in the branch-capability table to be offered as an option
MULTI_BRANCH = "MULTI_BRANCH"

#: the decision kinds and question names this package records under
KIND_PREPROCESSING, QUESTION_PREPROCESSING = "feature_preprocessing", "preprocessing"
KIND_GROUPING, QUESTION_GROUPING = "feature_grouping", "grouping_cut"
KIND_EXTRACTOR, QUESTION_EXTRACTOR = "group_extractor", "extractor"
KIND_CORE, QUESTION_CORE = "fused_core", "core"

#: Laya's sequence budget counts the options and the instructions too, and refuses rather than truncate
#: (`TOKEN_BUDGET_EXCEEDED`). A label is a hint; the key is the identity. So a long docstring line is shortened here,
#: and the record says it was.
LABEL_MAX_CHARS = 60

#: the decimals a grouping state is rendered with: a cut is compared on three correlation summaries, not on their
#: sixth decimal, and a shorter state leaves room under the same budget
GROUPING_STATE_DECIMALS = 3

INSTRUCTIONS_PREPROCESSING = ("Which preprocessing plugin suits this feature, given its measured profile?")
INSTRUCTIONS_GROUPING = ("Which cut of the feature groups should this pipeline use?")
INSTRUCTIONS_EXTRACTOR = ("Which extractor suits this group of features?")
INSTRUCTIONS_CORE = ("Which core should read the fused group branches?")

# --- refusal codes a caller may match on ---------------------------------------------------------------------------
REGISTRY_NOT_READ = "REGISTRY_NOT_READ"
MODULE_NOT_FOUND = "MODULE_NOT_FOUND"
CHOSEN_BY_MISMATCH = "CHOSEN_BY_MISMATCH"
NO_RECORD_TO_REPLAY = "NO_RECORD_TO_REPLAY"
CHOICE_NO_LONGER_DECLARED = "CHOICE_NO_LONGER_DECLARED"
DUPLICATE_PLUGIN_KEY = "DUPLICATE_PLUGIN_KEY"
NO_OPTIONS = "NO_OPTIONS"
NO_MULTI_BRANCH_CORE = "NO_MULTI_BRANCH_CORE"
UNKNOWN_PREPROCESSOR = "UNKNOWN_PREPROCESSOR"
UNKNOWN_EXTRACTOR = "UNKNOWN_EXTRACTOR"
UNKNOWN_CORE = "UNKNOWN_CORE"
FEATURE_WITHOUT_PREPROCESSING = "FEATURE_WITHOUT_PREPROCESSING"
GROUP_WITHOUT_EXTRACTOR = "GROUP_WITHOUT_EXTRACTOR"
DECISION_NOT_ON_DISK = "DECISION_NOT_ON_DISK"
DECISION_DOES_NOT_MATCH = "DECISION_DOES_NOT_MATCH"
FEATURE_ENG_NOT_INSTALLED = "FEATURE_ENG_NOT_INSTALLED"


class PipelineError(ValueError):
    """A catalog, a plan or a spec that cannot be read. Carries the refusal code in `.code`, as feature-eng does."""

    def __init__(self, code, why):
        super().__init__(f"{code}: {why}")
        self.code, self.why = code, why


def _refuse(code, why):
    raise PipelineError(code, why)


# --- where the registries are ----------------------------------------------------------------------------------------

#: the three registries WP18 reads, each with the environment variable that overrides its location. No path is
#: hard-coded: the search walks up from this package and looks for the sibling checkout.
REGISTRIES = {
    "predictor": {"env": "M5PHET_PREDICTOR_REPO", "directory": "predictor"},
    "preprocessor": {"env": "M5PHET_PREPROCESSOR_REPO", "directory": "preprocessor"},
    "feature-extractor": {"env": "M5PHET_FEATURE_EXTRACTOR_REPO", "directory": "feature-extractor"},
}

MANIFESTS = ("setup.py", "pyproject.toml")


def repo_root(registry, *, environ=None, start=None):
    """Where a registry's checkout is, or `None`. The environment wins; otherwise the sibling checkout is looked for.

    The search walks up from this file and from the working directory, so it finds the checkouts both when M5PHET is
    run from a sibling worktree and when it is installed non-editably in a venv somewhere else. Nothing about a
    machine is written into the repository: the variable names are declared here, the paths are not.
    """
    if registry not in REGISTRIES:
        _refuse("UNKNOWN_REGISTRY", f"{registry!r} is not one of {sorted(REGISTRIES)}")
    environ = os.environ if environ is None else environ
    declared = environ.get(REGISTRIES[registry]["env"])
    if declared:
        path = Path(os.path.expanduser(declared))
        return path if any((path / manifest).is_file() for manifest in MANIFESTS) else None
    starts = [Path(start).resolve()] if start is not None else [Path(__file__).resolve(), Path.cwd().resolve()]
    for here in starts:
        for parent in [here, *here.parents]:
            candidate = parent / REGISTRIES[registry]["directory"]
            if any((candidate / manifest).is_file() for manifest in MANIFESTS):
                return candidate
    return None


# --- reading declared entry points ------------------------------------------------------------------------------------

def entry_points_in_setup_py(text, group):
    """The `name=module:attr` entries a `setup.py` declares under `group`, in the order they are written."""
    block = re.search(r"['\"]" + re.escape(group) + r"['\"]\s*:\s*\[(.*?)\]", text, re.S)
    if block is None:
        return []
    entries = []
    for raw in re.findall(r"['\"]([^'\"]+)['\"]", block.group(1)):
        name, sep, target = raw.partition("=")
        if sep and name.strip() and target.strip():
            entries.append((name.strip(), target.strip()))
    return entries


def entry_points_in_pyproject(text, group):
    """The same, for a `[project.entry-points."<group>"]` table. Quoted or bare table names are both read."""
    pattern = (r"^\[project\.entry-points(?:\.\"" + re.escape(group) + r"\"|\." + re.escape(group) +
               r")\]\s*$(.*?)(?=^\[|\Z)")
    block = re.search(pattern, text, re.S | re.M)
    if block is None:
        return []
    entries = []
    for line in block.group(1).splitlines():
        match = re.match(r"\s*([A-Za-z0-9_.\-]+)\s*=\s*['\"]([^'\"]+)['\"]", line)
        if match:
            entries.append((match.group(1).strip(), match.group(2).strip()))
    return entries


def _declared_entries(root, group):
    """What one checkout declares for one group, and from which file. Never imports the package."""
    for manifest, reader in (("setup.py", entry_points_in_setup_py), ("pyproject.toml", entry_points_in_pyproject)):
        path = root / manifest
        if not path.is_file():
            continue
        try:
            entries = reader(path.read_text(encoding="utf-8", errors="replace"), group)
        except OSError as error:
            return [], manifest, f"{type(error).__name__}: {error}"
        if entries:
            return entries, manifest, None
    return [], None, "no manifest in this checkout declares this entry-point group"


def _first_line(text):
    for line in (text or "").strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def _shorten(label):
    """A label over the declared limit is cut on a word boundary and marked; the key is never touched."""
    if len(label) <= LABEL_MAX_CHARS:
        return label, False
    cut = label[:LABEL_MAX_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return (cut or label[:LABEL_MAX_CHARS]) + "...", True


def plugin_label(root, target):
    """The plugin's own docstring first line, read from the source WITHOUT importing it.

    Returns `(label, source, module_found)`: source is `class_docstring`, `module_docstring` or `not_found`, and
    `module_found` says whether the module the entry point names exists in the checkout at all. The class named by the
    entry point is asked first, because that is the plugin; its module is the fallback; when neither can be read the
    caller uses the key, which is always true.

    A declared entry point whose module is MISSING is still an option -- the repository declares it -- but the
    catalog says so, because a choice that lands on it cannot be executed until that repository fixes it.
    """
    module_name, _, attr = target.partition(":")
    relative = Path(*module_name.split("."))
    for candidate in (root / relative.with_suffix(".py"), root / relative / "__init__.py"):
        if not candidate.is_file():
            continue
        try:
            tree = ast.parse(candidate.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            return "", "not_found", True
        if attr:
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == attr:
                    line = _first_line(ast.get_docstring(node) or "")
                    if line:
                        return line, "class_docstring", True
        line = _first_line(ast.get_docstring(tree) or "")
        if line:
            return line, "module_docstring", True
        return "", "not_found", True
    return "", "not_found", False


def _catalog(role, sources, *, environ=None, start=None, keep=None, extra=None):
    """Read every declared source for one role into one catalog document.

    `keep(record) -> bool` filters (the cores are filtered by their probed branch capability). A source that cannot be
    read contributes nothing and lands in `problems`: the option list is then shorter, never invented.

    Three buckets, and a plugin is in exactly one: `options` -- offered to the chooser; `not_offerable` -- declared by
    the registry but impossible to execute (its module is not in the checkout); `excluded` -- readable, but filtered
    out by `keep` (a core that does not accept several branches). Nothing is silently dropped.
    """
    document = {"schema": CATALOG_SCHEMA, "role": role, "sources": [], "options": [], "excluded": [],
                "not_offerable": [], "problems": []}
    seen = {}
    for registry, group in sources:
        root = repo_root(registry, environ=environ, start=start)
        if root is None:
            document["sources"].append({"registry": registry, "group": group, "path": None, "status": "NOT_READ",
                                        "why": "no checkout of this repository was found; set "
                                               f"{REGISTRIES[registry]['env']} to declare where it is"})
            document["problems"].append(f"{REGISTRY_NOT_READ}: {registry} ({group})")
            continue
        entries, manifest, why = _declared_entries(root, group)
        document["sources"].append({"registry": registry, "group": group, "path": manifest,
                                    "status": "READ" if entries else "NOT_READ",
                                    "why": why or f"{len(entries)} entry points declared"})
        if not entries:
            document["problems"].append(f"{REGISTRY_NOT_READ}: {registry} ({group}): {why}")
            continue
        for key, target in entries:
            label, label_source, module_found = plugin_label(root, target)
            label, shortened = _shorten(label or key)
            record = {"key": key, "label": label, "label_source": label_source if label_source != "not_found" else "key",
                      "label_shortened": shortened, "module_found": module_found, "registry": registry,
                      "group": group, "target": target}
            if key in seen:
                _refuse(DUPLICATE_PLUGIN_KEY,
                        f"{key!r} is declared by {seen[key]} and again by {registry}; the two registries share the "
                        f"entry-point group {group!r}, so a choice between them could not be executed unambiguously")
            seen[key] = registry
            if not module_found:
                # Declared, and not offerable: a choice that landed on it could not be executed, so it is never put
                # in front of the chooser. It is kept here by name -- the registry declares it and this says why it
                # cannot be used -- rather than dropped, which would hide a broken registration.
                record["why"] = f"{registry} declares {key!r} as {target} and that module is not in the checkout"
                document["not_offerable"].append(record)
                document["problems"].append(f"{MODULE_NOT_FOUND}: {registry} declares {key!r} as {target} and that "
                                            f"module is not in the checkout; it is NOT offered as an option, because "
                                            f"a choice that landed on it could not be executed")
                continue
            if keep is not None and not keep(record):
                document["excluded"].append(record)
                continue
            document["options"].append(record)
    if extra:
        document.update(extra)
    if not document["options"]:
        document["problems"].append(f"{NO_OPTIONS}: nothing was declared for {role}, so nothing can be chosen")
    return document


def catalog_preprocessors(*, environ=None, start=None):
    """The preprocessors the repositories DECLARE, as options: predictor's and preprocessor's `preprocessor.plugins`.

    The group is shared between the two checkouts on purpose (predictor's AGENTS.md warns that co-installing them
    mixes it). Here that sharing is read, not resolved: two registries declaring the same key is refused as
    `DUPLICATE_PLUGIN_KEY`, because a decision between them could not be executed.
    """
    return _catalog("preprocessing", [("predictor", "preprocessor.plugins"), ("preprocessor", "preprocessor.plugins")],
                    environ=environ, start=start)


def catalog_extractors(*, environ=None, start=None):
    """The extractors feature-extractor DECLARES: its `feature_extractor.encoders` entry points.

    An encoder is the half of that repository's autoencoder that produces a representation of a window, which is what
    WP18's per-group extractor is. Its decoders are declared in their own group and are not options here.

    The "grouped extractor" WP18 points at as a starting point is NOT in this repository: the only implementation of
    that idea in the owner's code is `agent-multi/agent_plugins/grouped_features_extractor.py`, the RL branch
    extractor, which is not registered in any entry-point group and is therefore not offered as an option. `docs/
    PIPELINE.md` records where it is and what it does.
    """
    return _catalog("extractor", [("feature-extractor", "feature_extractor.encoders")], environ=environ, start=start)


def branch_capability(path=None):
    """The probed verdict table: which `predictor.plugins` accept several input branches.

    Produced by `predictor/tests/test_wp18_branch_capability.py`, never by this module: whether a Keras model can be
    built from two branches is a measurement, and it is made where the plugins and TensorFlow live.
    """
    path = Path(path) if path is not None else Path(__file__).with_name("branch_capability.json")
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        _refuse("BRANCH_CAPABILITY_UNREADABLE", f"{path} is not a readable capability table: {error}")
    if document.get("schema") != BRANCH_CAPABILITY_SCHEMA:
        _refuse("WRONG_SCHEMA", f"{path} carries schema {document.get('schema')!r}, not {BRANCH_CAPABILITY_SCHEMA!r}")
    return document


def catalog_cores(*, environ=None, start=None, capability_path=None):
    """The cores that may be chosen: `predictor.plugins` RESTRICTED to the probed `MULTI_BRANCH` ones.

    With no capability table the restriction cannot be applied, so the option list is empty and `problems` says why --
    a core that has not been shown to accept branches is not offered on the assumption that it might.
    """
    table = branch_capability(capability_path)
    verdicts = (table or {}).get("verdicts", {})
    catalog = _catalog("core", [("predictor", "predictor.plugins")], environ=environ, start=start,
                       keep=lambda record: verdicts.get(record["key"], {}).get("verdict") == MULTI_BRANCH,
                       extra={"capability": {"probed": table is not None,
                                             "probed_on": (table or {}).get("probed_on"),
                                             "probe": (table or {}).get("probe"),
                                             "source": "predictor/tests/test_wp18_branch_capability.py"}})
    for record in catalog["excluded"]:
        record["verdict"] = verdicts.get(record["key"], {}).get("verdict", "NOT_PROBED")
        record["why"] = verdicts.get(record["key"], {}).get("why", "this plugin does not appear in the probed table")
    if table is None:
        catalog["problems"].insert(0, f"{NO_MULTI_BRANCH_CORE}: no branch-capability table was found, so no core can "
                                      f"be declared to accept several input branches")
    elif not catalog["options"]:
        catalog["problems"].insert(0, f"{NO_MULTI_BRANCH_CORE}: the probe of {catalog['sources'][0]['registry']} on "
                                      f"{table.get('probed_on')} found no plugin that builds a model with two or more "
                                      f"inputs from two branches")
    return catalog


def options(catalog):
    """`[[key, label], ...]` for `decide.ask`, from a catalog document or from an already-made pair list."""
    if isinstance(catalog, dict):
        return [[record["key"], record["label"]] for record in catalog.get("options", [])]
    return [[str(key), str(label)] for key, label in catalog]


# --- the choices ------------------------------------------------------------------------------------------------------

def _metrics():
    try:
        from feature_eng_m5phet import metrics
    except ImportError as error:
        _refuse(FEATURE_ENG_NOT_INSTALLED, f"the metric sheet's own reader is needed to build a state: {error}")
    return metrics


def _grouping():
    try:
        from feature_eng_m5phet import grouping
    except ImportError as error:
        _refuse(FEATURE_ENG_NOT_INSTALLED, f"the groups document's own reader is needed to build a state: {error}")
    return grouping


def _now(as_of):
    return as_of if as_of is not None else datetime.now(timezone.utc).isoformat()


def load_records(record_dir):
    """Index the decision records in a directory by `(kind, question, state_sha256)`, for replay.

    A decision is bound to the exact state it was made on, so rebuilding an artifact from records is not a matter of
    trusting a file name: the state is rendered again here and only a record made on THAT state answers. Replay
    therefore asks nothing and invents nothing -- and a state that has changed since simply has no record.
    """
    folder = Path(os.path.expanduser(str(record_dir)))
    index = {}
    for path in sorted(folder.glob("*.json")):
        try:
            record = decide.load(path)
        except decide.DecisionError:
            continue                                    # a file that is not a valid record is not a decision
        index[(record["kind"], record["question"], record["state_sha256"])] = (record, str(path))
    return index


def _ask_one(engine, state_text, *, question, instructions, pairs, kind, as_of, record_dir, replay=None):
    """One decision, asked and read back into the shape every plan below records.

    With a `replay` index (`load_records`) nothing is asked: the record already made on this exact state answers, and
    a state with no record is refused as `NO_RECORD_TO_REPLAY` rather than quietly asked again.
    """
    if replay is not None:
        found = replay.get((kind, question, decide.state_sha256(state_text)))
        if found is None:
            return {"status": "REFUSED", "refusal": NO_RECORD_TO_REPLAY,
                    "why": f"no {kind}/{question} decision was recorded on this state",
                    "state_sha256": decide.state_sha256(state_text)}
        record, path = found
        asked = [list(pair) for pair in record["options"]]
        now = [list(pair) for pair in pairs]
        if record["chosen"] not in [pair[0] for pair in now]:
            return {"status": "REFUSED", "refusal": CHOICE_NO_LONGER_DECLARED,
                    "why": f"the recorded choice {record['chosen']!r} is not in the option list the registries "
                           f"declare now ({[pair[0] for pair in now]}); it cannot be carried into a spec",
                    "state_sha256": record["state_sha256"]}
        entry = {"status": "OK", "chosen": record["chosen"], "probabilities": record["probabilities"],
                 "probability_decimals": record["probability_decimals"], "checkpoint": record["checkpoint"],
                 "state_sha256": record["state_sha256"], "decision_sha256": decide.decision_sha256(record),
                 "record_path": path, "decision": record, "replayed": True, "options_at_decision": asked}
        if asked != now:
            # The registries moved under the record. The choice still stands -- it was made among THOSE candidates --
            # so it is carried with both lists visible, never rewritten as if it had been made among these ones.
            asked_keys, now_keys = {pair[0] for pair in asked}, {pair[0] for pair in now}
            entry["options_changed"] = {"added": sorted(now_keys - asked_keys),
                                        "removed": sorted(asked_keys - now_keys),
                                        "labels_differ": sorted(key for key in asked_keys & now_keys
                                                                if dict(asked)[key] != dict(now)[key])}
        return entry
    answered = decide.ask(engine, state_text, {question: {"options": pairs, "instructions": instructions}},
                          kind=kind, as_of=as_of, record_dir=record_dir)
    entry = answered[question]
    if entry.get("status") != "OK":
        return {"status": "REFUSED", "refusal": entry.get("refusal"), "why": entry.get("why"),
                "state_sha256": decide.state_sha256(state_text)}
    decision = entry["decision"]
    return {"status": "OK", "chosen": decision["chosen"], "probabilities": decision["probabilities"],
            "probability_decimals": decision["probability_decimals"], "checkpoint": decision["checkpoint"],
            "state_sha256": decision["state_sha256"], "decision_sha256": decide.decision_sha256(decision),
            "record_path": entry.get("record_path"), "decision": decision}


def _no_options(schema, role, catalog):
    return {"schema": schema, "status": "REFUSED", "refusal": NO_OPTIONS,
            "why": f"the registries declared no {role}; nothing was asked, because a choice needs declared options",
            "options": [], "problems": (catalog or {}).get("problems", []) if isinstance(catalog, dict) else []}


def choose_preprocessing(engine, feature_metrics, catalog, *, as_of=None, record_dir=None, features=None,
                         replay=None):
    """Step 2: one decision per feature -- which declared preprocessor suits the profile the metric sheet measured.

    The state is exactly `feature_eng_m5phet.metrics.decision_payload(sheet, feature)` rendered by
    `decide.decision_state`: measurements, no rows, the decimals the sheet declared. One `decide.ask` per feature, so
    one recorded decision per feature and no feature's choice can borrow another's state.
    """
    pairs = options(catalog)
    if not pairs:
        return _no_options(PREPROCESSING_PLAN_SCHEMA, "preprocessor", catalog)
    metrics = _metrics()
    stamped = _now(as_of)
    names = list(features) if features is not None else sorted(feature_metrics["features"])
    plan = {"schema": PREPROCESSING_PLAN_SCHEMA, "status": "OK", "kind": KIND_PREPROCESSING,
            "question": QUESTION_PREPROCESSING, "instructions": INSTRUCTIONS_PREPROCESSING, "options": pairs,
            "source": {"metric_sheet_schema": feature_metrics.get("schema"),
                       "dataset": feature_metrics.get("dataset"), "target": feature_metrics.get("target")},
            "as_of": stamped, "features": {}, "decisions": {},
            "note": "one uncalibrated choice per feature; nothing here was fitted or measured"}
    for feature in names:
        state = decide.decision_state("feature_profile", metrics.decision_payload(feature_metrics, feature))
        entry = _ask_one(engine, state, question=QUESTION_PREPROCESSING, instructions=INSTRUCTIONS_PREPROCESSING,
                         pairs=pairs, kind=KIND_PREPROCESSING, as_of=stamped, record_dir=record_dir, replay=replay)
        plan["features"][feature] = entry
        if entry["status"] == "OK":
            plan["decisions"][feature] = entry["decision_sha256"]
    return plan


def cut_summary(document, k):
    """One line describing a cut: its shape and its three summary numbers.

    Short on purpose. Laya's pinned SDK keeps at most 48 tokens of each option and refuses the whole question rather
    than truncate one (`TOKEN_BUDGET_EXCEEDED`), so the membership of each group lives in the STATE, where it is
    written once with aliases, and the label carries what tells the cuts apart at a glance.
    """
    block = document["cuts"][str(k)]
    sizes = "|".join(str(len(group["members"])) for group in block["groups"])
    return (f"{k} groups, sizes {sizes}; within {block['within_group_mean_abs_correlation']:.3f}, between "
            f"{block['between_group_mean_abs_correlation']:.3f}, silhouette {block['silhouette']:.3f}")


def cuts_state_payload(document):
    """The state for the grouping decision: the composition of every declared cut, with an alias legend.

    All the cuts must fit in ONE state under the provider's sequence budget, because the whole question is which of
    them to use. Three things make them fit without anything being truncated: feature names are long and repeat in
    every cut, so each feature gets an alias (`f1`...) and the legend is written once; each cut's shape is one line;
    and the three summary numbers of a cut (within, between, silhouette) travel in its OPTION LABEL -- which Laya
    reads as part of the question -- instead of being repeated here. Nothing is dropped, only written once.
    """
    features = list(document["features"])
    alias = {feature: f"f{index + 1}" for index, feature in enumerate(features)}
    cuts = {f"k={key}": " | ".join(" ".join(alias[member] for member in group["members"])
                                   for group in document["cuts"][key]["groups"])
            for key in sorted(document["cuts"], key=int)}
    return {"kind": "feature_grouping_cuts",
            "target": document["source"]["target"],
            "legend": {value: key for key, value in alias.items()},
            "distance": document["distance"]["kind"],
            "linkage": document["linkage"]["method"],
            "cuts": cuts,
            "recommended_k": document["recommended_k"]}


def confirm_grouping(engine, groups_document, *, as_of=None, record_dir=None, replay=None):
    """Step 3b: ONE decision among the cuts the grouping job declared (`k = 2..K`), with their summaries as state.

    The cuts are deterministic -- feature-eng computed them from the cross-metrics. What Laya adds is the cut, and only
    from the ones already on the document: `CUT_NOT_IN_DOCUMENT` is impossible here because the options ARE the
    document's cuts.
    """
    grouping = _grouping()
    grouping.validate(groups_document)
    ks = sorted(groups_document["cuts"], key=int)
    pairs = [[f"k={k}", cut_summary(groups_document, k)] for k in ks]
    if len(pairs) < 2:
        return {"schema": GROUPING_CHOICE_SCHEMA, "status": "REFUSED", "refusal": NO_OPTIONS,
                "why": f"this groups document declares {len(pairs)} cut(s); one cut is not a choice", "options": pairs}
    stamped = _now(as_of)
    state = decide.decision_state("feature_grouping_cuts", cuts_state_payload(groups_document),
                                  decimals=GROUPING_STATE_DECIMALS)
    entry = _ask_one(engine, state, question=QUESTION_GROUPING, instructions=INSTRUCTIONS_GROUPING, pairs=pairs,
                     kind=KIND_GROUPING, as_of=stamped, record_dir=record_dir, replay=replay)
    choice = {"schema": GROUPING_CHOICE_SCHEMA, "options": pairs, "as_of": stamped,
              "recommended_k": groups_document["recommended_k"],
              "source": dict(groups_document["source"]), "features": list(groups_document["features"]), **entry}
    if entry["status"] == "OK":
        k = int(entry["chosen"].split("=", 1)[1])
        block = groups_document["cuts"][str(k)]
        choice["cut"] = {"k": k, "groups": [{"group_id": group["group_id"], "members": list(group["members"]),
                                             "within_mean_abs_correlation": group["within_mean_abs_correlation"],
                                             "dominant_stationarity": group["dominant_stationarity"]["verdict"],
                                             "shared_acf_peaks": list(group["shared_acf_peaks"]),
                                             "summary": group["summary"]} for group in block["groups"]],
                         "within_group_mean_abs_correlation": block["within_group_mean_abs_correlation"],
                         "between_group_mean_abs_correlation": block["between_group_mean_abs_correlation"],
                         "silhouette": block["silhouette"],
                         "target": groups_document["source"]["target"],
                         "features": list(groups_document["features"])}
    return choice


def group_state_payload(cut, group):
    """One group as a state: what it holds, how tight it is, what it shares. No rows."""
    return {"kind": "feature_group", "k": cut["k"], "group_id": group["group_id"], "members": list(group["members"]),
            "member_count": len(group["members"]), "target": cut["target"],
            "within_mean_abs_correlation": group["within_mean_abs_correlation"],
            "dominant_stationarity": group["dominant_stationarity"],
            "shared_acf_peaks": list(group["shared_acf_peaks"]),
            "summary": group["summary"]}


def choose_extractors(engine, groups_cut, catalog, *, as_of=None, record_dir=None, replay=None):
    """Step 4: one decision per group -- which declared extractor reads that group's branch."""
    pairs = options(catalog)
    if not pairs:
        return _no_options(EXTRACTOR_PLAN_SCHEMA, "extractor", catalog)
    stamped = _now(as_of)
    plan = {"schema": EXTRACTOR_PLAN_SCHEMA, "status": "OK", "kind": KIND_EXTRACTOR, "question": QUESTION_EXTRACTOR,
            "instructions": INSTRUCTIONS_EXTRACTOR, "options": pairs, "as_of": stamped, "k": groups_cut["k"],
            "groups": {}, "decisions": {},
            "note": "one uncalibrated choice per group; no extractor was fitted here"}
    for group in groups_cut["groups"]:
        state = decide.decision_state("feature_group", group_state_payload(groups_cut, group))
        entry = _ask_one(engine, state, question=QUESTION_EXTRACTOR, instructions=INSTRUCTIONS_EXTRACTOR, pairs=pairs,
                         kind=KIND_EXTRACTOR, as_of=stamped, record_dir=record_dir, replay=replay)
        plan["groups"][group["group_id"]] = entry
        if entry["status"] == "OK":
            plan["decisions"][group["group_id"]] = entry["decision_sha256"]
    return plan


def core_state_payload(groups_cut, extractor_plan=None):
    """The state for the core decision: how many branches it would read, of what, and how tight each one is."""
    chosen = (extractor_plan or {}).get("groups", {})
    return {"kind": "fused_core", "k": groups_cut["k"], "target": groups_cut["target"],
            "branches": [{"group_id": group["group_id"], "members": list(group["members"]),
                          "within_mean_abs_correlation": group["within_mean_abs_correlation"],
                          "extractor": chosen.get(group["group_id"], {}).get("chosen", "NOT_CHOSEN")}
                         for group in groups_cut["groups"]],
            "branch_count": len(groups_cut["groups"])}


def inline_encoders(core_key, catalog, *, environ=None, start=None):
    """The encoders a fusing core implements ITSELF, read from its module's `ENCODERS` mapping without importing it.

    WP18 step 4 chooses an extractor per group from feature-extractor's registry; a fusing core implements its own
    per-branch encoders. The two namespaces are not the same namespace, and this is how the spec finds out what the
    core actually offers instead of assuming the names match.

    Returns `{name: label}`, or `None` when the core's module or its `ENCODERS` mapping cannot be read.
    """
    record = next((option for option in (catalog.get("options", []) if isinstance(catalog, dict) else [])
                   if option["key"] == core_key), None)
    if record is None:
        return None
    root = repo_root(record["registry"], environ=environ, start=start)
    if root is None:
        return None
    module_name = record["target"].partition(":")[0]
    relative = Path(*module_name.split("."))
    for candidate in (root / relative.with_suffix(".py"), root / relative / "__init__.py"):
        if not candidate.is_file():
            continue
        try:
            tree = ast.parse(candidate.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            return None
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else (
                [node.target] if isinstance(node, ast.AnnAssign) else [])
            if any(isinstance(target, ast.Name) and target.id == "ENCODERS" for target in targets):
                value = node.value
                if isinstance(value, ast.Dict):
                    try:
                        return {ast.literal_eval(key): ast.literal_eval(item)
                                for key, item in zip(value.keys, value.values)}
                    except ValueError:
                        return None
        return None
    return None


def map_encoders(core_key, extractor_plan, catalog, *, environ=None, start=None):
    """Which group branches the chosen core can actually encode with the extractor that was chosen for them.

    A branch is `MAPPED` only when the extractor key IS one of the core's own inline encoders -- the same name, not a
    name that looks similar. Anything else is `NOT_MAPPED` and NAMES the extractor: feature-extractor's `rnn` is not
    an inline encoder of `fused_branches`, and translating it into one would be inventing a pipeline nobody chose.
    """
    encoders = inline_encoders(core_key, catalog, environ=environ, start=start)
    branches, chosen = {}, extractor_plan.get("groups", {})
    for group_id, entry in chosen.items():
        extractor = entry.get("chosen") if entry.get("status") == "OK" else None
        if extractor is None:
            branches[group_id] = {"extractor": None, "encoder": None, "status": NOT_MAPPED,
                                  "why": "no extractor was chosen for this group"}
        elif encoders is None:
            branches[group_id] = {"extractor": extractor, "encoder": None, "status": NOT_MAPPED,
                                  "why": f"the inline encoders of {core_key!r} could not be read from its module"}
        elif extractor in encoders:
            branches[group_id] = {"extractor": extractor, "encoder": extractor, "status": MAPPED,
                                  "why": f"{extractor!r} is an inline encoder of {core_key!r}: {encoders[extractor]}"}
        else:
            branches[group_id] = {"extractor": extractor, "encoder": None, "status": NOT_MAPPED,
                                  "why": f"the chosen extractor {extractor!r} is not one of the inline encoders "
                                         f"{sorted(encoders)} of {core_key!r}; no mapping is invented for it"}
    return {"status": MAPPED if branches and all(branch["status"] == MAPPED for branch in branches.values())
                      else NOT_MAPPED,
            "core": core_key,
            "inline_encoders": sorted(encoders) if encoders is not None else None,
            "branches": branches}


def choose_core(engine, groups_cut, catalog, *, as_of=None, record_dir=None, extractor_plan=None, replay=None):
    """Step 5: the core over the fused branches, among cores PROBED to accept several input branches.

    Three cases, and each says plainly how the core got there, because only one of them is a decision:

    **No candidate.** The probe found no plugin that accepts several branches. Nothing is asked and the document
    carries `core: NOT_AVAILABLE_MULTI_BRANCH` with the plan to add one. Asking Laya to pick a single-tensor core for
    a multi-branch pipeline would record a choice that cannot be executed.

    **One candidate.** Nothing is asked either, and `chosen_by` is `ONLY_CANDIDATE`. `decide` refuses an option list
    with fewer than two entries (`MALFORMED_OPTIONS`: one option is not a choice, it is an instruction) and it is
    right to: sending one option and recording the answer would manufacture a decision out of a foregone conclusion,
    with probabilities that mean nothing. The spec then carries `decision: null` -- there is no decision to name.

    **Two or more candidates.** One decision, asked and recorded exactly as the other steps are.
    """
    pairs = options(catalog)
    if not pairs:
        document = _no_options(CORE_CHOICE_SCHEMA, "core that accepts several input branches", catalog)
        document.update(refusal=NO_MULTI_BRANCH_CORE, core=NOT_AVAILABLE_MULTI_BRANCH, chosen_by=NOT_AVAILABLE,
                        why="every declared predictor.plugins core was probed and none builds a Keras model with two "
                            "or more inputs from two branches, so there is nothing to choose between",
                        plan="a fusing core has to be added to predictor as a new predictor.plugins entry point (one "
                             "Input per group branch, a declared fusion, then the existing multi-horizon heads) and "
                             "the probe re-run; that work is not part of WP18 steps 2-6")
        return document
    stamped = _now(as_of)
    if len(pairs) == 1:
        key, label = pairs[0]
        return {"schema": CORE_CHOICE_SCHEMA, "status": "OK", "chosen_by": ONLY_CANDIDATE, "options": pairs,
                "as_of": stamped, "k": groups_cut["k"], "chosen": key, "core": key, "decision_sha256": None,
                "record_path": None, "probabilities": None, "probability_decimals": None, "checkpoint": None,
                "state_sha256": None,
                "why": f"{key!r} ({label}) is the only declared core that the probe found to accept several input "
                       f"branches; no decision was asked, because one option is not a choice"}
    state = decide.decision_state("fused_core", core_state_payload(groups_cut, extractor_plan))
    entry = _ask_one(engine, state, question=QUESTION_CORE, instructions=INSTRUCTIONS_CORE, pairs=pairs,
                     kind=KIND_CORE, as_of=stamped, record_dir=record_dir, replay=replay)
    return {"schema": CORE_CHOICE_SCHEMA, "options": pairs, "as_of": stamped, "k": groups_cut["k"], **entry,
            "chosen_by": LAYA_DECISION if entry["status"] == "OK" else NOT_AVAILABLE,
            "core": entry.get("chosen") if entry["status"] == "OK" else NOT_AVAILABLE_MULTI_BRANCH}


# --- the spec -----------------------------------------------------------------------------------------------------------

SPEC_KEYS = ("schema", "as_of", "representation", "representation_id", "features", "preprocessing", "grouping",
             "extractors", "core", "catalogs", "decisions", "provenance", "fitted", "execution_authorized", "note")


def _spec_entry(entry):
    """One chosen plugin as the spec carries it, saying so when the registry moved under its decision record."""
    spec_entry = {"plugin": entry["chosen"], "decision": entry["decision_sha256"]}
    if entry.get("options_changed"):
        spec_entry["options_changed_since_decision"] = entry["options_changed"]
    return spec_entry


def build_pipeline_spec(representation, preprocessing_plan, grouping_choice, extractor_plan, core_choice, *,
                        catalogs=None, as_of=None):
    """Step 6: one `m5phet.pipeline.v1` document -- the representation, the four choices, and their decision digests.

    Every plugin key in it came from a decision record, and every decision digest names a file on disk. A fit job can
    read this and nothing else; `validate_pipeline` is what reads it back.
    """
    representation_module = _representation()
    representation_module.validate_spec(representation)
    cut = grouping_choice.get("cut") or {}
    features = list(cut.get("features") or grouping_choice.get("features") or sorted(preprocessing_plan["features"]))
    preprocessing = {feature: _spec_entry(entry) for feature, entry in preprocessing_plan.get("features", {}).items()
                     if entry.get("status") == "OK"}
    extractors = {group_id: _spec_entry(entry) for group_id, entry in extractor_plan.get("groups", {}).items()
                  if entry.get("status") == "OK"}
    catalogs = catalogs or {}
    if core_choice.get("status") == "OK":
        core = {"key": core_choice["chosen"], "chosen_by": core_choice.get("chosen_by", LAYA_DECISION),
                "decision": core_choice.get("decision_sha256"), "why": core_choice.get("why"),
                "encoder_mapping": map_encoders(core_choice["chosen"], extractor_plan, catalogs.get("core", {}))}
    else:
        core = {"key": NOT_AVAILABLE_MULTI_BRANCH, "chosen_by": NOT_AVAILABLE, "decision": None,
                "why": core_choice.get("why"), "plan": core_choice.get("plan"),
                "encoder_mapping": {"status": NOT_MAPPED, "core": NOT_AVAILABLE_MULTI_BRANCH,
                                    "inline_encoders": None, "branches": {},
                                    "why": "no core was chosen, so no branch has an encoder"}}
    spec = {
        "schema": PIPELINE_SCHEMA,
        "as_of": _now(as_of),
        "representation": representation,
        "representation_id": representation_module.spec_id(representation),
        "features": features,
        "preprocessing": preprocessing,
        "grouping": {"k": cut.get("k"), "decision": grouping_choice.get("decision_sha256"),
                     "groups": [{"group_id": group["group_id"], "members": list(group["members"])}
                                for group in cut.get("groups", [])]},
        "extractors": extractors,
        "core": core,
        "catalogs": {"preprocessing": [pair[0] for pair in options(catalogs.get("preprocessing", []))] or
                                      [pair[0] for pair in preprocessing_plan.get("options", [])],
                     "extractor": [pair[0] for pair in options(catalogs.get("extractor", []))] or
                                  [pair[0] for pair in extractor_plan.get("options", [])],
                     "core": [pair[0] for pair in options(catalogs.get("core", []))] or
                             [pair[0] for pair in core_choice.get("options", [])]},
        "decisions": sorted({digest for digest in
                             [*preprocessing_plan.get("decisions", {}).values(),
                              grouping_choice.get("decision_sha256"),
                              *extractor_plan.get("decisions", {}).values(),
                              core_choice.get("decision_sha256")] if digest}),
        "provenance": representation.get("provenance", "DEVELOPMENT"),
        "fitted": False,
        "execution_authorized": False,
        "note": "a configuration chosen by an uncalibrated chooser; nothing in it has been fitted, scored or "
                "authorized, and the closure table of WP18 step 7 is what judges it",
    }
    return spec


def _representation():
    try:
        from feature_eng_m5phet import representation
    except ImportError as error:
        _refuse(FEATURE_ENG_NOT_INSTALLED, f"the representation spec's own reader is needed: {error}")
    return representation


def validate_pipeline(spec, *, catalogs=None, record_dir=None):
    """Read one pipeline spec, or refuse it BY NAME. Returns the spec when it is whole.

    What is refused, and under which name:

    `WRONG_SCHEMA`, `MISSING_KEY`, `UNKNOWN_KEY`   the document is not a `m5phet.pipeline.v1`
    the representation's own refusals               passed through from `feature_eng_m5phet.representation`
    `FEATURE_WITHOUT_PREPROCESSING`                 a feature of the pipeline that no decision covers
    `GROUP_WITHOUT_EXTRACTOR`                       a group of the chosen cut that no decision covers
    `UNKNOWN_PREPROCESSOR` / `UNKNOWN_EXTRACTOR` / `UNKNOWN_CORE`   a plugin key outside the declared list
    `CHOSEN_BY_MISMATCH`                            the core's `chosen_by` does not match what the lists show: a
                                                    sole candidate among several declared cores, a sole candidate
                                                    that still names a decision, an unavailable core that claims to
                                                    have been chosen
    `DECISION_NOT_ON_DISK` / `DECISION_DOES_NOT_MATCH`              a digest with no record, or a record that
                                                                    chose something else

    `catalogs` (a mapping of role to catalog document or pair list) checks the keys against the registries as they
    are NOW; without it the spec's own recorded lists are used, which is what a later reader of an archived spec has.
    """
    if not isinstance(spec, dict):
        _refuse("BAD_TYPE", "a pipeline spec is a JSON object")
    if spec.get("schema") != PIPELINE_SCHEMA:
        _refuse("WRONG_SCHEMA", f"schema is {spec.get('schema')!r} and this reader only reads {PIPELINE_SCHEMA!r}")
    missing = [key for key in SPEC_KEYS if key not in spec]
    if missing:
        _refuse("MISSING_KEY", f"the pipeline spec has no {missing}")
    unknown = [key for key in spec if key not in SPEC_KEYS]
    if unknown:
        _refuse("UNKNOWN_KEY", f"the pipeline spec carries {unknown}, which this reader does not declare")
    if spec["execution_authorized"] is not False:
        _refuse("EXECUTION_AUTHORIZED", "a pipeline spec authorizes nothing; `execution_authorized` must be false")

    _representation().validate_spec(spec["representation"])

    declared = {role: [pair[0] for pair in options(catalogs[role])] if catalogs and role in catalogs
                      else list(spec["catalogs"].get(role, []))
                for role in ("preprocessing", "extractor", "core")}

    for feature in spec["features"]:
        if feature not in spec["preprocessing"]:
            _refuse(FEATURE_WITHOUT_PREPROCESSING,
                    f"{feature!r} is a feature of this pipeline and no preprocessing decision covers it")
    for feature, entry in spec["preprocessing"].items():
        if entry["plugin"] not in declared["preprocessing"]:
            _refuse(UNKNOWN_PREPROCESSOR, f"{entry['plugin']!r} (chosen for {feature!r}) is not one of the declared "
                                          f"preprocessors {declared['preprocessing']}")

    groups = spec["grouping"].get("groups", [])
    for group in groups:
        if group["group_id"] not in spec["extractors"]:
            _refuse(GROUP_WITHOUT_EXTRACTOR,
                    f"group {group['group_id']!r} of the chosen cut has no extractor decision")
    for group_id, entry in spec["extractors"].items():
        if entry["plugin"] not in declared["extractor"]:
            _refuse(UNKNOWN_EXTRACTOR, f"{entry['plugin']!r} (chosen for group {group_id!r}) is not one of the "
                                       f"declared extractors {declared['extractor']}")

    core = spec["core"]["key"]
    chosen_by = spec["core"].get("chosen_by")
    if chosen_by not in CHOSEN_BY:
        _refuse(CHOSEN_BY_MISMATCH, f"the core says it was chosen by {chosen_by!r}, which is not one of "
                                    f"{list(CHOSEN_BY)}")
    if core == NOT_AVAILABLE_MULTI_BRANCH:
        if chosen_by != NOT_AVAILABLE:
            _refuse(CHOSEN_BY_MISMATCH, f"a core that is {NOT_AVAILABLE_MULTI_BRANCH} was chosen by nothing, not by "
                                        f"{chosen_by!r}")
    else:
        if core not in declared["core"]:
            _refuse(UNKNOWN_CORE, f"{core!r} is not one of the declared cores {declared['core']}")
        if chosen_by == ONLY_CANDIDATE:
            if len(declared["core"]) != 1:
                _refuse(CHOSEN_BY_MISMATCH, f"the core says it was the only candidate and the declared list holds "
                                            f"{declared['core']}")
            if spec["core"]["decision"] is not None:
                _refuse(CHOSEN_BY_MISMATCH, "a core that was the only candidate names a decision; no decision was "
                                            "asked, so it must carry none")
        elif chosen_by == NOT_AVAILABLE:
            _refuse(CHOSEN_BY_MISMATCH, f"{core!r} is a declared core, so {NOT_AVAILABLE!r} does not describe how it "
                                        f"got here")

    if record_dir is not None:
        folder = Path(os.path.expanduser(str(record_dir)))
        expected = [("preprocessing", feature, entry) for feature, entry in spec["preprocessing"].items()]
        expected += [("extractor", group_id, entry) for group_id, entry in spec["extractors"].items()]
        if core != NOT_AVAILABLE_MULTI_BRANCH and chosen_by == LAYA_DECISION:
            expected.append(("core", "core", {"decision": spec["core"]["decision"], "plugin": core}))
        for role, where, entry in expected:
            _check_record(folder, entry["decision"], entry["plugin"], f"the {role} of {where!r}")
        if spec["grouping"].get("decision"):
            _check_record(folder, spec["grouping"]["decision"], f"k={spec['grouping']['k']}", "the grouping cut")
        for digest in spec["decisions"]:
            if not (folder / f"{digest}.json").is_file():
                _refuse(DECISION_NOT_ON_DISK, f"the spec names decision {digest} and {folder} does not hold it")
    return spec


def _check_record(folder, digest, chosen, where):
    if not digest:
        _refuse(DECISION_NOT_ON_DISK, f"{where} carries no decision digest")
    path = folder / f"{digest}.json"
    if not path.is_file():
        _refuse(DECISION_NOT_ON_DISK, f"{where} names decision {digest} and {folder} does not hold it")
    try:
        record = decide.load(path)
    except decide.DecisionError as error:
        _refuse(DECISION_NOT_ON_DISK, f"{where} names decision {digest}, which does not read back: {error}")
    if record["chosen"] != chosen:
        _refuse(DECISION_DOES_NOT_MATCH,
                f"{where} says {chosen!r} and its decision record {digest} chose {record['chosen']!r}")
