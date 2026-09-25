"""WP15: a sentence may name a dataset from the data lake, and only a dataset the catalog already holds.

The owner's picture: instead of pressing `+` and attaching a file, a person writes "usa el dataset de consumo
eléctrico del data lake" and the framework knows which resource that is. The danger in that picture is obvious --
a language model asked "which dataset?" will always produce a name -- so the rule here is the rule of `interpret`:
**the model may only CHOOSE among candidates the catalog declares, never introduce one**. Everything else is
deterministic.

Three things live in this module.

**The catalog** (`m5phet.dataset_catalog.v1`). One entry per resource of the data foundation, built from that
resource's own `DATA.json`: its id (the directory name), a name, a description composed from the manifest's own
words, the columns it declares, how many rows it has, its provenance and whether it is governed. The catalog holds
**descriptions, never rows**. When a manifest does not declare a count, the declared file is opened only to count
its rows and read its column names, and nothing read is kept. `python -m m5phet.datasets index --root <dir>` writes
it to `~/.local/state/m5phet/dataset_catalog.json`.

**The resolution** (`resolve`). In this order and nothing else: an explicit id (in the envelope's state or written
verbatim in the sentence) wins; then deterministic word matching against id, name, description and columns with the
same normalisation `interpret` uses, plus a small alias table; then, and only when more than one candidate remains,
**Laya** is asked to choose AMONG THE CANDIDATE IDS through `m5phet.decide` (the owner's order of 2026-09-25: the
chooser of a configuration is Laya, the first cognitive layer of every area, not the sentence interpreter). The
choice is a `dataset_choice` decision record -- state digest, the option set, the chosen id, the model's own
uncalibrated probabilities, the checkpoint -- written beside the proposal. A label outside the candidates, an answer
from a fixture, an unreachable worker: each refuses `AMBIGUOUS` naming the candidates, and NOTHING falls back to the
interpreter, because a choice made by a non-model establishes nothing. Nothing matched is `NOT_FOUND` naming the
closest three. `source_of_choice` says which of the three settled it, so the person reviewing the proposal reads
whether their own words or Laya chose their data.

**The access**. An ungoverned foundation resource is read from disk when the run needs it. A governed one resolves
and is described exactly like any other -- a person must be able to see that it exists and that it is governed --
but its rows are refused by name: `GOVERNED_ACCESS_NOT_CONFIGURED`, naming the environment variables a successor
must bind to reach data-gov with the owner's identity. This package implements no governed access and holds no
credential; it refuses instead of downgrading to an ungoverned read, which is the failure that would matter.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

CATALOG_SCHEMA = "m5phet.dataset_catalog.v1"

#: where the built catalog is kept, and the variable that overrides it (verification instances and tests use the
#: variable, so the owner's own catalog is never written by a test)
DEFAULT_CATALOG_PATH = "~/.local/state/m5phet/dataset_catalog.json"
PATH_VARIABLE = "M5PHET_DATASET_CATALOG"

#: the data foundation this host carries. A root is scanned one level deep for directories holding a `DATA.json`.
DEFAULT_ROOTS = ("~/.local/state/crispdm-data-foundation",)

#: the manifest file that declares a resource
MANIFEST = "DATA.json"

#: what the resolver may say
OK = "OK"
AMBIGUOUS = "AMBIGUOUS"
NOT_FOUND = "NOT_FOUND"
NOT_ASKED = "NOT_ASKED"

#: how a dataset came to be chosen -- the three sources, and no fourth
EXPLICIT_ID = "EXPLICIT_ID"
QUESTION_TEXT = "QUESTION_TEXT"
LAYA = "LAYA"

#: the kind of decision a dataset choice is recorded under, and where the records are kept
DECISION_KIND = "dataset_choice"
DECISION_INSTRUCTIONS = "Which dataset fits the problem described?"
DEFAULT_RECORD_DIR = "~/.local/state/m5phet/decisions"

#: a panel a fitted experiment wrote is not a table of readings: `Xs` of `e1_household_dev_pilot_v1` is the slice
#: ALREADY standardized by the scaler its own manifest declares (mean 0, sd 1), while its labels stay in kW. Handing
#: it to an engine as raw rows makes that engine scale it a second time, and the first run of this path answered
#: "21.49 kW" for a household whose declared mean is 0.93 kW. So the rows of such a panel are refused by name until
#: WP16's `window_from_rows` -- which loads the bundle's OWN scaler and checks its digest -- exists to read them.
SCALE_REFUSAL = "ROWS_SCALE_NOT_DECLARED"

#: governed access is NOT implemented here. A governed dataset resolves; its rows are refused by this name.
GOVERNED_REFUSAL = "GOVERNED_ACCESS_NOT_CONFIGURED"
#: the variables a successor must bind for data-gov to be reached with the owner's identity. Their VALUES are never
#: read, printed or written by this module; only these names ever appear.
GOVERNED_VARIABLES = ("DATA_GOV_BASE_URL", "DATA_GOV_USER", "DATA_GOV_API_KEY_FILE")

#: how data-gov would be listed if it were reachable. Its installed console script in anaconda is shadowed by another
#: project's top-level `app` package, so the repository's own checkout is the way to run it; and its listing is an
#: HTTP call needing the owner's key, which is why the catalog carries a stub and not an entry.
DATA_GOV_COMMAND = ("PYTHONPATH=<data-gov checkout> python -m app.main --load_config examples/config/default.json "
                    "  # then DataGovClient(base_url=$DATA_GOV_BASE_URL).lakes() / .resources(lake_id)")

#: ordinary phrasings that name one resource. A phrase maps to an ID because two household resources exist on this
#: host -- an ungoverned development pilot and a governed delivery -- and "consumo eléctrico" names the one a person
#: can actually read; the governed one is reached by its id. The table is data: a catalog file may replace it.
DEFAULT_ALIASES = {
    "consumo eléctrico": "e1_household_dev_pilot_v1",
    "consumo electrico": "e1_household_dev_pilot_v1",
    "household power": "e1_household_dev_pilot_v1",
    "household electric power": "e1_household_dev_pilot_v1",
    "electricidad": "e1_household_dev_pilot_v1",
    "potencia del hogar": "e1_household_dev_pilot_v1",
}

#: words that say a person is naming a dataset at all. Without this gate every sentence would be matched against the
#: catalog, and "¿qué tono tiene esta noticia?" would acquire a dataset nobody asked for.
_MENTIONS = ("dataset", "datasets", "data lake", "datalake", "data set", "conjunto de datos", "conjuntos de datos",
             "lago de datos", "del data lake", "catálogo de datos", "catalogo de datos")

#: words too common to identify anything. Matching on them would make every dataset a candidate for every sentence.
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "data", "dataset", "rows", "row", "columns", "column",
    "input", "target", "channel", "schema", "panel", "slice", "window", "horizon", "value", "values", "at", "of",
    "de", "del", "la", "el", "los", "las", "un", "una", "y", "con", "para", "por", "que", "en", "usa", "usar",
    "lake", "closure", "origins", "local", "gives", "are", "block", "v1", "v2", "v3",
}

_WORD = re.compile(r"[a-z0-9]+")


class GovernedAccess(PermissionError):
    """A governed resource whose rows this package will not read. Refused by name, never downgraded."""


class RowsNotUsable(ValueError):
    """Rows this package can count and describe but must not hand to an engine. Refused by name, never guessed."""


class CatalogError(ValueError):
    """A catalog that cannot be read or built as written."""


# --- normalisation: the same as `interpret` ---------------------------------------------------------------------

def _present(lowered, token):
    """Whether `token` appears in an already-lowercased text as a whole word. Identical to `interpret._candidates`."""
    token = str(token).lower()
    if not token:
        return False
    return bool(re.search(r"(?<![a-z0-9_])" + re.escape(token) + r"(?![a-z0-9_])", lowered))


def _words(text):
    """The identifying words of a description: lowercased, split on anything else, stopwords and one-letter dropped."""
    return {w for w in _WORD.findall(str(text).lower()) if len(w) > 2 and w not in _STOPWORDS}


# --- building the catalog ----------------------------------------------------------------------------------------

def _manifest_rows(manifest):
    """The row count the manifest itself declares, or None. `slice_rows` is a [start, end) row range of the panel."""
    for key in ("rows", "n_rows", "row_count"):
        value = manifest.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    span = manifest.get("slice_rows")
    if isinstance(span, list) and len(span) == 2 and all(isinstance(v, int) for v in span) and span[1] >= span[0]:
        return span[1] - span[0]
    value = manifest.get("panel_rows")
    return value if isinstance(value, int) else None


def _manifest_step_seconds(manifest):
    for key in ("step_seconds", "sampling_seconds", "seconds_per_row", "step"):
        value = manifest.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _describe(manifest, columns):
    """A description composed from the manifest's OWN words. Nothing here is written about a dataset from outside it."""
    parts = []
    if manifest.get("schema"):
        parts.append(str(manifest["schema"]))
    if manifest.get("derived_from"):
        parts.append(str(manifest["derived_from"]))
    if manifest.get("data_access"):
        parts.append("data_access " + str(manifest["data_access"]))
    if columns:
        parts.append("input_columns " + ", ".join(str(c) for c in columns))
    channel = manifest.get("target_channel")
    if isinstance(channel, int) and 0 <= channel < len(columns):
        parts.append(f"target_channel {channel} ({columns[channel]})")
    for key in ("window", "horizon", "panel_rows"):
        if isinstance(manifest.get(key), int):
            parts.append(f"{key} {manifest[key]}")
    span = manifest.get("slice_rows")
    if isinstance(span, list) and len(span) == 2:
        parts.append(f"slice_rows {span[0]}-{span[1]}")
    return "; ".join(parts)


def _is_governed(manifest):
    if manifest.get("governed") is True:
        return True
    if "GOVERNED" in str(manifest.get("data_access") or "").upper():
        return True
    return bool(manifest.get("delivery"))


#: which file of a resource directory carries the rows, most declarative first
_DATA_FILES = ("DATA.csv", "DATA.parquet", "DATA.npz", "BLOCK_DATA.npz")


_FILENAME = re.compile(r"[\w.-]+\.(?:npz|csv|parquet)")


def _data_file(folder, manifest):
    """The file this resource's rows live in, as the manifest declares it or as the directory shows it.

    The manifest's own words come first: `derived_from` of the block resources says "BLOCK_DATA.npz at closure", and
    that is where their panel is -- their `DATA.npz` carries the labels and the scaler and no panel at all. Reading
    the conventional name instead would have reported a resource with no rows."""
    declared = (manifest.get("delivery") or {}).get("path") if isinstance(manifest.get("delivery"), dict) else None
    for name in _FILENAME.findall(str(manifest.get("derived_from") or "")):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    for name in _DATA_FILES:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    for pattern in ("*.csv", "*.parquet", "cache/public_panels/*.parquet", "*.npz"):
        found = sorted(folder.glob(pattern))
        if found:
            return found[0]
    if declared and Path(str(declared)).is_file():
        return Path(str(declared))
    return None


def _count_and_columns(path):
    """Open a declared file ONLY to count its rows and read its column names. Not one row is kept or returned."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return 0, []
            return sum(1 for _ in reader), [h.strip() for h in header]
    if suffix == ".npz":
        import numpy
        with numpy.load(path, allow_pickle=False) as archive:
            best = None
            for key in archive.files:
                shape = archive[key].shape
                if len(shape) == 2 and (best is None or shape[0] > best[1]):
                    best = (key, shape[0], shape[1])
            # no panel in this archive: the count is UNKNOWN, which is not the same as zero rows
            return (best[1], []) if best else (None, [])
    if suffix == ".parquet":
        try:
            import pandas
        except ImportError:                                             # pragma: no cover - environment dependent
            raise CatalogError(f"{path.name}: reading parquet needs pandas with a parquet engine; none is installed")
        frame = pandas.read_parquet(path)
        return int(len(frame)), [str(c) for c in frame.columns]
    raise CatalogError(f"{path.name}: this package reads .csv, .parquet and .npz")


def _scale_of(path):
    """What this package can SAY about the units of a resource's stored file, which is not the same as knowing them.

    A `.npz` is a fitted experiment's panel: its arrays are whatever that experiment scaled them to, and the manifest
    declares the scaler it used. A `.csv` or `.parquet` is a table, like a file a person attaches -- undeclared, and
    read exactly as such. Nothing here measures a column; measuring would mean reading rows the catalog must not
    hold."""
    if path is None:
        return "UNKNOWN"
    return "FITTED_EXPERIMENT_PANEL" if path.suffix.lower() == ".npz" else "UNDECLARED"


def resource_entry(folder):
    """One catalog entry from one resource directory, or None when it declares no manifest."""
    folder = Path(folder)
    manifest_path = folder / MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CatalogError(f"{manifest_path}: not a readable JSON manifest ({error})") from None
    if not isinstance(manifest, dict):
        raise CatalogError(f"{manifest_path}: a manifest must be a JSON object")
    columns = [str(c) for c in (manifest.get("input_columns") or []) if isinstance(c, (str, int))]
    rows = _manifest_rows(manifest)
    path = _data_file(folder, manifest)
    if rows is None and path is not None:
        counted, read_columns = _count_and_columns(path)
        rows = counted
        columns = columns or read_columns
    return {"id": folder.name, "name": folder.name.replace("_", " "),
            "description": _describe(manifest, columns), "columns": columns, "rows": rows,
            "scale": _scale_of(path), "step_seconds": _manifest_step_seconds(manifest),
            "provenance": str(manifest.get("data_access") or manifest.get("derived_from") or "FOUNDATION_MANIFEST"),
            "governed": _is_governed(manifest), "source": "foundation",
            "path_or_ref": str(path) if path is not None else str(folder)}


def file_entry(name, data):
    """A file attached to a chat, described in the same shape so one resolver reads one vocabulary."""
    from .orchestrate import dataset_profile
    profile = dataset_profile(data)
    return {"id": "file:" + name, "name": name, "description": f"attached to this question; {profile['kind']}",
            "columns": list(profile.get("columns") or []), "rows": profile.get("rows", 0), "scale": "UNDECLARED",
            "step_seconds": None,
            "provenance": "ATTACHED_BY_THE_PERSON", "governed": False, "source": "file", "path_or_ref": name}


def data_gov_source(available=False):
    """data-gov as a SOURCE, not as entries: its listing is an HTTP call needing the owner's key, so nothing is
    invented about what it holds. The stub says how the owner would reach it."""
    return {"source": "data-gov", "available": bool(available), "command": DATA_GOV_COMMAND,
            "why": ("data-gov's listing (`DataGovClient.lakes()` / `.resources()`) is an HTTP call that needs "
                    f"{', '.join(GOVERNED_VARIABLES)}; none is bound here, so no dataset of it is listed"),
            "variables": list(GOVERNED_VARIABLES)}


def build_catalog(roots=None, *, files=None):
    """Every resource under every root that carries a manifest, described and never read for its content."""
    roots = [Path(os.path.expanduser(str(r))) for r in (roots if roots is not None else DEFAULT_ROOTS)]
    entries, scanned = [], []
    for root in roots:
        scanned.append(str(root))
        if not root.is_dir():
            continue
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            entry = resource_entry(folder)
            if entry is not None:
                entries.append(entry)
    for name, data in (files or {}).items():
        entries.append(file_entry(name, data))
    return {"schema": CATALOG_SCHEMA, "roots": scanned, "datasets": entries,
            "sources": [{"source": "foundation", "available": True, "roots": scanned}, data_gov_source()],
            "aliases": dict(DEFAULT_ALIASES),
            "reading": "descriptions of what exists; no catalog entry carries a row of anybody's data"}


def entry(catalog, dataset_id):
    """One entry by id, or None."""
    for item in (catalog or {}).get("datasets") or []:
        if item.get("id") == dataset_id:
            return item
    return None


# --- persistence --------------------------------------------------------------------------------------------------

def catalog_path(environ=None):
    env = os.environ if environ is None else environ
    named = env.get(PATH_VARIABLE)
    if named:
        return Path(named)
    home = env.get("HOME")
    return Path(home) / DEFAULT_CATALOG_PATH[2:] if home else Path(os.path.expanduser(DEFAULT_CATALOG_PATH))


def write_catalog(catalog, path=None, environ=None):
    target = Path(path) if path is not None else catalog_path(environ)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def load_catalog(path=None, environ=None, configuration=None):
    """The catalog this installation has, or an empty one. A missing catalog is not an error: it means no dataset
    can be named by a sentence yet, and every sentence is then answered exactly as it was before WP15."""
    env = os.environ if environ is None else environ
    if path is None and configuration is not None:
        path = (configuration.datasets or {}).get("catalog")
    target = Path(os.path.expanduser(str(path))) if path is not None else catalog_path(env)
    if not target.is_file():
        return {"schema": CATALOG_SCHEMA, "datasets": [], "sources": [data_gov_source()], "aliases": {},
                "why": f"no dataset catalog at {target}; build one with `python -m m5phet.datasets index`"}
    try:
        catalog = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CatalogError(f"{target}: not a readable dataset catalog ({error})") from None
    if not isinstance(catalog, dict) or catalog.get("schema") != CATALOG_SCHEMA:
        raise CatalogError(f"{target}: schema {(catalog or {}).get('schema')!r} is not {CATALOG_SCHEMA!r}")
    catalog.setdefault("aliases", dict(DEFAULT_ALIASES))
    return catalog


# --- resolution ---------------------------------------------------------------------------------------------------

def mentions_dataset(sentence, catalog=None):
    """Whether these words name a dataset at all: a trigger phrase, a declared alias, or a catalog id written out."""
    lowered = str(sentence or "").lower()
    if any(phrase in lowered for phrase in _MENTIONS):
        return True
    for phrase in (catalog or {}).get("aliases") or DEFAULT_ALIASES:
        if phrase.lower() in lowered:
            return True
    return any(_present(lowered, item.get("id", "")) for item in (catalog or {}).get("datasets") or [])


def _sentence_and_explicit(subject):
    """A sentence, or an envelope state that may carry `dataset: <id>`. Returns (text, explicit id or None)."""
    if isinstance(subject, dict):
        explicit = subject.get("dataset")
        text = " ".join(str(v) for v in subject.values() if isinstance(v, str))
        return text, (explicit if isinstance(explicit, str) and explicit.strip() else None)
    return str(subject or ""), None


def _overlap(item, lowered):
    """How many of this entry's identifying words the sentence says. Deterministic, and the same for every entry."""
    haystack = " ".join([str(item.get("id", "")), str(item.get("name", "")), str(item.get("description", "")),
                         " ".join(str(c) for c in item.get("columns") or [])])
    return sum(1 for word in _words(haystack) if _present(lowered, word))


def _closest(catalog, lowered, how_many=3):
    items = list((catalog or {}).get("datasets") or [])
    ranked = sorted(items, key=lambda i: (-_overlap(i, lowered), str(i.get("id"))))
    return [i["id"] for i in ranked[:how_many]]


def _result(status, dataset=None, candidates=(), why=None, source_of_choice=None, decision=None, record_path=None):
    return {"status": status, "dataset": dataset, "candidates": list(candidates), "why": why,
            "source_of_choice": source_of_choice, "decision": decision, "record_path": record_path}


def _laya_chooses(text, candidates, decider, *, area, record_dir, as_of):
    """Laya chooses among the candidate ids, or the choice is refused. There is no path from here to a fallback.

    The state describes the problem and each candidate the way the catalog describes it -- id, name, description,
    columns, rows, step_seconds -- and the option set is exactly the candidate ids. `m5phet.decide` checks the answer
    back against that set, refuses an answer that did not come from Laya's own backend, and writes the record."""
    from . import decide
    ids = [item["id"] for item in candidates]
    state = decide.decision_state(DECISION_KIND, {
        "problem": text, "area": area,
        "candidates": [{"id": item["id"], "name": item["name"], "description": item["description"],
                        "columns": list(item.get("columns") or []), "rows": item.get("rows"),
                        "step_seconds": item.get("step_seconds")} for item in candidates]})
    options = [[item["id"], item["name"]] for item in candidates]
    try:
        asked = decide.ask(decider, state, {"dataset": {"options": options, "instructions": DECISION_INSTRUCTIONS}},
                           kind=DECISION_KIND, record_dir=record_dir, as_of=as_of)
    except decide.DecisionError as error:
        return _result(AMBIGUOUS, None, ids, f"these words fit {ids} and the choice could not be asked ({error}); "
                                             "name the dataset by its id")
    entry = asked["dataset"]
    if entry.get("status") != "OK":
        return _result(AMBIGUOUS, None, ids,
                       f"these words fit {ids} and Laya did not make a decision "
                       f"({entry.get('refusal')}: {entry.get('why')}); nothing falls back to the sentence "
                       "interpreter, because a choice made by a non-model establishes nothing -- name the id")
    decision = entry["decision"]
    return _result(OK, entry_by_id(candidates, decision["chosen"]), ids, None, LAYA, decision, entry.get("record_path"))


def entry_by_id(items, dataset_id):
    return next((item for item in items if item.get("id") == dataset_id), None)


def resolve(subject, catalog, decider=None, *, area=None, record_dir=None, as_of=None):
    """Which dataset these words name: an explicit id, then the words, then -- only between candidates -- Laya.

    `decider` is an `m5phet.web.engine.Engine` (so a decision takes the private worker route to the real checkpoint)
    or a bare `Registry`. With none, two candidates are refused rather than guessed."""
    text, explicit = _sentence_and_explicit(subject)
    lowered = text.lower()
    items = list((catalog or {}).get("datasets") or [])
    aliases = (catalog or {}).get("aliases")
    aliases = DEFAULT_ALIASES if aliases is None else aliases

    # (1) an explicit id wins over every word, including words that name another dataset
    named = explicit or next((i["id"] for i in items if _present(lowered, i.get("id", ""))), None)
    if named:
        found = entry(catalog, named)
        if found is not None:
            return _result(OK, found, [named], None, EXPLICIT_ID)
        return _result(NOT_FOUND, None, _closest(catalog, lowered),
                       f"no dataset {named!r} is in the catalog; the closest are "
                       f"{_closest(catalog, lowered)}. Build or refresh it with "
                       "`python -m m5phet.datasets index`")

    if not mentions_dataset(text, catalog):
        return _result(NOT_ASKED, None, [], "these words name no dataset; nothing was resolved")

    # (2) a declared alias names one resource; then the words themselves
    for phrase, dataset_id in sorted(aliases.items(), key=lambda kv: -len(kv[0])):
        if phrase.lower() in lowered and entry(catalog, dataset_id) is not None:
            return _result(OK, entry(catalog, dataset_id), [dataset_id], None, QUESTION_TEXT)

    scored = [(item, _overlap(item, lowered)) for item in items]
    best = max((score for _, score in scored), default=0)
    if best == 0:
        closest = _closest(catalog, lowered)
        return _result(NOT_FOUND, None, closest,
                       f"these words name no dataset the catalog holds; the closest three are {closest}")
    candidates = [item for item, score in scored if score == best]
    if len(candidates) == 1:
        return _result(OK, candidates[0], [candidates[0]["id"]], None, QUESTION_TEXT)

    # (3) more than one candidate: LAYA chooses AMONG THEM, and among nothing else
    ids = [item["id"] for item in candidates]
    if decider is None:
        return _result(AMBIGUOUS, None, ids,
                       f"these words fit {ids}; say which one, or configure the classification engine so Laya can "
                       "choose between them")
    return _laya_chooses(text, candidates, decider, area=area,
                         record_dir=DEFAULT_RECORD_DIR if record_dir is None else record_dir, as_of=as_of)


def proposal_view(resolution):
    """What the proposal and the review panel carry: id, source, rows, columns -- and who chose. Never a row."""
    found = (resolution or {}).get("dataset")
    if not found:
        return None
    view = {"id": found["id"], "source": found["source"], "rows": found["rows"], "columns": list(found["columns"]),
            "governed": found["governed"], "scale": found.get("scale", "UNKNOWN"),
            "source_of_choice": resolution.get("source_of_choice")}
    if resolution.get("decision") is not None:
        # the record beside the proposal: what Laya was shown (by digest), what it could choose from, what it chose
        # and with which uncalibrated probabilities. The state TEXT is not here; its digest binds the record to it.
        view["decision"] = resolution["decision"]
        view["decision_record"] = resolution.get("record_path")
    return view


def profile_of(found):
    """The shape an engine's data requirement is checked against, from the catalog's description alone."""
    return {"kind": "table" if found.get("columns") else "dataset", "columns": list(found.get("columns") or []),
            "rows": found.get("rows") or 0, "dataset": found["id"], "source": found["source"]}


# --- the rows, when a run actually needs them ---------------------------------------------------------------------

def load_rows(found):
    """The rows of a resolved dataset, as a list of row mappings. Governed access is refused, never downgraded."""
    if found.get("governed"):
        raise GovernedAccess(
            f"{GOVERNED_REFUSAL}: {found['id']!r} is a governed resource and this package implements no governed "
            f"access. A successor must bind {', '.join(GOVERNED_VARIABLES)} and read it through data-gov with the "
            "owner's identity; the values of those variables are never written into this repository nor printed. "
            "Until then the run is refused rather than served from an ungoverned copy.")
    path = Path(str(found.get("path_or_ref") or ""))
    if not path.is_file():
        raise CatalogError(f"{found['id']}: its declared file {path} is not there; refresh the catalog")
    if found.get("scale") == "FITTED_EXPERIMENT_PANEL":
        raise RowsNotUsable(
            f"{SCALE_REFUSAL}: {found['id']!r} stores its panel as a fitted experiment's arrays ({path.name}), "
            "already scaled by the scaler its own manifest declares, while its labels stay in the engine's units. "
            "Handing those rows to an engine would have it scale them a second time and answer a number in no "
            "scale at all. Reading such a panel is WP16's `window_from_rows`, which loads the bundle's own scaler "
            "and checks its digest; until that is installed the run is refused rather than answered wrongly. A "
            "resource stored as .csv or .parquet is read here as an attached file is.")
    suffix = path.suffix.lower()
    columns = list(found.get("columns") or [])
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".npz":
        import numpy
        with numpy.load(path, allow_pickle=False) as archive:
            chosen = None
            for key in archive.files:
                values = archive[key]
                if values.ndim == 2 and (not columns or values.shape[1] == len(columns)):
                    if chosen is None or values.shape[0] > chosen.shape[0]:
                        chosen = values
            if chosen is None:
                raise CatalogError(f"{found['id']}: {path.name} holds no 2-D array matching its {len(columns)} "
                                   "declared columns")
            names = columns or [f"c{i}" for i in range(chosen.shape[1])]
            return [dict(zip(names, (float(v) for v in row))) for row in chosen]
    if suffix == ".parquet":
        try:
            import pandas
        except ImportError:                                             # pragma: no cover - environment dependent
            raise CatalogError(f"{found['id']}: reading parquet needs pandas with a parquet engine; none is installed")
        return pandas.read_parquet(path).to_dict("records")
    raise CatalogError(f"{found['id']}: this package reads .csv, .parquet and .npz, not {path.suffix!r}")


# --- the command line ---------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m m5phet.datasets", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="build the dataset catalog from the manifests under each root")
    index.add_argument("--root", action="append", default=None,
                       help=f"a data foundation to scan (repeatable; default {DEFAULT_ROOTS[0]})")
    index.add_argument("--out", default=None, help=f"where to write it (default {DEFAULT_CATALOG_PATH})")
    show = sub.add_parser("show", help="print the catalog this installation has")
    show.add_argument("--catalog", default=None)
    args = parser.parse_args(argv)
    if args.command == "index":
        catalog = build_catalog(args.root)
        target = write_catalog(catalog, args.out)
        print(json.dumps({"schema": catalog["schema"], "datasets": len(catalog["datasets"]),
                          "governed": sum(1 for d in catalog["datasets"] if d["governed"]),
                          "roots": catalog["roots"], "written": str(target)}, ensure_ascii=False))
        return 0
    print(json.dumps(load_catalog(args.catalog), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":                                              # pragma: no cover - the command line
    sys.exit(main())
