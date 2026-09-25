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
and its rows are read **through data-gov**, with `data_gov.client.DataGovClient`, under the owner's identity bound
in the environment (`DATA_GOV_BASE_URL`, `DATA_GOV_USER`, `DATA_GOV_API_KEY_FILE`). This module never holds a
credential: it reads the key file at the moment of the call, hands it to the client and keeps nothing; no value of
any of those variables is ever printed, logged, returned in a receipt or written into this repository.

Three outcomes and no fourth. Nothing bound (or no key file where the variable points, or no data-gov client
importable) is `GOVERNED_ACCESS_NOT_CONFIGURED`, naming the variables. data-gov answered no -- an unauthenticated
key, a policy that does not grant `download`, a range under holdout, a resource it does not serve -- is
`GOVERNED_ACCESS_REFUSED` **carrying the server's own reason**; the run stops there and no ungoverned copy of the
same bytes is read in its place, which is the failure that would matter. data-gov answered yes and the run carries
a `governance` receipt -- profile `GOVERNED`, the lake, the resource, the delivery's digest and when it was asked --
beside its answers.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

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
#: ALREADY standardized, while its labels stay in kW. Handing it to an engine as raw rows makes that engine scale it
#: a second time, and the first run of this path answered "21.49 kW" for a household whose declared mean is 0.93 kW.
#:
#: There are two such panels and they are not the same case. When the manifest DECLARES the scaler that standardized
#: it -- mean and sd, one per input column -- the transformation is known and invertible, so the rows are returned in
#: original units (`raw = Xs * sd + mean`, in float64) and WP16's `window_from_rows` applies the BUNDLE's own scaler
#: to them and checks its digest, which is the check that matters. When no scaler is declared, the scale is simply
#: unknown and the rows are refused by this name; guessing it would be the same mistake with more steps.
SCALE_REFUSAL = "ROWS_SCALE_NOT_DECLARED"

#: what `rows_receipt` says about rows that were inverted, so an answer records how its inputs came to be
INVERTED_ORIGIN = "inverse_standardized_from_manifest_scaler"
STORED_ORIGIN = "as_stored"

#: governed access with nothing bound to reach data-gov with. The dataset still resolves and is described; only its
#: rows are refused, by this name.
GOVERNED_REFUSAL = "GOVERNED_ACCESS_NOT_CONFIGURED"
#: governed access attempted and DENIED BY DATA-GOV. A different thing from the one above and never merged with it:
#: this one means the governance server was reached, was asked, and said no. It is never downgraded to a local read.
GOVERNED_DENIED = "GOVERNED_ACCESS_REFUSED"
#: the profile a run carries once its rows came through data-gov, against `LOCAL_UNGOVERNED` for every other run
GOVERNED_PROFILE = "GOVERNED"
#: the variables the owner binds for data-gov to be reached with his identity. Their VALUES are never printed,
#: logged, put in a receipt or written into this repository by this module; only these names ever appear.
GOVERNED_VARIABLES = ("DATA_GOV_BASE_URL", "DATA_GOV_USER", "DATA_GOV_API_KEY_FILE")
#: where a checkout of data-gov may be found when its distribution is not installed in this interpreter
CHECKOUT_VARIABLE = "DATA_GOV_CHECKOUT"
#: where delivered bytes are kept. Content-addressed by data-gov's client, outside every repository.
GOVERNED_CACHE = "~/.cache/m5phet/data-gov"
CACHE_VARIABLE = "M5PHET_DATA_GOV_CACHE"
#: the experiment key these reads are recorded under in data-gov's accounting
GOVERNED_EXPERIMENT_KEY = "m5phet-chat"

#: how data-gov is reached. The client is data-gov's OWN (`data_gov.client.DataGovClient`), installed in this
#: environment or loaded from a checkout named by DATA_GOV_CHECKOUT; the listing is an authenticated HTTP call.
DATA_GOV_COMMAND = ("pip install <data-gov checkout>   # or export DATA_GOV_CHECKOUT=<data-gov checkout>\n"
                    "  # then DataGovClient(base_url=$DATA_GOV_BASE_URL, api_key_file=$DATA_GOV_API_KEY_FILE)"
                    ".lakes() / .resources(lake_id)")

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
    """A governed resource whose rows cannot be read as this installation stands. Refused by name, never downgraded."""


class GovernedRefused(GovernedAccess):
    """data-gov was reached, was asked, and said no. Its own reason travels in the message; nothing falls back."""


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


def _declared_scaler(manifest, columns):
    """The scaler the manifest declares, when it lines up with the input columns; otherwise None.

    A scaler with fewer statistics than columns cannot invert them, and a zero sd cannot be divided back out. Both
    leave the panel's scale unknown, which is a refusal and never an approximation."""
    scaler = manifest.get("scaler")
    if not isinstance(scaler, dict) or not columns:
        return None
    mean, sd = scaler.get("mean"), scaler.get("sd")
    if not isinstance(mean, list) or not isinstance(sd, list):
        return None
    if len(mean) != len(columns) or len(sd) != len(columns):
        return None
    try:
        mean = [float(v) for v in mean]
        sd = [float(v) for v in sd]
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(v) for v in mean + sd) or any(v == 0.0 for v in sd):
        return None
    return {"mean": mean, "sd": sd}


def scaler_sha256(scaler):
    """The digest of the declared statistics, so the catalog can name a scaler without carrying its numbers."""
    canonical = json.dumps({"mean": scaler["mean"], "sd": scaler["sd"]}, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _scale_of(path, scaler=None):
    """What this package can SAY about the units of a resource's stored file, which is not the same as knowing them.

    A `.npz` is a fitted experiment's panel: its arrays are whatever that experiment scaled them to. When the
    manifest declares that scaler the transformation is known and invertible; when it does not, the scale is unknown.
    A `.csv` or `.parquet` is a table, like a file a person attaches -- undeclared, and read exactly as such. Nothing
    here measures a column; measuring would mean reading rows the catalog must not hold."""
    if path is None:
        return "UNKNOWN"
    if path.suffix.lower() != ".npz":
        return "UNDECLARED"
    return "STANDARDIZED_BY_DECLARED_SCALER" if scaler else "FITTED_EXPERIMENT_PANEL"


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
    scaler = _declared_scaler(manifest, columns)
    return {"id": folder.name, "name": folder.name.replace("_", " "),
            "description": _describe(manifest, columns), "columns": columns, "rows": rows,
            "scale": _scale_of(path, scaler), "scaler_sha256": scaler_sha256(scaler) if scaler else None,
            "target_channel": manifest.get("target_channel") if isinstance(manifest.get("target_channel"), int)
            else None,
            "step_seconds": _manifest_step_seconds(manifest),
            "provenance": str(manifest.get("data_access") or manifest.get("derived_from") or "FOUNDATION_MANIFEST"),
            "governed": _is_governed(manifest), "source": "foundation",
            "manifest_ref": str(manifest_path), "path_or_ref": str(path) if path is not None else str(folder)}


def file_entry(name, data):
    """A file attached to a chat, described in the same shape so one resolver reads one vocabulary."""
    from .orchestrate import dataset_profile
    profile = dataset_profile(data)
    return {"id": "file:" + name, "name": name, "description": f"attached to this question; {profile['kind']}",
            "columns": list(profile.get("columns") or []), "rows": profile.get("rows", 0), "scale": "UNDECLARED",
            "scaler_sha256": None, "target_channel": None, "manifest_ref": None, "step_seconds": None,
            "provenance": "ATTACHED_BY_THE_PERSON", "governed": False, "source": "file", "path_or_ref": name}


def data_gov_source(available=False, *, why=None, lakes=None, resources=None, bound=0):
    """data-gov as a SOURCE. Reachable, it says what it listed; unreachable, it says what is missing and never
    invents an entry. No value of any governed variable appears here -- not the base URL, not the key path."""
    record = {"source": "data-gov", "available": bool(available), "command": DATA_GOV_COMMAND,
              "variables": list(GOVERNED_VARIABLES)}
    if available:
        record.update(lakes=lakes, resources=resources, bound_to_foundation_resources=bound,
                      why=why or "listed through DataGovClient.lakes() / .resources() with the bound identity")
    else:
        record["why"] = why or ("data-gov's listing (`DataGovClient.lakes()` / `.resources()`) is an authenticated "
                                f"HTTP call that needs {', '.join(GOVERNED_VARIABLES)}; they are not all bound "
                                "here, so no dataset of it is listed")
    return record


# --- reaching data-gov ---------------------------------------------------------------------------------------------

def governed_configuration(environ=None):
    """`(configuration, missing)` from the environment. `missing` names variables, never shows a value.

    A variable bound to a key file that is not there is as missing as one not bound at all: what the owner meant is
    not available, and pretending otherwise would fail later with a confusing message."""
    env = os.environ if environ is None else environ
    values = {name: str(env.get(name) or "").strip() for name in GOVERNED_VARIABLES}
    missing = [name for name in GOVERNED_VARIABLES if not values[name]]
    key_file = None
    if values["DATA_GOV_API_KEY_FILE"]:
        key_file = Path(os.path.expanduser(values["DATA_GOV_API_KEY_FILE"]))
        if not key_file.is_file():
            missing.append("DATA_GOV_API_KEY_FILE")
    if missing:
        return None, sorted(set(missing))
    return {"base_url": values["DATA_GOV_BASE_URL"], "user": values["DATA_GOV_USER"],
            "key_file": str(key_file)}, []


def _not_configured(subject, missing, extra=""):
    return GovernedAccess(
        f"{GOVERNED_REFUSAL}: {subject!r} is a governed resource and this installation cannot reach data-gov: "
        f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not usable here. Bind "
        f"{', '.join(GOVERNED_VARIABLES)} to the owner's identity (the key lives in a file outside every "
        f"repository; no value of them is ever printed). {extra}"
        "Until then the run is refused rather than served from an ungoverned copy.")


def data_gov_client_class(environ=None):
    """data-gov's OWN client class: the installed distribution, else a checkout named by DATA_GOV_CHECKOUT.

    M5PHET writes no HTTP client for governance. A second implementation of a governed download is a second set of
    rules about what a delivery is, and only data-gov's rules govern."""
    try:
        from data_gov.client import DataGovClient                       # the installed distribution
        return DataGovClient
    except ImportError:
        pass
    env = os.environ if environ is None else environ
    checkout = str(env.get(CHECKOUT_VARIABLE) or "").strip()
    if not checkout:
        return None
    module_path = Path(os.path.expanduser(checkout)) / "data_gov" / "client.py"
    if not module_path.is_file():
        return None
    import importlib.util
    package = module_path.parent
    name = "m5phet_data_gov"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, package / "__init__.py", submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    import importlib
    return importlib.import_module(f"{name}.client").DataGovClient


def governed_client(configuration, environ=None, *, experiment_key=GOVERNED_EXPERIMENT_KEY):
    """One client, built for one call, reading the key file AT THE MOMENT OF THE CALL and keeping nothing."""
    cls = data_gov_client_class(environ)
    if cls is None:
        raise _not_configured(
            "this installation", ["the data-gov client"],
            "Install data-gov in this environment or bind "
            f"{CHECKOUT_VARIABLE} to its checkout. ")
    return cls(base_url=configuration["base_url"], api_key_file=configuration["key_file"],
               experiment_key=experiment_key)


def _refused(subject, reason, where=""):
    return GovernedRefused(
        f"{GOVERNED_DENIED}: data-gov was asked for {subject!r}{where} and refused: {reason}. The run is refused "
        "by that reason; the same bytes are NOT read from an ungoverned copy.")


def data_gov_listing(client):
    """`[(lake_id, resource_id, lake)]` -- everything the bound identity may see. Its own failure is its own reason."""
    status, payload = client.lakes()
    if status != 200:
        raise _refused("the lake listing", _reason(status, payload))
    listing = []
    for lake in (payload or {}).get("lakes") or []:
        lake_id = lake.get("lake_id")
        if not lake_id:
            continue
        status, resources = client.resources(lake_id)
        if status != 200:
            continue                                                    # a lake this identity may not inventory
        for resource in (resources or {}).get("resources") or []:
            resource_id = resource.get("resource_id")
            if resource_id:
                listing.append((lake_id, resource_id, lake))
    return listing


def _reason(status, payload):
    """data-gov's own words for a refusal, never replaced by ours."""
    if isinstance(payload, dict):
        for key in ("error", "reason", "detail", "message"):
            if payload.get(key):
                return f"{payload[key]} (HTTP {status})"
    return f"HTTP {status}"


def governed_binding(found, client):
    """Which (lake, resource) of data-gov IS this catalog entry. Declared wins; else exactly one match, else refused.

    The deterministic rule: a data-gov resource binds to a foundation entry when the resource's file name without
    its suffix IS the entry's id. Nothing fuzzy: two matches or none is refused, naming what was seen, so nobody's
    run silently reads a different resource than the one the person reviewed."""
    if found.get("lake") and found.get("resource"):
        return found["lake"], found["resource"]
    dataset_id = found["id"]
    matches = [(lake_id, resource_id) for lake_id, resource_id, _ in data_gov_listing(client)
               if resource_id == dataset_id or PurePosixPath(resource_id).stem == dataset_id]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise _refused(dataset_id, "no resource of any lake this identity may see is named that way")
    named = ", ".join(f"{lake}/{resource}" for lake, resource in sorted(matches))
    raise _refused(dataset_id, f"{len(matches)} resources answer to that name ({named}); the catalog entry must "
                               "declare which lake and resource it is")


def governed_rows(found, environ=None):
    """`(rows, governance)` for a governed resource, through data-gov and through nothing else."""
    env = os.environ if environ is None else environ
    configuration, missing = governed_configuration(env)
    if missing:
        raise _not_configured(found["id"], missing)
    client = governed_client(configuration, env)
    lake, resource = governed_binding(found, client)
    requested_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cache = Path(os.path.expanduser(str(env.get(CACHE_VARIABLE) or GOVERNED_CACHE)))
    status, info = client.download(lake, resource, str(cache))
    if status != 200:
        raise _refused(found["id"], _reason(status, info), where=f" ({lake}/{resource})")
    delivered = Path(str(info.get("path") or ""))
    if not delivered.is_file():
        raise _refused(found["id"], "the delivery reported success but left no file",
                       where=f" ({lake}/{resource})")
    rows = _rows_from_path(delivered, found)
    governance = {
        "profile": GOVERNED_PROFILE, "lake": lake, "resource": resource,
        # Flow v2 deliveries carry no receipt id; the digest of the delivered bytes IS the receipt data-gov and this
        # answer share, and it is what its accounting recorded for this identity.
        "receipt_id": info.get("delivery_id"),
        "response_digest": info.get("sha256"),
        "source_digest": info.get("source_sha256") or None,
        "delivery": info.get("delivery"), "bytes": info.get("bytes"),
        "cached": bool(info.get("cached")), "requested_at": requested_at,
        "user": configuration["user"], "client": "data_gov.client.DataGovClient",
        "reading": ("these rows were delivered by data-gov to this identity and its accounting recorded the "
                    "delivery; the receipt names no address and carries no credential"),
    }
    return rows, governance


def build_catalog(roots=None, *, files=None, environ=None, data_gov=None):
    """Every resource under every root that carries a manifest, described and never read for its content.

    When the identity to reach data-gov is bound (and `data_gov` is not False), its listing is folded in too: a
    resource whose name IS a foundation resource's id binds to that entry -- one dataset, now addressable through
    governance, never a second entry competing with the first in the resolver -- and a resource with no counterpart
    on this disk becomes its own entry. Still descriptions only: a listing gives names and sizes, not rows."""
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
    source = data_gov_source() if data_gov is False else _fold_in_data_gov(entries, environ)
    return {"schema": CATALOG_SCHEMA, "roots": scanned, "datasets": entries,
            "sources": [{"source": "foundation", "available": True, "roots": scanned}, source],
            "aliases": dict(DEFAULT_ALIASES),
            "reading": "descriptions of what exists; no catalog entry carries a row of anybody's data"}


def _fold_in_data_gov(entries, environ=None):
    """Bind or add data-gov's resources, and say plainly when it could not be listed. Never raises."""
    configuration, missing = governed_configuration(environ)
    if missing:
        return data_gov_source(False, why=("not listed: " + ", ".join(missing) + " "
                                           + ("is" if len(missing) == 1 else "are") + " not usable here"))
    try:
        client = governed_client(configuration, environ)
        listing = data_gov_listing(client)
    except GovernedAccess as refusal:
        return data_gov_source(False, why=str(refusal))
    except OSError as error:                                            # the server is simply not up
        return data_gov_source(False, why=f"not listed: data-gov did not answer ({type(error).__name__})")
    by_id = {item["id"]: item for item in entries}
    bound, lakes = 0, set()
    for lake_id, resource_id, lake in listing:
        lakes.add(lake_id)
        existing = by_id.get(PurePosixPath(resource_id).stem) or by_id.get(resource_id)
        if existing is not None:
            existing.update(lake=lake_id, resource=resource_id, governed=True)
            bound += 1
            continue
        entries.append({
            "id": f"data-gov:{lake_id}/{resource_id}", "name": PurePosixPath(resource_id).stem.replace("_", " "),
            "description": f"{lake.get('title') or lake_id}: {lake.get('description') or 'governed resource'}",
            "columns": [], "rows": None, "scale": "UNKNOWN", "scaler_sha256": None, "target_channel": None,
            "step_seconds": None, "provenance": "DATA_GOV_LISTING", "governed": True, "source": "data-gov",
            "manifest_ref": None, "path_or_ref": None, "lake": lake_id, "resource": resource_id})
    return data_gov_source(True, lakes=sorted(lakes), resources=len(listing), bound=bound)


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
    view["rows_receipt"] = rows_receipt(found)
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

def load_rows(found, environ=None):
    """The rows of a resolved dataset. Kept for callers that do not record governance; see `load_rows_with_receipt`."""
    return load_rows_with_receipt(found, environ)[0]


def load_rows_with_receipt(found, environ=None):
    """`(rows, governance)`. `governance` is None for an ungoverned read and a receipt for a governed delivery.

    A governed resource goes through data-gov and through nothing else: configured, it is delivered and the receipt
    travels with the answer; not configured, `GOVERNED_ACCESS_NOT_CONFIGURED`; refused by the server,
    `GOVERNED_ACCESS_REFUSED` with the server's reason. There is no branch that reads a governed resource's bytes
    off the disk they happen to also sit on."""
    if found.get("governed"):
        return governed_rows(found, environ)
    path = Path(str(found.get("path_or_ref") or ""))
    if not path.is_file():
        raise CatalogError(f"{found['id']}: its declared file {path} is not there; refresh the catalog")
    return _rows_from_path(path, found), None


def _rows_from_path(path, found):
    """The rows a delivered or local file holds, in the one vocabulary the engines are handed.

    What can be said about the units is a statement about the bytes IN HAND. For a local read those are the
    resource's declared file and this changes nothing; a governed delivery may hand over the same resource in
    another form -- the panel as a parquet table rather than the fitted experiment's `.npz` arrays -- and a table
    is read as a table, exactly as an attached file is, never put through an inversion meant for those arrays."""
    scale = found.get("scale", "UNKNOWN") if path.suffix.lower() == ".npz" else _scale_of(path)
    if scale == "FITTED_EXPERIMENT_PANEL":
        raise RowsNotUsable(
            f"{SCALE_REFUSAL}: {found['id']!r} stores its panel as a fitted experiment's arrays ({path.name}) and "
            "its manifest declares no scaler for them, so what units those numbers are in is unknown. Handing them "
            "to an engine would have it scale them again and answer a number in no scale at all. A panel whose "
            "manifest DOES declare its scaler is inverted to original units and read; a resource stored as .csv or "
            ".parquet is read as an attached file is. This one is refused rather than guessed.")
    if scale == "STANDARDIZED_BY_DECLARED_SCALER":
        return _inverted_rows(found, path)
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


def _panel(archive, columns):
    """The 2-D array of a fitted experiment's archive that matches the declared columns, or None."""
    chosen = None
    for key in archive.files:
        values = archive[key]
        if values.ndim == 2 and values.shape[1] == len(columns):
            if chosen is None or values.shape[0] > chosen.shape[0]:
                chosen = values
    return chosen


def _inverted_rows(found, path):
    """`raw = Xs * sd + mean` in float64: the panel back in the units it was recorded in.

    This is the only arithmetic this module does to anybody's data, and it is the exact inverse of the one the
    manifest declares. WP16's `window_from_rows` then applies the BUNDLE's own scaler to these rows and checks its
    digest -- if the two scalers are not the same one it refuses `SCALER_DIGEST_MISMATCH`, which is its check to
    make and not this module's to pre-empt."""
    import numpy
    scaler = _manifest_scaler(found)
    if scaler is None:
        raise CatalogError(f"{found['id']}: its manifest no longer declares the scaler the catalog was built from; "
                           "refresh the catalog")
    columns = list(found.get("columns") or [])
    with numpy.load(path, allow_pickle=False) as archive:
        panel = _panel(archive, columns)
        if panel is None:
            raise CatalogError(f"{found['id']}: {path.name} holds no 2-D array matching its {len(columns)} "
                               "declared columns")
        raw = panel.astype(numpy.float64) * numpy.asarray(scaler["sd"]) + numpy.asarray(scaler["mean"])
    return [dict(zip(columns, (float(v) for v in row))) for row in raw]


def _manifest_scaler(found):
    """The declared statistics, re-read from the manifest: the catalog carries their digest, never their numbers."""
    reference = found.get("manifest_ref")
    if not reference or not Path(str(reference)).is_file():
        return None
    try:
        manifest = json.loads(Path(str(reference)).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    scaler = _declared_scaler(manifest, list(found.get("columns") or []))
    if scaler is None or scaler_sha256(scaler) != found.get("scaler_sha256"):
        return None
    return scaler


def rows_sha256(rows):
    """The digest of the rows actually handed to an engine, in the order they were handed over.

    The envelope path binds an answer to its request digest, not to a population digest, so without this an answer
    from a resolved dataset would record which dataset it read and not WHICH ROWS. It is computed only where the
    rows already exist -- at execution -- and never at proposal time, where reading them would be reading data to
    describe it.

    Non-finite cells are rendered as their JSON literals rather than refused: this panel carries one non-finite row
    in 50,400, and a digest of what was handed over must cover it. Whether such a row can be USED is the engine's
    own check (`NON_NUMERIC`, on the window it takes), not this digest's business."""
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def rows_receipt(found):
    """How this dataset's rows come to be what the engine is handed. It travels with the proposal and the answer.

    It is computed from the catalog alone: no row is read to produce it, and it says which column the target is,
    because that is the one question a reader of the answer will have. The label array `Y` of these resources is the
    value at the HORIZON, not at the row, and no manifest declares it as a co-temporal column -- so it is never
    offered as one, and the target is the inverted input column the manifest's `target_channel` points at."""
    if not found:
        return None
    if found.get("scale") != "STANDARDIZED_BY_DECLARED_SCALER":
        return {"rows_origin": STORED_ORIGIN, "scaler_sha256": None,
                "why": "the rows are handed to the engine as the file stores them"}
    columns = list(found.get("columns") or [])
    channel = found.get("target_channel")
    target = columns[channel] if isinstance(channel, int) and 0 <= channel < len(columns) else None
    return {"rows_origin": INVERTED_ORIGIN, "scaler_sha256": found.get("scaler_sha256"),
            "target_from": "inverted_input_column", "target_column": target,
            "why": ("the panel is stored standardized by the scaler its manifest declares; it is inverted to "
                    "original units (raw = Xs * sd + mean, float64) and the engine applies its own scaler to that. "
                    "The label array Y is the value at the horizon, not at the row, and no manifest declares it as "
                    "a co-temporal column, so it is not offered as one")}


# --- the command line ---------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m m5phet.datasets", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="build the dataset catalog from the manifests under each root")
    index.add_argument("--root", action="append", default=None,
                       help=f"a data foundation to scan (repeatable; default {DEFAULT_ROOTS[0]})")
    index.add_argument("--out", default=None, help=f"where to write it (default {DEFAULT_CATALOG_PATH})")
    index.add_argument("--no-data-gov", action="store_true",
                       help="do not list data-gov even when the identity to reach it is bound")
    show = sub.add_parser("show", help="print the catalog this installation has")
    show.add_argument("--catalog", default=None)
    args = parser.parse_args(argv)
    if args.command == "index":
        catalog = build_catalog(args.root, data_gov=not args.no_data_gov)
        target = write_catalog(catalog, args.out)
        listed = next((s for s in catalog["sources"] if s["source"] == "data-gov"), {})
        print(json.dumps({"schema": catalog["schema"], "datasets": len(catalog["datasets"]),
                          "governed": sum(1 for d in catalog["datasets"] if d["governed"]),
                          "data_gov": {"available": listed.get("available", False),
                                       "resources": listed.get("resources"),
                                       "bound_to_foundation_resources": listed.get(
                                           "bound_to_foundation_resources")},
                          "roots": catalog["roots"], "written": str(target)}, ensure_ascii=False))
        return 0
    print(json.dumps(load_catalog(args.catalog), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":                                              # pragma: no cover - the command line
    sys.exit(main())
