"""WP15: a sentence may name a dataset from the data lake, and the catalog holds descriptions -- never rows.

Every test here is about one of two promises. The catalog describes what exists (id, words, columns, a row COUNT) and
carries no cell of anybody's data. The resolution is deterministic first and only asks a model to choose BETWEEN
candidates the catalog already holds, so a model cannot introduce a dataset any more than it can introduce a horizon.
"""

import json

import pytest

from m5phet import datasets, decide
from m5phet.interpret import Interpreter
from m5phet.orchestrate import check_proposal, route
from m5phet.questions import catalog as question_catalog
from m5phet.runtime import Registry


class Fixed(Interpreter):
    """An interpreter whose answer is written by the test, so the rule it is held to is what is being measured."""

    def __init__(self, reply):
        super().__init__(command="fixture", model="fixture-v1", environ={})
        self.reply, self.asked = reply, []

    @property
    def available(self):
        return True

    def _ask(self, text):
        self.asked.append(text)
        return self.reply


def write_resource(root, name, manifest, rows=None, header=None):
    """One foundation-shaped resource: a DATA.json and, when the manifest declares no count, a CSV to count."""
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "DATA.json").write_text(json.dumps(manifest), encoding="utf-8")
    if rows is not None:
        lines = [",".join(header)] + [",".join(str(v) for v in row) for row in rows]
        (folder / "DATA.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder


HOUSEHOLD = {"schema": "df_e1_pilot_data.v1",
             "input_columns": ["Global_reactive_power", "Voltage", "Global_intensity", "Global_active_power"],
             "target_channel": 3, "window": 60, "horizon": 60,
             "panel_rows": 2075259, "slice_rows": [1412361, 1462761],
             "derived_from": "household electric power consumption panel"}

MARKET = {"schema": "df_market.v1", "input_columns": ["open", "high", "low", "close", "volume"],
          "target_channel": 3, "derived_from": "EURUSD hourly candles"}


@pytest.fixture
def roots(tmp_path):
    write_resource(tmp_path, "e1_household_dev_pilot_v1", HOUSEHOLD)
    write_resource(tmp_path, "eurusd_hourly_v2", MARKET,
                   rows=[[1, 2, 0, 1.5, 10], [1.1, 2.2, 0.5, 1.6, 11], [1.2, 2.3, 0.6, 1.7, 12]],
                   header=["open", "high", "low", "close", "volume"])
    return [tmp_path]


# --- the catalog -------------------------------------------------------------------------------------------------

def test_the_catalog_describes_every_manifest_and_names_its_schema(roots):
    catalog = datasets.build_catalog(roots)
    assert catalog["schema"] == datasets.CATALOG_SCHEMA
    ids = sorted(d["id"] for d in catalog["datasets"])
    assert ids == ["e1_household_dev_pilot_v1", "eurusd_hourly_v2"]
    household = datasets.entry(catalog, "e1_household_dev_pilot_v1")
    assert household["source"] == "foundation"
    assert household["columns"] == HOUSEHOLD["input_columns"]
    assert household["rows"] == 50400, "the manifest's own slice declaration is the count; nothing is opened"
    assert "household electric power consumption" in household["description"], "the manifest's own words"
    assert household["governed"] is False


def test_a_manifest_without_a_count_is_counted_by_opening_the_file_and_keeping_nothing(roots):
    market = datasets.entry(datasets.build_catalog(roots), "eurusd_hourly_v2")
    assert market["rows"] == 3 and market["columns"] == ["open", "high", "low", "close", "volume"]
    assert market["path_or_ref"].endswith("DATA.csv")


def test_the_file_the_manifest_s_own_words_name_is_the_one_counted(tmp_path):
    folder = write_resource(tmp_path, "blocks_v1", {"input_columns": ["a", "b"],
                                                    "derived_from": "BLOCK_DATA.csv at closure"})
    (folder / "DATA.csv").write_text("a,b\n", encoding="utf-8")                 # the conventional name, and empty
    (folder / "BLOCK_DATA.csv").write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    counted = datasets.entry(datasets.build_catalog([tmp_path]), "blocks_v1")
    assert counted["rows"] == 3 and counted["path_or_ref"].endswith("BLOCK_DATA.csv")


def test_a_resource_whose_count_cannot_be_read_says_so_instead_of_saying_zero(tmp_path):
    write_resource(tmp_path, "empty_v1", {"input_columns": ["a"], "derived_from": "nowhere"})
    unknown = datasets.entry(datasets.build_catalog([tmp_path]), "empty_v1")
    assert unknown["rows"] is None, "an unknown count is not zero rows"


def test_the_catalog_holds_no_row_of_anybody_s_data(roots):
    text = json.dumps(datasets.build_catalog(roots))
    for cell in ("1.5", "1.6", "1.7", "2.2", "2.3"):
        assert cell not in text, "a catalog that carries a cell has shipped data it was only meant to describe"


def test_data_gov_is_declared_as_an_unavailable_source_with_the_command_that_would_reach_it(roots):
    sources = {s["source"]: s for s in datasets.build_catalog(roots)["sources"]}
    assert sources["data-gov"]["available"] is False
    assert sources["data-gov"]["command"], "a stub that does not say how to reach it is not a stub, it is a silence"
    assert not any(d["source"] == "data-gov" for d in datasets.build_catalog(roots)["datasets"])


def test_a_file_attached_to_a_chat_is_described_in_the_same_shape(roots):
    attached = datasets.file_entry("ventas.csv", [{"fecha": "2026-01-01", "ventas": "120.5"}])
    assert attached["source"] == "file" and attached["rows"] == 1 and attached["columns"] == ["fecha", "ventas"]
    assert "120.5" not in json.dumps(attached)


# --- resolution --------------------------------------------------------------------------------------------------

def test_an_explicit_id_in_the_envelope_state_wins_over_every_word(roots):
    catalog = datasets.build_catalog(roots)
    out = datasets.resolve({"dataset": "eurusd_hourly_v2", "why": "consumo eléctrico del data lake"}, catalog, None)
    assert out["status"] == "OK" and out["dataset"]["id"] == "eurusd_hourly_v2"
    assert out["source_of_choice"] == "EXPLICIT_ID"


def test_an_explicit_id_written_in_the_sentence_wins(roots):
    catalog = datasets.build_catalog(roots)
    out = datasets.resolve("usa el dataset eurusd_hourly_v2 por favor", catalog, None)
    assert out["status"] == "OK" and out["dataset"]["id"] == "eurusd_hourly_v2"
    assert out["source_of_choice"] == "EXPLICIT_ID"


def test_an_explicit_id_the_catalog_does_not_have_is_refused_naming_what_it_does_have(roots):
    catalog = datasets.build_catalog(roots)
    out = datasets.resolve({"dataset": "e1_household_dev_pilot_v9"}, catalog, None)
    assert out["status"] == datasets.NOT_FOUND and "e1_household_dev_pilot_v9" in out["why"]


def test_the_words_alone_resolve_one_dataset_and_say_so(roots):
    catalog = datasets.build_catalog(roots)
    out = datasets.resolve("¿cuánta potencia habrá en la próxima hora? usa el dataset de consumo eléctrico "
                           "del data lake", catalog, None)
    assert out["status"] == "OK" and out["dataset"]["id"] == "e1_household_dev_pilot_v1"
    assert out["source_of_choice"] == "QUESTION_TEXT", "no model is consulted when the words settle it"


def test_a_column_name_resolves_the_dataset_that_declares_it(roots):
    catalog = datasets.build_catalog(roots)
    out = datasets.resolve("con el dataset que tiene volume y close", catalog, None)
    assert out["status"] == "OK" and out["dataset"]["id"] == "eurusd_hourly_v2"


# --- ambiguity: the model chooses AMONG the candidates, and nothing else -------------------------------------------

#: words that fit BOTH household resources and neither uniquely: the case Laya exists for
AMBIGUOUS_SENTENCE = "usa el dataset household power del data lake"


class FakeLaya:
    """A classification provider with Laya's answer shape, as `tests/test_decide.py` uses. `backend` is what decides
    whether a decision may exist at all; `raises` is an unreachable worker."""

    name, area = "laya_news", "classification"

    def __init__(self, label, probabilities, backend="laya", raises=False):
        self.label, self.probabilities, self.backend, self.raises = label, probabilities, backend, raises
        self.seen = []

    def capabilities(self):
        return {"provider": self.name, "operations": ["infer"], "families": ["classification"],
                "output_kinds": ["typed_questions"], "uncertainty_methods": ["UNCALIBRATED_CLASS_PROBABILITIES"],
                "supported": [{"operation": "infer", "family": "classification",
                               "output_kind": "typed_questions"}], "known_states": ["laya-checkpoint:abc"],
                "backend": self.backend}

    def question_types(self):
        return {"choice": {"required": ["options"], "optional": ["instructions"]}}

    def answer_questions(self, state, questions, data, as_of):
        if self.raises:
            raise OSError("the private worker is unreachable")
        self.seen.append(state)
        answers = {name: {"type": "choice", "status": "OK", "label": self.label, "backend": self.backend,
                          "instructions": question.get("instructions"),
                          "options": [list(o) for o in question["options"]],
                          "uncalibrated_probabilities": dict(self.probabilities), "probability_decimals": 4,
                          "calibration": "UNCALIBRATED", "execution_authorized": False,
                          **({"non_model_fixture": True} if self.backend == "fixture" else {})}
                   for name, question in questions.items()}
        answers["__state_ref__"] = "laya-checkpoint:abc"
        return answers


def laya(label, backend="laya", raises=False):
    registry = Registry()
    registry.register(FakeLaya(label, {"e1_household_dev_pilot_v1": 0.2871, "e1_household_successor_v3": 0.7129},
                               backend=backend, raises=raises))
    return registry


def ambiguous_catalog(tmp_path):
    write_resource(tmp_path, "e1_household_dev_pilot_v1", HOUSEHOLD)
    write_resource(tmp_path, "e1_household_successor_v3", dict(HOUSEHOLD, data_access="GOVERNED_DELIVERY"))
    catalog = datasets.build_catalog([tmp_path])
    catalog["aliases"] = {}                   # no alias declares which household resource; only Laya can choose
    return catalog


def test_when_two_candidates_remain_laya_chooses_among_them_and_the_choice_is_recorded(tmp_path):
    catalog = ambiguous_catalog(tmp_path)
    registry = laya("e1_household_successor_v3")
    out = datasets.resolve(AMBIGUOUS_SENTENCE, catalog, registry, area="forecasting",
                           record_dir=tmp_path / "decisions")
    assert out["status"] == "OK" and out["dataset"]["id"] == "e1_household_successor_v3"
    assert out["source_of_choice"] == datasets.LAYA
    assert sorted(out["candidates"]) == ["e1_household_dev_pilot_v1", "e1_household_successor_v3"]
    decision = out["decision"]
    assert decision["kind"] == "dataset_choice" and decision["chosen"] == "e1_household_successor_v3"
    assert decision["backend"] == "laya" and decision["execution_authorized"] is False
    assert decision["probabilities"] == {"e1_household_dev_pilot_v1": 0.2871,
                                         "e1_household_successor_v3": 0.7129}, "the model's own numbers, verbatim"
    assert [pair[0] for pair in decision["options"]] == ["e1_household_dev_pilot_v1", "e1_household_successor_v3"]
    assert "state_sha256" in decision and "problem" not in json.dumps(decision), "the record carries the digest, not the text"
    assert decide.load(out["record_path"])["chosen"] == "e1_household_successor_v3"


def test_the_state_laya_is_shown_describes_the_candidates_and_carries_no_row(tmp_path):
    registry = laya("e1_household_dev_pilot_v1")
    datasets.resolve(AMBIGUOUS_SENTENCE, ambiguous_catalog(tmp_path), registry, area="forecasting",
                     record_dir=tmp_path / "decisions")
    state = registry.get("laya_news").seen[0]["news"]
    assert state.startswith("decision_kind: dataset_choice")
    assert "e1_household_dev_pilot_v1" in state and "e1_household_successor_v3" in state
    assert "area: forecasting" in state and AMBIGUOUS_SENTENCE in state
    assert "Global_active_power" in state, "the columns are a description"


def test_a_label_outside_the_candidates_is_refused_never_accepted(tmp_path):
    out = datasets.resolve(AMBIGUOUS_SENTENCE, ambiguous_catalog(tmp_path), laya("eurusd_hourly_v2"),
                           record_dir=tmp_path / "decisions")
    assert out["status"] == datasets.AMBIGUOUS and out["dataset"] is None
    assert "CHOICE_OUTSIDE_OPTIONS" in out["why"] and "e1_household_dev_pilot_v1" in out["why"]


def test_an_answer_from_a_fixture_is_not_a_decision_and_nothing_falls_back(tmp_path):
    out = datasets.resolve(AMBIGUOUS_SENTENCE, ambiguous_catalog(tmp_path),
                           laya("e1_household_successor_v3", backend="fixture"), record_dir=tmp_path / "decisions")
    assert out["status"] == datasets.AMBIGUOUS and out["dataset"] is None
    assert "NON_MODEL_FIXTURE" in out["why"]
    assert not list((tmp_path / "decisions").glob("*.json")) if (tmp_path / "decisions").is_dir() else True


def test_an_unreachable_worker_refuses_naming_the_candidates_and_never_asks_the_interpreter(tmp_path):
    out = datasets.resolve(AMBIGUOUS_SENTENCE, ambiguous_catalog(tmp_path),
                           laya("e1_household_dev_pilot_v1", raises=True), record_dir=tmp_path / "decisions")
    assert out["status"] == datasets.AMBIGUOUS and out["dataset"] is None
    assert "PROVIDER_ERROR" in out["why"]
    assert "e1_household_dev_pilot_v1" in out["why"] and "e1_household_successor_v3" in out["why"]
    assert "interpreter" in out["why"], "the refusal says that nothing falls back to the sentence interpreter"


def test_without_a_decider_two_candidates_are_refused_naming_both(tmp_path):
    out = datasets.resolve(AMBIGUOUS_SENTENCE, ambiguous_catalog(tmp_path), None)
    assert out["status"] == datasets.AMBIGUOUS
    assert "e1_household_dev_pilot_v1" in out["why"] and "e1_household_successor_v3" in out["why"]


def test_nothing_matched_is_refused_naming_the_closest_three(tmp_path):
    for name in ("alpha_sales_v1", "beta_traffic_v1", "gamma_weather_v1", "delta_energy_v1"):
        write_resource(tmp_path, name, {"input_columns": ["a", "b"], "derived_from": name.replace("_", " ")})
    out = datasets.resolve("usa el dataset de criptomonedas del data lake", datasets.build_catalog([tmp_path]), None)
    assert out["status"] == datasets.NOT_FOUND
    assert len(out["candidates"]) == 3 and all(c in out["why"] for c in out["candidates"])


def test_a_sentence_that_names_no_dataset_asks_for_none(roots):
    out = datasets.resolve("¿qué tono tiene esta noticia?", datasets.build_catalog(roots), None)
    assert out["status"] == datasets.NOT_ASKED and out["dataset"] is None


# --- governed access is resolved, described, and refused at execution ----------------------------------------------

def test_a_governed_dataset_resolves_but_its_rows_are_refused_naming_the_variables(tmp_path):
    write_resource(tmp_path, "e1_household_successor_v3", dict(HOUSEHOLD, data_access="GOVERNED_DELIVERY"))
    catalog = datasets.build_catalog([tmp_path])
    governed = datasets.entry(catalog, "e1_household_successor_v3")
    assert governed["governed"] is True
    out = datasets.resolve({"dataset": "e1_household_successor_v3"}, catalog, None)
    assert out["status"] == "OK"
    with pytest.raises(datasets.GovernedAccess) as raised:
        datasets.load_rows(governed)
    message = str(raised.value)
    assert datasets.GOVERNED_REFUSAL in message
    for variable in datasets.GOVERNED_VARIABLES:
        assert variable in message, "a refusal that does not name what is missing cannot be acted on"
    assert "=" not in message, "the variables are named; no value of any of them is ever printed"


# --- the rows themselves ------------------------------------------------------------------------------------------

def test_the_rows_are_loaded_from_the_declared_file_only_when_asked(roots):
    market = datasets.entry(datasets.build_catalog(roots), "eurusd_hourly_v2")
    rows = datasets.load_rows(market)
    assert len(rows) == 3 and rows[0]["close"] == "1.5"


def test_a_fitted_experiment_s_panel_is_described_counted_and_refused_by_name(tmp_path):
    """The panel of `e1_household_dev_pilot_v1` is stored ALREADY scaled by the scaler its manifest declares. An
    engine handed those rows scales them again and answers a number in no scale at all -- which is what the first
    run of this path did (21.49 "kW" for a household whose declared mean is 0.93 kW). It is refused by name."""
    folder = write_resource(tmp_path, "panel_v1", dict(HOUSEHOLD, scaler={"mean": [0.0], "sd": [1.0]}))
    numpy = pytest.importorskip("numpy")
    numpy.savez(folder / "DATA.npz", Xs=numpy.zeros((7, 4), dtype="float32"))
    stored = datasets.entry(datasets.build_catalog([tmp_path]), "panel_v1")
    assert stored["scale"] == "FITTED_EXPERIMENT_PANEL" and stored["rows"] == 50400
    with pytest.raises(datasets.RowsNotUsable) as raised:
        datasets.load_rows(stored)
    assert datasets.SCALE_REFUSAL in str(raised.value) and "declares no scaler" in str(raised.value)


def standardized_resource(tmp_path, name="panel_v1", columns=None, seed=7, dtype="float64"):
    """A foundation-shaped resource whose panel is standardized with the scaler its own manifest declares."""
    numpy = pytest.importorskip("numpy")
    columns = columns or ["Voltage", "Global_intensity", "Global_active_power"]
    generator = numpy.random.default_rng(seed)
    raw = generator.normal(loc=[241.0, 4.0, 0.93], scale=[2.2, 3.8, 0.91], size=(80, len(columns)))
    mean, sd = raw.mean(0), raw.std(0)
    folder = write_resource(tmp_path, name, {"input_columns": columns, "target_channel": len(columns) - 1,
                                             "window": 4, "horizon": 4,
                                             "scaler": {"mean": mean.tolist(), "sd": sd.tolist(),
                                                        "fitted_on": "train windows only"}})
    numpy.savez(folder / "DATA.npz", Xs=((raw - mean) / sd).astype(dtype), Y=raw[:, -1].astype("float64"))
    return datasets.entry(datasets.build_catalog([tmp_path]), name), raw, columns


def test_a_panel_standardized_by_its_own_declared_scaler_is_inverted_back_to_original_units(tmp_path):
    """WP16 takes RAW rows and applies the bundle's own scaler. A panel stored already standardized, with the very
    scaler that standardized it declared in its manifest, is therefore inverted here -- exactly, not approximately."""
    stored, raw, columns = standardized_resource(tmp_path)
    assert stored["scale"] == "STANDARDIZED_BY_DECLARED_SCALER"
    rows = datasets.load_rows(stored)
    assert len(rows) == len(raw) and list(rows[0]) == columns
    worst = max(abs(rows[i][name] - raw[i][j])
                for i in range(len(raw)) for j, name in enumerate(columns))
    assert worst < 1e-12, f"the round trip must be exact; the worst column differs by {worst}"


def test_a_panel_stored_as_float32_round_trips_to_the_precision_float32_kept(tmp_path):
    """The real panels are float32, so the inverse recovers what the file HOLDS, not more. The arithmetic is exact
    in float64; what it cannot restore is the precision the experiment discarded when it stored the panel."""
    pytest.importorskip("numpy")
    stored, raw, columns = standardized_resource(tmp_path, name="panel32_v1", dtype="float32")
    rows = datasets.load_rows(stored)
    worst = max(abs(rows[i][name] - raw[i][j]) / max(abs(raw[i][j]), 1.0)
                for i in range(len(raw)) for j, name in enumerate(columns))
    assert worst < 1e-6, f"float32 storage allows ~1e-7 relative; this is {worst}"


def test_the_receipt_says_where_the_rows_came_from_and_which_column_the_target_is(tmp_path):
    stored, _raw, columns = standardized_resource(tmp_path)
    receipt = datasets.rows_receipt(stored)
    assert receipt["rows_origin"] == "inverse_standardized_from_manifest_scaler"
    assert len(receipt["scaler_sha256"]) == 64
    assert receipt["target_from"] == "inverted_input_column" and receipt["target_column"] == columns[-1]
    assert "Y" not in json.dumps(receipt) or "not" in receipt["why"], "the label array is not offered as a column"


def test_the_scaler_digest_is_of_the_declared_statistics_and_travels_without_them(tmp_path):
    stored, _raw, _columns = standardized_resource(tmp_path)
    assert len(stored["scaler_sha256"]) == 64
    assert "mean" not in json.dumps(stored), "the catalog carries the digest of the scaler, not its numbers"


def test_a_panel_with_no_declared_scaler_still_refuses_because_its_scale_is_unknown(tmp_path):
    numpy = pytest.importorskip("numpy")
    folder = write_resource(tmp_path, "unscaled_v1", {"input_columns": ["a", "b"], "target_channel": 1})
    numpy.savez(folder / "DATA.npz", Xs=numpy.zeros((9, 2), dtype="float32"))
    stored = datasets.entry(datasets.build_catalog([tmp_path]), "unscaled_v1")
    assert stored["scale"] == "FITTED_EXPERIMENT_PANEL" and stored["scaler_sha256"] is None
    with pytest.raises(datasets.RowsNotUsable) as raised:
        datasets.load_rows(stored)
    assert datasets.SCALE_REFUSAL in str(raised.value)


def test_a_scaler_that_does_not_line_up_with_the_columns_is_not_a_scaler(tmp_path):
    numpy = pytest.importorskip("numpy")
    folder = write_resource(tmp_path, "mismatched_v1", {"input_columns": ["a", "b", "c"],
                                                        "scaler": {"mean": [0.0, 1.0], "sd": [1.0, 1.0]}})
    numpy.savez(folder / "DATA.npz", Xs=numpy.zeros((9, 3), dtype="float32"))
    stored = datasets.entry(datasets.build_catalog([tmp_path]), "mismatched_v1")
    assert stored["scale"] == "FITTED_EXPERIMENT_PANEL", "two statistics cannot invert three columns"
    with pytest.raises(datasets.RowsNotUsable):
        datasets.load_rows(stored)


def test_a_table_resource_is_read_exactly_as_an_attached_file_is(roots):
    assert datasets.entry(datasets.build_catalog(roots), "eurusd_hourly_v2")["scale"] == "UNDECLARED"


# --- wiring: a resolved dataset satisfies the data requirement -----------------------------------------------------

class Forecaster:
    name, area = "fake_forecaster", "forecasting"

    def capabilities(self):
        return {"provider": self.name, "operations": ["infer"], "families": ["regression_forecasting"],
                "output_kinds": ["point_forecast"], "uncertainty_methods": ["none"],
                "supported": [{"operation": "infer", "family": "regression_forecasting",
                               "output_kind": "point_forecast"}], "known_states": ["s1"]}

    def question_types(self):
        return {"point_forecast": {"required": ["horizon"], "optional": ["target"]}}

    def data_requirement(self):
        return {"required": True, "why": "the bundle forecasts from the rows it is given",
                "shape": "a table with the fitted columns"}

    def answer_questions(self, state, questions, data, as_of):
        return {n: {"type": "point_forecast", "values": [0.5412], "unit": "kW"} for n in questions}


def registry():
    r = Registry()
    r.register(Forecaster())
    return r


PROPOSAL = {"area": "forecasting", "state": {"target_variable": "Global_active_power"},
            "questions": {"p": {"type": "point_forecast", "horizon": 60}}}


def test_an_area_that_needs_data_is_refused_when_nothing_is_attached_and_nothing_resolved():
    task, problems = check_proposal(PROPOSAL, question_catalog(registry()), {"kind": "none", "columns": [], "rows": 0})
    assert task is None and any("needs data attached" in p for p in problems)


def test_route_with_a_resolved_dataset_passes_the_data_requirement(roots):
    catalog = datasets.build_catalog(roots)
    out = route("¿cuánta potencia habrá en la próxima hora? usa el dataset de consumo eléctrico del data lake",
                None, registry(), interpreter=Fixed(json.dumps(PROPOSAL)), datasets=catalog)
    assert out["status"] == "OK", out.get("problems")
    assert out["dataset"]["id"] == "e1_household_dev_pilot_v1"
    assert out["dataset"]["source_of_choice"] == "QUESTION_TEXT"
    assert out["dataset"]["rows"] == 50400 and out["dataset"]["columns"] == HOUSEHOLD["input_columns"]
    assert out["profile"]["kind"] == "table" and out["profile"]["rows"] == 50400
    assert out["task"]["state"]["dataset"] == "e1_household_dev_pilot_v1", "what runs records which dataset it read"


def test_a_sentence_with_an_attachment_never_reaches_the_resolver(roots):
    rows = [{"Global_active_power": "0.5", "Voltage": "241.0"}]
    out = route("usa el dataset de consumo eléctrico del data lake", rows, registry(),
                interpreter=Fixed(json.dumps({"area": "forecasting", "state": {},
                                              "questions": {"p": {"type": "point_forecast", "horizon": 60}}})),
                datasets=datasets.build_catalog(roots))
    assert out["dataset"] is None, "an attached file is the data; the catalog is not consulted behind the person's back"
    assert out["profile"]["rows"] == 1
