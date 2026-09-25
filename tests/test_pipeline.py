"""WP18: the pipeline's catalogs, its four choices and the spec that carries them.

Written against the same FAKE classification provider as `tests/test_decide.py`, and against FIXTURE registries --
`setup.py` and `pyproject.toml` texts written into a temporary directory. That is the point: these tests prove that
the option lists are READ from whatever the repositories declare, not that this machine happens to hold a particular
plugin. What the real registries declare today is in `docs/PIPELINE.md`, and the one thing this suite does assert
about the shipped world is that the committed branch-capability table is internally consistent.

Nothing here fits anything, and a passing suite says nothing about whether a chosen pipeline is any good.
"""

import json

import pytest

from m5phet import decide, pipeline
from test_decide import FakeLaya, engine_with

pytest.importorskip("feature_eng_m5phet", reason="the metric sheet, groups and representation readers live there")


# --- fixture registries ---------------------------------------------------------------------------------------------

PREDICTOR_SETUP = '''
from setuptools import setup
setup(name="predictor", entry_points={
    "predictor.plugins": [
        "ann=predictor_plugins.predictor_plugin_ann:Plugin",
        "fused=predictor_plugins.predictor_plugin_fused:Plugin",
    ],
    "preprocessor.plugins": [
        "default_preprocessor=preprocessor_plugins.default_preprocessor:PreprocessorPlugin",
        "stl_preprocessor=preprocessor_plugins.stl_preprocessor:PreprocessorPlugin",
    ],
})
'''

PREPROCESSOR_SETUP = '''
from setuptools import setup
setup(name="preprocessor", entry_points={"preprocessor.plugins": [
    "normalizer=app.plugins.plugin_normalizer:Plugin",
]})
'''

EXTRACTOR_PYPROJECT = '''
[project]
name = "feature-extractor"

[project.entry-points."feature_extractor.encoders"]
cnn = "app.plugins.encoder_plugin_cnn:Plugin"
lstm = "app.plugins.encoder_plugin_lstm:Plugin"

[project.entry-points."feature_extractor.decoders"]
cnn = "app.plugins.decoder_plugin_cnn:Plugin"
'''


def _module(root, dotted, text):
    path = root.joinpath(*dotted.split("."))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".py").write_text(text, encoding="utf-8")


@pytest.fixture
def registries(tmp_path, monkeypatch):
    """Three fixture checkouts, declared through the environment, with plugin modules that carry docstrings."""
    predictor, preprocessor, extractor = (tmp_path / "predictor", tmp_path / "preprocessor",
                                          tmp_path / "feature-extractor")
    for root in (predictor, preprocessor, extractor):
        root.mkdir()
    (predictor / "setup.py").write_text(PREDICTOR_SETUP, encoding="utf-8")
    (preprocessor / "setup.py").write_text(PREPROCESSOR_SETUP, encoding="utf-8")
    (extractor / "pyproject.toml").write_text(EXTRACTOR_PYPROJECT, encoding="utf-8")
    _module(predictor, "predictor_plugins.predictor_plugin_ann",
            '"""A dense multi-horizon predictor."""\nclass Plugin:\n    """A dense core over one window."""\n')
    _module(predictor, "predictor_plugins.predictor_plugin_fused",
            'ENCODERS = {"cnn": "a conv stack", "lstm": "stacked LSTM"}\n'
            'class Plugin:\n    """A core that fuses one branch per group."""\n')
    _module(predictor, "preprocessor_plugins.default_preprocessor",
            '"""Sliding windows and normalization."""\nclass PreprocessorPlugin:\n    pass\n')
    # stl_preprocessor is declared and its module is NOT written: a label that cannot be read falls back to the key
    _module(preprocessor, "app.plugins.plugin_normalizer",
            'class Plugin:\n    """Z-score normalization of every column."""\n')
    _module(extractor, "app.plugins.encoder_plugin_cnn", 'class Plugin:\n    """A CNN encoder."""\n')
    _module(extractor, "app.plugins.encoder_plugin_lstm", 'class Plugin:\n    """An LSTM encoder."""\n')
    monkeypatch.setenv("M5PHET_PREDICTOR_REPO", str(predictor))
    monkeypatch.setenv("M5PHET_PREPROCESSOR_REPO", str(preprocessor))
    monkeypatch.setenv("M5PHET_FEATURE_EXTRACTOR_REPO", str(extractor))
    return {"predictor": predictor, "preprocessor": preprocessor, "feature-extractor": extractor}


def capability_table(tmp_path, verdicts):
    path = tmp_path / "branch_capability.json"
    path.write_text(json.dumps({
        "schema": pipeline.BRANCH_CAPABILITY_SCHEMA, "group": "predictor.plugins", "registry": "predictor/setup.py",
        "probed_on": "2026-09-25", "probe": {"window": 8}, "verdicts": verdicts,
        "multi_branch": sorted(k for k, v in verdicts.items() if v["verdict"] == pipeline.MULTI_BRANCH)}),
        encoding="utf-8")
    return path


# --- the catalogs are READ, never typed --------------------------------------------------------------------------------

def test_the_preprocessor_catalog_is_read_from_both_registries_with_each_plugins_own_docstring(registries):
    catalog = pipeline.catalog_preprocessors()
    assert [record["key"] for record in catalog["options"]] == ["default_preprocessor", "normalizer"]
    labels = {record["key"]: (record["label"], record["label_source"]) for record in catalog["options"]}
    assert labels["default_preprocessor"] == ("Sliding windows and normalization.", "module_docstring")
    assert labels["normalizer"] == ("Z-score normalization of every column.", "class_docstring")
    assert all(record["module_found"] for record in catalog["options"])


def test_a_declared_plugin_whose_module_is_missing_is_named_but_never_offered(registries):
    """`stl_preprocessor` is declared by the fixture registry and its module is not there."""
    catalog = pipeline.catalog_preprocessors()
    assert "stl_preprocessor" not in [record["key"] for record in catalog["options"]]
    not_offerable = {record["key"]: record for record in catalog["not_offerable"]}
    assert list(not_offerable) == ["stl_preprocessor"]
    assert not_offerable["stl_preprocessor"]["module_found"] is False
    # the label is still the key -- never an invented description -- and the reason names the module that is missing
    assert not_offerable["stl_preprocessor"]["label"] == "stl_preprocessor"
    assert "preprocessor_plugins.stl_preprocessor" in not_offerable["stl_preprocessor"]["why"]
    assert [problem.split(":")[0] for problem in catalog["problems"]] == [pipeline.MODULE_NOT_FOUND]
    assert "NOT offered" in catalog["problems"][0]
    assert pipeline.options(catalog)[0] == ["default_preprocessor", "Sliding windows and normalization."]


def test_the_extractor_catalog_reads_pyproject_entry_points_and_only_the_encoders(registries):
    catalog = pipeline.catalog_extractors()
    assert [record["key"] for record in catalog["options"]] == ["cnn", "lstm"]
    assert [record["group"] for record in catalog["options"]] == ["feature_extractor.encoders"] * 2


def test_a_registry_that_cannot_be_read_contributes_nothing_and_says_so(registries, tmp_path, monkeypatch):
    monkeypatch.setenv("M5PHET_PREPROCESSOR_REPO", str(tmp_path / "nowhere"))
    catalog = pipeline.catalog_preprocessors()
    assert [record["key"] for record in catalog["options"]] == ["default_preprocessor"]
    assert any(problem.startswith(pipeline.REGISTRY_NOT_READ) for problem in catalog["problems"])
    assert [source["status"] for source in catalog["sources"]] == ["READ", "NOT_READ"]


def test_two_registries_declaring_the_same_key_are_refused_rather_than_silently_merged(registries):
    (registries["preprocessor"] / "setup.py").write_text(
        PREPROCESSOR_SETUP.replace("normalizer=", "stl_preprocessor="), encoding="utf-8")
    with pytest.raises(pipeline.PipelineError) as refused:
        pipeline.catalog_preprocessors()
    assert refused.value.code == pipeline.DUPLICATE_PLUGIN_KEY


def test_the_core_catalog_offers_only_the_plugins_the_probe_found_multi_branch(registries, tmp_path):
    path = capability_table(tmp_path, {
        "fused": {"verdict": pipeline.MULTI_BRANCH, "why": "built a model with 2 inputs"},
        "ann": {"verdict": "SINGLE_BRANCH_ONLY", "why": "one tensor only"}})
    catalog = pipeline.catalog_cores(capability_path=path)
    assert [record["key"] for record in catalog["options"]] == ["fused"]
    assert [(record["key"], record["verdict"]) for record in catalog["excluded"]] == [("ann", "SINGLE_BRANCH_ONLY")]
    assert catalog["capability"]["probed"] is True


def test_the_core_catalog_offers_nothing_when_no_plugin_was_probed_multi_branch(registries, tmp_path):
    path = capability_table(tmp_path, {"ann": {"verdict": "SINGLE_BRANCH_ONLY", "why": "one tensor only"},
                                       "fused": {"verdict": "NOT_PROBED", "why": "no GPU here"}})
    catalog = pipeline.catalog_cores(capability_path=path)
    assert catalog["options"] == []
    assert catalog["problems"][0].startswith(pipeline.NO_MULTI_BRANCH_CORE)


def test_without_a_capability_table_no_core_is_offered_on_the_assumption_that_it_might_qualify(registries, tmp_path):
    catalog = pipeline.catalog_cores(capability_path=tmp_path / "absent.json")
    assert catalog["options"] == []
    assert catalog["capability"]["probed"] is False


def test_the_committed_capability_table_says_the_same_thing_twice():
    """The table shipped with this package is the probe's own output and is internally consistent."""
    table = pipeline.branch_capability()
    assert table is not None and table["group"] == "predictor.plugins"
    assert table["multi_branch"] == sorted(key for key, record in table["verdicts"].items()
                                           if record["verdict"] == pipeline.MULTI_BRANCH)


# --- the documents the household artifacts have ---------------------------------------------------------------------

FEATURES = ["price", "volume", "spread"]


def sheet_fixture():
    """A metric sheet in the shape `feature_eng_m5phet.metrics.decision_payload` reads."""
    def block(name, index):
        return {"is_target": name == "price", "rows": 500, "missing_fraction": 0.0,
                "stationarity": {"verdict": "STATIONARY", "verdict_source": "adf+kpss agree"},
                "acf": {"peaks": [{"lag": 24 + index, "autocorrelation": 0.5}], "decay_lag": 30 + index,
                        "computed_on": "levels"},
                "distribution": {"mean": 1.0 + index, "std": 0.5, "skew": 0.1, "excess_kurtosis": 0.2,
                                 "minimum": 0.0, "maximum": 2.0, "quantile_01": 0.1, "quantile_99": 1.9},
                "scale": {"magnitude": index}, "dtype": {"values_are_integral": False},
                "cross_correlation_to_target": {"by_lag": [{"lag": 1, "correlation": 0.3 - 0.1 * index}]}}
    return {"schema": "m5phet.feature_metrics.v1", "target": "price",
            "dataset": {"path": "fixture.csv", "sha256": "0" * 64},
            "sampling": {"step_seconds": 60, "regular_fraction": 1.0},
            "decimals": {"correlation": 6, "distribution": 6, "fraction": 6, "statistic": 6},
            "features": {name: block(name, index) for index, name in enumerate(FEATURES)}}


def group_block(group_id, members):
    return {"dominant_stationarity": {"counts": {"STATIONARY": len(members)}, "members": len(members),
                                      "members_with_it": len(members), "unanimous": True, "verdict": "STATIONARY"},
            "group_id": group_id, "members": list(members), "shared_acf_peaks": [24],
            "shared_acf_peaks_rule": "a lag every member has a peak within 10% of", "size": len(members),
            "summary": f"{group_id}: {', '.join(members)}", "within_mean_abs_correlation": 0.5}


def cut_block(k, groups):
    return {"between_group_mean_abs_correlation": 0.1, "between_rule": "mean |corr| across groups",
            "groups": groups, "k": k, "silhouette": 0.3, "silhouette_rule": "mean silhouette",
            "sizes": [len(group["members"]) for group in groups],
            "within_group_mean_abs_correlation": 0.5, "within_rule": "mean |corr| within groups"}


def groups_fixture():
    """A groups document `feature_eng_m5phet.grouping.validate` accepts, with two declared cuts."""
    return {"schema": "m5phet.feature_groups.v1",
            "source": {"metric_sheet_schema": "m5phet.feature_metrics.v1", "dataset": {"path": "fixture.csv"},
                       "target": "price", "lags": [1]},
            "features": list(FEATURES), "excluded_features": {},
            "distance": {"kind": "correlation", "rule": "1 - |pearson|", "matrix": [], "order": list(FEATURES)},
            "absolute_correlation": {"matrix": [], "order": list(FEATURES), "rule": "|pearson|"},
            "linkage": {"method": "average", "backend": "fixture", "merges": [], "merge_format": "fixture",
                        "cut_rule": "fixture"},
            "max_k": 3, "max_k_rule": "min(6, features - 1)",
            "cuts": {"2": cut_block(2, [group_block("g1", ["price", "volume"]), group_block("g2", ["spread"])]),
                     "3": cut_block(3, [group_block("g1", ["price"]), group_block("g2", ["volume"]),
                                        group_block("g3", ["spread"])])},
            "silhouette_by_k": {"2": 0.3, "3": 0.2}, "recommended_k": 2,
            "recommended_k_rule": "the highest silhouette", "recommendation_status": "DETERMINISTIC_RECOMMENDATION",
            "decimals": {"correlation": 6, "distance": 6, "silhouette": 6},
            "environment": {"python": "3.12"}, "fitted": "NOTHING"}


REPRESENTATION = {"schema": "m5phet.representation.v1", "sampling": {"step_seconds": 60, "timezone": "UTC"},
                  "target": {"column": "price", "transform": "level"}, "windows": [8], "lags": [1],
                  "differencing": {"order": 0}, "calendar": {"clock": "receipt", "columns": []}, "features": [],
                  "holdout": {"fraction": 0.2}, "provenance": "DEVELOPMENT"}


def laya(label, keys):
    """A fake Laya that chooses `label` and carries a probability for every declared key, and nothing else."""
    share = round(1.0 / len(keys), 4)
    return FakeLaya(label=label, probabilities={key: share for key in keys})


# --- one decision per item, with the declared options -------------------------------------------------------------------

def test_choose_preprocessing_asks_one_decision_per_feature_with_exactly_the_declared_options(registries, tmp_path):
    catalog = pipeline.catalog_preprocessors()
    keys = [record["key"] for record in catalog["options"]]
    provider = laya("normalizer", keys)
    plan = pipeline.choose_preprocessing(engine_with(provider), sheet_fixture(), catalog, record_dir=tmp_path)

    assert len(provider.seen) == len(FEATURES)                          # one ask per feature, never one for all
    for call in provider.seen:
        assert list(call["questions"]) == [pipeline.QUESTION_PREPROCESSING]
        assert [list(option) for option in call["questions"]["preprocessing"]["options"]] == pipeline.options(catalog)
    assert {call["state"]["news"] for call in provider.seen}.__len__() == len(FEATURES)     # a state per feature
    assert sorted(plan["features"]) == sorted(FEATURES)
    assert {entry["chosen"] for entry in plan["features"].values()} == {"normalizer"}
    assert sorted(plan["decisions"]) == sorted(FEATURES)
    for feature, digest in plan["decisions"].items():
        assert (tmp_path / f"{digest}.json").is_file()
        assert decide.load(tmp_path / f"{digest}.json")["question"] == pipeline.QUESTION_PREPROCESSING


def test_a_feature_state_is_the_metric_sheets_own_payload_rendered_by_decide(registries):
    from feature_eng_m5phet import metrics
    catalog = pipeline.catalog_preprocessors()
    provider = laya("normalizer", [record["key"] for record in catalog["options"]])
    pipeline.choose_preprocessing(engine_with(provider), sheet_fixture(), catalog, features=["volume"])
    expected = decide.decision_state("feature_profile", metrics.decision_payload(sheet_fixture(), "volume"))
    assert provider.seen[0]["state"]["news"] == expected


def test_a_chosen_label_outside_the_declared_options_is_refused_and_nothing_is_recorded(registries, tmp_path):
    catalog = pipeline.catalog_preprocessors()
    records = tmp_path / "records"
    provider = laya("a_preprocessor_nobody_declared", [record["key"] for record in catalog["options"]])
    plan = pipeline.choose_preprocessing(engine_with(provider), sheet_fixture(), catalog, record_dir=records,
                                         features=["price"])
    entry = plan["features"]["price"]
    assert entry["status"] == "REFUSED" and entry["refusal"] == decide.CHOICE_OUTSIDE_OPTIONS
    assert plan["decisions"] == {} and not records.exists()


def test_without_declared_options_nothing_is_asked_at_all(registries, tmp_path, monkeypatch):
    monkeypatch.setenv("M5PHET_PREDICTOR_REPO", str(tmp_path / "nowhere"))
    monkeypatch.setenv("M5PHET_PREPROCESSOR_REPO", str(tmp_path / "nowhere"))
    provider = laya("normalizer", ["normalizer"])
    plan = pipeline.choose_preprocessing(engine_with(provider), sheet_fixture(), pipeline.catalog_preprocessors())
    assert plan["status"] == "REFUSED" and plan["refusal"] == pipeline.NO_OPTIONS
    assert provider.seen == []


def test_confirm_grouping_asks_one_decision_among_the_declared_cuts(registries):
    document = groups_fixture()
    provider = laya("k=3", ["k=2", "k=3"])
    choice = pipeline.confirm_grouping(engine_with(provider), document)

    assert len(provider.seen) == 1
    asked = provider.seen[0]["questions"][pipeline.QUESTION_GROUPING]
    assert [option[0] for option in asked["options"]] == ["k=2", "k=3"]
    state = provider.seen[0]["state"]["news"]
    assert "legend:" in state and "f1: price" in state                  # the composition travels with aliases
    assert choice["status"] == "OK" and choice["chosen"] == "k=3" and choice["cut"]["k"] == 3
    assert [group["group_id"] for group in choice["cut"]["groups"]] == ["g1", "g2", "g3"]
    assert choice["recommended_k"] == 2                                 # the chooser is free to differ from it


def test_a_groups_document_with_one_cut_is_not_a_choice(registries):
    document = groups_fixture()
    document["cuts"].pop("3")
    document["silhouette_by_k"].pop("3")
    provider = laya("k=2", ["k=2"])
    choice = pipeline.confirm_grouping(engine_with(provider), document)
    assert choice["status"] == "REFUSED" and choice["refusal"] == pipeline.NO_OPTIONS and provider.seen == []


def test_choose_extractors_asks_one_decision_per_group(registries):
    catalog = pipeline.catalog_extractors()
    cut = pipeline.confirm_grouping(engine_with(laya("k=3", ["k=2", "k=3"])), groups_fixture())["cut"]
    provider = laya("lstm", [record["key"] for record in catalog["options"]])
    plan = pipeline.choose_extractors(engine_with(provider), cut, catalog)

    assert len(provider.seen) == 3
    assert sorted(plan["groups"]) == ["g1", "g2", "g3"]
    assert {entry["chosen"] for entry in plan["groups"].values()} == {"lstm"}
    for call in provider.seen:
        assert [list(option) for option in call["questions"]["extractor"]["options"]] == pipeline.options(catalog)


def test_choose_core_asks_nothing_when_no_declared_core_accepts_several_branches(registries, tmp_path):
    catalog = pipeline.catalog_cores(capability_path=capability_table(
        tmp_path, {"ann": {"verdict": "SINGLE_BRANCH_ONLY", "why": "one tensor only"}}))
    cut = pipeline.confirm_grouping(engine_with(laya("k=2", ["k=2", "k=3"])), groups_fixture())["cut"]
    provider = laya("ann", ["ann"])
    choice = pipeline.choose_core(engine_with(provider), cut, catalog)
    assert provider.seen == []
    assert choice["status"] == "REFUSED" and choice["refusal"] == pipeline.NO_MULTI_BRANCH_CORE
    assert choice["core"] == pipeline.NOT_AVAILABLE_MULTI_BRANCH and choice["plan"]


def test_choose_core_asks_one_decision_when_a_core_was_probed_multi_branch(registries, tmp_path):
    catalog = pipeline.catalog_cores(capability_path=capability_table(
        tmp_path, {"fused": {"verdict": pipeline.MULTI_BRANCH, "why": "built a model with 2 inputs"},
                   "ann": {"verdict": pipeline.MULTI_BRANCH, "why": "built a model with 2 inputs"}}))
    cut = pipeline.confirm_grouping(engine_with(laya("k=2", ["k=2", "k=3"])), groups_fixture())["cut"]
    provider = laya("fused", ["fused", "ann"])
    choice = pipeline.choose_core(engine_with(provider), cut, catalog)
    assert len(provider.seen) == 1 and choice["core"] == "fused"
    assert choice["chosen_by"] == pipeline.LAYA_DECISION and choice["decision_sha256"]
    assert "branch_count: 2" in provider.seen[0]["state"]["news"]


# --- the spec -------------------------------------------------------------------------------------------------------

def full_run(registries, tmp_path, *, core_verdict=pipeline.MULTI_BRANCH, cores=None):
    """Steps 2 to 6 end to end against the fake provider, with every record written to `tmp_path`.

    `cores` declares the probe's verdict per plugin; by default both declared cores are multi-branch, so step 5 is a
    real decision. One multi-branch core exercises the `ONLY_CANDIDATE` path instead.
    """
    verdicts = cores or {"fused": core_verdict, "ann": core_verdict}
    preprocessors = pipeline.catalog_preprocessors()
    extractors = pipeline.catalog_extractors()
    cores = pipeline.catalog_cores(capability_path=capability_table(
        tmp_path, {key: {"verdict": verdict, "why": "probe"} for key, verdict in verdicts.items()}))
    plan = pipeline.choose_preprocessing(
        engine_with(laya("normalizer", [r["key"] for r in preprocessors["options"]])), sheet_fixture(),
        preprocessors, record_dir=tmp_path)
    grouping = pipeline.confirm_grouping(engine_with(laya("k=2", ["k=2", "k=3"])), groups_fixture(),
                                         record_dir=tmp_path)
    extraction = pipeline.choose_extractors(engine_with(laya("cnn", [r["key"] for r in extractors["options"]])),
                                            grouping["cut"], extractors, record_dir=tmp_path)
    core = pipeline.choose_core(engine_with(laya("fused", ["fused", "ann"])), grouping["cut"], cores,
                                record_dir=tmp_path)
    spec = pipeline.build_pipeline_spec(REPRESENTATION, plan, grouping, extraction, core,
                                        catalogs={"preprocessing": preprocessors, "extractor": extractors,
                                                  "core": cores})
    return spec, {"preprocessing": preprocessors, "extractor": extractors, "core": cores}


def test_a_full_spec_round_trips_through_json_and_validates_against_the_records_on_disk(registries, tmp_path):
    spec, catalogs = full_run(registries, tmp_path)
    assert spec["schema"] == pipeline.PIPELINE_SCHEMA and spec["execution_authorized"] is False
    assert sorted(spec["preprocessing"]) == sorted(FEATURES)
    assert sorted(spec["extractors"]) == ["g1", "g2"]
    assert spec["core"]["key"] == "fused"
    assert len(spec["decisions"]) == len(FEATURES) + 1 + 2 + 1                   # features, cut, groups, core

    reloaded = json.loads(json.dumps(spec))
    assert pipeline.validate_pipeline(reloaded, catalogs=catalogs, record_dir=tmp_path) is reloaded


def test_a_spec_whose_core_could_not_be_chosen_is_still_a_valid_spec_and_says_so(registries, tmp_path):
    spec, catalogs = full_run(registries, tmp_path, core_verdict="SINGLE_BRANCH_ONLY")
    assert spec["core"]["key"] == pipeline.NOT_AVAILABLE_MULTI_BRANCH and spec["core"]["decision"] is None
    assert spec["core"]["chosen_by"] == pipeline.NOT_AVAILABLE
    assert pipeline.validate_pipeline(spec, catalogs=catalogs, record_dir=tmp_path) is spec


@pytest.mark.parametrize("break_it, code", [
    (lambda spec: spec["preprocessing"]["price"].update(plugin="a_preprocessor_nobody_declared"),
     pipeline.UNKNOWN_PREPROCESSOR),
    (lambda spec: spec["extractors"]["g1"].update(plugin="an_extractor_nobody_declared"), pipeline.UNKNOWN_EXTRACTOR),
    (lambda spec: spec["core"].update(key="a_core_nobody_declared"), pipeline.UNKNOWN_CORE),
    (lambda spec: spec["preprocessing"].pop("volume"), pipeline.FEATURE_WITHOUT_PREPROCESSING),
    (lambda spec: spec["extractors"].pop("g2"), pipeline.GROUP_WITHOUT_EXTRACTOR),
    (lambda spec: spec.update(schema="m5phet.pipeline.v2"), "WRONG_SCHEMA"),
    (lambda spec: spec.pop("core"), "MISSING_KEY"),
    (lambda spec: spec.update(surprise=True), "UNKNOWN_KEY"),
    (lambda spec: spec.update(execution_authorized=True), "EXECUTION_AUTHORIZED"),
])
def test_a_spec_that_does_not_hold_together_is_refused_by_name(registries, tmp_path, break_it, code):
    spec, catalogs = full_run(registries, tmp_path)
    break_it(spec)
    with pytest.raises(pipeline.PipelineError) as refused:
        pipeline.validate_pipeline(spec, catalogs=catalogs, record_dir=tmp_path)
    assert refused.value.code == code


def test_a_digest_with_no_record_on_disk_is_refused(registries, tmp_path):
    spec, catalogs = full_run(registries, tmp_path)
    empty = tmp_path / "elsewhere"
    empty.mkdir()
    with pytest.raises(pipeline.PipelineError) as refused:
        pipeline.validate_pipeline(spec, catalogs=catalogs, record_dir=empty)
    assert refused.value.code == pipeline.DECISION_NOT_ON_DISK


def test_a_plugin_key_that_its_own_decision_record_did_not_choose_is_refused(registries, tmp_path):
    spec, catalogs = full_run(registries, tmp_path)
    # the digest still points at the record that chose `normalizer`; the spec now claims something else was chosen
    spec["preprocessing"]["price"]["plugin"] = "default_preprocessor"
    with pytest.raises(pipeline.PipelineError) as refused:
        pipeline.validate_pipeline(spec, catalogs=catalogs, record_dir=tmp_path)
    assert refused.value.code == pipeline.DECISION_DOES_NOT_MATCH


def test_an_archived_spec_validates_against_the_lists_it_carries_when_no_catalog_is_given(registries, tmp_path):
    spec, _ = full_run(registries, tmp_path)
    assert spec["catalogs"]["preprocessing"] == ["default_preprocessor", "normalizer"]
    assert pipeline.validate_pipeline(json.loads(json.dumps(spec)), record_dir=tmp_path)


def test_a_single_candidate_core_is_recorded_as_the_only_candidate_and_no_decision_is_asked(registries, tmp_path):
    """`decide` refuses a one-option choice, and rightly: a foregone conclusion is not a decision."""
    catalog = pipeline.catalog_cores(capability_path=capability_table(
        tmp_path, {"fused": {"verdict": pipeline.MULTI_BRANCH, "why": "built a model with 2 inputs"},
                   "ann": {"verdict": "SINGLE_BRANCH_ONLY", "why": "one tensor only"}}))
    cut = pipeline.confirm_grouping(engine_with(laya("k=2", ["k=2", "k=3"])), groups_fixture())["cut"]
    provider = laya("fused", ["fused"])
    choice = pipeline.choose_core(engine_with(provider), cut, catalog, record_dir=tmp_path)

    assert provider.seen == []                                  # nothing was asked of the model
    assert choice["status"] == "OK" and choice["core"] == "fused"
    assert choice["chosen_by"] == pipeline.ONLY_CANDIDATE
    assert choice["decision_sha256"] is None and choice["probabilities"] is None


def test_a_spec_whose_core_was_the_only_candidate_validates_with_no_decision(registries, tmp_path):
    spec, catalogs = full_run(registries, tmp_path, cores={"fused": pipeline.MULTI_BRANCH,
                                                           "ann": "SINGLE_BRANCH_ONLY"})
    assert spec["core"] == {"key": "fused", "chosen_by": pipeline.ONLY_CANDIDATE, "decision": None,
                            "why": spec["core"]["why"], "encoder_mapping": spec["core"]["encoder_mapping"]}
    assert pipeline.validate_pipeline(json.loads(json.dumps(spec)), catalogs=catalogs, record_dir=tmp_path)


def test_a_sole_candidate_claim_is_refused_when_the_declared_list_holds_several(registries, tmp_path):
    spec, catalogs = full_run(registries, tmp_path)               # two declared cores, a real decision
    spec["core"].update(chosen_by=pipeline.ONLY_CANDIDATE, decision=None)
    with pytest.raises(pipeline.PipelineError) as refused:
        pipeline.validate_pipeline(spec, catalogs=catalogs, record_dir=tmp_path)
    assert refused.value.code == pipeline.CHOSEN_BY_MISMATCH


# --- the core's own inline encoders -----------------------------------------------------------------------------------

def test_the_cores_inline_encoders_are_read_from_its_module_not_assumed(registries, tmp_path):
    catalog = pipeline.catalog_cores(capability_path=capability_table(
        tmp_path, {"fused": {"verdict": pipeline.MULTI_BRANCH, "why": "probe"}}))
    assert pipeline.inline_encoders("fused", catalog) == {"cnn": "a conv stack", "lstm": "stacked LSTM"}
    # `ann`'s module declares no ENCODERS mapping, so nothing is known about it -- and nothing is guessed
    assert pipeline.inline_encoders("ann", catalog) is None


def test_a_branch_is_mapped_only_when_its_extractor_is_one_of_the_cores_own_encoders(registries, tmp_path):
    catalog = pipeline.catalog_cores(capability_path=capability_table(
        tmp_path, {"fused": {"verdict": pipeline.MULTI_BRANCH, "why": "probe"}}))
    plan = {"groups": {"g1": {"status": "OK", "chosen": "lstm"}, "g2": {"status": "OK", "chosen": "vae_small"}}}
    mapping = pipeline.map_encoders("fused", plan, catalog)

    assert mapping["status"] == pipeline.NOT_MAPPED and mapping["inline_encoders"] == ["cnn", "lstm"]
    assert mapping["branches"]["g1"] == {"extractor": "lstm", "encoder": "lstm", "status": pipeline.MAPPED,
                                         "why": mapping["branches"]["g1"]["why"]}
    assert mapping["branches"]["g2"]["status"] == pipeline.NOT_MAPPED
    assert mapping["branches"]["g2"]["encoder"] is None
    assert "vae_small" in mapping["branches"]["g2"]["why"]        # the extractor is NAMED, never translated


def test_every_branch_mapped_makes_the_mapping_mapped(registries, tmp_path):
    catalog = pipeline.catalog_cores(capability_path=capability_table(
        tmp_path, {"fused": {"verdict": pipeline.MULTI_BRANCH, "why": "probe"}}))
    plan = {"groups": {"g1": {"status": "OK", "chosen": "cnn"}, "g2": {"status": "OK", "chosen": "lstm"}}}
    assert pipeline.map_encoders("fused", plan, catalog)["status"] == pipeline.MAPPED


# --- replaying recorded decisions --------------------------------------------------------------------------------------

def test_a_plan_can_be_rebuilt_from_the_records_without_asking_the_model_again(registries, tmp_path):
    catalog = pipeline.catalog_preprocessors()
    keys = [record["key"] for record in catalog["options"]]
    first = pipeline.choose_preprocessing(engine_with(laya("normalizer", keys)), sheet_fixture(), catalog,
                                          record_dir=tmp_path)
    replay = pipeline.load_records(tmp_path)
    provider = laya("normalizer", keys)
    again = pipeline.choose_preprocessing(engine_with(provider), sheet_fixture(), catalog, replay=replay)

    assert provider.seen == []                                   # the records answered, not the model
    assert again["decisions"] == first["decisions"]
    assert all(entry["replayed"] for entry in again["features"].values())


def test_a_state_with_no_record_is_refused_rather_than_quietly_asked_again(registries, tmp_path):
    catalog = pipeline.catalog_preprocessors()
    provider = laya("normalizer", [record["key"] for record in catalog["options"]])
    plan = pipeline.choose_preprocessing(engine_with(provider), sheet_fixture(), catalog,
                                         replay=pipeline.load_records(tmp_path))
    assert provider.seen == []
    assert {entry["refusal"] for entry in plan["features"].values()} == {pipeline.NO_RECORD_TO_REPLAY}


def test_a_replayed_choice_says_when_the_registry_moved_under_its_record(registries, tmp_path):
    """The choice stands -- it was made among THOSE candidates -- and the spec shows both lists."""
    catalog = pipeline.catalog_preprocessors()
    keys = [record["key"] for record in catalog["options"]]
    pipeline.choose_preprocessing(engine_with(laya("normalizer", keys)), sheet_fixture(), catalog,
                                  record_dir=tmp_path)
    replay = pipeline.load_records(tmp_path)

    # the registry gains a plugin after the decisions were recorded
    (registries["preprocessor"] / "setup.py").write_text(
        PREPROCESSOR_SETUP.replace('"normalizer=app.plugins.plugin_normalizer:Plugin",',
                                   '"normalizer=app.plugins.plugin_normalizer:Plugin",\n'
                                   '    "scaler=app.plugins.plugin_scaler:Plugin",'), encoding="utf-8")
    _module(registries["preprocessor"], "app.plugins.plugin_scaler",
            'class Plugin:\n    """A scaler that did not exist when the decision was made."""\n')
    wider = pipeline.catalog_preprocessors()
    plan = pipeline.choose_preprocessing(engine_with(laya("normalizer", keys)), sheet_fixture(), wider, replay=replay)

    entry = plan["features"]["price"]
    assert entry["status"] == "OK" and entry["chosen"] == "normalizer"
    assert entry["options_changed"]["added"] == ["scaler"] and entry["options_changed"]["removed"] == []
    assert [pair[0] for pair in entry["options_at_decision"]] == keys


def test_a_recorded_choice_the_registries_no_longer_declare_cannot_be_replayed(registries, tmp_path):
    catalog = pipeline.catalog_preprocessors()
    keys = [record["key"] for record in catalog["options"]]
    pipeline.choose_preprocessing(engine_with(laya("normalizer", keys)), sheet_fixture(), catalog,
                                  record_dir=tmp_path, features=["price"])
    replay = pipeline.load_records(tmp_path)
    # the chosen plugin is withdrawn from the registry that declared it
    (registries["preprocessor"] / "setup.py").write_text(
        PREPROCESSOR_SETUP.replace("normalizer=app.plugins.plugin_normalizer:Plugin", "other=app.plugins.other:Plugin"),
        encoding="utf-8")
    _module(registries["preprocessor"], "app.plugins.other", 'class Plugin:\n    """Another plugin."""\n')
    plan = pipeline.choose_preprocessing(engine_with(laya("normalizer", keys)), sheet_fixture(),
                                         pipeline.catalog_preprocessors(), replay=replay, features=["price"])
    assert plan["features"]["price"]["refusal"] == pipeline.CHOICE_NO_LONGER_DECLARED
