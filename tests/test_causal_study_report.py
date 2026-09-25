"""The DML arm of WP22's closure table: a causal study written as a report, with its refusal intact.

Three properties decide whether this row can be trusted in a table beside two forecast stages:

* every number in it is COPIED from the `prepare-study` result -- the report computes nothing;
* the corpus is sealed over the study's own rows, and the seal identifies the LABELS, so two studies of the same
  releases under different designs (a different window, a different estimator) come out over one seal and can be put
  side by side. A seal that changed with the design would call two readings of one corpus two corpora;
* `causal_accuracy` stays refused, so `compare_stages` ranks nothing here and the cell says why.
"""

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation import causal_study_report as csr                                                        # noqa: E402
from evaluation import compare_stages                                                                    # noqa: E402

COLUMNS = "event_key,published_at,surprise,log_return,realized_vol,pre_event_vol_high,other_surprises_before"


def table(tmp_path, *, rows=120, name="fit.csv", jitter=0.0):
    """A table of the shape `feature_eng_m5phet.event_study_dataset` writes; only the design columns move."""
    lines = [COLUMNS]
    for index in range(rows):
        lines.append(f"claims@2019-{1 + index % 12:02d}-{1 + index % 28:02d}T12:30:00+00:00#{index},"
                     f"2019-{1 + index % 12:02d}-{1 + index % 28:02d}T12:30:00+00:00,"
                     f"{(index % 7) - 3.0},{index * 1e-5},{index * 1e-9},{index % 2},{jitter + index * 0.01}")
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def study(*, estimate=0.002, low=-0.001, high=0.005, status="OK", engine="econml.dml.LinearDML"):
    return {
        "study_id": "wp22-test", "state_ref": "causal-ate:" + "a" * 64, "spec_sha256": "b" * 64,
        "spec": {"treatment_kind": "continuous"},
        "result": {"status": status, "reason": None if status == "OK" else "refused",
                   "payload": {"estimand": "CATE", "estimate": estimate, "interval": [low, high],
                               "unit": "log return per one standard deviation of the surprise",
                               "assumptions": ["consistency", "positivity", "sufficient_adjustment"],
                               "conditional_effects": [
                                   {"modifier": "pre_event_vol_high", "level": 0, "subgroup": "pre_event_vol_high == 0",
                                    "estimate": 0.001, "interval": [-0.002, 0.004], "n_subgroup": 60,
                                    "n_treated": None, "n_control": None, "treatment_std": 2.0}],
                               "diagnostics": {"engine": engine, "treatment_kind": "continuous",
                                               "uncertainty_method": "econml_statsmodels_HC1_normal",
                                               "confidence_level": 0.95, "n_rows": 120, "seed": 1729}}},
    }


def build(tmp_path, **kwargs):
    arguments = {"stage": "dml", "outcome": "log_return", "horizon": "h+30min",
                 "scale": "log return per one standard deviation of the surprise"}
    path = kwargs.pop("table_path", None) or table(tmp_path)
    arguments.update(kwargs)
    return csr.build(study(), path, **arguments)


def test_every_number_in_the_report_is_the_study_s_own(tmp_path):
    document = build(tmp_path)
    metrics = document["metric_sets"][0]["values"]
    assert metrics["estimate"] == 0.002 and metrics["interval"] == [-0.001, 0.005]
    assert metrics["estimand"] == "CATE"
    assert document["causal_study"]["estimator"] == "econml.dml.LinearDML"
    assert document["causal_study"]["treatment_kind"] == "continuous"
    assert document["causal_study"]["execution_authorized"] is False
    assert document["sealed_row_count"] == 120


def test_causal_accuracy_is_refused_and_the_table_says_so_rather_than_ranking_the_row(tmp_path):
    path = tmp_path / "dml.json"
    path.write_text(json.dumps(build(tmp_path)), encoding="utf-8")
    table_document = compare_stages.compare([compare_stages.load_stage(path)])
    area = next(entry for entry in table_document["areas"] if entry["area"] == "causal")
    row = area["rows"][0]
    assert row["status"] == compare_stages.NO_NEW_MEASUREMENT
    assert row["refusal"]["metric"] == "causal_accuracy"
    assert row["rank"] is None
    assert "causal_accuracy" in row["metric"]


def test_two_designs_over_the_same_releases_share_one_seal_when_the_labels_are_named(tmp_path):
    """The point of `--label-source`: a different window is a different STUDY, not a different corpus."""
    bars = "bars.csv sha256 " + "c" * 64
    first = build(tmp_path, table_path=table(tmp_path, name="hand.csv", jitter=0.0), label_source=bars,
                  label_producer="the price bars")
    second = build(tmp_path, table_path=table(tmp_path, name="laya.csv", jitter=5.0), label_source=bars,
                   label_producer="the price bars")
    assert first["corpus_seal"] == second["corpus_seal"]
    assert first["protocol_digest"] == second["protocol_digest"]


def test_without_a_named_label_source_the_table_path_is_used_and_two_designs_are_two_corpora(tmp_path):
    first = build(tmp_path, table_path=table(tmp_path, name="hand.csv"))
    second = build(tmp_path, table_path=table(tmp_path, name="laya.csv"))
    assert first["corpus_seal"] != second["corpus_seal"]


def test_the_identification_caveat_is_carried_verbatim_into_the_report(tmp_path):
    caveat = "NOT_IDENTIFIED | ASSUMED_PUBLICATION_CLOCK: nobody observed when anything was published"
    document = build(tmp_path, caveat=caveat)
    assert document["causal_study"]["identification_caveat"] == caveat
    assert any(caveat in rule for rule in document["annotation_rules"])


def test_a_study_that_was_refused_produces_no_report(tmp_path):
    with pytest.raises(csr.CausalReportError) as refused:
        csr.build(study(status="NOT_IDENTIFIED"), table(tmp_path), stage="dml", outcome="log_return",
                  horizon="h+30min", scale="x")
    assert "NOT_IDENTIFIED" in str(refused.value)


def test_a_table_without_the_columns_a_seal_needs_is_refused_by_name(tmp_path):
    path = tmp_path / "thin.csv"
    path.write_text("event_key,published_at\nk,2019-01-01T00:00:00+00:00\n", encoding="utf-8")
    with pytest.raises(csr.CausalReportError) as refused:
        csr.build(study(), path, stage="dml", outcome="log_return", horizon="h+30min", scale="x")
    assert "log_return" in str(refused.value)


def test_a_repeated_event_key_is_refused_rather_than_sealed_twice(tmp_path):
    path = tmp_path / "dup.csv"
    rows = [COLUMNS] + ["k,2019-01-01T00:00:00+00:00,1,0.1,1e-9,0,0.5"] * 2
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(csr.CausalReportError) as refused:
        csr.build(study(), path, stage="dml", outcome="log_return", horizon="h+30min", scale="x")
    assert "twice" in str(refused.value)


def test_two_runs_over_the_same_study_write_the_same_bytes(tmp_path):
    path = table(tmp_path)
    first = json.dumps(csr.build(study(), path, stage="dml", outcome="log_return", horizon="h+30min", scale="x"),
                       sort_keys=True)
    second = json.dumps(csr.build(study(), path, stage="dml", outcome="log_return", horizon="h+30min", scale="x"),
                        sort_keys=True)
    assert first == second
