"""A report cannot be read without the conditions it was computed under, and it never quietly drops a weak one."""

import dataclasses
import json

import pytest
from conftest import INDEPENDENT_LABELS, LABELS, PREDICTIONS, TRUTH, build_protocol

from m5phet_evaluation import (AUTHOR_WRITTEN_SMOKE, UNDERPOWERED, PopulationMismatch, ProtocolError, SealBroken,
                               build_report, score_classification, seal_corpus)


def test_a_smoke_corpus_is_labelled_as_such_everywhere_the_report_can_be_read(classification_case):
    protocol, seal, metrics = classification_case
    report = build_report(protocol=protocol, seal=seal, metric_sets=[metrics])
    assert report.is_smoke and AUTHOR_WRITTEN_SMOKE in report.flags
    assert AUTHOR_WRITTEN_SMOKE in report.headline()
    assert AUTHOR_WRITTEN_SMOKE in json.loads(report.to_json())["label_provenance"]
    statement = next(line for line in report.statements if line.startswith(AUTHOR_WRITTEN_SMOKE))
    assert "self-consistency" in statement and "fidelity, never accuracy" in statement


def test_independent_labels_name_their_source_and_producer():
    protocol = build_protocol(**INDEPENDENT_LABELS)
    seal = seal_corpus(TRUTH, protocol=protocol)
    metrics = score_classification(protocol=protocol, seal=seal, truth=TRUTH, predictions=PREDICTIONS, labels=LABELS)
    report = build_report(protocol=protocol, seal=seal, metric_sets=[metrics])
    assert AUTHOR_WRITTEN_SMOKE not in report.flags and not report.is_smoke
    assert any(INDEPENDENT_LABELS["label_producer"] in line for line in report.statements)


def test_the_report_carries_the_digest_the_seal_and_the_counts(classification_case):
    protocol, seal, metrics = classification_case
    payload = json.loads(build_report(protocol=protocol, seal=seal, metric_sets=[metrics]).to_json())
    assert payload["protocol_digest"] == protocol.digest
    assert payload["corpus_seal"] == seal.seal and payload["sealed_row_count"] == 6
    assert payload["metric_sets"][0]["counts"]["scored_rows"] == 5
    assert seal.seal in " ".join(payload["statements"])


def test_a_metric_over_another_population_is_refused(classification_case):
    protocol, seal, metrics = classification_case
    elsewhere = dataclasses.replace(metrics, population=("r1", "r2", "r3"))
    with pytest.raises(PopulationMismatch, match="not the protocol population"):
        build_report(protocol=protocol, seal=seal, metric_sets=[elsewhere])
    with pytest.raises(ProtocolError, match="is a forecast metric under a classification protocol"):
        build_report(protocol=protocol, seal=seal, metric_sets=[dataclasses.replace(metrics, family="forecast")])


def test_an_underpowered_report_is_emitted_and_says_so_with_its_count():
    protocol = build_protocol(minimum_rows=400)
    seal = seal_corpus(TRUTH, protocol=protocol)
    metrics = score_classification(protocol=protocol, seal=seal, truth=TRUTH, predictions=PREDICTIONS, labels=LABELS)
    report = build_report(protocol=protocol, seal=seal, metric_sets=[metrics])
    assert report.is_underpowered
    assert f"{UNDERPOWERED}:classification:5/400" in report.flags
    statement = next(line for line in report.statements if line.startswith(UNDERPOWERED))
    assert "5 scored rows" in statement and "minimum of 400" in statement
    assert "do not support a claim about quality" in statement
    assert UNDERPOWERED in report.headline()


def test_a_report_with_enough_rows_carries_no_underpowered_flag(classification_case):
    protocol, seal, metrics = classification_case
    report = build_report(protocol=protocol, seal=seal, metric_sets=[metrics])
    assert not report.is_underpowered and not any(flag.startswith(UNDERPOWERED) for flag in report.flags)


def test_a_report_refuses_a_foreign_seal_no_metrics_and_countless_metrics(classification_case):
    protocol, seal, metrics = classification_case
    with pytest.raises(SealBroken, match="was taken under protocol"):
        build_report(protocol=build_protocol(minimum_rows=400), seal=seal, metric_sets=[metrics])
    with pytest.raises(ProtocolError, match="no metric set is not a report"):
        build_report(protocol=protocol, seal=seal, metric_sets=[])
    countless = dataclasses.replace(metrics, counts={"declared_rows": 6})
    with pytest.raises(ProtocolError, match="no scored_rows count"):
        build_report(protocol=protocol, seal=seal, metric_sets=[countless])
