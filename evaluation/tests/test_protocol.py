"""The protocol refuses the declarations that would make a later number unreadable."""

import dataclasses

import pytest
from conftest import INDEPENDENT_LABELS, ROWS, build_protocol

from m5phet_evaluation import AUTHOR_WRITTEN_SMOKE, ProtocolError


def test_digest_is_stable_and_covers_every_declared_condition():
    base = build_protocol()
    assert build_protocol().digest == base.digest
    for change in ({"population": ROWS[:4], "split": {"test": ROWS[:4]}}, {"metrics": ("macro_f1",)},
                   {"baseline": "stratified_random"}, {"minimum_rows": 400},
                   {"split": {"train": ROWS[:4], "test": ROWS[4:]}},
                   {"ambiguity_adjudication": "Resolved by a second reader."}, INDEPENDENT_LABELS):
        assert build_protocol(**change).digest != base.digest, change


def test_labels_with_no_independent_source_can_only_be_smoke():
    with pytest.raises(ProtocolError) as refusal:
        build_protocol(label_provenance="INDEPENDENT_HUMAN", label_producer="an annotator")
    assert AUTHOR_WRITTEN_SMOKE in str(refusal.value)
    assert build_protocol().is_smoke and not build_protocol().independent_labels


def test_independent_labels_must_name_who_produced_them():
    with pytest.raises(ProtocolError, match="label_producer"):
        build_protocol(label_source="eurusd-relevance-annotation-2026-09", label_provenance="INDEPENDENT_HUMAN")
    declared = build_protocol(**INDEPENDENT_LABELS)
    assert declared.independent_labels and not declared.is_smoke


def test_a_row_cannot_sit_in_two_splits():
    with pytest.raises(ProtocolError, match="both contain row"):
        build_protocol(split={"train": ROWS[:4], "test": ROWS[3:]})


def test_every_population_row_belongs_to_exactly_one_split():
    with pytest.raises(ProtocolError, match="belong to no split"):
        build_protocol(split={"test": ROWS[:4]})


def test_rows_outside_the_population_and_duplicate_identities_are_refused():
    with pytest.raises(ProtocolError, match="not in the declared population"):
        build_protocol(split={"test": ROWS + ("r7",)})
    with pytest.raises(ProtocolError, match="duplicate row identity"):
        build_protocol(population=ROWS + ("r6",), split={"test": ROWS})


def test_incomplete_declarations_are_refused():
    with pytest.raises(ProtocolError, match="family"):
        build_protocol(family="ranking")
    with pytest.raises(ProtocolError, match="annotation_rules"):
        build_protocol(annotation_rules=())
    with pytest.raises(ProtocolError, match="metrics"):
        build_protocol(metrics=())
    with pytest.raises(ProtocolError, match="minimum_rows"):
        build_protocol(minimum_rows=0)
    with pytest.raises(ProtocolError, match="baseline"):
        build_protocol(baseline="  ")


def test_the_protocol_cannot_be_edited_after_it_is_digested():
    declared = build_protocol()
    with pytest.raises(dataclasses.FrozenInstanceError):
        declared.baseline = "a friendlier baseline"
    assert declared.to_dict()["digest"] == declared.digest
    assert declared.split_of("r1") == "test"
