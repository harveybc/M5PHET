"""The seal is what makes editing the corpus after seeing the scores visible instead of invisible."""

import pytest
from conftest import LABELS, PREDICTIONS, ROWS, TRUTH, build_protocol

from m5phet_evaluation import (PopulationMismatch, SealBroken, read_seal, require_intact, score_classification,
                               seal_corpus)


def test_a_seal_covers_exactly_the_declared_population(protocol):
    with pytest.raises(PopulationMismatch, match="declared rows absent"):
        seal_corpus({row: TRUTH[row] for row in ROWS[:4]}, protocol=protocol)
    with pytest.raises(PopulationMismatch, match="undeclared rows present"):
        seal_corpus(dict(TRUTH, r7="relevant"), protocol=protocol)


def test_seal_survives_a_round_trip_through_a_file(protocol, tmp_path):
    seal = seal_corpus(TRUTH, protocol=protocol)
    reread = read_seal(seal.write(tmp_path / "corpus.seal.json"))
    assert reread == seal
    reread.verify(TRUTH)


def test_a_relabelled_row_is_named_in_the_refusal(protocol):
    seal = seal_corpus(TRUTH, protocol=protocol)
    with pytest.raises(SealBroken, match="r4") as refusal:
        seal.verify(dict(TRUTH, r4="relevant"))
    assert "1 relabelled" in str(refusal.value)


def test_added_and_removed_rows_are_refused(protocol):
    seal = seal_corpus(TRUTH, protocol=protocol)
    with pytest.raises(SealBroken, match="added"):
        seal.verify(dict(TRUTH, r7="relevant"))
    with pytest.raises(SealBroken, match="removed"):
        seal.verify({row: TRUTH[row] for row in ROWS[:5]})


def test_scoring_a_mutated_corpus_against_the_old_seal_refuses(protocol):
    """The whole point, end to end: the labels move to suit the predictions and the score never arrives."""
    seal = seal_corpus(TRUTH, protocol=protocol)
    improved = dict(TRUTH, r2="irrelevant", r4="relevant")
    with pytest.raises(SealBroken, match="2 relabelled"):
        score_classification(protocol=protocol, seal=seal, truth=improved, predictions=PREDICTIONS, labels=LABELS)


def test_a_seal_belongs_to_one_protocol_only(protocol):
    seal = seal_corpus(TRUTH, protocol=protocol)
    other = build_protocol(minimum_rows=400)
    with pytest.raises(SealBroken, match="was taken under protocol"):
        require_intact(seal, TRUTH, other)
    with pytest.raises(SealBroken, match="taken before the scores were seen"):
        require_intact("a digest someone pasted", TRUTH, protocol)


def test_a_seal_file_from_another_version_is_unknown_not_weaker(protocol, tmp_path):
    seal = seal_corpus(TRUTH, protocol=protocol)
    path = seal.write(tmp_path / "corpus.seal.json")
    path.write_text(path.read_text().replace(seal.version, "some-other-seal/9"))
    with pytest.raises(SealBroken, match="version"):
        read_seal(path)
