"""Shared corpus for the tests, small enough that every expected number below is worked out by hand.

The sources are put on the path rather than installed: this machinery has to be reviewable and runnable from a checkout,
on a host where nothing can be built. The corpus itself is deliberately author-written, because that is the only kind of
corpus that exists today and the tests must show it being labelled as such rather than passing for a measurement.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from m5phet_evaluation import ABSTAINED, EvaluationProtocol, seal_corpus, score_classification  # noqa: E402

ROWS = ("r1", "r2", "r3", "r4", "r5", "r6")
LABELS = ("relevant", "irrelevant")

TRUTH = {"r1": "relevant", "r2": "relevant", "r3": "irrelevant",
         "r4": "irrelevant", "r5": "relevant", "r6": "irrelevant"}

#: one hit, one miss, one false positive, two correct rejections and one abstention: every cell of the matrix is exercised
PREDICTIONS = {"r1": "relevant", "r2": "irrelevant", "r3": "irrelevant",
               "r4": "relevant", "r5": ABSTAINED, "r6": "irrelevant"}

INDEPENDENT_LABELS = {
    "label_source": "eurusd-relevance-annotation-2026-09",
    "label_producer": "two annotators outside the M5PHET team",
    "label_provenance": "INDEPENDENT_HUMAN",
}


def build_protocol(**overrides) -> EvaluationProtocol:
    declared = {
        "family": "classification",
        "population": ROWS,
        "label_source": None,
        "label_producer": None,
        "label_provenance": "AUTHOR_WRITTEN_SMOKE",
        "annotation_rules": ("A row is relevant when the release names EURUSD or one of its legs.",),
        "ambiguity_adjudication": "Ambiguous rows were decided by the author against the written rule; no second reader existed.",
        "split": {"test": ROWS},
        "split_frozen_at": "2026-09-24T00:00:00Z",
        "split_frozen_by": "Row order fixed by the corpus file digest before any model was run.",
        "metrics": ("macro_f1", "accuracy", "coverage"),
        "baseline": "majority_class",
        "minimum_rows": 3,
    }
    declared.update(overrides)
    return EvaluationProtocol(**declared)


@pytest.fixture
def protocol() -> EvaluationProtocol:
    return build_protocol()


@pytest.fixture
def classification_case():
    """A protocol, the seal taken before scoring, and the metric set. The order is the one the package enforces."""
    declared = build_protocol()
    seal = seal_corpus(TRUTH, protocol=declared, sealed_at="2026-09-24T00:00:00Z")
    metrics = score_classification(protocol=declared, seal=seal, truth=TRUTH, predictions=PREDICTIONS, labels=LABELS)
    return declared, seal, metrics
