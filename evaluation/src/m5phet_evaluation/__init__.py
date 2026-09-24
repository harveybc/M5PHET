"""Evaluation machinery for the M5PHET task families: the instrument, not a measurement.

Nothing in this package computes a score for any real M5PHET model, and no measured quality of any family is claimed
anywhere in it. Every fitted state in the workbench is DEVELOPMENT provenance and the only labelled corpus that exists
today was written by the author of the system under test. What is built here is the protocol, the seal, the metrics and
the report, specified and tested in advance, so that the day independent labels exist a score can be computed without
anyone inventing a protocol while under time pressure to produce a number.

The parts, and the dishonest path each one closes:

* protocol  - the declared conditions, frozen and digested. A corpus with no independent label source can only be
              AUTHOR_WRITTEN_SMOKE, and a split that leaves rows unassigned is refused, so rows cannot move after scoring.
* freeze    - the corpus seal, taken before any score is seen. Relabelling, dropping or adding a row is named in the
              refusal, and the seal in every report makes a variant chosen after the fact visible.
* scoring   - metrics per family that refuse rather than guess: baselines on the same rows, undefined values left
              undefined, and no accuracy at all for causal or policy, by name and with the reason.
* report    - the digest, the seal, the counts and the label provenance travelling with the numbers, plus an explicit
              UNDERPOWERED flag carrying its own row count.

Agreement between a wrapper and its engine is fidelity, never accuracy; this package will not let the two be reported as
the same thing.
"""

from .freeze import CorpusSeal, read_seal, require_intact, seal_corpus
from .protocol import (AUTHOR_WRITTEN_SMOKE, FAMILIES, LABEL_PROVENANCES, EvaluationError, EvaluationProtocol,
                       NotEvaluable, PopulationMismatch, ProtocolError, SealBroken)
from .report import UNDERPOWERED, EvaluationReport, build_report
from .scoring import (ABSTAINED, REFUSED_METRICS, MetricSet, causal_accuracy, policy_profitability, regime_accuracy,
                      score_causal, score_classification, score_forecast, score_policy, score_regimes)

__all__ = [
    "ABSTAINED", "AUTHOR_WRITTEN_SMOKE", "FAMILIES", "LABEL_PROVENANCES", "REFUSED_METRICS", "UNDERPOWERED",
    "CorpusSeal", "EvaluationError", "EvaluationProtocol", "EvaluationReport", "MetricSet", "NotEvaluable",
    "PopulationMismatch", "ProtocolError", "SealBroken", "build_report", "causal_accuracy", "policy_profitability",
    "read_seal", "regime_accuracy", "require_intact", "seal_corpus", "score_causal", "score_classification",
    "score_forecast", "score_policy", "score_regimes",
]
