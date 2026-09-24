"""The frozen conditions a score is allowed to be computed under (INT01, INT05).

A protocol exists so that a number can never travel without the conditions that produced it: which family answered, which
rows were in scope, where the labels came from and who wrote them, how ambiguous rows were adjudicated, how the split was
frozen, which metrics were promised in advance and which baseline they must be read against. All of it is digested, so a
report computed under one protocol cannot later be presented as the answer to another; changing the population, the split,
the metric list or the baseline changes the digest and the mismatch surfaces instead of passing.

One rule is the reason this module is a type and not a dictionary. A corpus whose labels no independent source produced is
AUTHOR_WRITTEN_SMOKE, and there is no default, ordering or convenience constructor by which it can be described as anything
else. That is the state of the existing relevance corpus, and a report built on it has to say so in its own text.

Nothing in this package measures a model. It is the instrument; taking the measurement needs labels that do not exist yet.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

FAMILIES = ("classification", "forecast", "regimes", "causal", "policy")

#: labels written by the author of the system under test; self-consistency, never accuracy
AUTHOR_WRITTEN_SMOKE = "AUTHOR_WRITTEN_SMOKE"
LABEL_PROVENANCES = (AUTHOR_WRITTEN_SMOKE, "INDEPENDENT_HUMAN", "INDEPENDENT_SYSTEM", "REALISED_OUTCOME")


class EvaluationError(ValueError):
    """Base of every refusal here. A refusal names what was missing; it never degrades into a default score."""


class ProtocolError(EvaluationError):
    """The declared conditions are incomplete or contradict each other, so no score computed under them would mean anything."""


class PopulationMismatch(EvaluationError):
    """A number was computed over rows other than the ones it is being reported for. Two populations silently compared is
    the classic way a baseline is beaten: the model scored on the easy rows, the baseline on all of them."""


class SealBroken(EvaluationError):
    """The corpus is not the one that was sealed. Either the labels moved after the scores were seen, or the seal is stale."""


class NotEvaluable(EvaluationError):
    """The quantity asked for does not exist for this family. Not unimplemented: undefined, and named as such."""


def canonical_digest(obj) -> str:
    """One canonical serialization for every digest in the package, so the same content never yields two different seals."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _text(value, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{what}: a nonempty declaration is required")
    return value.strip()


def _identities(values, what: str) -> tuple:
    try:
        rows = tuple(values)
    except TypeError:
        raise ProtocolError(f"{what}: a sequence of row identities is required") from None
    if not rows:
        raise ProtocolError(f"{what}: at least one row identity is required")
    for row in rows:
        if isinstance(row, bool) or not isinstance(row, (str, int)) or (isinstance(row, str) and not row.strip()):
            raise ProtocolError(f"{what}: row identities must be nonempty strings or integers, got {row!r}")
    seen = set()
    for row in rows:
        if row in seen:
            # two rows sharing an identity make every count ambiguous: a label or a prediction could belong to either
            raise ProtocolError(f"{what}: duplicate row identity {row!r}")
        seen.add(row)
    return rows


@dataclasses.dataclass(frozen=True)
class EvaluationProtocol:
    """Declared before scoring, frozen afterwards. Its digest is the identity every report and seal must carry."""

    family: str
    population: tuple
    label_source: str | None
    label_producer: str | None
    label_provenance: str
    annotation_rules: tuple
    ambiguity_adjudication: str
    split: tuple
    split_frozen_at: str
    split_frozen_by: str
    metrics: tuple
    baseline: str
    minimum_rows: int
    digest: str = dataclasses.field(default="", init=False, compare=False)

    def __post_init__(self) -> None:
        family = _text(self.family, "family")
        if family not in FAMILIES:
            raise ProtocolError(f"family: {family!r} is not one of {FAMILIES}")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "population", _identities(self.population, "population"))

        source = self.label_source.strip() if isinstance(self.label_source, str) and self.label_source.strip() else None
        producer = self.label_producer.strip() if isinstance(self.label_producer, str) and self.label_producer.strip() else None
        provenance = _text(self.label_provenance, "label_provenance")
        if provenance not in LABEL_PROVENANCES:
            raise ProtocolError(f"label_provenance: {provenance!r} is not one of {LABEL_PROVENANCES}")
        if source is None and provenance != AUTHOR_WRITTEN_SMOKE:
            # the only way an unsourced corpus could be reported as measured accuracy, so it is closed here
            raise ProtocolError(
                f"label_provenance: no independent label source is named, so the provenance must be "
                f"{AUTHOR_WRITTEN_SMOKE!r}, not {provenance!r}")
        if provenance != AUTHOR_WRITTEN_SMOKE and producer is None:
            raise ProtocolError("label_producer: an independent provenance must name who produced the labels")
        object.__setattr__(self, "label_source", source)
        object.__setattr__(self, "label_producer", producer)
        object.__setattr__(self, "label_provenance", provenance)

        rules = tuple(_text(rule, "annotation_rules") for rule in (self.annotation_rules or ()))
        if not rules:
            raise ProtocolError("annotation_rules: state the rules, or state in one line that no row was annotated")
        object.__setattr__(self, "annotation_rules", rules)
        object.__setattr__(self, "ambiguity_adjudication", _text(self.ambiguity_adjudication, "ambiguity_adjudication"))

        items = self.split.items() if isinstance(self.split, dict) else tuple(self.split or ())
        assigned: dict = {}
        normalized = []
        for entry in items:
            try:
                name, rows = entry
            except (TypeError, ValueError):
                raise ProtocolError("split: each entry is a (name, rows) pair") from None
            name = _text(name, "split name")
            rows = _identities(rows, f"split {name!r}")
            for row in rows:
                if row not in self.population:
                    raise ProtocolError(f"split {name!r}: row {row!r} is not in the declared population")
                if row in assigned:
                    # a row in two splits is trained on and tested on; the score would be memory, not generalization
                    raise ProtocolError(f"split {name!r} and split {assigned[row]!r} both contain row {row!r}")
                assigned[row] = name
            normalized.append((name, rows))
        if not normalized:
            raise ProtocolError("split: at least one named split is required")
        missing = [row for row in self.population if row not in assigned]
        if missing:
            # an unassigned row is a row somebody can still move into or out of the test set after seeing the scores
            raise ProtocolError(f"split: {len(missing)} population rows belong to no split, first {missing[0]!r}")
        object.__setattr__(self, "split", tuple(sorted(normalized)))
        object.__setattr__(self, "split_frozen_at", _text(self.split_frozen_at, "split_frozen_at"))
        object.__setattr__(self, "split_frozen_by", _text(self.split_frozen_by, "split_frozen_by"))

        metrics = tuple(_text(metric, "metrics") for metric in (self.metrics or ()))
        if not metrics:
            raise ProtocolError("metrics: declare the metrics before scoring, not after reading them")
        if len(set(metrics)) != len(metrics):
            raise ProtocolError("metrics: duplicate metric name")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "baseline", _text(self.baseline, "baseline"))

        if isinstance(self.minimum_rows, bool) or not isinstance(self.minimum_rows, int) or self.minimum_rows < 1:
            raise ProtocolError("minimum_rows: declare, as a positive integer, the count below which a report is underpowered")
        object.__setattr__(self, "digest", canonical_digest(self.as_dict()))

    def as_dict(self) -> dict:
        """The digested content. The digest itself is excluded, because a digest cannot cover itself."""
        return {
            "family": self.family,
            "population": list(self.population),
            "label_source": self.label_source,
            "label_producer": self.label_producer,
            "label_provenance": self.label_provenance,
            "annotation_rules": list(self.annotation_rules),
            "ambiguity_adjudication": self.ambiguity_adjudication,
            "split": [[name, list(rows)] for name, rows in self.split],
            "split_frozen_at": self.split_frozen_at,
            "split_frozen_by": self.split_frozen_by,
            "metrics": list(self.metrics),
            "baseline": self.baseline,
            "minimum_rows": self.minimum_rows,
        }

    def to_dict(self) -> dict:
        return dict(self.as_dict(), digest=self.digest)

    @property
    def is_smoke(self) -> bool:
        return self.label_provenance == AUTHOR_WRITTEN_SMOKE

    @property
    def independent_labels(self) -> bool:
        """True only when somebody outside the system under test produced the labels."""
        return not self.is_smoke and self.label_source is not None

    def split_of(self, row):
        for name, rows in self.split:
            if row in rows:
                return name
        raise PopulationMismatch(f"row {row!r} is not in the declared population")
