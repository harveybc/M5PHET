"""The only object meant to leave this package, and the reason the numbers inside it cannot be read out of context.

A report carries four things no metric carries on its own: the protocol digest it was computed under, the seal of the
corpus it was computed against, the counts every ratio rests on, and the provenance of the labels. The provenance is not a
footnote. A corpus the author of the system labelled measures self-consistency, and the report says that in its own
statements, its flags and its JSON, so quoting a number without the caveat requires deleting the caveat by hand.

Two things a report refuses, and one it never hides. It refuses a metric computed over a population other than the
protocol's, because two populations compared as one is how a baseline gets beaten on paper. It refuses a seal taken under
another protocol. And when the evidence is thinner than the minimum declared in advance it is still emitted, flagged
UNDERPOWERED with the row count in the flag, because suppressing a weak result and reporting a strong one is selection.
"""

from __future__ import annotations

import dataclasses
import json
import time

from .freeze import CorpusSeal
from .protocol import AUTHOR_WRITTEN_SMOKE, EvaluationProtocol, PopulationMismatch, ProtocolError, SealBroken

REPORT_VERSION = "m5phet-evaluation-report/1"
UNDERPOWERED = "UNDERPOWERED"


@dataclasses.dataclass(frozen=True)
class EvaluationReport:
    """Built only by build_report, so no path exists that produces a report without its digest, seal and counts."""

    version: str
    protocol_digest: str
    family: str
    label_provenance: str
    label_source: str | None
    label_producer: str | None
    corpus_seal: str
    sealed_row_count: int
    sealed_at: str
    minimum_rows: int
    metric_sets: tuple
    flags: tuple
    statements: tuple
    generated_at: str

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "protocol_digest": self.protocol_digest,
            "family": self.family,
            "label_provenance": self.label_provenance,
            "label_source": self.label_source,
            "label_producer": self.label_producer,
            "corpus_seal": self.corpus_seal,
            "sealed_row_count": self.sealed_row_count,
            "sealed_at": self.sealed_at,
            "minimum_rows": self.minimum_rows,
            "flags": list(self.flags),
            "statements": list(self.statements),
            "metric_sets": [metrics.as_dict() for metrics in self.metric_sets],
            "generated_at": self.generated_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, indent=2, default=str)

    @property
    def is_smoke(self) -> bool:
        return self.label_provenance == AUTHOR_WRITTEN_SMOKE

    @property
    def is_underpowered(self) -> bool:
        return any(flag.startswith(UNDERPOWERED) for flag in self.flags)

    def headline(self) -> str:
        """One line that cannot be quoted without its conditions, because they are inside the line."""
        scored = min((metrics.counts.get("scored_rows", 0) for metrics in self.metric_sets), default=0)
        return (f"{self.family} evaluated on {scored} scored rows of {self.sealed_row_count} sealed "
                f"[labels: {self.label_provenance}] [protocol {self.protocol_digest[:12]}] "
                f"[seal {self.corpus_seal[:12]}]" + (f" [{'] ['.join(self.flags)}]" if self.flags else ""))


def build_report(*, protocol: EvaluationProtocol, seal: CorpusSeal, metric_sets, generated_at: str | None = None) -> EvaluationReport:
    """Assemble the report, re-checking what each metric set claims instead of trusting how it was produced."""
    if not isinstance(protocol, EvaluationProtocol):
        raise ProtocolError("build_report: a frozen EvaluationProtocol is required")
    if not isinstance(seal, CorpusSeal):
        raise SealBroken("build_report: the corpus seal taken before scoring is required")
    seal.matches_protocol(protocol)
    sets = tuple(metric_sets or ())
    if not sets:
        # an empty report would read as an evaluation that found nothing to object to, rather than one never performed
        raise ProtocolError("build_report: a report with no metric set is not a report")

    flags, statements = [], []
    for metrics in sets:
        if metrics.family != protocol.family:
            raise ProtocolError(f"metric set {metrics.name!r} is a {metrics.family} metric under a {protocol.family} protocol")
        if tuple(metrics.population) != tuple(protocol.population):
            raise PopulationMismatch(
                f"metric set {metrics.name!r} was computed over {len(tuple(metrics.population))} rows that are not the "
                f"protocol population of {len(protocol.population)} rows")
        scored = metrics.counts.get("scored_rows")
        if not isinstance(scored, int):
            raise ProtocolError(f"metric set {metrics.name!r} carries no scored_rows count, so its ratios have no denominator")
        if scored < protocol.minimum_rows:
            flags.append(f"{UNDERPOWERED}:{metrics.name}:{scored}/{protocol.minimum_rows}")
            statements.append(
                f"{UNDERPOWERED}: {metrics.name} rests on {scored} scored rows against a declared minimum of "
                f"{protocol.minimum_rows}. The numbers are reported because withholding weak results and publishing "
                f"strong ones is selection, but they do not support a claim about quality.")

    if protocol.is_smoke:
        flags.append(AUTHOR_WRITTEN_SMOKE)
        statements.append(
            f"{AUTHOR_WRITTEN_SMOKE}: no independent label source is named for this corpus. The labels were written by "
            "the author of the system under test, so these numbers measure self-consistency and not accuracy. Agreement "
            "between a wrapper and its own engine is fidelity, never accuracy.")
    else:
        statements.append(
            f"Labels come from {protocol.label_source!r}, produced by {protocol.label_producer!r}, under provenance "
            f"{protocol.label_provenance}.")
    statements.append(
        f"Seal {seal.seal} over {seal.row_count} rows, sealed at {seal.sealed_at}, under protocol {protocol.digest}. A "
        "variant or prompt chosen after these scores were seen would carry this same seal, so the order is checkable.")

    return EvaluationReport(
        version=REPORT_VERSION,
        protocol_digest=protocol.digest,
        family=protocol.family,
        label_provenance=protocol.label_provenance,
        label_source=protocol.label_source,
        label_producer=protocol.label_producer,
        corpus_seal=seal.seal,
        sealed_row_count=seal.row_count,
        sealed_at=seal.sealed_at,
        minimum_rows=protocol.minimum_rows,
        metric_sets=sets,
        flags=tuple(flags),
        statements=tuple(statements),
        generated_at=generated_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
