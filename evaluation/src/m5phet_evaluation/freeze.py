"""Sealing a labelled corpus before anyone sees a score.

Order is the whole point: hash the rows and their labels first, score second. Without a seal the cheapest way to improve a
result is to edit the corpus — relabel the rows the model got wrong, drop the awkward ones, or keep running variants and
keep the corpus that suited the winner. None of that leaves a trace in the numbers themselves. It leaves a trace here: the
seal is recomputed at scoring time and any row whose label moved, appeared or vanished is named in the refusal, and every
report carries the seal it was computed against, so a prompt or variant chosen after the fact can be seen against the seal
that was in force when the choice was made.

A seal is bound to one protocol digest. Re-sealing the same rows under a different population, split or label provenance
produces a different seal, so a corpus cannot be quietly carried from the conditions it was sealed under into other ones.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path

from .protocol import EvaluationProtocol, PopulationMismatch, ProtocolError, SealBroken, canonical_digest

SEAL_VERSION = "m5phet-evaluation-seal/1"


def _row_digest(row, label) -> str:
    return canonical_digest({"row": row, "label": label})


@dataclasses.dataclass(frozen=True)
class CorpusSeal:
    """The fingerprint of one labelled corpus under one protocol. Compared, never trusted from memory."""

    protocol_digest: str
    label_provenance: str
    row_digests: tuple
    seal: str
    row_count: int
    sealed_at: str
    version: str = SEAL_VERSION

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "protocol_digest": self.protocol_digest,
            "label_provenance": self.label_provenance,
            "row_digests": [[row, digest] for row, digest in self.row_digests],
            "seal": self.seal,
            "row_count": self.row_count,
            "sealed_at": self.sealed_at,
        }

    def verify(self, rows) -> None:
        """Raise naming what moved. The message carries the row identities because "the corpus changed" is not actionable
        evidence and cannot be reviewed; "row 41 was relabelled" can."""
        if not isinstance(rows, dict):
            raise SealBroken("a seal is verified against a row->label mapping")
        current = {row: _row_digest(row, label) for row, label in rows.items()}
        sealed = dict(self.row_digests)
        added = sorted((str(row) for row in current if row not in sealed))
        removed = sorted((str(row) for row in sealed if row not in current))
        changed = sorted((str(row) for row in current if row in sealed and current[row] != sealed[row]))
        if added or removed or changed:
            raise SealBroken(
                f"the corpus no longer matches seal {self.seal[:12]}: "
                f"{len(changed)} relabelled {changed[:5]}, {len(added)} added {added[:5]}, "
                f"{len(removed)} removed {removed[:5]}")

    def matches_protocol(self, protocol: EvaluationProtocol) -> None:
        if self.protocol_digest != protocol.digest:
            raise SealBroken(
                f"seal {self.seal[:12]} was taken under protocol {self.protocol_digest[:12]}, "
                f"not under protocol {protocol.digest[:12]}")

    def write(self, path) -> Path:
        """Written atomically: a half-written seal would be indistinguishable from a tampered one at the next read."""
        path = Path(path)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(self.as_dict(), sort_keys=True, indent=2) + "\n")
        os.replace(tmp, path)
        return path


def seal_corpus(rows, *, protocol: EvaluationProtocol, sealed_at: str | None = None) -> CorpusSeal:
    """Seal exactly the declared population. A seal over a subset would leave the remaining rows free to be edited."""
    if not isinstance(rows, dict) or not rows:
        raise ProtocolError("seal_corpus: a nonempty row->label mapping is required")
    missing = [row for row in protocol.population if row not in rows]
    extra = [row for row in rows if row not in protocol.population]
    if missing or extra:
        raise PopulationMismatch(
            f"the corpus is not the declared population: {len(missing)} declared rows absent "
            f"{[str(r) for r in missing[:5]]}, {len(extra)} undeclared rows present {[str(r) for r in extra[:5]]}")
    digests = tuple((row, _row_digest(row, rows[row])) for row in protocol.population)
    seal = canonical_digest({"protocol": protocol.digest, "rows": [[row, digest] for row, digest in digests]})
    return CorpusSeal(
        protocol_digest=protocol.digest,
        label_provenance=protocol.label_provenance,
        row_digests=digests,
        seal=seal,
        row_count=len(digests),
        sealed_at=sealed_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def read_seal(path) -> CorpusSeal:
    payload = json.loads(Path(path).read_text())
    found = payload.get("version")
    if found != SEAL_VERSION:
        # a seal written by another version is not a weaker seal but an unknown one; reading it would invent agreement
        raise SealBroken(f"seal file version {found!r} is not {SEAL_VERSION!r}")
    return CorpusSeal(
        protocol_digest=payload["protocol_digest"],
        label_provenance=payload["label_provenance"],
        row_digests=tuple((row, digest) for row, digest in payload["row_digests"]),
        seal=payload["seal"],
        row_count=payload["row_count"],
        sealed_at=payload["sealed_at"],
    )


def require_intact(seal: CorpusSeal, rows, protocol: EvaluationProtocol) -> None:
    """The single gate every scorer passes through, so no family can acquire a way to score an unsealed corpus."""
    if not isinstance(seal, CorpusSeal):
        raise SealBroken("scoring requires the corpus seal taken before the scores were seen")
    seal.matches_protocol(protocol)
    seal.verify(rows)
