"""Tamper-evident sequencing for a deployment's episode log.

EU AI Act Article 12 requires automatic, tamper-evident recording over a
high-risk system's lifetime. The failure mode it is aimed at is not corruption
but *selection*: a vendor quietly dropping the episodes that went badly before
handing the log to an auditor. A per-record hash does not catch that -- every
surviving record is individually valid. A chain does, because each entry commits
to the one before it, so a deletion leaves a gap that cannot be closed without
recomputing everything after it.

    entry_hash(n) = sha256( prev_hash(n) || canonical_json(record without hashes) )
    prev_hash(n)  = entry_hash(n-1),  or GENESIS for the first record

What chaining alone does and does not prove
-------------------------------------------
It proves the log is internally consistent: no record has been altered,
reordered or removed *relative to the head you already hold*.

It does not, by itself, stop the party who holds the chain. A vendor who wants
to drop a failed episode can delete it and recompute every subsequent hash, and
the result verifies perfectly. Chaining is only tamper-*evident* once the head
has been committed somewhere the vendor cannot rewrite -- sent to the bureau,
counter-signed, or published. That is what `Checkpoint` is for, and any claim
of Article 12 compliance rests on the checkpoints being anchored, not on the
chain existing.

Saying so plainly matters more than the feature: an auditor who discovers this
themselves will discount everything else.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Iterable

GENESIS = "0" * 64

# Fields that are part of the chain's identity but not of the hashed body.
_UNHASHED = ("entry_hash",)


def canonical(payload: dict[str, Any]) -> str:
    """Deterministic encoding. Any two agreeing parties must produce this."""
    body = {k: v for k, v in payload.items() if k not in _UNHASHED}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=_default)


def _default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not canonically encodable: {type(value).__name__}")


def entry_hash(payload: dict[str, Any]) -> str:
    """The hash of one entry, given its prev_hash is already in the payload."""
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


@dataclass
class ChainState:
    """Running head of one deployment's chain.

    Persist this alongside the salt. A recorder restarted with a fresh state
    starts a new chain, which verification will report as a discontinuity
    rather than silently accept.
    """

    seq: int = 0
    head: str = GENESIS

    def advance(self, payload: dict[str, Any]) -> tuple[int, str, str]:
        """Return (seq, prev_hash, entry_hash) for the next record."""
        prev = self.head
        staged = dict(payload, seq=self.seq, prev_hash=prev)
        digest = entry_hash(staged)
        self.seq += 1
        self.head = digest
        return staged["seq"], prev, digest


@dataclass(frozen=True)
class Checkpoint:
    """A commitment to the chain head at a point in time.

    Only useful once it has left the vendor's control. Send it, have it
    counter-signed, or publish it -- an un-anchored checkpoint proves nothing
    the vendor could not have manufactured.
    """

    deployment_id: str
    seq: int
    head: str
    created_at: float

    def to_wire(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "seq": self.seq,
            "head": self.head,
            "created_at": self.created_at,
        }


def checkpoint(deployment_id: str, state: ChainState) -> Checkpoint:
    return Checkpoint(
        deployment_id=deployment_id,
        seq=state.seq,
        head=state.head,
        created_at=time.time(),
    )


@dataclass
class VerificationResult:
    ok: bool
    checked: int
    first_bad_seq: int | None = None
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


def verify(
    payloads: Iterable[dict[str, Any]],
    *,
    expected_head: str | None = None,
    start_hash: str = GENESIS,
) -> VerificationResult:
    """Recompute the chain and report the first place it disagrees.

    `expected_head` should come from a checkpoint the verifier already held.
    Without it this only proves the sequence is self-consistent -- which the
    vendor could have produced from scratch.
    """
    prev = start_hash
    expected_seq = 0
    count = 0

    for payload in payloads:
        seq = payload.get("seq")
        if seq != expected_seq:
            return VerificationResult(
                False, count, seq if isinstance(seq, int) else None,
                f"sequence gap: expected {expected_seq}, found {seq}",
            )
        if payload.get("prev_hash") != prev:
            return VerificationResult(
                False, count, seq, "prev_hash does not match the previous entry"
            )
        recomputed = entry_hash(payload)
        if recomputed != payload.get("entry_hash"):
            return VerificationResult(
                False, count, seq, "entry_hash does not match the record body"
            )
        prev = recomputed
        expected_seq += 1
        count += 1

    if expected_head is not None and prev != expected_head:
        return VerificationResult(
            False, count, None,
            "chain is self-consistent but does not reach the checkpointed head "
            "-- entries are missing from the end, or this is a rewritten chain",
        )
    return VerificationResult(True, count)
