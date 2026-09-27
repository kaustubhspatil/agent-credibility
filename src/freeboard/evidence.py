"""Typed, content free log entries, and the timeline an auditor can check.

Four entry types share one log, so one witnessed history can say "this
configuration passed evaluation commitment X at T, and here is every
monitored episode since", and prove the "every":

    episode          a validated wire record, exactly as `to_wire` emits it
    eval_commitment  the hash of an evaluation held elsewhere, and its verdict
    evidence_pack    the digest of an evidence bundle held elsewhere
    revalidation     reason codes that call for evaluating again

Like the wire format, entries are an allow list: every value is a number, a
bool, a value from a fixed enum, or a hex digest, and unknown fields are
refused. Nothing an evaluation or evidence pack contains goes into the log,
only a hash of it.

Entries are encoded as canonical JSON and the leaf commits to those exact
bytes. `decode_entry` refuses any encoding that is not canonical, so there
is only one byte string per entry to argue about.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .chain import GENESIS, entry_hash as _chain_entry_hash
from .episode import EpisodeRecord
from .tlog import TransparencyLog, leaf_hash, root_from_range
from .witness import LogCheckpoint, SignedCheckpoint, Signer, _check_name, sign_checkpoint
from .wire import _CLASS_CODE, _HEX16, _HEX64, WireViolation, _check_str, validate

__all__ = [
    "ENTRY_VERSION",
    "EntryType",
    "EvidenceLog",
    "PackKind",
    "ReasonCode",
    "Timeline",
    "Verdict",
    "commitment_hash",
    "decode_entry",
    "encode_entry",
    "episode_entry",
    "eval_commitment_entry",
    "evidence_pack_entry",
    "revalidation_entry",
    "validate_entry",
    "verify_timeline",
]

ENTRY_VERSION = 1


class EntryType(str, Enum):
    EPISODE = "episode"
    EVAL_COMMITMENT = "eval_commitment"
    EVIDENCE_PACK = "evidence_pack"
    REVALIDATION = "revalidation"


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class PackKind(str, Enum):
    EVAL_REPORT = "eval_report"
    RED_TEAM = "red_team"
    INCIDENT_REVIEW = "incident_review"
    ATTESTATION = "attestation"
    AUDIT = "audit"
    OTHER = "other"


class ReasonCode(str, Enum):
    MODEL_CHANGE = "model_change"
    PROMPT_CHANGE = "prompt_change"
    TOOL_MANIFEST_CHANGE = "tool_manifest_change"
    POLICY_CHANGE = "policy_change"
    DRIFT_DETECTED = "drift_detected"
    DIVERGENCE_ALERT = "divergence_alert"
    ANOMALY_ALERT = "anomaly_alert"
    INCIDENT = "incident"
    SCHEDULED = "scheduled"
    AUDITOR_REQUEST = "auditor_request"
    COMMITMENT_EXPIRED = "commitment_expired"


_TYPES = {t.value for t in EntryType}
_VERDICTS = {v.value for v in Verdict}
_PACKS = {p.value for p in PackKind}
_REASONS = {r.value for r in ReasonCode}

_COMMON: dict[str, tuple[Any, bool]] = {
    "type": (_TYPES, False),
    "v": ("int", False),
    "ts": ("int", False),
    "deployment_id": (_HEX16, False),
}

_BY_TYPE: dict[str, dict[str, tuple[Any, bool]]] = {
    "episode": {"record": ("record", False)},
    "eval_commitment": {
        "commitment": (_HEX64, False),
        "verdict": (_VERDICTS, False),
        "manifest_hash": (_HEX16, False),
        "envelope_class": (_CLASS_CODE, False),
    },
    "evidence_pack": {
        "digest": (_HEX64, False),
        "pack_kind": (_PACKS, False),
        "commitment": (_HEX64, True),
        "n_items": ("int", False),
    },
    "revalidation": {
        "reasons": ("reasons", False),
        "commitment": (_HEX64, True),
    },
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def validate_entry(entry: Mapping[str, Any]) -> None:
    """Refuse anything the entry format does not explicitly permit."""
    if not isinstance(entry, Mapping):
        raise WireViolation("an entry is a JSON object")
    kind = entry.get("type")
    if not isinstance(kind, str) or kind not in _BY_TYPE:
        raise WireViolation(f"type: {kind!r} not in permitted values")
    rules = {**_COMMON, **_BY_TYPE[kind]}
    unknown = set(entry) - set(rules)
    if unknown:
        raise WireViolation(f"unknown fields: {sorted(unknown)}")
    missing = set(rules) - set(entry)
    if missing:
        raise WireViolation(f"missing fields: {sorted(missing)}")

    for name, (rule, nullable) in rules.items():
        value = entry[name]
        if value is None:
            if not nullable:
                raise WireViolation(f"{name}: null not permitted")
            continue
        if rule == "int":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise WireViolation(f"{name}: expected non negative int")
        elif rule == "record":
            if not isinstance(value, dict):
                raise WireViolation("record: expected an episode wire record")
            validate(value)
            if value["deployment_id"] != entry["deployment_id"]:
                raise WireViolation("record: deployment_id differs from the entry's")
        elif rule == "reasons":
            if not isinstance(value, list) or not value:
                raise WireViolation("reasons: expected a non empty list")
            for item in value:
                _check_str("reasons", item, _REASONS)
            if value != sorted(set(value)):
                raise WireViolation("reasons: must be sorted and unique")
        else:
            _check_str(name, value, rule)

    if entry["v"] != ENTRY_VERSION:
        raise WireViolation(f"v: entry version {entry['v']} is not {ENTRY_VERSION}")


def _canonical(entry: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            entry, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
    except ValueError as exc:
        raise WireViolation(f"entry is not canonically encodable: {exc}") from exc
    return text.encode("ascii")


def encode_entry(entry: Mapping[str, Any]) -> bytes:
    """Validate, then encode canonically. These bytes are what the leaf hashes."""
    validate_entry(entry)
    return _canonical(entry)


def decode_entry(data: bytes) -> dict[str, Any]:
    """Parse, validate, and insist the bytes were the canonical encoding."""
    try:
        entry = json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise WireViolation(f"entry is not JSON: {exc}") from exc
    validate_entry(entry)
    if _canonical(entry) != bytes(data):
        raise WireViolation("entry is not canonically encoded")
    return entry


# --- builders --------------------------------------------------------------


def _ts(ts: int | None) -> int:
    return _now_ms() if ts is None else ts


def episode_entry(record: EpisodeRecord | Mapping[str, Any], *, ts: int | None = None) -> dict:
    if isinstance(record, EpisodeRecord):
        payload = asdict(record)
        payload["observed_capabilities"] = list(record.observed_capabilities)
    else:
        payload = dict(record)
    entry = {
        "type": EntryType.EPISODE.value,
        "v": ENTRY_VERSION,
        "ts": _ts(ts),
        "deployment_id": payload.get("deployment_id"),
        "record": payload,
    }
    validate_entry(entry)
    return entry


def eval_commitment_entry(
    *,
    deployment_id: str,
    commitment: str,
    verdict: Verdict | str,
    manifest_hash: str,
    envelope_class: str,
    ts: int | None = None,
) -> dict:
    entry = {
        "type": EntryType.EVAL_COMMITMENT.value,
        "v": ENTRY_VERSION,
        "ts": _ts(ts),
        "deployment_id": deployment_id,
        "commitment": commitment,
        "verdict": Verdict(verdict).value,
        "manifest_hash": manifest_hash,
        "envelope_class": envelope_class,
    }
    validate_entry(entry)
    return entry


def evidence_pack_entry(
    *,
    deployment_id: str,
    digest: str,
    pack_kind: PackKind | str,
    commitment: str | None = None,
    n_items: int = 0,
    ts: int | None = None,
) -> dict:
    entry = {
        "type": EntryType.EVIDENCE_PACK.value,
        "v": ENTRY_VERSION,
        "ts": _ts(ts),
        "deployment_id": deployment_id,
        "digest": digest,
        "pack_kind": PackKind(pack_kind).value,
        "commitment": commitment,
        "n_items": n_items,
    }
    validate_entry(entry)
    return entry


def revalidation_entry(
    *,
    deployment_id: str,
    reasons: Iterable[ReasonCode | str],
    commitment: str | None = None,
    ts: int | None = None,
) -> dict:
    entry = {
        "type": EntryType.REVALIDATION.value,
        "v": ENTRY_VERSION,
        "ts": _ts(ts),
        "deployment_id": deployment_id,
        "reasons": sorted({ReasonCode(r).value for r in reasons}),
        "commitment": commitment,
    }
    validate_entry(entry)
    return entry


def commitment_hash(payload: bytes, nonce: bytes) -> str:
    """A hiding commitment to an evaluation or evidence pack held elsewhere.

    The nonce matters: without one, a low entropy payload (a pass/fail
    report from a known template) can be confirmed by guessing. Keep the
    nonce with the payload and reveal both to whoever should check it.
    """
    if not isinstance(nonce, bytes) or len(nonce) < 16:
        raise ValueError("use a nonce of at least 16 random bytes")
    h = hashlib.sha256(b"freeboard-commitment/v1\x00")
    h.update(len(nonce).to_bytes(8, "big"))
    h.update(nonce)
    h.update(payload)
    return h.hexdigest()


# --- the operator's side ---------------------------------------------------


class EvidenceLog:
    """A transparency log of evidence entries, with signed checkpoints.

    Usable as `Recorder(sink=...)`: each finished episode is appended. With
    `path`, entries are appended to a JSON Lines file in canonical form and
    replayed on open; editing that file changes the root, which the next
    witness to see a checkpoint refuses.

    Many recorders can write to one EvidenceLog, which is the point for
    deployments where every container keeps its own hash chain: the log is
    where their records get one order, and `verify_timeline` reports each
    chain's continuity inside it.
    """

    def __init__(
        self,
        origin: str,
        signer: Signer,
        *,
        path: str | Path | None = None,
        fsync: bool = False,
        clock: Callable[[], int] = _now_ms,
    ) -> None:
        _check_name("origin", origin)
        self.origin = origin
        self._signer = signer
        self._clock = clock
        self._fsync = fsync
        self.log = TransparencyLog()
        self._handle = None
        if path is not None:
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self._replay()
            self._handle = open(self.path, "a", encoding="ascii", buffering=1)

    def _replay(self) -> None:
        with open(self.path, "rb") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.rstrip(b"\r\n")
                if not line:
                    continue
                try:
                    decode_entry(line)
                except WireViolation as exc:
                    raise ValueError(f"{self.path}: line {lineno}: {exc}") from exc
                self.log.append(line)

    # appending

    def append(self, entry: Mapping[str, Any]) -> int:
        data = encode_entry(entry)
        if self._handle is not None:
            self._handle.write(data.decode("ascii") + "\n")
            if self._fsync:
                self._handle.flush()
                os.fsync(self._handle.fileno())
        return self.log.append(data)

    def write(self, record: EpisodeRecord | Mapping[str, Any]) -> None:
        """The Recorder sink protocol."""
        self.append(episode_entry(record, ts=self._clock()))

    def commit_evaluation(
        self, recorder: Any, commitment: str, verdict: Verdict | str = Verdict.PASS
    ) -> int:
        """Log that `recorder`'s current configuration was evaluated."""
        return self.append(
            eval_commitment_entry(
                deployment_id=recorder.deployment_hash,
                commitment=commitment,
                verdict=verdict,
                manifest_hash=recorder.envelope.manifest_hash,
                envelope_class=recorder.envelope.class_code,
                ts=self._clock(),
            )
        )

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> EvidenceLog:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # reading and proving

    @property
    def size(self) -> int:
        return self.log.size

    def root(self, size: int | None = None) -> bytes:
        return self.log.root(size)

    def entry(self, index: int) -> bytes:
        return self.log.entry(index)

    def entries(self, start: int = 0, end: int | None = None) -> list[bytes]:
        return self.log.entries(start, end)

    def inclusion_proof(self, index: int, size: int | None = None) -> list[bytes]:
        return self.log.inclusion_proof(index, size)

    def consistency_proof(self, old_size: int, new_size: int | None = None) -> list[bytes]:
        return self.log.consistency_proof(old_size, new_size)

    def frontier(self, size: int | None = None) -> list[bytes]:
        return self.log.frontier(size)

    def checkpoint(self) -> SignedCheckpoint:
        cp = LogCheckpoint(self.origin, self.size, self.root(), self._clock())
        return sign_checkpoint(cp, self._signer)

    def cosign_with(self, *witnesses: Any) -> SignedCheckpoint:
        """Sign a checkpoint and collect cosignatures from witnesses in this process.

        Each witness gets a consistency proof from the size it last cosigned
        and checks it itself; a refusal propagates as WitnessRefusal.
        """
        signed = self.checkpoint()
        size = signed.checkpoint.size
        for witness in witnesses:
            prev = witness.latest(self.origin)
            old = prev.checkpoint.size if prev is not None else size
            proof = self.consistency_proof(old, size) if old <= size else []
            signed = signed.with_cosignature(witness.cosign(signed, proof))
        return signed

    def find_commitment(self, commitment: str) -> int | None:
        """Index of the first eval_commitment entry carrying `commitment`."""
        for index, data in enumerate(self.log.entries()):
            entry = json.loads(data)
            if entry["type"] == "eval_commitment" and entry["commitment"] == commitment:
                return index
        return None

    def export_timeline(self, commitment: str, size: int | None = None) -> tuple[int, list[bytes], list[bytes]]:
        """(start, frontier, entries) for an auditor to pass to verify_timeline."""
        size = self.size if size is None else size
        start = self.find_commitment(commitment)
        if start is None or start >= size:
            raise KeyError(f"commitment {commitment} is not in the first {size} entries")
        return start, self.frontier(start), self.entries(start, size)


# --- the auditor's side ----------------------------------------------------


@dataclass
class Timeline:
    """What a range of the log says about one deployment since a commitment.

    `ok` means the range is authentic and complete up to the checkpoint and
    starts with the commitment. The counts are for a person to judge; none
    of them is folded into `ok`.

    Chain counts use the recorder's own hash chain inside each episode:
    `chains_started` is records at seq 0 (a fresh process or container),
    `unlinked` is records whose predecessor is not in the range (carried in
    from before the commitment, or never logged; with start 0 only the
    latter), `chain_forks` is two records claiming the same predecessor, and
    `chain_breaks` is a record whose own hash or seq does not add up.
    """

    ok: bool
    reason: str = ""
    size: int = 0
    start: int = 0
    deployment_id: str | None = None
    commitment_ts: int | None = None
    verdict: str | None = None
    episodes: int = 0
    first_episode_ts: int | None = None
    last_episode_ts: int | None = None
    manifest_mismatches: int = 0
    evidence_packs: int = 0
    revalidations: list[tuple[int, int, tuple[str, ...]]] = field(default_factory=list)
    later_commitments: list[tuple[int, str, str]] = field(default_factory=list)
    chains_started: int = 0
    unlinked: int = 0
    chain_forks: int = 0
    chain_breaks: int = 0
    ts_regressions: int = 0

    def __bool__(self) -> bool:
        return self.ok


def verify_timeline(
    checkpoint: LogCheckpoint | SignedCheckpoint,
    start: int,
    frontier: Sequence[bytes],
    entries: Sequence[bytes],
    *,
    commitment: str,
) -> Timeline:
    """Check entries [start, checkpoint.size) against a checkpoint, offline.

    Verify the checkpoint's signatures first (`verify_cosigned`): this
    function trusts its (size, root) and proves the entries against it. A
    match means nothing was dropped, inserted, altered or reordered between
    the commitment and the checkpoint.
    """
    cp = checkpoint.checkpoint if isinstance(checkpoint, SignedCheckpoint) else checkpoint
    got = root_from_range(start, frontier, (leaf_hash(bytes(e)) for e in entries))
    if got is None:
        return Timeline(False, "frontier has the wrong shape for the start index", cp.size, start)
    size, root = got
    if size != cp.size:
        return Timeline(
            False, f"range ends at {size}, checkpoint is at {cp.size}", cp.size, start
        )
    if root != cp.root:
        return Timeline(
            False,
            "entries do not reproduce the checkpointed root: something was "
            "dropped, inserted, altered or reordered",
            cp.size, start,
        )

    decoded = []
    for offset, data in enumerate(entries):
        try:
            decoded.append(decode_entry(bytes(data)))
        except WireViolation as exc:
            return Timeline(False, f"entry {start + offset}: {exc}", cp.size, start)

    if not decoded or decoded[0]["type"] != "eval_commitment" or decoded[0]["commitment"] != commitment:
        return Timeline(False, "range does not start with that commitment", cp.size, start)

    first = decoded[0]
    t = Timeline(
        True, "", cp.size, start,
        deployment_id=first["deployment_id"],
        commitment_ts=first["ts"],
        verdict=first["verdict"],
    )
    dep, manifest = first["deployment_id"], first["manifest_hash"]
    last_ts = first["ts"]
    known: dict[str, int] = {}
    consumed: set[str] = set()

    for offset, entry in enumerate(decoded[1:], start=1):
        if entry["deployment_id"] != dep:
            continue
        index = start + offset
        if entry["ts"] < last_ts:
            t.ts_regressions += 1
        last_ts = max(last_ts, entry["ts"])
        kind = entry["type"]
        if kind == "episode":
            rec = entry["record"]
            t.episodes += 1
            if t.first_episode_ts is None:
                t.first_episode_ts = entry["ts"]
            t.last_episode_ts = entry["ts"]
            if rec["manifest_hash"] != manifest:
                t.manifest_mismatches += 1
            _thread(t, rec, known, consumed)
        elif kind == "revalidation":
            if entry["commitment"] in (None, commitment):
                t.revalidations.append((index, entry["ts"], tuple(entry["reasons"])))
        elif kind == "evidence_pack":
            if entry["commitment"] == commitment:
                t.evidence_packs += 1
        elif kind == "eval_commitment":
            t.later_commitments.append((index, entry["commitment"], entry["verdict"]))
    return t


def _thread(t: Timeline, rec: dict, known: dict[str, int], consumed: set[str]) -> None:
    if _chain_entry_hash(rec) != rec["entry_hash"]:
        t.chain_breaks += 1
        return
    prev, seq = rec["prev_hash"], rec["seq"]
    if prev == GENESIS and seq == 0:
        t.chains_started += 1
    elif prev in known:
        if known[prev] + 1 != seq:
            t.chain_breaks += 1
    else:
        t.unlinked += 1
    if prev in consumed and prev != GENESIS:
        t.chain_forks += 1
    consumed.add(prev)
    known[rec["entry_hash"]] = seq
