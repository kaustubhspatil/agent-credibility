"""Signed checkpoints, and witnesses that cosign only what extends what they saw.

A checkpoint is (origin, size, root, timestamp), signed by the log. On its own
it proves only that the log operator said so: an operator who rewrites history
signs the new root just as happily. A witness is a separate party that keeps
the last checkpoint it cosigned for each log and cosigns a new one only after
checking a consistency proof from that stored checkpoint. After that, a
rewrite of anything the witness has seen needs a hash collision to get past
it, and a relying party that insists on witness cosignatures cannot be handed
a rewritten tree.

What one witness cannot see is a split view: the operator showing tree A to
one witness and tree B to another. Each branch is consistent with itself.
That is caught by requiring a quorum of witnesses, or by comparing checkpoints
between parties (`compare_checkpoints`). See docs/WITNESSED_LOG.md.

Signed messages are domain separated text, in the spirit of C2SP signed notes
but not wire compatible with them:

    freeboard-checkpoint/v1\\n<origin>\\n<size>\\n<root hex>\\n<timestamp ms>\\n
    freeboard-cosignature/v1\\n<witness>\\n<timestamp ms>\\n + checkpoint body

`HmacSigner` is for tests and demos. It is not a public key signature: anyone
who can verify it can forge it. `Ed25519Signer` is the real one and needs the
optional `cryptography` package.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from .state import StateError
from .tlog import HASH_SIZE, verify_consistency

try:  # optional: real public key signatures
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives import serialization as _serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey as _Ed25519PrivateKey,
        Ed25519PublicKey as _Ed25519PublicKey,
    )

    HAVE_ED25519 = True
except ImportError:  # pragma: no cover, depends on the environment
    HAVE_ED25519 = False

__all__ = [
    "HAVE_ED25519",
    "CheckpointVerdict",
    "Cosignature",
    "Ed25519Signer",
    "Ed25519Verifier",
    "ForkCheck",
    "HmacSigner",
    "LogCheckpoint",
    "LogSignature",
    "SignedCheckpoint",
    "Signer",
    "Verifier",
    "Witness",
    "WitnessRefusal",
    "compare_checkpoints",
    "cosign_message",
    "sign_checkpoint",
    "verify_cosigned",
    "verify_log_signature",
]

# Printable, no whitespace: a newline in an origin would make two different
# checkpoints encode to the same signed body.
_NAME = re.compile(r"\A[a-z0-9][a-z0-9_.:/\-]{0,127}\Z")
_KEY_ID = re.compile(r"\A[0-9a-f]{16}\Z")

CHECKPOINT_TAG = "freeboard-checkpoint/v1"
COSIGNATURE_TAG = "freeboard-cosignature/v1"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _check_name(kind: str, value: str) -> None:
    if not isinstance(value, str) or not _NAME.match(value):
        raise ValueError(f"{kind} {value!r} must match {_NAME.pattern}")


def _check_ms(kind: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{kind} must be a non negative int of milliseconds")


# --- checkpoints -----------------------------------------------------------


@dataclass(frozen=True)
class LogCheckpoint:
    """(origin, size, root, timestamp). The timestamp is the log's claim."""

    origin: str
    size: int
    root: bytes
    timestamp: int

    def __post_init__(self) -> None:
        _check_name("origin", self.origin)
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("size must be a non negative int")
        if not isinstance(self.root, bytes) or len(self.root) != HASH_SIZE:
            raise ValueError("root must be 32 bytes")
        _check_ms("timestamp", self.timestamp)

    def body(self) -> bytes:
        return (
            f"{CHECKPOINT_TAG}\n{self.origin}\n{self.size}\n"
            f"{self.root.hex()}\n{self.timestamp}\n"
        ).encode("ascii")

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "size": self.size,
            "root": self.root.hex(),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LogCheckpoint:
        return cls(
            origin=data["origin"],
            size=data["size"],
            root=bytes.fromhex(data["root"]),
            timestamp=data["timestamp"],
        )


@dataclass(frozen=True)
class LogSignature:
    key_id: str
    algorithm: str
    value: bytes

    def to_dict(self) -> dict[str, Any]:
        return {"key_id": self.key_id, "algorithm": self.algorithm, "value": self.value.hex()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LogSignature:
        return cls(data["key_id"], data["algorithm"], bytes.fromhex(data["value"]))


@dataclass(frozen=True)
class Cosignature:
    witness: str
    key_id: str
    algorithm: str
    timestamp: int
    value: bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "witness": self.witness,
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "timestamp": self.timestamp,
            "value": self.value.hex(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Cosignature:
        return cls(
            data["witness"], data["key_id"], data["algorithm"],
            data["timestamp"], bytes.fromhex(data["value"]),
        )


@dataclass(frozen=True)
class SignedCheckpoint:
    checkpoint: LogCheckpoint
    signature: LogSignature
    cosignatures: tuple[Cosignature, ...] = field(default_factory=tuple)

    def with_cosignature(self, cosig: Cosignature) -> SignedCheckpoint:
        kept = tuple(c for c in self.cosignatures if c.witness != cosig.witness)
        return replace(self, cosignatures=kept + (cosig,))

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint.to_dict(),
            "signature": self.signature.to_dict(),
            "cosignatures": [c.to_dict() for c in self.cosignatures],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SignedCheckpoint:
        return cls(
            LogCheckpoint.from_dict(data["checkpoint"]),
            LogSignature.from_dict(data["signature"]),
            tuple(Cosignature.from_dict(c) for c in data.get("cosignatures", ())),
        )


def cosign_message(checkpoint: LogCheckpoint, witness: str, timestamp: int) -> bytes:
    _check_name("witness", witness)
    _check_ms("timestamp", timestamp)
    head = f"{COSIGNATURE_TAG}\n{witness}\n{timestamp}\n".encode("ascii")
    return head + checkpoint.body()


# --- signers ---------------------------------------------------------------


@runtime_checkable
class Verifier(Protocol):
    key_id: str
    algorithm: str

    def verify(self, message: bytes, signature: bytes) -> bool: ...


@runtime_checkable
class Signer(Protocol):
    key_id: str
    algorithm: str

    def sign(self, message: bytes) -> bytes: ...

    def verifier(self) -> Verifier: ...


class HmacSigner:
    """HMAC sha256 over the message. FOR TESTS AND DEMOS ONLY.

    This is a shared secret, not a public key signature. Whoever can verify
    it holds the key and can therefore forge it, so it proves nothing to a
    third party and a "fork proof" made of HMAC signatures convinces no one
    but the key holder. Use `Ed25519Signer` for anything real.
    """

    algorithm = "hmac-sha256"

    def __init__(self, key: bytes) -> None:
        if not isinstance(key, bytes) or len(key) < 32:
            raise ValueError("an HMAC key must be at least 32 random bytes")
        self._key = key
        self.key_id = hashlib.sha256(b"freeboard-hmac-key-id/v1\n" + key).hexdigest()[:16]

    @classmethod
    def generate(cls) -> HmacSigner:
        return cls(os.urandom(32))

    def sign(self, message: bytes) -> bytes:
        return hmac.new(self._key, message, hashlib.sha256).digest()

    def verify(self, message: bytes, signature: bytes) -> bool:
        return isinstance(signature, bytes) and hmac.compare_digest(self.sign(message), signature)

    def verifier(self) -> HmacSigner:
        return self


def _ed25519_key_id(public: bytes) -> str:
    return hashlib.sha256(public).hexdigest()[:16]


def _require_ed25519() -> None:
    if not HAVE_ED25519:
        raise RuntimeError(
            "Ed25519 needs the optional 'cryptography' package; "
            "freeboard itself stays dependency free"
        )


class Ed25519Verifier:
    algorithm = "ed25519"

    def __init__(self, public_key: bytes) -> None:
        _require_ed25519()
        if not isinstance(public_key, bytes) or len(public_key) != 32:
            raise ValueError("an Ed25519 public key is 32 bytes")
        self.public_key = public_key
        self.key_id = _ed25519_key_id(public_key)
        self._key = _Ed25519PublicKey.from_public_bytes(public_key)

    def verify(self, message: bytes, signature: bytes) -> bool:
        try:
            self._key.verify(signature, message)
        except (_InvalidSignature, TypeError, ValueError):
            return False
        return True


class Ed25519Signer:
    algorithm = "ed25519"

    def __init__(self, private_key: bytes) -> None:
        _require_ed25519()
        if not isinstance(private_key, bytes) or len(private_key) != 32:
            raise ValueError("an Ed25519 private key is 32 bytes")
        self._key = _Ed25519PrivateKey.from_private_bytes(private_key)
        self.public_key = self._key.public_key().public_bytes(
            _serialization.Encoding.Raw, _serialization.PublicFormat.Raw
        )
        self.key_id = _ed25519_key_id(self.public_key)

    @classmethod
    def generate(cls) -> Ed25519Signer:
        _require_ed25519()
        return cls(os.urandom(32))

    def sign(self, message: bytes) -> bytes:
        return self._key.sign(message)

    def verifier(self) -> Ed25519Verifier:
        return Ed25519Verifier(self.public_key)


# --- signing and verifying -------------------------------------------------


def sign_checkpoint(checkpoint: LogCheckpoint, signer: Signer) -> SignedCheckpoint:
    sig = LogSignature(signer.key_id, signer.algorithm, signer.sign(checkpoint.body()))
    return SignedCheckpoint(checkpoint, sig)


def _matches(verifier: Verifier, key_id: str, algorithm: str) -> bool:
    return verifier.key_id == key_id and verifier.algorithm == algorithm


def verify_log_signature(signed: SignedCheckpoint, verifier: Verifier) -> bool:
    sig = signed.signature
    if not _matches(verifier, sig.key_id, sig.algorithm):
        return False
    return bool(verifier.verify(signed.checkpoint.body(), sig.value))


@dataclass
class CheckpointVerdict:
    ok: bool
    reason: str
    witnesses: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.ok


def verify_cosigned(
    signed: SignedCheckpoint,
    log_verifier: Verifier,
    witnesses: Mapping[str, Verifier],
    quorum: int | None = None,
    *,
    max_age_ms: int | None = None,
    now_ms: int | None = None,
) -> CheckpointVerdict:
    """The relying party's check: the log signed it and `quorum` known
    witnesses cosigned it. A checkpoint that fails this is the operator's
    word only.

    `quorum` defaults to a strict majority of `witnesses`. Each honest
    witness cosigns one linear history, so two histories that fork cannot
    both reach a majority. A smaller quorum lets an operator show different
    histories to different relying parties (E1 measures this).

    With `max_age_ms`, cosignatures older than that do not count, so an
    operator cannot stop asking the witnesses and keep serving an old
    checkpoint as current.
    """
    if quorum is None:
        quorum = len(witnesses) // 2 + 1
    if quorum < 1:
        raise ValueError("quorum must be at least 1; zero witnesses is no witness")
    if not verify_log_signature(signed, log_verifier):
        return CheckpointVerdict(False, "log signature does not verify")
    oldest = None
    if max_age_ms is not None:
        oldest = (_now_ms() if now_ms is None else now_ms) - max_age_ms
    good = []
    for c in signed.cosignatures:
        verifier = witnesses.get(c.witness)
        if verifier is None or c.witness in good:
            continue
        if not _matches(verifier, c.key_id, c.algorithm):
            continue
        if oldest is not None and (not isinstance(c.timestamp, int) or c.timestamp < oldest):
            continue
        try:
            message = cosign_message(signed.checkpoint, c.witness, c.timestamp)
        except ValueError:
            continue
        if verifier.verify(message, c.value):
            good.append(c.witness)
    if len(good) < quorum:
        return CheckpointVerdict(
            False, f"{len(good)} valid cosignatures, quorum is {quorum}", tuple(good)
        )
    return CheckpointVerdict(True, "", tuple(good))


# --- the witness -----------------------------------------------------------


class WitnessRefusal(Exception):
    """The witness would not cosign. `code` says why; `evidence` holds the
    signed checkpoints that show it when there is something to show."""

    def __init__(self, code: str, message: str, evidence: tuple = ()) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.evidence = evidence


WITNESS_STATE_VERSION = 1


class Witness:
    """Cosigns a log's checkpoint only if it extends the last one cosigned.

    State is one signed checkpoint per origin, persisted before any
    cosignature is returned. A witness that forgets its state can be rolled
    back, so a damaged state file raises instead of resetting.

    The first checkpoint seen for an origin is accepted without a proof
    (there is nothing to be consistent with). Enrol witnesses when the log
    is created, or anything before first contact is unprotected.
    """

    def __init__(
        self,
        name: str,
        signer: Signer,
        logs: Mapping[str, Verifier],
        *,
        state_path: str | Path | None = None,
        clock: Callable[[], int] = _now_ms,
        max_skew_ms: int = 5 * 60 * 1000,
    ) -> None:
        _check_name("witness", name)
        for origin in logs:
            _check_name("origin", origin)
        self.name = name
        self._signer = signer
        self._logs = dict(logs)
        self._clock = clock
        self._max_skew_ms = max_skew_ms
        self._state_path = Path(state_path) if state_path is not None else None
        self._latest: dict[str, SignedCheckpoint] = {}
        if self._state_path is not None and self._state_path.exists():
            self._load()

    def latest(self, origin: str) -> SignedCheckpoint | None:
        return self._latest.get(origin)

    def verifier(self) -> Verifier:
        return self._signer.verifier()

    def cosign(self, signed: SignedCheckpoint, proof: Sequence[bytes] = ()) -> Cosignature:
        """Check, persist, then sign. Raises WitnessRefusal otherwise.

        The old size is always the witness's own stored one. A proof "from"
        some other size, chosen by the caller, would let the operator pick a
        point before the rewrite.
        """
        cp = signed.checkpoint
        verifier = self._logs.get(cp.origin)
        if verifier is None:
            raise WitnessRefusal("unknown_origin", f"not witnessing {cp.origin!r}")
        if not verify_log_signature(signed, verifier):
            raise WitnessRefusal("bad_signature", "log signature does not verify")

        now = self._clock()
        if cp.timestamp > now + self._max_skew_ms:
            raise WitnessRefusal("future_timestamp", f"{cp.timestamp} is ahead of {now}")

        prev_signed = self._latest.get(cp.origin)
        if prev_signed is not None:
            prev = prev_signed.checkpoint
            if cp.timestamp < prev.timestamp:
                raise WitnessRefusal(
                    "stale_timestamp", f"{cp.timestamp} is before {prev.timestamp}"
                )
            if cp.size < prev.size:
                raise WitnessRefusal(
                    "rollback", f"size {cp.size} after cosigning {prev.size}",
                    (prev_signed, signed),
                )
            if cp.size == prev.size and cp.root != prev.root:
                raise WitnessRefusal(
                    "fork", f"two roots signed for size {cp.size}", (prev_signed, signed)
                )
            if cp.size > prev.size and not verify_consistency(
                prev.size, cp.size, prev.root, cp.root, list(proof)
            ):
                raise WitnessRefusal(
                    "inconsistent",
                    f"no valid consistency proof from {prev.size} to {cp.size}",
                    (prev_signed, signed),
                )

        self._latest[cp.origin] = SignedCheckpoint(cp, signed.signature)
        if self._state_path is not None:
            self._save()
        value = self._signer.sign(cosign_message(cp, self.name, now))
        return Cosignature(self.name, self._signer.key_id, self._signer.algorithm, now, value)

    # persistence

    def _save(self) -> None:
        data = {
            "version": WITNESS_STATE_VERSION,
            "witness": self.name,
            "logs": {o: s.to_dict() for o, s in sorted(self._latest.items())},
        }
        _atomic_write_json(self._state_path, data)

    def _load(self) -> None:
        path = self._state_path
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["version"] != WITNESS_STATE_VERSION:
                raise StateError(f"witness state at {path} is version {data['version']}")
            if data["witness"] != self.name:
                raise StateError(
                    f"witness state at {path} belongs to {data['witness']!r}, not {self.name!r}"
                )
            latest = {o: SignedCheckpoint.from_dict(s) for o, s in data["logs"].items()}
        except StateError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise StateError(f"witness state at {path} is unreadable: {exc}") from exc
        for origin, signed in latest.items():
            if origin != signed.checkpoint.origin:
                raise StateError(f"witness state at {path} files {origin!r} under the wrong origin")
        self._latest = latest


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# --- comparing views -------------------------------------------------------


@dataclass
class ForkCheck:
    """`consistent`, `fork` (two roots signed for one size, which is proof),
    `unproven` (the log owes a consistency proof and did not supply a valid
    one; an honest log always can), or `invalid` (bad signature or origin)."""

    status: str
    detail: str = ""

    @property
    def split_view(self) -> bool:
        return self.status in ("fork", "unproven")


def compare_checkpoints(
    a: SignedCheckpoint,
    b: SignedCheckpoint,
    log_verifier: Verifier,
    proof: Sequence[bytes] = (),
) -> ForkCheck:
    """Gossip: do two parties' checkpoints describe one history?

    `proof` is a consistency proof from the smaller size to the larger,
    obtained from the log. This is how split views between witnesses, or
    between a witness and an auditor, come to light.
    """
    for s in (a, b):
        if not verify_log_signature(s, log_verifier):
            return ForkCheck("invalid", "a log signature does not verify")
    if a.checkpoint.origin != b.checkpoint.origin:
        return ForkCheck("invalid", "different origins")
    small, large = sorted((a.checkpoint, b.checkpoint), key=lambda c: c.size)
    if small.size == large.size:
        if small.root == large.root:
            return ForkCheck("consistent")
        return ForkCheck("fork", f"two roots signed for size {small.size}")
    if verify_consistency(small.size, large.size, small.root, large.root, list(proof)):
        return ForkCheck("consistent")
    return ForkCheck("unproven", f"no valid consistency proof from {small.size} to {large.size}")
