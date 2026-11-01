"""C2SP formats and the witness protocol, so freeboard's logs can be cosigned by existing witness networks.

freeboard's own checkpoints (witness.py) are domain separated text that only freeboard reads. This module speaks
the community formats instead:

- c2sp.org/signed-note: a text ending in a newline, an empty line, then signature lines
  "— <key name> <base64(4 byte key ID || signature)>". An Ed25519 key ID is
  SHA-256(name || 0x0A || 0x01 || public key)[:4]; a verifier key is "<name>+<hex key ID>+<base64(0x01 || key)>".
- c2sp.org/tlog-checkpoint: the note text is the origin, the tree size in decimal and the base64 root, one per
  line. The log signs that text with an ordinary Ed25519 note signature.
- c2sp.org/tlog-cosignature: a witness signs "cosignature/v1\\ntime <seconds>\\n" followed by the checkpoint
  text; the signature line carries the eight byte big endian timestamp, then the signature, under key ID
  SHA-256(name || 0x0A || 0x04 || public key)[:4]. Timestamps are POSIX seconds, not freeboard's milliseconds.
- c2sp.org/tlog-witness: POST <prefix>/add-checkpoint with "old <size>", up to 63 base64 consistency proof
  hashes, an empty line and the checkpoint note. 200 returns cosignature lines; 409 (text/x.tlog.size) returns the
  size the witness holds; 422 a proof that does not verify; 404 an unknown log; 403 no valid log signature; 400 a
  malformed request or an old size beyond the checkpoint.

Ed25519 only (the optional `cryptography` package). ML-DSA-44, which the specs recommend for new witnesses, is not
implemented. Standard library otherwise.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import struct
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import tlog
from .witness import _require_ed25519

__all__ = [
    "CosignatureSigner",
    "CosignatureVerifier",
    "NoteError",
    "NoteSigner",
    "NoteVerifier",
    "NoteWitness",
    "add_checkpoint",
    "checkpoint_text",
    "format_note",
    "log_note",
    "parse_checkpoint",
    "parse_note",
    "serve_witness",
    "verify_note",
]

DASH = "—"
TYPE_ED25519 = 0x01
TYPE_COSIGNATURE = 0x04
MAX_PROOF = 63
SIZE_TYPE = "text/x.tlog.size"


class NoteError(ValueError):
    """A note, key or request that does not follow the format, or a signature that does not verify."""


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    """Standard base64 with padding; a non canonical encoding is refused, as the specs require."""
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise NoteError(f"not base64: {text!r}") from exc
    if _b64(data) != text:
        raise NoteError(f"not canonical base64: {text!r}")
    return data


def _check_name(name: str) -> None:
    if not name or any(c.isspace() for c in name) or "+" in name:
        raise NoteError(f"a key name is non empty, with no spaces and no '+': {name!r}")


def key_id(name: str, kind: int, public_key: bytes) -> bytes:
    return hashlib.sha256(name.encode("utf-8") + b"\n" + bytes([kind]) + public_key).digest()[:4]


def _ed25519_private(seed: bytes):
    _require_ed25519()
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if len(seed) != 32:
        raise NoteError("an Ed25519 private key is 32 bytes")
    return Ed25519PrivateKey.from_private_bytes(seed)


def _ed25519_verify(public_key: bytes, signature: bytes, message: bytes) -> bool:
    _require_ed25519()
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except (InvalidSignature, ValueError):
        return False
    return True


def _public_bytes(private) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


# --- keys -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NoteVerifier:
    """An Ed25519 note verifier: a log's key (type 0x01) or a witness's cosignature key (type 0x04)."""

    name: str
    public_key: bytes
    kind: int = TYPE_ED25519

    def __post_init__(self) -> None:
        _check_name(self.name)
        if len(self.public_key) != 32:
            raise NoteError("an Ed25519 public key is 32 bytes")
        if self.kind not in (TYPE_ED25519, TYPE_COSIGNATURE):
            raise NoteError(f"signature type {self.kind:#x} is not Ed25519 (0x01) or an Ed25519 cosignature (0x04)")

    @property
    def key_id(self) -> bytes:
        return key_id(self.name, self.kind, self.public_key)

    def vkey(self) -> str:
        return f"{self.name}+{self.key_id.hex()}+{_b64(bytes([self.kind]) + self.public_key)}"

    @classmethod
    def from_vkey(cls, vkey: str) -> NoteVerifier:
        parts = vkey.split("+", 2)  # base64 may itself hold '+'
        if len(parts) != 3 or len(parts[1]) != 8:
            raise NoteError(f"a verifier key is <name>+<8 hex key ID>+<base64>: {vkey!r}")
        name, hexid, b64 = parts
        raw = _unb64(b64)
        if len(raw) != 33:
            raise NoteError("an Ed25519 verifier key is a type byte and 32 bytes")
        v = cls(name, raw[1:], raw[0])
        if v.key_id.hex() != hexid.lower() or hexid != hexid.lower():
            raise NoteError(f"key ID {hexid} does not match the key")
        return v

    def check(self, text: str, signature: bytes) -> int | None:
        """True when `signature` (the bytes after the key ID) signs `text`. For a cosignature key the text is the
        checkpoint text and the result is the cosignature's timestamp in seconds; otherwise 0."""
        if self.kind == TYPE_COSIGNATURE:
            if len(signature) != 8 + 64:
                return None
            (ts,) = struct.unpack(">Q", signature[:8])
            ok = _ed25519_verify(self.public_key, signature[8:], cosigned_message(text, ts))
            return ts if ok else None
        return 0 if _ed25519_verify(self.public_key, signature, text.encode("utf-8")) else None


class NoteSigner:
    """Signs note text with Ed25519 (type 0x01): a log signing its checkpoints."""

    kind = TYPE_ED25519

    def __init__(self, name: str, seed: bytes) -> None:
        _check_name(name)
        self.name = name
        self._key = _ed25519_private(seed)
        self.public_key = _public_bytes(self._key)

    @classmethod
    def generate(cls, name: str) -> NoteSigner:
        return cls(name, os.urandom(32))

    @classmethod
    def from_private_key(cls, skey: str) -> NoteSigner:
        """The Go x/mod/sumdb/note private key form: PRIVATE+KEY+<name>+<hex key ID>+<base64(0x01 || seed)>."""
        parts = skey.split("+", 4)
        if len(parts) != 5 or parts[:2] != ["PRIVATE", "KEY"]:
            raise NoteError("a private key is PRIVATE+KEY+<name>+<hex key ID>+<base64>")
        raw = _unb64(parts[4])
        if len(raw) != 33 or raw[0] != cls.kind:
            raise NoteError("an Ed25519 private key is the type byte 0x01 and a 32 byte seed")
        signer = cls(parts[2], raw[1:])
        if signer.verifier().key_id.hex() != parts[3]:
            raise NoteError(f"key ID {parts[3]} does not match the key")
        return signer

    def verifier(self) -> NoteVerifier:
        return NoteVerifier(self.name, self.public_key, self.kind)

    def _line(self, value: bytes) -> str:
        return f"{DASH} {self.name} {_b64(self.verifier().key_id + value)}\n"

    def sign(self, text: str) -> str:
        """One signature line over the note text."""
        _check_text(text)
        return self._line(self._key.sign(text.encode("utf-8")))


def cosigned_message(checkpoint: str, timestamp: int) -> bytes:
    return f"cosignature/v1\ntime {timestamp}\n{checkpoint}".encode()


class CosignatureSigner(NoteSigner):
    """A witness's checkpoint cosignature key (type 0x04)."""

    kind = TYPE_COSIGNATURE

    def sign(self, text: str, timestamp: int | None = None) -> str:  # type: ignore[override]
        parse_checkpoint(text)
        ts = int(time.time()) if timestamp is None else int(timestamp)
        return self._line(struct.pack(">Q", ts) + self._key.sign(cosigned_message(text, ts)))


CosignatureVerifier = NoteVerifier  # with kind=TYPE_COSIGNATURE


# --- notes ------------------------------------------------------------------------------------------------------


def _check_text(text: str) -> None:
    if not text.endswith("\n"):
        raise NoteError("note text ends in a newline")
    if any(ord(c) < 0x20 and c != "\n" for c in text) or "\x7f" in text:
        raise NoteError("note text has no control characters other than newline")


@dataclass(frozen=True)
class Signature:
    name: str
    key_id: bytes
    value: bytes


def format_note(text: str, lines: Iterable[str]) -> str:
    _check_text(text)
    lines = list(lines)
    if not lines:
        raise NoteError("a signed note has at least one signature line")
    return text + "\n" + "".join(lines)


def parse_note(note: str) -> tuple[str, list[Signature]]:
    """The text and the signatures. The separator is the last empty line."""
    cut = note.rfind("\n\n")
    if cut < 0 or not note.endswith("\n"):
        raise NoteError("a signed note is text, an empty line, and signature lines ending in a newline")
    text, rest = note[: cut + 1], note[cut + 2 :]
    _check_text(text)
    sigs = []
    seen = set()
    for line in rest[:-1].split("\n"):
        if not line.startswith(DASH + " "):
            raise NoteError(f"a signature line starts with an em dash and a space: {line!r}")
        try:
            name, b64 = line[2:].split(" ")
        except ValueError as exc:
            raise NoteError(f"a signature line is '— <name> <base64>': {line!r}") from exc
        _check_name(name)
        raw = _unb64(b64)
        if len(raw) < 5:
            raise NoteError("a signature is a 4 byte key ID and at least one byte")
        if (name, raw[:4]) in seen:
            raise NoteError(f"two signatures by {name} with one key ID")
        seen.add((name, raw[:4]))
        sigs.append(Signature(name, raw[:4], raw[4:]))
    return text, sigs


def verify_note(note: str, verifiers: Sequence[NoteVerifier]) -> tuple[str, list[tuple[NoteVerifier, int]]]:
    """The text, and each known key whose signature verifies with its timestamp (0 for a log signature). Unknown
    signatures are ignored; a known key whose signature fails rejects the note, and so does a note with none."""
    text, sigs = parse_note(note)
    known = {(v.name, v.key_id): v for v in verifiers}
    good = []
    for s in sigs:
        v = known.get((s.name, s.key_id))
        if v is None:
            continue
        ts = v.check(text, s.value)
        if ts is None:
            raise NoteError(f"the signature by {v.name} does not verify")
        good.append((v, ts))
    if not good:
        raise NoteError("no signature by a known key")
    return text, good


# --- checkpoints ------------------------------------------------------------------------------------------------


def checkpoint_text(origin: str, size: int, root: bytes, extensions: Sequence[str] = ()) -> str:
    _check_text(origin + "\n")
    if not origin or "\n" in origin:
        raise NoteError("an origin is one non empty line")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0 or len(root) != 32:
        raise NoteError("a checkpoint has a non negative size and a 32 byte root")
    if any(not e or "\n" in e for e in extensions):
        raise NoteError("extension lines are non empty")
    return "".join(f"{x}\n" for x in (origin, size, _b64(root), *extensions))


def parse_checkpoint(text: str) -> tuple[str, int, bytes, list[str]]:
    _check_text(text)
    lines = text[:-1].split("\n")
    if len(lines) < 3 or not all(lines):
        raise NoteError("a checkpoint is the origin, the size and the root, each on a non empty line")
    origin, size, root = lines[:3]
    if not size.isdigit() or (size != "0" and size.startswith("0")):
        raise NoteError(f"a tree size is decimal with no leading zeroes: {size!r}")
    raw = _unb64(root)
    if len(raw) != 32:
        raise NoteError("a root is 32 bytes")
    return origin, int(size), raw, lines[3:]


def log_note(log, origin: str, signer: NoteSigner, size: int | None = None) -> str:
    """A C2SP checkpoint note for a freeboard log (TransparencyLog, EvidenceLog or MerkleTree), signed by the log.
    The signer's name should be the origin."""
    size = log.size if size is None else size
    text = checkpoint_text(origin, size, log.root(size))
    return format_note(text, [signer.sign(text)])


# --- the witness ------------------------------------------------------------------------------------------------


class _Refused(Exception):
    def __init__(self, status: int, message: str, body: bytes | None = None, ctype: str = "text/plain") -> None:
        super().__init__(message)
        self.status, self.body, self.ctype = status, body if body is not None else (message + "\n").encode(), ctype


class NoteWitness:
    """A tlog-witness: it keeps, for each log it knows, the size and root it last cosigned, and cosigns a new
    checkpoint only when the request's old size is that size and a consistency proof links them. The first
    checkpoint of a log is accepted without proof (old size 0): enrol a witness before anything needs protecting.
    State is a JSON file written atomically; keep it on the witness's side, not the log's."""

    def __init__(self, signer: CosignatureSigner, logs: Mapping[str, NoteVerifier],
                 state_path: str | Path | None = None, clock: Callable[[], float] = time.time) -> None:
        if signer.kind != TYPE_COSIGNATURE:
            raise NoteError("a witness signs with a cosignature key (CosignatureSigner)")
        self.signer, self.logs, self._clock = signer, dict(logs), clock
        self.state_path = Path(state_path) if state_path is not None else None
        self._lock = threading.Lock()
        self._state: dict[str, dict] = {}
        if self.state_path is not None and self.state_path.exists():
            self._state = json.loads(self.state_path.read_text(encoding="utf-8"))

    def latest(self, origin: str) -> tuple[int, bytes] | None:
        st = self._state.get(origin)
        return None if st is None else (st["size"], bytes.fromhex(st["root"]))

    def _save(self) -> None:
        if self.state_path is None:
            return
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(self._state, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def add_checkpoint(self, body: bytes) -> tuple[int, bytes, str]:
        """The tlog-witness add-checkpoint call: (status, response body, content type)."""
        try:
            return 200, self._add(body).encode("utf-8"), "text/plain; charset=utf-8"
        except _Refused as r:
            return r.status, r.body, r.ctype

    def _add(self, body: bytes) -> str:
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _Refused(400, "the request is not UTF-8") from exc
        head, sep, note = text.partition("\n\n")
        if not sep:
            raise _Refused(400, "the request is the old size, proof lines, an empty line and the note")
        lines = head.split("\n")
        if not lines[0].startswith("old ") or not lines[0][4:].isdigit() or \
                (lines[0][4:] != "0" and lines[0][4:].startswith("0")):
            raise _Refused(400, "the first line is 'old <size>' in decimal with no leading zeroes")
        old = int(lines[0][4:])
        if len(lines) - 1 > MAX_PROOF:
            raise _Refused(400, f"at most {MAX_PROOF} proof lines")
        try:
            proof = [_unb64(p) for p in lines[1:]]
            if any(len(p) != 32 for p in proof):
                raise NoteError("a proof hash is 32 bytes")
            cp_text, _ = parse_note(note)
            origin, size, root, _ = parse_checkpoint(cp_text)
        except NoteError as exc:
            raise _Refused(400, str(exc)) from exc
        log_key = self.logs.get(origin)
        if log_key is None:
            raise _Refused(404, f"unknown log {origin}")
        try:
            verify_note(note, [log_key])
        except NoteError as exc:
            raise _Refused(403, str(exc)) from exc
        if old > size:
            raise _Refused(400, "the old size is beyond the checkpoint")
        with self._lock:
            held = self.latest(origin)
            held_size, held_root = held if held is not None else (0, None)
            if old != held_size:
                raise _Refused(409, "old size mismatch", f"{held_size}\n".encode(), SIZE_TYPE)
            if old == 0 and proof:
                raise _Refused(422, "a proof from size 0 is empty")
            if held_root is not None and not tlog.verify_consistency(old, size, held_root, root, proof):
                raise _Refused(422, "the consistency proof does not verify")
            self._state[origin] = {"size": size, "root": root.hex()}
            self._save()
            return self.signer.sign(cp_text, int(self._clock()))


def serve_witness(witness: NoteWitness, host: str = "127.0.0.1", port: int = 0,
                  prefix: str = "") -> ThreadingHTTPServer:
    """An HTTP server for POST <prefix>/add-checkpoint. Call serve_forever() (or run it in a thread)."""
    path = prefix.rstrip("/") + "/add-checkpoint"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n <= 1 << 20 else b""
            if self.path != path:
                status, out, ctype = 404, b"no such path\n", "text/plain"
            elif n > 1 << 20:
                status, out, ctype = 400, b"request too large\n", "text/plain"
            else:
                status, out, ctype = witness.add_checkpoint(body)
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *_args) -> None:
            pass

    return ThreadingHTTPServer((host, port), Handler)


def add_checkpoint(url: str, old_size: int, proof: Sequence[bytes], note: str, timeout: float = 10) -> str:
    """Ask a tlog-witness to cosign; `url` is its submission prefix. Returns its cosignature lines. A 409 raises
    NoteError carrying the size the witness holds in `.size`."""
    body = f"old {old_size}\n" + "".join(_b64(p) + "\n" for p in proof) + "\n" + note
    req = urllib.request.Request(url.rstrip("/") + "/add-checkpoint", data=body.encode("utf-8"), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        text = err.read().decode("utf-8", "replace")
        e = NoteError(f"the witness refused ({err.code}): {text.strip()}")
        e.status = err.code  # type: ignore[attr-defined]
        if err.code == 409 and text.strip().isdigit():
            e.size = int(text.strip())  # type: ignore[attr-defined]
        raise e from None
