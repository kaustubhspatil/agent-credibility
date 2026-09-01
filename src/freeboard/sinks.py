"""Somewhere for records to go, and a way to get them back.

`Recorder(sink=...)` accepts anything with a `.write(record)` method, but the
package shipped no implementation, so a pilot had nowhere to put anything. This
is the file format a design partner hands back: one validated JSON object per
line, append-only, in chain order.

JSON Lines rather than a single JSON array, for one reason that matters here:
an append-only file has no closing bracket to rewrite, so a process killed
mid-run leaves a readable file with a complete prefix of the chain. A truncated
array is unparseable, and losing the whole log because a container was
rescheduled is not an acceptable failure mode for an audit trail.

Validation runs on the way out *and* on the way back in. Reading is where a
tampered or hand-edited file gets caught, and refusing to parse it is the point
-- `verify()` can then be run over the result to check the chain itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

from .wire import to_wire, validate

__all__ = ["JsonlSink", "read_jsonl", "write_checkpoint", "read_checkpoint"]


class JsonlSink:
    """Append validated episode records to a JSON Lines file.

    Usable directly as a `Recorder(sink=...)`, or as a context manager when the
    caller wants the handle closed deterministically.
    """

    def __init__(self, path: str | Path, *, fsync: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered append: each record is a complete line as soon as it is
        # written, so a reader never sees half a record.
        self._handle = open(self.path, "a", encoding="utf-8", buffering=1)
        self._fsync = fsync
        self.written = 0

    def write(self, record: Any) -> None:
        """Validate, then append. An invalid record raises and is not written."""
        payload = record if isinstance(record, dict) else to_wire(record)
        if isinstance(record, dict):
            validate(payload)
        line = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        self._handle.write(line + "\n")
        if self._fsync:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        self.written += 1

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> JsonlSink:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_jsonl(path: str | Path, *, strict: bool = True) -> Iterator[dict[str, Any]]:
    """Yield validated records from a JSON Lines file, in order.

    With `strict` (the default) a malformed or non-conforming line raises,
    naming the line number. Feed the result straight to `verify()` to check the
    chain:

        from freeboard import read_jsonl, verify
        verify(read_jsonl("episodes.jsonl"), expected_head=head)
    """
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                validate(payload)
            except Exception as exc:
                if strict:
                    raise ValueError(f"{path}: line {lineno}: {exc}") from exc
                continue
            yield payload


def write_checkpoint(path: str | Path, checkpoint: Any) -> None:
    """Write a chain-head commitment next to the records.

    A checkpoint sitting in the same directory as the records it commits to
    proves nothing on its own -- anyone who can edit one can edit the other.
    It becomes evidence only once a copy has been sent somewhere the writer
    cannot reach. This function is the local half; anchoring is the other.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint.to_wire() if hasattr(checkpoint, "to_wire") else checkpoint
    path.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8"
    )


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("deployment_id", "seq", "head"):
        if key not in data:
            raise ValueError(f"{path}: checkpoint is missing {key!r}")
    if not isinstance(data["head"], str) or len(data["head"]) != 64:
        raise ValueError(f"{path}: checkpoint head is not a sha256 digest")
    return data
