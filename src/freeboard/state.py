"""Durable recorder state, so an ephemeral container does not break the chain.

Agents run in things that restart. Without persistence every restart begins a
fresh chain with a fresh salt, which has two consequences and both are bad: the
hash chain shows a discontinuity an auditor has to triage, and the task hashes
change, so the same ticket looks like a different task and the deployment's own
experience is silently split in two. An auditor drowning in false-positive
discontinuities stops reading them, which is worse than not chaining at all.

Two values have to survive a restart:

    salt   the per-deployment pseudonymisation key. If it changes, identifiers
           stop being linkable and credibility loses the exposure it had.
    chain  seq and head, so the next record commits to the last one written.

The salt is the closest thing this SDK has to a secret: anyone holding it can
confirm a guessed task id by recomputing its hash. It is not a capability --
it grants no access -- but it should be stored with the deployment's other
configuration secrets rather than in a repo.

Corruption is loud on purpose. A truncated or edited state file raises rather
than quietly starting a new chain, because silently restarting is exactly the
behaviour a vendor would want if they were trying to lose a bad episode.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from .chain import GENESIS, ChainState

STATE_VERSION = 1


class StateError(RuntimeError):
    """The state file exists but cannot be trusted."""


@dataclass
class RecorderState:
    deployment_id: str
    salt: str
    seq: int = 0
    head: str = GENESIS

    def chain_state(self) -> ChainState:
        return ChainState(seq=self.seq, head=self.head)

    def absorb(self, chain: ChainState) -> None:
        self.seq = chain.seq
        self.head = chain.head

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": STATE_VERSION,
            "deployment_id": self.deployment_id,
            "salt": self.salt,
            "seq": self.seq,
            "head": self.head,
        }

    def save(self, path: str | Path) -> None:
        """Atomic write: a crash mid-save must not corrupt the chain head."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        try:
            os.chmod(path, 0o600)  # best effort; a no-op on some filesystems
        except OSError:
            pass

    @classmethod
    def load(cls, path: str | Path, deployment_id: str) -> RecorderState | None:
        """Return the stored state, or None if there is nothing yet.

        Raises rather than returning None when the file exists but is wrong:
        starting a fresh chain over a damaged one destroys the evidence that
        something happened to it.
        """
        path = Path(path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StateError(f"recorder state at {path} is unreadable: {exc}") from exc

        for key in ("version", "deployment_id", "salt", "seq", "head"):
            if key not in data:
                raise StateError(f"recorder state at {path} is missing {key!r}")
        if data["version"] != STATE_VERSION:
            raise StateError(
                f"recorder state at {path} is version {data['version']}, "
                f"expected {STATE_VERSION}"
            )
        if data["deployment_id"] != deployment_id:
            raise StateError(
                f"recorder state at {path} belongs to deployment "
                f"{data['deployment_id']!r}, not {deployment_id!r} -- refusing to "
                "continue another deployment's chain"
            )
        if not isinstance(data["seq"], int) or data["seq"] < 0:
            raise StateError(f"recorder state at {path} has a bad seq")
        if not isinstance(data["head"], str) or len(data["head"]) != 64:
            raise StateError(f"recorder state at {path} has a bad head")

        return cls(
            deployment_id=data["deployment_id"],
            salt=data["salt"],
            seq=data["seq"],
            head=data["head"],
        )

    @classmethod
    def load_or_create(
        cls, path: str | Path | None, deployment_id: str, salt: str = ""
    ) -> RecorderState:
        if path is not None:
            existing = cls.load(path, deployment_id)
            if existing is not None:
                return existing
        return cls(deployment_id=deployment_id, salt=salt or uuid.uuid4().hex)
