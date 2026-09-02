"""Persistence for the bureau.

SQLite on purpose. The bureau runs on one small box, the write volume is a
handful of batches per deployment per day, and a single file that can be copied,
checksummed and restored is worth more here than throughput. It is also stdlib,
so the box needs nothing installed.

What is stored is exactly what the SDK emits: validated wire records, which are
counts, enums and hashes. No prompts, no payloads, no tool names, no raw
identifiers. That is worth stating precisely, because the moment a server exists
the privacy claim changes shape -- it stops being "nothing leaves your process"
and becomes "what leaves your process contains no content, and here is the
schema that enforces it."

Two tables carry the design:

    episodes     every accepted record, with an `admitted` flag
    deployments  one row per contributing deployment: chain head, counts,
                 admission state

The `admitted` flag is the whole anti-poisoning design. Accepting a record and
letting it move a published base rate are different decisions, and conflating
them would mean anyone could rewrite the asset with a curl command.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS deployments (
    deployment_id   TEXT PRIMARY KEY,
    role            TEXT NOT NULL,
    envelope_class  TEXT NOT NULL,
    manifest_hash   TEXT NOT NULL,
    head            TEXT NOT NULL,
    next_seq        INTEGER NOT NULL,
    n_episodes      INTEGER NOT NULL DEFAULT 0,
    first_seen      REAL NOT NULL,
    last_seen       REAL NOT NULL,
    admitted        INTEGER NOT NULL DEFAULT 0,
    admitted_at     REAL
);

CREATE TABLE IF NOT EXISTS episodes (
    deployment_id   TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    entry_hash      TEXT NOT NULL,
    role            TEXT NOT NULL,
    envelope_class  TEXT NOT NULL,
    task_hash       TEXT NOT NULL,
    pass_index      INTEGER NOT NULL,
    resolved        INTEGER,
    escalated       INTEGER NOT NULL,
    n_actions       INTEGER NOT NULL,
    error_rate      REAL NOT NULL,
    repeat_rate     REAL NOT NULL,
    tool_entropy    REAL NOT NULL,
    n_irreversible  INTEGER NOT NULL,
    duration_ms     INTEGER NOT NULL,
    unreported_actions INTEGER NOT NULL DEFAULT 0,
    attestation_source TEXT NOT NULL DEFAULT 'none',
    received_at     REAL NOT NULL,
    PRIMARY KEY (deployment_id, seq)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    deployment_id   TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    head            TEXT NOT NULL,
    created_at      REAL NOT NULL,
    received_at     REAL NOT NULL,
    PRIMARY KEY (deployment_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_episodes_role ON episodes(role, envelope_class);
CREATE INDEX IF NOT EXISTS idx_deployments_role ON deployments(role, admitted);
"""

# The columns pulled out of a wire record into their own SQL fields. Anything
# else in the record is kept in neither -- the wire schema is the contract, and
# storing a subset of it is deliberate: the bureau needs frequency inputs, not
# a mirror of the client.
EPISODE_FIELDS = (
    "seq", "entry_hash", "role", "envelope_class", "task_hash", "pass_index",
    "resolved", "escalated", "n_actions", "error_rate", "repeat_rate",
    "tool_entropy", "n_irreversible", "duration_ms", "unreported_actions",
    "attestation_source",
)


@dataclass(frozen=True)
class IngestResult:
    accepted: int
    rejected: int
    reason: str | None
    head: str
    next_seq: int
    admitted: bool


class Store:
    def __init__(self, path: str | Path = "bureau.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- deployments -------------------------------------------------------

    def deployment(self, deployment_id: str) -> sqlite3.Row | None:
        cur = self._conn.execute(
            "SELECT * FROM deployments WHERE deployment_id = ?", (deployment_id,)
        )
        return cur.fetchone()

    def upsert_deployment(self, record: dict[str, Any], head: str, next_seq: int,
                          n_new: int) -> None:
        now = time.time()
        existing = self.deployment(record["deployment_id"])
        if existing is None:
            self._conn.execute(
                "INSERT INTO deployments (deployment_id, role, envelope_class,"
                " manifest_hash, head, next_seq, n_episodes, first_seen, last_seen)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (record["deployment_id"], record["role"], record["envelope_class"],
                 record["manifest_hash"], head, next_seq, n_new, now, now),
            )
        else:
            self._conn.execute(
                "UPDATE deployments SET head = ?, next_seq = ?,"
                " n_episodes = n_episodes + ?, last_seen = ?, envelope_class = ?,"
                " manifest_hash = ? WHERE deployment_id = ?",
                (head, next_seq, n_new, now, record["envelope_class"],
                 record["manifest_hash"], record["deployment_id"]),
            )

    def set_admitted(self, deployment_id: str, admitted: bool) -> None:
        self._conn.execute(
            "UPDATE deployments SET admitted = ?, admitted_at = ?"
            " WHERE deployment_id = ?",
            (1 if admitted else 0, time.time() if admitted else None, deployment_id),
        )
        self._conn.commit()

    # -- episodes ----------------------------------------------------------

    def insert_episodes(self, deployment_id: str, records: Iterable[dict]) -> int:
        now = time.time()
        rows = [
            tuple([deployment_id] + [r.get(f) for f in EPISODE_FIELDS] + [now])
            for r in records
        ]
        if not rows:
            return 0
        placeholders = ",".join("?" * (len(EPISODE_FIELDS) + 2))
        self._conn.executemany(
            f"INSERT OR IGNORE INTO episodes (deployment_id,"
            f" {','.join(EPISODE_FIELDS)}, received_at) VALUES ({placeholders})",
            rows,
        )
        return len(rows)

    def insert_checkpoint(self, payload: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO checkpoints (deployment_id, seq, head,"
            " created_at, received_at) VALUES (?,?,?,?,?)",
            (payload["deployment_id"], payload["seq"], payload["head"],
             payload.get("created_at", 0.0), time.time()),
        )
        self._conn.commit()

    def commit(self) -> None:
        self._conn.commit()

    # -- reading for the prior --------------------------------------------

    def class_observations(
        self, role: str, admitted_only: bool = True
    ) -> tuple[list[float], list[str]]:
        """Loss indicator and deployment id for every episode in a role class.

        Only admitted deployments count toward a published prior. Everything
        else is stored and ignored until it earns its way in.
        """
        sql = (
            "SELECT e.deployment_id AS did, e.resolved AS resolved"
            " FROM episodes e JOIN deployments d"
            " ON d.deployment_id = e.deployment_id"
            " WHERE e.role = ? AND e.resolved IS NOT NULL"
        )
        if admitted_only:
            sql += " AND d.admitted = 1"
        cur = self._conn.execute(sql, (role,))
        values: list[float] = []
        ids: list[str] = []
        for row in cur:
            values.append(0.0 if row["resolved"] else 1.0)
            ids.append(row["did"])
        return values, ids

    def role_summary(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT role,"
            " COUNT(*) AS deployments,"
            " SUM(admitted) AS admitted,"
            " SUM(n_episodes) AS episodes"
            " FROM deployments GROUP BY role ORDER BY episodes DESC"
        )
        return [dict(r) for r in cur]

    def stats(self) -> dict[str, Any]:
        d = self._conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(admitted),0) a FROM deployments"
        ).fetchone()
        e = self._conn.execute("SELECT COUNT(*) n FROM episodes").fetchone()
        c = self._conn.execute("SELECT COUNT(*) n FROM checkpoints").fetchone()
        return {
            "deployments": d["n"],
            "admitted_deployments": d["a"],
            "episodes": e["n"],
            "checkpoints": c["n"],
        }
