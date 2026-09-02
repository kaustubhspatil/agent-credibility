"""The bureau's ingest endpoint.

Three routes and no accounts:

    POST /v1/episodes     submit a batch of wire records, get your class prior back
    POST /v1/checkpoints  anchor a chain head outside your own control
    GET  /v1/priors       read a role's pooled prior
    GET  /v1/health       counts

The reciprocity is the response body, not a policy document. You POST your
sufficient statistics and the same round trip returns the pooled prior for your
class, with the credibility weight your own exposure has earned. Contribute and
you can price day zero; do not, and you wait to accumulate your own history.
That is the entire give-to-get, and it costs the contributor nothing they were
keeping.

`POST /v1/checkpoints` is the route that makes the SDK's tamper-evidence claim
true rather than aspirational. A hash chain only proves anything once its head
has been committed somewhere the writer cannot rewrite. Until this endpoint
existed, nothing in the system could make that claim honestly.

Standard library only: this runs on one small always-on box, and a server that
needs nothing installed is a server that still runs after an unattended reboot.
Terminate TLS at a reverse proxy in front of it -- there is deliberately no
crypto here beyond hashing.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from freeboard import GENESIS, WireViolation, validate, verify
from freeboard.estimate import components

from .admission import capped_weights, evaluate
from .divergence import class_divergence
from .store import Store

log = logging.getLogger("bureau")

MAX_BODY = 8 * 1024 * 1024   # an accountless endpoint needs a hard ceiling
MAX_BATCH = 5_000


class BureauError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# --- core logic, independent of HTTP so it can be tested directly ----------


def ingest_episodes(store: Store, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise BureauError(400, "empty batch")
    if len(records) > MAX_BATCH:
        raise BureauError(413, f"batch of {len(records)} exceeds {MAX_BATCH}")

    for i, record in enumerate(records):
        try:
            validate(record)
        except WireViolation as exc:
            raise BureauError(400, f"record {i}: {exc}") from exc

    deployment_ids = {r["deployment_id"] for r in records}
    if len(deployment_ids) != 1:
        raise BureauError(400, "a batch must come from one deployment")
    deployment_id = records[0]["deployment_id"]

    roles = {r["role"] for r in records}
    if len(roles) != 1:
        raise BureauError(400, "a batch must come from one role")
    role = records[0]["role"]

    records.sort(key=lambda r: r["seq"])

    existing = store.deployment(deployment_id)
    expected_seq = existing["next_seq"] if existing else 0
    start_hash = existing["head"] if existing else GENESIS

    if records[0]["seq"] != expected_seq:
        # Not an error the client can fix by retrying blindly: tell it where we are.
        raise BureauError(
            409,
            f"expected seq {expected_seq}, batch starts at {records[0]['seq']}",
        )

    result = verify(records, start_hash=start_hash, start_seq=expected_seq)
    if not result:
        raise BureauError(422, f"chain rejected: {result.reason}")

    head = records[-1]["entry_hash"]
    next_seq = records[-1]["seq"] + 1

    n_new = store.insert_episodes(deployment_id, records)
    store.upsert_deployment(records[0], head, next_seq, n_new)
    store.commit()

    row = store.deployment(deployment_id)
    decision = evaluate(
        n_episodes=row["n_episodes"],
        first_seen=row["first_seen"],
        n_checkpoints=_checkpoint_count(store, deployment_id),
        chain_intact=True,
    )
    if decision.admitted and not row["admitted"]:
        store.set_admitted(deployment_id, True)

    return {
        "accepted": n_new,
        "head": head,
        "next_seq": next_seq,
        "n_episodes": row["n_episodes"],
        "pool": {
            "admitted": bool(decision.admitted),
            "status": decision.summary,
        },
        "prior": class_prior(store, role, own_n=row["n_episodes"]),
    }


def _checkpoint_count(store: Store, deployment_id: str) -> int:
    cur = store._conn.execute(
        "SELECT COUNT(*) n FROM checkpoints WHERE deployment_id = ?",
        (deployment_id,),
    )
    return cur.fetchone()["n"]


def class_prior(store: Store, role: str, own_n: float = 0.0) -> dict[str, Any]:
    """The pooled prior for a role, with single-contributor influence capped."""
    values, ids = store.class_observations(role, admitted_only=True)
    n_deployments = len(set(ids))
    if n_deployments < 2:
        return {
            "role": role,
            "available": False,
            "reason": f"{n_deployments} admitted deployment(s); need at least 2",
            "n_deployments": n_deployments,
        }

    exposure: dict[str, float] = {}
    for rid in ids:
        exposure[rid] = exposure.get(rid, 0.0) + 1.0
    order = list(exposure)
    caps = dict(zip(order, capped_weights([exposure[r] for r in order])))
    weights = [caps[rid] / exposure[rid] for rid in ids]

    comp = components(values, ids, weights)
    out = {
        "role": role,
        "available": True,
        "mu": round(comp.mu, 6),
        "k": None if comp.k == float("inf") else round(comp.k, 4),
        "n_deployments": comp.n_risks,
        "n_episodes": int(sum(exposure.values())),
        "influence_capped": any(caps[r] < exposure[r] for r in order),
    }
    div = divergence_priors(store, role)
    if div:
        out["divergence"] = div
    if own_n and comp.k != float("inf"):
        z = comp.z(own_n)
        out["your_z"] = round(z, 4)
        out["episodes_to_z50"] = round(comp.episodes_for_z(0.5), 1)
    return out


def divergence_priors(store: Store, role: str) -> list[dict[str, Any]]:
    """What divergence looks like for this role, per attestation source.

    Never pooled across sources: a kernel observer counts execve, an egress
    proxy counts requests, an audit log counts whatever the platform chose to
    log. Averaging them produces a number about nothing.
    """
    out = []
    for source in sorted(store.attestation_sources(role)):
        diverged, magnitude, ids = store.divergence_observations(role, source)
        out.append(
            class_divergence(diverged, magnitude, ids, role, source).as_dict()
        )
    return out


def ingest_checkpoint(store: Store, payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("deployment_id", "seq", "head"):
        if key not in payload:
            raise BureauError(400, f"checkpoint missing {key!r}")
    if not isinstance(payload["head"], str) or len(payload["head"]) != 64:
        raise BureauError(400, "head is not a sha256 digest")

    row = store.deployment(payload["deployment_id"])
    if row is None:
        raise BureauError(404, "unknown deployment; submit episodes first")
    if payload["seq"] == row["next_seq"] and payload["head"] != row["head"]:
        raise BureauError(
            409, "checkpoint head disagrees with the chain we hold at that seq"
        )

    store.insert_checkpoint(payload)
    return {"anchored": True, "seq": payload["seq"],
            "checkpoints": _checkpoint_count(store, payload["deployment_id"])}


# --- HTTP -----------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "freeboard-bureau/0.1"
    store: Store

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("%s %s", self.address_string(), fmt % args)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise BureauError(411, "Content-Length required")
        if length > MAX_BODY:
            raise BureauError(413, f"body exceeds {MAX_BODY} bytes")
        try:
            return json.loads(self.rfile.read(length))
        except ValueError as exc:
            raise BureauError(400, f"invalid JSON: {exc}") from exc

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            payload = self._read_json()
            if route == "/v1/episodes":
                records = payload.get("episodes") if isinstance(payload, dict) else payload
                if not isinstance(records, list):
                    raise BureauError(400, "expected a list of episodes")
                self._send(200, ingest_episodes(self.store, records))
            elif route == "/v1/checkpoints":
                self._send(200, ingest_checkpoint(self.store, payload))
            else:
                raise BureauError(404, "no such route")
        except BureauError as exc:
            self._send(exc.status, {"error": exc.message})
        except Exception as exc:  # noqa: BLE001
            # Without this the handler drops the connection and a proxy in front
            # reports a bare 502, which says nothing about what went wrong.
            log.exception("unhandled")
            self._send(500, {"error": f"{type(exc).__name__}"})
        except Exception as exc:  # noqa: BLE001
            log.exception("unhandled")
            self._send(500, {"error": f"{type(exc).__name__}"})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/v1/priors":
                role = (parse_qs(parsed.query).get("role") or [None])[0]
                if not role:
                    self._send(200, {"roles": self.store.role_summary()})
                else:
                    self._send(200, class_prior(self.store, role))
            elif parsed.path == "/v1/divergence":
                role = (parse_qs(parsed.query).get("role") or [None])[0]
                if not role:
                    raise BureauError(400, "role is required")
                self._send(200, {"role": role,
                                 "sources": divergence_priors(self.store, role)})
            elif parsed.path == "/v1/health":
                self._send(200, {"ok": True, **self.store.stats()})
            else:
                raise BureauError(404, "no such route")
        except BureauError as exc:
            self._send(exc.status, {"error": exc.message})


def serve(db: str = "bureau.db", host: str = "127.0.0.1", port: int = 8080) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    Handler.store = Store(db)
    server = ThreadingHTTPServer((host, port), Handler)
    log.info("bureau listening on %s:%d, db=%s", host, port, db)
    log.info("bind to 127.0.0.1 and terminate TLS at a proxy; there is none here")
    server.serve_forever()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="bureau.db")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    serve(args.db, args.host, args.port)
