"""Talking to a bureau, without ever putting the agent at risk.

The governing constraint is not throughput or latency. It is that **this must
never break the thing it is measuring.** A monitoring SDK that raises inside a
customer's agent loop when a remote server is slow, down, or misconfigured is
uninstallable, and deservedly so. Everything here fails open: on any transport
error the episode is already durably on local disk, the exception is swallowed,
and the agent carries on.

That ordering is the design. Records are written to a local spool *first* and
submitted *second*. If the bureau is unreachable for a week, nothing is lost --
the spool holds, and the next successful flush carries the backlog. If the
process dies mid-flush, the chain state and the spool are both on disk and the
bureau's own sequence check resolves the overlap.

    from freeboard import Recorder, BureauSink

    sink = BureauSink("https://bureau.example", spool="/var/lib/freeboard/acme.jsonl")
    rec = Recorder(deployment_id="acme-prod", role="customer_support",
                   tools=[...], sink=sink, state_path="/var/lib/freeboard/acme.state")

    ...

    sink.flush()              # or let it batch automatically
    print(sink.last_prior)    # what the bureau sent back
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .sinks import JsonlSink, read_jsonl
from .wire import to_wire

log = logging.getLogger("freeboard.client")

DEFAULT_TIMEOUT = 10.0
DEFAULT_BATCH = 100


class BureauError(RuntimeError):
    """The bureau answered, and the answer was no."""

    def __init__(self, status: int, message: str, payload: dict | None = None) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message
        self.payload = payload or {}


class BureauClient:
    """Thin HTTP client. Raises; `BureauSink` is the one that swallows."""

    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: Any = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                payload = json.loads(raw or b"{}")
            except ValueError:
                payload = {"error": raw.decode(errors="replace")[:200]}
            raise BureauError(exc.code, payload.get("error", "unknown"), payload) from exc

    # -- routes ------------------------------------------------------------

    def submit(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("POST", "/v1/episodes", {"episodes": records})

    def anchor(self, checkpoint: Any) -> dict[str, Any]:
        payload = (
            checkpoint.to_wire() if hasattr(checkpoint, "to_wire") else checkpoint
        )
        return self._request("POST", "/v1/checkpoints", payload)

    def prior(self, role: str) -> dict[str, Any]:
        from urllib.parse import quote

        return self._request("GET", f"/v1/priors?role={quote(role)}")

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/v1/health")


class BureauSink:
    """A `Recorder` sink that spools locally and forwards in batches.

    Usable as `Recorder(sink=...)`. Never raises into the agent loop.
    """

    def __init__(
        self,
        base_url: str,
        spool: str | Path,
        batch_size: int = DEFAULT_BATCH,
        timeout: float = DEFAULT_TIMEOUT,
        client: BureauClient | None = None,
    ) -> None:
        self.client = client or BureauClient(base_url, timeout)
        self.spool_path = Path(spool)
        self._spool = JsonlSink(self.spool_path)
        self.batch_size = batch_size
        self.sent_through: int | None = None  # last seq the bureau confirmed
        self.last_prior: dict[str, Any] | None = None
        self.last_error: str | None = None
        self._pending = 0

    # -- the sink interface ------------------------------------------------

    def write(self, record: Any) -> None:
        """Durability first: on disk before the network is even attempted."""
        self._spool.write(record)
        self._pending += 1
        if self._pending >= self.batch_size:
            self.flush()

    # -- forwarding --------------------------------------------------------

    def _unsent(self) -> list[dict[str, Any]]:
        records = list(read_jsonl(self.spool_path))
        if self.sent_through is None:
            return records
        return [r for r in records if r["seq"] > self.sent_through]

    def flush(self, retry_on_conflict: bool = True) -> dict[str, Any] | None:
        """Send everything not yet confirmed. Returns the bureau's reply, or None.

        Swallows every transport failure by design. The spool is the source of
        truth, so a failed flush costs nothing but a later retry.
        """
        try:
            records = self._unsent()
        except Exception as exc:  # a corrupt spool must not kill the agent
            self.last_error = f"spool unreadable: {exc}"
            log.warning("freeboard: %s", self.last_error)
            return None
        if not records:
            self._pending = 0
            return None

        try:
            reply = self.client.submit(records)
        except BureauError as exc:
            if exc.status == 409 and retry_on_conflict:
                # The bureau is ahead of what we thought it had confirmed --
                # usually a flush that succeeded server-side after our
                # connection dropped. It tells us where it is; resync and retry
                # once rather than wedging forever on a conflict we can resolve.
                expected = _expected_seq(exc.message)
                if expected is not None:
                    self.sent_through = expected - 1
                    log.info("freeboard: resyncing to seq %d", expected)
                    return self.flush(retry_on_conflict=False)
            self.last_error = str(exc)
            log.warning("freeboard: bureau refused a batch (%s)", exc)
            return None
        except Exception as exc:  # noqa: BLE001 -- unreachable bureau is normal
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.info("freeboard: bureau unreachable, spooled (%s)", self.last_error)
            return None

        self.sent_through = records[-1]["seq"]
        self.last_prior = reply.get("prior")
        self.last_error = None
        self._pending = 0
        return reply

    def anchor(self, checkpoint: Any) -> dict[str, Any] | None:
        """Commit a chain head to the bureau. Also never raises."""
        try:
            return self.client.anchor(checkpoint)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.info("freeboard: could not anchor (%s)", self.last_error)
            return None

    def close(self) -> None:
        self.flush()
        self._spool.close()

    def __enter__(self) -> BureauSink:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _expected_seq(message: str) -> int | None:
    """Pull the sequence number out of the bureau's 409 message."""
    import re

    match = re.search(r"expected seq (\d+)", message)
    return int(match.group(1)) if match else None
