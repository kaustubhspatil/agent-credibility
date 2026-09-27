"""Shared fixtures for the witnessed log experiments.

Records are real wire records: one is produced by a Recorder and the rest
vary its outcome and counts, then get chained exactly as Recorder._emit
chains them. Every one passes wire.validate and is logged as a typed
episode entry, so the attacks run against the real formats.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from freeboard import Capability, Recorder, Reversibility, ToolSpec, to_wire  # noqa: E402
from freeboard.chain import ChainState  # noqa: E402
from freeboard.evidence import encode_entry, episode_entry  # noqa: E402

OUT = ROOT / "out"

_TOOLS = [
    ToolSpec("search_orders", capabilities=frozenset({Capability.PRIVATE_DATA}),
             reversibility=Reversibility.REVERSIBLE),
    ToolSpec("issue_refund", capabilities=frozenset({Capability.EXTERNAL_EFFECT}),
             reversibility=Reversibility.IRREVERSIBLE),
]


def _template() -> dict:
    rec = Recorder(deployment_id="acme-prod", role="customer_support", tools=_TOOLS, salt="x" * 32)
    with rec.episode("seed") as ep:
        ep.action("search_orders")
        ep.resolve(success=True)
    body = to_wire(rec.records[0])
    for key in ("seq", "prev_hash", "entry_hash"):
        body.pop(key)
    return body


TEMPLATE = _template()


def make_bodies(n: int, rng: random.Random) -> list[dict]:
    """Unchained record bodies with varied outcomes."""
    out = []
    for _ in range(n):
        b = dict(TEMPLATE)
        b["episode_id"] = "%016x" % rng.getrandbits(64)
        b["task_hash"] = "%016x" % rng.getrandbits(64)
        b["resolved"] = rng.random() < 0.8
        b["n_actions"] = rng.randint(1, 12)
        b["duration_ms"] = rng.randint(100, 60000)
        out.append(b)
    return out


def chain(bodies: list[dict]) -> list[dict]:
    """Chain bodies from genesis, the way the recorder does."""
    state = ChainState()
    chained = []
    for body in bodies:
        seq, prev, digest = state.advance(body)
        chained.append({**body, "seq": seq, "prev_hash": prev, "entry_hash": digest})
    return chained


def entries(chained: list[dict], ts0: int = 1_790_000_000_000) -> list[bytes]:
    return [encode_entry(episode_entry(r, ts=ts0 + i)) for i, r in enumerate(chained)]
