"""The client must never break the thing it is measuring.

Every test here is really one property restated: an unreachable, slow, broken
or hostile bureau must cost the agent nothing. A monitoring SDK that raises
inside a customer's agent loop is uninstallable.
"""

import json
import random
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from bureau.app import Handler
from bureau.store import Store
from freeboard import (
    BureauClient,
    BureauError,
    BureauSink,
    Capability,
    Recorder,
    Reversibility,
    ToolSpec,
    read_jsonl,
)

TOOLS = [
    ToolSpec("read_case", capabilities=frozenset({Capability.PRIVATE_DATA}),
             reversibility=Reversibility.REVERSIBLE)
]


@pytest.fixture()
def bureau(tmp_path):
    Handler.store = Store(tmp_path / "bureau.db")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    Handler.store.close()


def run_agent(rec, n, rate=0.3, seed=0):
    rng = random.Random(seed)
    for i in range(n):
        with rec.episode(f"t{i}") as ep:
            ep.action("read_case")
            ep.resolve(success=rng.random() > rate)


# --- the happy path --------------------------------------------------------


def test_records_reach_the_bureau(bureau, tmp_path):
    sink = BureauSink(bureau, spool=tmp_path / "spool.jsonl", batch_size=10)
    rec = Recorder(deployment_id="acme", role="ops", tools=TOOLS, sink=sink,
                   state_path=str(tmp_path / "s.json"))
    run_agent(rec, 25)
    reply = sink.flush()
    assert reply["n_episodes"] == 25
    assert sink.sent_through == 24
    assert sink.last_error is None
    assert BureauClient(bureau).health()["episodes"] == 25


def test_the_prior_comes_back_in_the_same_round_trip(bureau, tmp_path):
    sink = BureauSink(bureau, spool=tmp_path / "spool.jsonl", batch_size=1000)
    rec = Recorder(deployment_id="acme", role="ops", tools=TOOLS, sink=sink,
                   state_path=str(tmp_path / "s.json"))
    run_agent(rec, 12)
    sink.flush()
    assert sink.last_prior is not None
    assert sink.last_prior["role"] == "ops"


def test_batching_forwards_without_an_explicit_flush(bureau, tmp_path):
    sink = BureauSink(bureau, spool=tmp_path / "spool.jsonl", batch_size=5)
    rec = Recorder(deployment_id="acme", role="ops", tools=TOOLS, sink=sink,
                   state_path=str(tmp_path / "s.json"))
    run_agent(rec, 20)
    assert BureauClient(bureau).health()["episodes"] >= 15


def test_checkpoints_anchor(bureau, tmp_path):
    sink = BureauSink(bureau, spool=tmp_path / "spool.jsonl")
    rec = Recorder(deployment_id="acme", role="ops", tools=TOOLS, sink=sink,
                   state_path=str(tmp_path / "s.json"))
    run_agent(rec, 8)
    sink.flush()
    assert sink.anchor(rec.checkpoint())["anchored"] is True


# --- failing open ----------------------------------------------------------


def test_an_unreachable_bureau_does_not_break_the_agent(tmp_path):
    sink = BureauSink("http://127.0.0.1:1", spool=tmp_path / "spool.jsonl",
                      batch_size=5, timeout=0.4)
    rec = Recorder(deployment_id="acme", role="ops", tools=TOOLS, sink=sink,
                   state_path=str(tmp_path / "s.json"))
    run_agent(rec, 20)          # must not raise
    assert sink.flush() is None
    assert sink.last_error is not None
    # and nothing was lost: it is all on disk
    assert len(list(read_jsonl(tmp_path / "spool.jsonl"))) == 20


def test_a_backlog_is_delivered_once_the_bureau_returns(bureau, tmp_path):
    spool = tmp_path / "spool.jsonl"
    down = BureauSink("http://127.0.0.1:1", spool=spool, batch_size=5, timeout=0.4)
    rec = Recorder(deployment_id="acme", role="ops", tools=TOOLS, sink=down,
                   state_path=str(tmp_path / "s.json"))
    run_agent(rec, 15)
    down._spool.close()

    back_up = BureauSink(bureau, spool=spool, batch_size=1000)
    reply = back_up.flush()
    assert reply["n_episodes"] == 15


def test_a_refused_batch_does_not_raise_into_the_agent(bureau, tmp_path):
    sink = BureauSink(bureau, spool=tmp_path / "spool.jsonl", batch_size=1000)
    rec = Recorder(deployment_id="acme", role="ops", tools=TOOLS, sink=sink,
                   state_path=str(tmp_path / "s.json"))
    run_agent(rec, 5)
    sink.flush()
    # corrupt the spool with a record the bureau will reject
    with open(tmp_path / "spool.jsonl", "a", encoding="utf-8") as fh:
        fh.write("not json at all\n")
    sink.sent_through = None
    assert sink.flush() is None          # swallowed
    assert sink.last_error is not None


def test_a_conflict_resyncs_instead_of_wedging(bureau, tmp_path):
    """A flush that succeeded server-side after the connection dropped leaves
    the client believing it sent nothing. It must recover, not loop."""
    sink = BureauSink(bureau, spool=tmp_path / "spool.jsonl", batch_size=1000)
    rec = Recorder(deployment_id="acme", role="ops", tools=TOOLS, sink=sink,
                   state_path=str(tmp_path / "s.json"))
    run_agent(rec, 10)
    sink.flush()
    assert sink.sent_through == 9

    sink.sent_through = None      # pretend we never got the reply
    run_agent(rec, 5)
    reply = sink.flush()
    assert reply is not None
    assert reply["n_episodes"] == 15
    assert sink.sent_through == 14


# --- the client itself -----------------------------------------------------


def test_client_raises_where_the_sink_swallows(bureau, tmp_path):
    client = BureauClient(bureau)
    with pytest.raises(BureauError) as exc:
        client.submit([{"nonsense": True}])
    assert exc.value.status == 400


def test_prior_route(bureau, tmp_path):
    assert BureauClient(bureau).prior("ops")["available"] is False
