"""Persistence: an ephemeral container must not break the chain.

Without this, every restart begins a fresh chain and a fresh salt. The chain
shows a discontinuity an auditor has to triage, and task hashes change, so the
same ticket looks like a new task and the deployment's own experience is split
in two. False-positive discontinuities are worse than none: an auditor who
learns to ignore them will ignore a real one.
"""

import json

import pytest

from freeboard import (
    Capability,
    Recorder,
    RecorderState,
    Reversibility,
    StateError,
    ToolSpec,
    to_wire,
    verify,
)

TOOLS = [
    ToolSpec(
        "read_case",
        capabilities=frozenset({Capability.PRIVATE_DATA}),
        reversibility=Reversibility.REVERSIBLE,
    )
]


def _run(state_path, task_ids):
    rec = Recorder(
        deployment_id="acme", role="ops", tools=TOOLS, state_path=str(state_path)
    )
    for t in task_ids:
        with rec.episode(t) as ep:
            ep.action("read_case")
            ep.resolve(success=True)
    return rec, [to_wire(r) for r in rec.records]


def test_chain_continues_across_a_restart(tmp_path):
    sp = tmp_path / "state.json"
    first, w1 = _run(sp, ["t0", "t1", "t2"])
    second, w2 = _run(sp, ["t3", "t4"])

    assert [w["seq"] for w in w2] == [3, 4]
    assert w2[0]["prev_hash"] == w1[-1]["entry_hash"]
    assert verify(w1 + w2, expected_head=second.checkpoint().head)


def test_salt_survives_so_the_same_task_keeps_its_hash(tmp_path):
    sp = tmp_path / "state.json"
    _, w1 = _run(sp, ["ticket-1"])
    _, w2 = _run(sp, ["ticket-1"])
    assert w1[0]["task_hash"] == w2[0]["task_hash"]
    assert w1[0]["deployment_id"] == w2[0]["deployment_id"]


def test_without_a_state_path_each_process_starts_fresh(tmp_path):
    """Documented behaviour, not an accident -- fine for a notebook."""
    a = Recorder(deployment_id="acme", role="ops", tools=TOOLS)
    b = Recorder(deployment_id="acme", role="ops", tools=TOOLS)
    for rec in (a, b):
        with rec.episode("ticket-1") as ep:
            ep.resolve(success=True)
    assert a.records[0].task_hash != b.records[0].task_hash
    assert a.records[0].seq == b.records[0].seq == 0


def test_state_is_written_atomically_and_leaves_no_temp_files(tmp_path):
    sp = tmp_path / "nested" / "state.json"
    _run(sp, ["t0", "t1"])
    assert sp.exists()
    leftovers = [p for p in sp.parent.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
    saved = json.loads(sp.read_text())
    assert saved["seq"] == 2
    assert len(saved["head"]) == 64


def test_corrupt_state_raises_rather_than_silently_restarting(tmp_path):
    """Quietly starting a new chain over a damaged one destroys the evidence
    that anything happened to it -- and is what a vendor losing a bad episode
    would want."""
    sp = tmp_path / "state.json"
    _run(sp, ["t0"])
    sp.write_text("{not json")
    with pytest.raises(StateError):
        Recorder(deployment_id="acme", role="ops", tools=TOOLS, state_path=str(sp))


def test_state_from_another_deployment_is_refused(tmp_path):
    sp = tmp_path / "state.json"
    _run(sp, ["t0"])
    with pytest.raises(StateError, match="belongs to deployment"):
        Recorder(deployment_id="globex", role="ops", tools=TOOLS, state_path=str(sp))


@pytest.mark.parametrize("field", ["version", "salt", "seq", "head"])
def test_a_missing_or_bad_field_is_refused(tmp_path, field):
    sp = tmp_path / "state.json"
    _run(sp, ["t0"])
    data = json.loads(sp.read_text())
    del data[field]
    sp.write_text(json.dumps(data))
    with pytest.raises(StateError):
        RecorderState.load(sp, "acme")


def test_bad_head_length_is_refused(tmp_path):
    sp = tmp_path / "state.json"
    _run(sp, ["t0"])
    data = json.loads(sp.read_text())
    sp.write_text(json.dumps({**data, "head": "deadbeef"}))
    with pytest.raises(StateError, match="bad head"):
        RecorderState.load(sp, "acme")


def test_head_is_persisted_before_the_sink_sees_the_record(tmp_path):
    """If the process dies between persisting and sending, the state must be
    ahead of what was transmitted -- a gap, which verification reports. The
    reverse would silently re-issue a sequence number."""
    seen = []
    sp = tmp_path / "state.json"

    class Sink:
        def write(self, record):
            seen.append(json.loads(sp.read_text())["seq"])

    rec = Recorder(
        deployment_id="acme", role="ops", tools=TOOLS,
        state_path=str(sp), sink=Sink(),
    )
    for t in ("t0", "t1"):
        with rec.episode(t) as ep:
            ep.resolve(success=True)
    assert seen == [1, 2]  # state already advanced past the record being sent
