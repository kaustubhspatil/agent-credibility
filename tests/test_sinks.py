"""The file a design partner hands back.

The round-trip is the point: records written by the SDK must come back as
something `verify()` can check, or the chain guarantees stop at the process
boundary and are worth nothing.
"""

import json

import pytest

from freeboard import (
    Capability,
    JsonlSink,
    Recorder,
    Reversibility,
    ToolSpec,
    read_checkpoint,
    read_jsonl,
    to_wire,
    verify,
    write_checkpoint,
)

TOOLS = [
    ToolSpec("read_case", capabilities=frozenset({Capability.PRIVATE_DATA}),
             reversibility=Reversibility.REVERSIBLE),
    ToolSpec("send_email", capabilities=frozenset({Capability.EXTERNAL_EFFECT}),
             reversibility=Reversibility.IRREVERSIBLE),
]


def _record_to(path, state_path=None, n=5, start=0):
    sink = JsonlSink(path)
    rec = Recorder(deployment_id="acme", role="ops", tools=TOOLS,
                   sink=sink, state_path=state_path)
    for i in range(start, start + n):
        with rec.episode(f"t{i}") as ep:
            ep.action("read_case")
            ep.resolve(success=(i % 2 == 0))
    sink.close()
    return rec


def test_round_trip_preserves_the_chain(tmp_path):
    path = tmp_path / "episodes.jsonl"
    rec = _record_to(path)
    back = list(read_jsonl(path))
    assert len(back) == 5
    assert verify(back, expected_head=rec.checkpoint().head)


def test_read_back_equals_what_was_written(tmp_path):
    path = tmp_path / "episodes.jsonl"
    rec = _record_to(path, n=3)
    assert list(read_jsonl(path)) == [to_wire(r) for r in rec.records]


def test_appending_across_a_restart_still_verifies(tmp_path):
    """A rescheduled container appends to the same file and the same chain."""
    path = tmp_path / "episodes.jsonl"
    state = tmp_path / "state.json"
    _record_to(path, state_path=str(state), n=3, start=0)
    second = _record_to(path, state_path=str(state), n=3, start=3)

    back = list(read_jsonl(path))
    assert len(back) == 6
    assert [r["seq"] for r in back] == [0, 1, 2, 3, 4, 5]
    assert verify(back, expected_head=second.checkpoint().head)


def test_a_hand_edited_line_is_caught_on_read(tmp_path):
    path = tmp_path / "episodes.jsonl"
    _record_to(path, n=3)
    lines = path.read_text(encoding="utf-8").splitlines()
    doctored = json.loads(lines[1])
    doctored["resolved"] = True
    lines[1] = json.dumps(doctored)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # the record still validates -- it is well-formed -- but the chain does not
    back = list(read_jsonl(path))
    result = verify(back)
    assert not result
    assert result.first_bad_seq == 1


def test_a_malformed_line_raises_with_its_line_number(tmp_path):
    path = tmp_path / "episodes.jsonl"
    _record_to(path, n=2)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    with pytest.raises(ValueError, match="line 3"):
        list(read_jsonl(path))


def test_a_smuggled_field_is_refused_on_read(tmp_path):
    path = tmp_path / "episodes.jsonl"
    _record_to(path, n=1)
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    payload["note"] = "an SSN would go here"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        list(read_jsonl(path))


def test_non_strict_mode_skips_bad_lines(tmp_path):
    path = tmp_path / "episodes.jsonl"
    _record_to(path, n=2)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("garbage\n")
    assert len(list(read_jsonl(path, strict=False))) == 2


def test_blank_lines_are_tolerated(tmp_path):
    path = tmp_path / "episodes.jsonl"
    _record_to(path, n=2)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n\n")
    assert len(list(read_jsonl(path))) == 2


def test_sink_refuses_a_record_that_would_not_validate(tmp_path):
    from freeboard import WireViolation

    path = tmp_path / "episodes.jsonl"
    with JsonlSink(path) as sink:
        with pytest.raises(WireViolation):
            sink.write({"episode_id": "nope"})
    assert not path.exists() or path.read_text(encoding="utf-8") == ""


def test_checkpoint_round_trips(tmp_path):
    path = tmp_path / "episodes.jsonl"
    cp_path = tmp_path / "checkpoint.json"
    rec = _record_to(path, n=4)
    write_checkpoint(cp_path, rec.checkpoint())

    cp = read_checkpoint(cp_path)
    assert cp["seq"] == 4
    assert len(cp["head"]) == 64
    assert verify(read_jsonl(path), expected_head=cp["head"])


def test_a_damaged_checkpoint_is_refused(tmp_path):
    cp_path = tmp_path / "checkpoint.json"
    cp_path.write_text(json.dumps({"deployment_id": "d", "seq": 1, "head": "short"}))
    with pytest.raises(ValueError, match="not a sha256"):
        read_checkpoint(cp_path)
