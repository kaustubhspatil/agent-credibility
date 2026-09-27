"""Typed evidence entries, the evidence log, and the auditor's timeline.

The claim under test: one witnessed log can show "passed commitment X at T,
then every monitored episode since", where "every" is proved rather than
asserted, and nothing in it is content.
"""

import json
import os

import pytest

from freeboard import (
    Capability,
    Recorder,
    Reversibility,
    ToolSpec,
    WireViolation,
    to_wire,
)
from freeboard.evidence import (
    EvidenceLog,
    PackKind,
    ReasonCode,
    Verdict,
    commitment_hash,
    decode_entry,
    encode_entry,
    episode_entry,
    eval_commitment_entry,
    evidence_pack_entry,
    revalidation_entry,
    validate_entry,
    verify_timeline,
)
from freeboard.tlog import leaf_hash, verify_inclusion
from freeboard.witness import HmacSigner, Witness, WitnessRefusal, verify_cosigned

ORIGIN = "log.example/acme"
CANARY = "123-45-6789"
DEP = "0123456789abcdef"
C1 = "a" * 64

TOOLS = [
    ToolSpec("read_case", capabilities=frozenset({Capability.PRIVATE_DATA}),
             reversibility=Reversibility.REVERSIBLE),
]
MORE_TOOLS = TOOLS + [
    ToolSpec("send_email", capabilities=frozenset({Capability.EXTERNAL_EFFECT}),
             reversibility=Reversibility.IRREVERSIBLE),
]


class Clock:
    def __init__(self, t=1_790_000_000_000):
        self.t = t

    def __call__(self):
        self.t += 10
        return self.t


def _log(tmp_path=None, **kw):
    clock = Clock()
    key = HmacSigner.generate()
    path = None if tmp_path is None else tmp_path / "evidence.jsonl"
    return EvidenceLog(ORIGIN, key, path=path, clock=clock, **kw), key, clock


def _recorder(sink, tools=TOOLS, salt="s" * 32, **kw):
    return Recorder(deployment_id="acme", role="ops", tools=tools, sink=sink, salt=salt, **kw)


def _run(rec, n, prefix="t"):
    for i in range(n):
        with rec.episode(f"{prefix}{i}") as ep:
            ep.action("read_case")
            ep.resolve(success=(i % 3 != 0))


def _commitment(tag=b"report"):
    return commitment_hash(tag, b"n" * 16)


# --- the entry format ---------------------------------------------------------------


def test_each_builder_makes_a_valid_entry():
    for entry in (
        eval_commitment_entry(deployment_id=DEP, commitment=C1, verdict="pass",
                              manifest_hash=DEP, envelope_class="C100-I0", ts=1),
        evidence_pack_entry(deployment_id=DEP, digest=C1, pack_kind=PackKind.RED_TEAM,
                            commitment=C1, n_items=3, ts=1),
        revalidation_entry(deployment_id=DEP, reasons=[ReasonCode.INCIDENT, "model_change"], ts=1),
    ):
        validate_entry(entry)
        assert decode_entry(encode_entry(entry)) == entry


def test_reason_codes_are_sorted_and_deduplicated_by_the_builder():
    e = revalidation_entry(deployment_id=DEP, reasons=["scheduled", "incident", "incident"], ts=1)
    assert e["reasons"] == ["incident", "scheduled"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("commitment", CANARY),
        ("commitment", "A" * 64),
        ("commitment", "a" * 63),
        ("verdict", "passed with notes"),
        ("manifest_hash", CANARY),
        ("envelope_class", CANARY),
        ("deployment_id", CANARY),
        ("ts", -1),
        ("ts", True),
        ("ts", 1.5),
        ("v", 2),
        ("type", ["eval_commitment"]),
    ],
)
def test_anything_outside_the_allow_list_is_refused(field, value):
    entry = eval_commitment_entry(deployment_id=DEP, commitment=C1, verdict="pass",
                                  manifest_hash=DEP, envelope_class="C100-I0", ts=1)
    entry[field] = value
    with pytest.raises(WireViolation):
        validate_entry(entry)


def test_unknown_and_missing_fields_are_refused():
    entry = revalidation_entry(deployment_id=DEP, reasons=["incident"], ts=1)
    with pytest.raises(WireViolation, match="unknown"):
        validate_entry({**entry, "note": CANARY})
    del entry["reasons"]
    with pytest.raises(WireViolation, match="missing"):
        validate_entry(entry)


@pytest.mark.parametrize("reasons", [[], ["incident", "incident"], ["scheduled", "incident"], [CANARY], "incident"])
def test_bad_reason_lists_are_refused(reasons):
    entry = revalidation_entry(deployment_id=DEP, reasons=["incident"], ts=1)
    entry["reasons"] = reasons
    with pytest.raises(WireViolation):
        validate_entry(entry)


def test_an_episode_entry_carries_the_validated_wire_record_and_nothing_else():
    log, _, _ = _log()
    rec = _recorder(log)
    _run(rec, 1)
    entry = decode_entry(log.entry(0))
    assert entry["record"] == to_wire(rec.records[0])
    smuggled = dict(entry, record={**entry["record"], "note": CANARY})
    with pytest.raises(WireViolation):
        validate_entry(smuggled)
    mismatched = dict(entry, deployment_id="f" * 16)
    with pytest.raises(WireViolation, match="differs"):
        validate_entry(mismatched)


def test_canary_never_reaches_the_log(tmp_path):
    log, _, _ = _log(tmp_path)
    rec = Recorder(deployment_id=CANARY, role="ops", sink=log,
                   tools=[ToolSpec(f"tool_{CANARY}", description=CANARY)])
    log.commit_evaluation(rec, _commitment())
    with rec.episode(task_id=CANARY) as ep:
        ep.action(f"tool_{CANARY}", output_chars=len(CANARY))
        ep.resolve(success=True)
    log.close()
    assert CANARY not in (tmp_path / "evidence.jsonl").read_text(encoding="ascii")
    assert all(CANARY.encode() not in e for e in log.entries())


def test_only_the_canonical_encoding_decodes():
    entry = revalidation_entry(deployment_id=DEP, reasons=["incident"], ts=1)
    data = encode_entry(entry)
    assert decode_entry(data) == entry
    spaced = json.dumps(entry, sort_keys=True).encode()
    unsorted = json.dumps(dict(reversed(list(entry.items()))), separators=(",", ":")).encode()
    for other in (spaced, unsorted, b"not json", b"\xff\xfe"):
        with pytest.raises(WireViolation):
            decode_entry(other)


def test_commitment_hash_needs_a_nonce_and_depends_on_it():
    with pytest.raises(ValueError):
        commitment_hash(b"report", b"short")
    a = commitment_hash(b"report", b"n" * 16)
    assert a == commitment_hash(b"report", b"n" * 16)
    assert a != commitment_hash(b"report", b"m" * 16)
    assert a != commitment_hash(b"report2", b"n" * 16)
    assert len(a) == 64


# --- the evidence log ------------------------------------------------------------------


def test_every_logged_episode_has_an_inclusion_proof_against_the_checkpoint():
    log, key, _ = _log()
    rec = _recorder(log)
    _run(rec, 13)
    signed = log.checkpoint()
    cp = signed.checkpoint
    for i in range(log.size):
        proof = log.inclusion_proof(i, cp.size)
        assert verify_inclusion(leaf_hash(log.entry(i)), i, cp.size, proof, cp.root)


def test_log_file_replays_to_the_same_root(tmp_path):
    log, key, clock = _log(tmp_path)
    rec = _recorder(log)
    log.commit_evaluation(rec, _commitment())
    _run(rec, 9)
    root, size = log.root(), log.size
    log.close()
    again = EvidenceLog(ORIGIN, key, path=tmp_path / "evidence.jsonl", clock=clock)
    assert (again.size, again.root()) == (size, root)


def test_editing_the_log_file_is_refused_by_the_next_witness(tmp_path):
    log, key, clock = _log(tmp_path)
    rec = _recorder(log)
    _run(rec, 6)
    w = Witness("w1", HmacSigner.generate(), {ORIGIN: key.verifier()}, clock=clock)
    w.cosign(log.checkpoint())
    log.close()

    path = tmp_path / "evidence.jsonl"
    lines = path.read_text(encoding="ascii").splitlines()
    entry = json.loads(lines[2])
    entry["record"]["resolved"] = not entry["record"]["resolved"]
    lines[2] = encode_entry(entry).decode()
    path.write_text("\n".join(lines) + "\n", encoding="ascii")

    doctored = EvidenceLog(ORIGIN, key, path=path, clock=clock)
    _recorder(doctored)  # the file still parses: every line is well formed
    with pytest.raises(WitnessRefusal):
        w.cosign(doctored.checkpoint(), doctored.consistency_proof(6))


def test_a_malformed_log_file_raises_with_its_line_number(tmp_path):
    log, key, clock = _log(tmp_path)
    _run(_recorder(log), 2)
    log.close()
    with (tmp_path / "evidence.jsonl").open("a", encoding="ascii") as fh:
        fh.write('{"type":"episode"}\n')
    with pytest.raises(ValueError, match="line 3"):
        EvidenceLog(ORIGIN, key, path=tmp_path / "evidence.jsonl")


# --- the timeline ------------------------------------------------------------------------


def _timeline_setup(before=4, after=10):
    log, key, clock = _log()
    rec = _recorder(log)
    _run(rec, before, prefix="pre")
    c = _commitment()
    log.commit_evaluation(rec, c)
    _run(rec, after)
    return log, key, clock, rec, c


def test_timeline_shows_the_commitment_then_every_episode_since():
    log, key, clock, rec, c = _timeline_setup()
    w = Witness("w1", HmacSigner.generate(), {ORIGIN: key.verifier()}, clock=clock)
    signed = log.checkpoint()
    signed = signed.with_cosignature(w.cosign(signed))
    assert verify_cosigned(signed, key.verifier(), {"w1": w.verifier()}, 1)

    start, frontier, entries = log.export_timeline(c)
    t = verify_timeline(signed, start, frontier, entries, commitment=c)
    assert t.ok, t.reason
    assert (t.start, t.size, t.verdict) == (4, 15, "pass")
    assert t.episodes == 10
    assert t.deployment_id == rec.deployment_hash
    assert t.commitment_ts <= t.first_episode_ts <= t.last_episode_ts
    assert (t.manifest_mismatches, t.chain_forks, t.chain_breaks, t.ts_regressions) == (0, 0, 0, 0)
    assert t.unlinked == 1  # the first record after the commitment links back before it


@pytest.mark.parametrize("attack", ["drop", "swap", "truncate", "alter", "insert"])
def test_a_doctored_range_does_not_verify(attack):
    log, key, clock, rec, c = _timeline_setup()
    signed = log.checkpoint()
    start, frontier, entries = log.export_timeline(c)
    doctored = list(entries)
    if attack == "drop":
        del doctored[5]
    elif attack == "swap":
        doctored[3], doctored[4] = doctored[4], doctored[3]
    elif attack == "truncate":
        doctored = doctored[:-2]
    elif attack == "alter":
        entry = json.loads(doctored[6])
        entry["record"]["resolved"] = not entry["record"]["resolved"]
        doctored[6] = encode_entry(entry)
    elif attack == "insert":
        doctored.insert(2, doctored[2])
    t = verify_timeline(signed, start, frontier, doctored, commitment=c)
    assert not t.ok
    assert t.episodes == 0


def test_the_range_must_start_with_the_named_commitment_and_frontier():
    log, key, clock, rec, c = _timeline_setup()
    signed = log.checkpoint()
    start, frontier, entries = log.export_timeline(c)
    assert not verify_timeline(signed, start, frontier, entries, commitment="b" * 64)
    assert not verify_timeline(signed, start - 1, log.frontier(start - 1), log.entries(start - 1), commitment=c)
    assert not verify_timeline(signed, start, frontier[:-1], entries, commitment=c)
    with pytest.raises(KeyError):
        log.export_timeline("b" * 64)


def test_revalidations_packs_and_later_commitments_are_reported():
    log, key, clock, rec, c = _timeline_setup(before=0, after=3)
    d = rec.deployment_hash
    log.append(evidence_pack_entry(deployment_id=d, digest="d" * 64, pack_kind="eval_report",
                                   commitment=c, n_items=2, ts=clock()))
    log.append(revalidation_entry(deployment_id=d, reasons=["tool_manifest_change"], ts=clock()))
    log.append(revalidation_entry(deployment_id=d, reasons=["scheduled"], commitment="e" * 64, ts=clock()))
    c2 = _commitment(b"second")
    log.commit_evaluation(rec, c2, Verdict.FAIL)
    t = verify_timeline(log.checkpoint(), *log.export_timeline(c), commitment=c)
    assert t.ok
    assert t.evidence_packs == 1
    assert [r[2] for r in t.revalidations] == [("tool_manifest_change",)]
    assert t.later_commitments == [(7, c2, "fail")]


def test_episodes_from_a_changed_configuration_count_as_manifest_mismatches():
    log, key, clock, rec, c = _timeline_setup(before=0, after=2)
    changed = _recorder(log, tools=MORE_TOOLS)  # same deployment and salt, new powers
    assert changed.deployment_hash == rec.deployment_hash
    _run(changed, 3, prefix="new")
    t = verify_timeline(log.checkpoint(), *log.export_timeline(c), commitment=c)
    assert t.ok and t.episodes == 5 and t.manifest_mismatches == 3


def test_other_deployments_in_the_same_log_are_ignored():
    log, key, clock, rec, c = _timeline_setup(before=0, after=4)
    other = _recorder(log, salt="o" * 32)
    _run(other, 5)
    t = verify_timeline(log.checkpoint(), *log.export_timeline(c), commitment=c)
    assert t.ok and t.episodes == 4


# --- many containers, one log -------------------------------------------------------------------


def test_containers_with_their_own_chains_interleave_in_one_log():
    """Serverless: each container starts its own chain. The log gives them
    one order, and each chain's continuity stays checkable inside it."""
    log, key, clock = _log()
    first = _recorder(log)
    c = _commitment()
    log.commit_evaluation(first, c)
    containers = [first, _recorder(log), _recorder(log)]
    for round_ in range(4):
        for k, rec in enumerate(containers):
            _run(rec, 1, prefix=f"c{k}r{round_}")
    t = verify_timeline(log.checkpoint(), *log.export_timeline(c), commitment=c)
    assert t.ok and t.episodes == 12
    assert (t.chains_started, t.unlinked, t.chain_forks, t.chain_breaks) == (3, 0, 0, 0)


def test_an_episode_withheld_from_the_log_shows_as_an_unlinked_record():
    log, key, clock = _log()

    class Filter:
        def __init__(self):
            self.n = 0

        def write(self, record):
            self.n += 1
            if self.n != 4:  # the recorder chained it, the log never saw it
                log.write(record)

    rec = _recorder(Filter())
    c = _commitment()
    log.commit_evaluation(rec, c)
    _run(rec, 8)
    t = verify_timeline(log.checkpoint(), *log.export_timeline(c), commitment=c)
    assert t.ok and t.episodes == 7
    assert t.chains_started == 1 and t.unlinked == 1


def test_two_containers_restored_from_one_snapshot_show_as_a_chain_fork(tmp_path):
    log, key, clock = _log()
    state = tmp_path / "state.json"
    seed = _recorder(log, state_path=str(state))
    c = _commitment()
    log.commit_evaluation(seed, c)
    _run(seed, 2)
    snapshot = state.read_bytes()
    a = _recorder(log, state_path=str(state))
    _run(a, 1, prefix="a")
    state.write_bytes(snapshot)
    b = _recorder(log, state_path=str(state))
    _run(b, 1, prefix="b")
    t = verify_timeline(log.checkpoint(), *log.export_timeline(c), commitment=c)
    assert t.ok and t.chain_forks == 1


def test_a_record_whose_own_hash_is_wrong_is_a_chain_break():
    log, key, clock = _log()
    rec = _recorder(log)
    c = _commitment()
    log.commit_evaluation(rec, c)
    _run(rec, 2)
    bad = to_wire(rec.records[-1])
    bad["resolved"] = not bad["resolved"]  # entry_hash no longer matches
    log.append(episode_entry(bad, ts=clock()))
    t = verify_timeline(log.checkpoint(), *log.export_timeline(c), commitment=c)
    assert t.ok and t.chain_breaks == 1


def test_cosign_with_collects_every_witness_and_passes_refusals_through():
    log, key, clock = _log()
    rec = _recorder(log)
    ws = [Witness(f"w{i}", HmacSigner.generate(), {ORIGIN: key.verifier()}, clock=clock) for i in range(3)]
    _run(rec, 3)
    first = log.cosign_with(*ws)
    _run(rec, 4)
    second = log.cosign_with(*ws)
    known = {w.name: w.verifier() for w in ws}
    assert verify_cosigned(first, key.verifier(), known)
    assert verify_cosigned(second, key.verifier(), known, 3)

    rewritten, _, _ = _log()
    rewritten._signer, rewritten._clock = key, clock
    for data in log.entries()[1:]:
        rewritten.log.append(data)
    _run(_recorder(rewritten), 3)
    with pytest.raises(WitnessRefusal):
        rewritten.cosign_with(*ws)


def test_the_witnessed_log_example_runs_and_catches_its_rewrite():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "witnessed_log.py"
    spec = importlib.util.spec_from_file_location("witnessed_log_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = module.run(episodes=20)
    assert out["trusted"] and out["timeline"].ok and out["timeline"].episodes == 20
    assert out["rewrite_refused"] == "inconsistent"
