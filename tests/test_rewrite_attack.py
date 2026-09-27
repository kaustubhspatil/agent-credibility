"""E1 in miniature: the operator rewrites history and recomputes everything.

Pins both halves of the result. What a witness catches, and what it does
not: a rewrite of entries no witness has seen yet, and a fork shown to two
witnesses that each see a consistent branch, which only gossip or a
majority quorum exposes.
"""

import random

import pytest

from freeboard import Capability, Recorder, Reversibility, ToolSpec, to_wire
from freeboard.chain import ChainState, verify
from freeboard.evidence import encode_entry, episode_entry
from freeboard.tlog import TransparencyLog, leaf_hash, root_from_range
from freeboard.witness import (
    HmacSigner,
    LogCheckpoint,
    SignedCheckpoint,
    Witness,
    WitnessRefusal,
    compare_checkpoints,
    sign_checkpoint,
    verify_cosigned,
    verify_log_signature,
)

ORIGIN = "log.example/acme"
N = 40
M = 25  # the size the witness has cosigned


def _bodies(n, seed=0):
    rec = Recorder(deployment_id="acme", role="ops", salt="x" * 32, tools=[
        ToolSpec("read_case", capabilities=frozenset({Capability.PRIVATE_DATA}),
                 reversibility=Reversibility.REVERSIBLE)])
    rng = random.Random(seed)
    for i in range(n):
        with rec.episode(f"t{i}") as ep:
            ep.action("read_case")
            ep.resolve(success=rng.random() < 0.8)
    out = []
    for r in rec.records:
        b = to_wire(r)
        for k in ("seq", "prev_hash", "entry_hash"):
            b.pop(k)
        out.append(b)
    return out


def _chain(bodies):
    state = ChainState()
    out = []
    for b in bodies:
        seq, prev, digest = state.advance(b)
        out.append({**b, "seq": seq, "prev_hash": prev, "entry_hash": digest})
    return out


class Op:
    def __init__(self):
        self.key = HmacSigner.generate()
        self.t = 1_790_000_000_000

    def clock(self):
        self.t += 1000
        return self.t

    def build(self, bodies):
        chained = _chain(bodies)
        log = TransparencyLog()
        for i, r in enumerate(chained):
            log.append(encode_entry(episode_entry(r, ts=i)))
        return chained, log

    def sign(self, log, size=None):
        size = log.size if size is None else size
        return sign_checkpoint(LogCheckpoint(ORIGIN, size, log.root(size), self.clock()), self.key)

    def witness(self, name="w1"):
        return Witness(name, HmacSigner.generate(), {ORIGIN: self.key.verifier()}, clock=self.clock)


def _rewrite(kind, bodies):
    b = list(bodies)
    if kind == "delete":
        del b[10]
    elif kind == "modify":
        b[10] = {**b[10], "resolved": not b[10]["resolved"]}
    elif kind == "reorder":
        b[10], b[11] = b[11], b[10]
    elif kind == "truncate":
        b = b[:20]
    elif kind == "delete_then_grow":
        del b[10]
        b += _bodies(10, seed=99)
    return b


@pytest.mark.parametrize("kind", ["delete", "modify", "reorder", "truncate", "delete_then_grow"])
def test_rewrite_of_the_witnessed_prefix_passes_the_chain_and_fails_the_witness(kind):
    op = Op()
    bodies = _bodies(N)
    honest_chain, honest = op.build(bodies)
    w = op.witness()
    w.cosign(op.sign(honest, M))

    new_chain, new = op.build(_rewrite(kind, bodies))
    assert new_chain != honest_chain
    assert verify(new_chain)  # the linear chain alone cannot tell
    assert not verify(new_chain[:M], expected_head=honest_chain[M - 1]["entry_hash"])

    signed = op.sign(new)
    attempts = [[]]
    if new.size >= M:
        attempts.append(new.consistency_proof(M))
    if M <= new.size <= honest.size:
        attempts.append(honest.consistency_proof(M, new.size))
    for proof in attempts:
        with pytest.raises(WitnessRefusal):
            w.cosign(signed, proof)


def test_limitation_a_rewrite_of_unwitnessed_entries_is_not_caught():
    op = Op()
    bodies = _bodies(N)
    _, honest = op.build(bodies)
    w = op.witness()
    w.cosign(op.sign(honest, M))
    b = list(bodies)
    del b[M + 3]
    _, new = op.build(b)
    w.cosign(op.sign(new), new.consistency_proof(M))  # accepted: it never saw M + 3


def _fork():
    op = Op()
    bodies = _bodies(N)
    _, a = op.build(bodies)
    w1, w2 = op.witness("w1"), op.witness("w2")
    common = op.sign(a, M)
    w1.cosign(common)
    w2.cosign(common)
    b_bodies = list(bodies)
    b_bodies[M + 5] = {**b_bodies[M + 5], "resolved": not b_bodies[M + 5]["resolved"]}
    _, b = op.build(b_bodies + _bodies(3, seed=5))
    a_signed, b_signed = op.sign(a), op.sign(b)
    ca = w1.cosign(a_signed, a.consistency_proof(M))
    cb = w2.cosign(b_signed, b.consistency_proof(M))
    return op, a, b, w1, w2, a_signed.with_cosignature(ca), b_signed.with_cosignature(cb)


def test_limitation_each_witness_accepts_its_own_branch_of_a_fork():
    _, a, b, w1, w2, _, _ = _fork()
    assert w1.latest(ORIGIN).checkpoint.root != w2.latest(ORIGIN).checkpoint.root


def test_gossip_between_witnesses_exposes_the_fork_whatever_proof_the_operator_gives():
    op, a, b, w1, w2, _, _ = _fork()
    s1, s2 = w1.latest(ORIGIN), w2.latest(ORIGIN)
    lo, hi = sorted((s1.checkpoint.size, s2.checkpoint.size))
    for proof in ([], b.consistency_proof(lo, hi)):
        assert compare_checkpoints(s1, s2, op.key.verifier(), proof).split_view


def test_a_majority_quorum_accepts_neither_branch_and_a_quorum_of_one_accepts_both():
    op, _, _, w1, w2, a_full, b_full = _fork()
    ws = {"w1": w1.verifier(), "w2": w2.verifier()}
    v = op.key.verifier()
    assert verify_cosigned(a_full, v, ws, 1) and verify_cosigned(b_full, v, ws, 1)
    assert not verify_cosigned(a_full, v, ws, 2) and not verify_cosigned(b_full, v, ws, 2)


def test_split_view_fools_an_auditor_who_trusts_the_log_signature_alone():
    op = Op()
    bodies = _bodies(N)
    _, honest = op.build(bodies)
    w = op.witness()
    real = op.sign(honest)
    cosig = w.cosign(real)
    _, fake = op.build(_rewrite("delete", bodies))
    fake_signed = op.sign(fake)
    stapled = SignedCheckpoint(fake_signed.checkpoint, fake_signed.signature, (cosig,))
    replay = root_from_range(0, [], [leaf_hash(e) for e in fake.entries()])

    assert verify_log_signature(stapled, op.key.verifier())
    assert replay == (stapled.checkpoint.size, stapled.checkpoint.root)  # fooled
    assert not verify_cosigned(stapled, op.key.verifier(), {"w1": w.verifier()}, 1)
    assert replay != (real.checkpoint.size, real.checkpoint.root)


def test_the_default_quorum_is_the_majority_that_blocks_a_fork():
    op, _, _, w1, w2, a_full, b_full = _fork()
    ws = {"w1": w1.verifier(), "w2": w2.verifier()}
    assert not verify_cosigned(a_full, op.key.verifier(), ws)
    assert not verify_cosigned(b_full, op.key.verifier(), ws)
