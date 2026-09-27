"""Checkpoints, signers and the witness rule.

The witness rule is the whole point: cosign a new checkpoint only if a
consistency proof links it to the last one this witness cosigned. Every
refusal path is pinned here, including the ones an operator would try.
"""

import json
import os
import subprocess
import sys

import pytest

from freeboard import StateError
from freeboard.tlog import TransparencyLog
from freeboard.witness import (
    HAVE_ED25519,
    Cosignature,
    Ed25519Signer,
    Ed25519Verifier,
    HmacSigner,
    LogCheckpoint,
    SignedCheckpoint,
    Witness,
    WitnessRefusal,
    compare_checkpoints,
    cosign_message,
    sign_checkpoint,
    verify_cosigned,
    verify_log_signature,
)

ORIGIN = "log.example/acme"


class Clock:
    def __init__(self, t=1_790_000_000_000):
        self.t = t

    def __call__(self):
        self.t += 1000
        return self.t


class Operator:
    """A log operator: a tree, a key and a clock."""

    def __init__(self, key=None, clock=None, entries=()):
        self.key = key or HmacSigner.generate()
        self.clock = clock or Clock()
        self.log = TransparencyLog()
        for e in entries:
            self.log.append(e)

    def add(self, n, tag=b"e"):
        for i in range(n):
            self.log.append(tag + str(self.log.size).encode())

    def checkpoint(self, size=None):
        size = self.log.size if size is None else size
        cp = LogCheckpoint(ORIGIN, size, self.log.root(size), self.clock())
        return sign_checkpoint(cp, self.key)


def _witness(op, name="w1", **kw):
    return Witness(name, HmacSigner.generate(), {ORIGIN: op.key.verifier()}, clock=op.clock, **kw)


# --- checkpoints and signatures ----------------------------------------------


def test_hmac_signer_signs_and_verifies_and_refuses_short_keys():
    s = HmacSigner.generate()
    sig = s.sign(b"m")
    assert s.verify(b"m", sig)
    assert not s.verify(b"m2", sig)
    assert not HmacSigner.generate().verify(b"m", sig)
    with pytest.raises(ValueError):
        HmacSigner(b"too short")
    assert s.key_id == HmacSigner(s._key).key_id and len(s.key_id) == 16


@pytest.mark.parametrize("origin", ["has\nnewline", "Upper", "", "sp ace", "x" * 200])
def test_an_origin_that_could_make_bodies_ambiguous_is_refused(origin):
    with pytest.raises(ValueError):
        LogCheckpoint(origin, 1, b"\x00" * 32, 0)


def test_checkpoint_fields_are_type_checked():
    with pytest.raises(ValueError):
        LogCheckpoint(ORIGIN, -1, b"\x00" * 32, 0)
    with pytest.raises(ValueError):
        LogCheckpoint(ORIGIN, 1, b"\x00" * 31, 0)
    with pytest.raises(ValueError):
        LogCheckpoint(ORIGIN, True, b"\x00" * 32, 0)
    with pytest.raises(ValueError):
        LogCheckpoint(ORIGIN, 1, b"\x00" * 32, 1.5)


def test_body_is_domain_separated_and_covers_every_field():
    base = LogCheckpoint(ORIGIN, 5, b"\x01" * 32, 7)
    body = base.body()
    assert body.startswith(b"freeboard-checkpoint/v1\n")
    variants = [
        LogCheckpoint(ORIGIN + "x", 5, b"\x01" * 32, 7),
        LogCheckpoint(ORIGIN, 6, b"\x01" * 32, 7),
        LogCheckpoint(ORIGIN, 5, b"\x02" * 32, 7),
        LogCheckpoint(ORIGIN, 5, b"\x01" * 32, 8),
    ]
    assert len({v.body() for v in variants} | {body}) == 5
    assert cosign_message(base, "w1", 9) != body


def test_log_signature_breaks_when_any_field_changes():
    op = Operator(entries=[b"a", b"b", b"c"])
    signed = op.checkpoint()
    assert verify_log_signature(signed, op.key.verifier())
    cp = signed.checkpoint
    for forged in (
        LogCheckpoint(cp.origin, cp.size - 1, cp.root, cp.timestamp),
        LogCheckpoint(cp.origin, cp.size, op.log.root(2), cp.timestamp),
        LogCheckpoint(cp.origin, cp.size, cp.root, cp.timestamp + 1),
    ):
        assert not verify_log_signature(SignedCheckpoint(forged, signed.signature), op.key.verifier())
    assert not verify_log_signature(signed, HmacSigner.generate())


def test_signed_checkpoint_round_trips_through_json():
    op = Operator(entries=[b"a"])
    w = _witness(op)
    signed = op.checkpoint()
    signed = signed.with_cosignature(w.cosign(signed))
    back = SignedCheckpoint.from_dict(json.loads(json.dumps(signed.to_dict())))
    assert back == signed


def test_with_cosignature_replaces_the_same_witness_rather_than_stacking():
    op = Operator(entries=[b"a"])
    w = _witness(op)
    signed = op.checkpoint()
    once = signed.with_cosignature(w.cosign(signed))
    twice = once.with_cosignature(w.cosign(signed))
    assert len(twice.cosignatures) == 1


# --- the witness rule --------------------------------------------------------------


def test_first_contact_is_accepted_and_extensions_need_a_valid_proof():
    op = Operator()
    op.add(10)
    w = _witness(op)
    w.cosign(op.checkpoint())
    op.add(7)
    signed = op.checkpoint()
    with pytest.raises(WitnessRefusal) as missing:
        w.cosign(signed)
    assert missing.value.code == "inconsistent"
    cosig = w.cosign(signed, op.log.consistency_proof(10))
    assert isinstance(cosig, Cosignature)
    assert w.latest(ORIGIN).checkpoint.size == 17


def test_a_rewritten_history_is_refused_even_with_its_own_valid_proof():
    op = Operator()
    op.add(20)
    w = _witness(op)
    w.cosign(op.checkpoint())

    # the operator deletes entry 7, rebuilds, and extends past the old size
    rewritten = Operator(key=op.key, clock=op.clock)
    for i, e in enumerate(op.log.entries()):
        if i != 7:
            rewritten.log.append(e)
    rewritten.add(5)
    signed = rewritten.checkpoint()
    proof = rewritten.log.consistency_proof(20)  # valid inside the rewritten tree
    with pytest.raises(WitnessRefusal) as refusal:
        w.cosign(signed, proof)
    assert refusal.value.code == "inconsistent"
    assert w.latest(ORIGIN).checkpoint.size == 20  # state did not move


def test_the_witness_uses_its_own_old_size_not_one_the_caller_picks():
    """A proof from a size before the rewrite point is valid in the
    rewritten tree. It must not help, because the witness compares against
    what it stored, not against a size the operator names."""
    op = Operator()
    op.add(20)
    w = _witness(op)
    w.cosign(op.checkpoint())
    rewritten = Operator(key=op.key, clock=op.clock, entries=op.log.entries()[:15])
    rewritten.add(10, tag=b"forged")
    proof_from_5 = rewritten.log.consistency_proof(5)
    with pytest.raises(WitnessRefusal):
        w.cosign(rewritten.checkpoint(), proof_from_5)


def test_rollback_is_refused_with_evidence():
    op = Operator()
    op.add(12)
    w = _witness(op)
    w.cosign(op.checkpoint())
    with pytest.raises(WitnessRefusal) as refusal:
        w.cosign(op.checkpoint(size=8))
    assert refusal.value.code == "rollback"
    assert len(refusal.value.evidence) == 2


def test_two_roots_for_one_size_is_a_fork_and_the_evidence_is_both_signed_heads():
    op = Operator()
    op.add(9)
    w = _witness(op)
    w.cosign(op.checkpoint())
    other = Operator(key=op.key, clock=op.clock, entries=op.log.entries()[:8] + [b"forged"])
    with pytest.raises(WitnessRefusal) as refusal:
        w.cosign(other.checkpoint())
    assert refusal.value.code == "fork"
    a, b = refusal.value.evidence
    assert a.checkpoint.size == b.checkpoint.size and a.checkpoint.root != b.checkpoint.root
    assert verify_log_signature(a, op.key.verifier()) and verify_log_signature(b, op.key.verifier())


def test_the_same_checkpoint_can_be_cosigned_again():
    op = Operator()
    op.add(4)
    w = _witness(op)
    signed = op.checkpoint()
    w.cosign(signed)
    w.cosign(signed)


def test_unknown_origin_and_bad_signature_are_refused():
    op = Operator()
    op.add(3)
    stranger = Witness("w1", HmacSigner.generate(), {"other.example/log": op.key.verifier()}, clock=op.clock)
    with pytest.raises(WitnessRefusal) as unknown:
        stranger.cosign(op.checkpoint())
    assert unknown.value.code == "unknown_origin"

    w = _witness(op)
    impostor = Operator(entries=op.log.entries())
    with pytest.raises(WitnessRefusal) as bad:
        w.cosign(impostor.checkpoint())
    assert bad.value.code == "bad_signature"


def test_timestamps_must_not_run_backwards_or_far_ahead():
    op = Operator()
    op.add(3)
    w = _witness(op)
    first = op.checkpoint()
    w.cosign(first)
    op.add(1)
    earlier = sign_checkpoint(
        LogCheckpoint(ORIGIN, 4, op.log.root(), first.checkpoint.timestamp - 1), op.key
    )
    with pytest.raises(WitnessRefusal) as stale:
        w.cosign(earlier, op.log.consistency_proof(3))
    assert stale.value.code == "stale_timestamp"
    ahead = sign_checkpoint(
        LogCheckpoint(ORIGIN, 4, op.log.root(), op.clock.t + 10**9), op.key
    )
    with pytest.raises(WitnessRefusal) as future:
        w.cosign(ahead, op.log.consistency_proof(3))
    assert future.value.code == "future_timestamp"


# --- persistence -------------------------------------------------------------------


def test_witness_state_survives_a_restart(tmp_path):
    op = Operator()
    op.add(10)
    key = HmacSigner.generate()
    path = tmp_path / "w.json"
    w = Witness("w1", key, {ORIGIN: op.key.verifier()}, state_path=path, clock=op.clock)
    w.cosign(op.checkpoint())

    again = Witness("w1", key, {ORIGIN: op.key.verifier()}, state_path=path, clock=op.clock)
    assert again.latest(ORIGIN).checkpoint.size == 10
    rolled = Operator(key=op.key, clock=op.clock, entries=op.log.entries()[:6])
    with pytest.raises(WitnessRefusal):
        again.cosign(rolled.checkpoint())


def test_state_is_persisted_before_the_cosignature_exists(tmp_path):
    op = Operator()
    op.add(5)
    path = tmp_path / "w.json"
    seen = []

    class Spy(HmacSigner):
        def sign(self, message):
            seen.append(json.loads(path.read_text())["logs"][ORIGIN]["checkpoint"]["size"])
            return super().sign(message)

    w = Witness("w1", Spy(os.urandom(32)), {ORIGIN: op.key.verifier()}, state_path=path, clock=op.clock)
    w.cosign(op.checkpoint())
    assert seen == [5]


def test_damaged_or_foreign_witness_state_raises(tmp_path):
    op = Operator()
    op.add(2)
    path = tmp_path / "w.json"
    w = Witness("w1", HmacSigner.generate(), {ORIGIN: op.key.verifier()}, state_path=path, clock=op.clock)
    w.cosign(op.checkpoint())
    with pytest.raises(StateError, match="belongs to"):
        Witness("w2", HmacSigner.generate(), {ORIGIN: op.key.verifier()}, state_path=path)
    path.write_text("{not json")
    with pytest.raises(StateError):
        Witness("w1", HmacSigner.generate(), {ORIGIN: op.key.verifier()}, state_path=path)


# --- relying parties ----------------------------------------------------------------


def test_verify_cosigned_counts_distinct_known_valid_witnesses():
    op = Operator()
    op.add(6)
    w1, w2 = _witness(op, "w1"), _witness(op, "w2")
    signed = op.checkpoint()
    signed = signed.with_cosignature(w1.cosign(signed))
    known = {"w1": w1.verifier(), "w2": w2.verifier()}
    assert verify_cosigned(signed, op.key.verifier(), known, 1)
    assert not verify_cosigned(signed, op.key.verifier(), known, 2)
    signed = signed.with_cosignature(w2.cosign(signed))
    verdict = verify_cosigned(signed, op.key.verifier(), known, 2)
    assert verdict and set(verdict.witnesses) == {"w1", "w2"}
    with pytest.raises(ValueError):
        verify_cosigned(signed, op.key.verifier(), known, 0)


def test_forged_and_unknown_cosignatures_do_not_count():
    op = Operator()
    op.add(6)
    w1 = _witness(op, "w1")
    signed = op.checkpoint()
    real = w1.cosign(signed)
    forged = Cosignature("w1", real.key_id, real.algorithm, real.timestamp + 1, real.value)
    stranger = _witness(op, "w9").cosign(signed)
    bundle = SignedCheckpoint(signed.checkpoint, signed.signature, (forged, stranger))
    assert not verify_cosigned(bundle, op.key.verifier(), {"w1": w1.verifier()}, 1)


def test_an_operator_signature_alone_is_not_enough_for_a_relying_party():
    op = Operator()
    op.add(3)
    verdict = verify_cosigned(op.checkpoint(), op.key.verifier(), {}, 1)
    assert not verdict and "quorum" in verdict.reason


# --- gossip -----------------------------------------------------------------------


def test_compare_checkpoints_finds_split_views():
    op = Operator()
    op.add(10)
    a = op.checkpoint()
    op.add(5)
    b = op.checkpoint()
    v = op.key.verifier()
    assert compare_checkpoints(a, b, v, op.log.consistency_proof(10, 15)).status == "consistent"

    fork = Operator(key=op.key, clock=op.clock, entries=op.log.entries()[:12])
    fork.add(3, tag=b"forged")
    c = fork.checkpoint()
    same_size = compare_checkpoints(b, c, v)
    assert same_size.status == "fork" and same_size.split_view

    # different sizes: the log owes a proof, and neither branch can supply one
    d = Operator(key=op.key, clock=op.clock, entries=[b"x"] + op.log.entries()[1:10])
    small = d.checkpoint()
    for proof in (op.log.consistency_proof(10, 15), d.log.consistency_proof(5, 10), []):
        check = compare_checkpoints(small, b, v, proof)
        assert check.status == "unproven" and check.split_view

    foreign = Operator(entries=op.log.entries()).checkpoint()
    assert compare_checkpoints(a, foreign, v).status == "invalid"


# --- the optional real signer -----------------------------------------------------------


@pytest.mark.skipif(not HAVE_ED25519, reason="cryptography is not installed")
def test_ed25519_signs_checkpoints_and_cosignatures():
    log_key = Ed25519Signer.generate()
    op = Operator(key=log_key)
    op.add(5)
    wkey = Ed25519Signer.generate()
    w = Witness("w1", wkey, {ORIGIN: Ed25519Verifier(log_key.public_key)}, clock=op.clock)
    signed = op.checkpoint()
    signed = signed.with_cosignature(w.cosign(signed))
    verdict = verify_cosigned(
        signed, Ed25519Verifier(log_key.public_key), {"w1": Ed25519Verifier(wkey.public_key)}, 1
    )
    assert verdict
    tampered = SignedCheckpoint(
        LogCheckpoint(ORIGIN, 4, signed.checkpoint.root, signed.checkpoint.timestamp),
        signed.signature, signed.cosignatures,
    )
    assert not verify_cosigned(tampered, Ed25519Verifier(log_key.public_key), {}, 1)


@pytest.mark.skipif(HAVE_ED25519, reason="cryptography is installed")
def test_ed25519_without_cryptography_fails_loudly_not_silently():
    with pytest.raises(RuntimeError, match="cryptography"):
        Ed25519Signer.generate()


def test_the_package_imports_without_cryptography():
    code = (
        "import sys; sys.modules['cryptography'] = None\n"
        "import freeboard, freeboard.witness\n"
        "assert freeboard.witness.HAVE_ED25519 is False\n"
        "print('ok')\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_default_quorum_is_a_strict_majority_of_known_witnesses():
    op = Operator()
    op.add(5)
    ws = [_witness(op, f"w{i}") for i in range(3)]
    known = {w.name: w.verifier() for w in ws}
    signed = op.checkpoint()
    one = signed.with_cosignature(ws[0].cosign(signed))
    assert not verify_cosigned(one, op.key.verifier(), known)
    two = one.with_cosignature(ws[1].cosign(signed))
    assert verify_cosigned(two, op.key.verifier(), known)
    assert not verify_cosigned(two, op.key.verifier(), {**known, "w3": HmacSigner.generate(), "w4": HmacSigner.generate()})


def test_stale_cosignatures_do_not_count_when_freshness_is_required():
    op = Operator()
    op.add(5)
    w = _witness(op)
    signed = op.checkpoint()
    cosig = w.cosign(signed)
    signed = signed.with_cosignature(cosig)
    known = {"w1": w.verifier()}
    later = cosig.timestamp + 10 * 60 * 1000
    assert verify_cosigned(signed, op.key.verifier(), known)
    assert not verify_cosigned(signed, op.key.verifier(), known, max_age_ms=60_000, now_ms=later)
    assert verify_cosigned(signed, op.key.verifier(), known, max_age_ms=3_600_000, now_ms=later)
