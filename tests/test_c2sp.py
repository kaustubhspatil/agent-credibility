"""C2SP signed notes, checkpoints, cosignatures and the tlog-witness protocol.

The signed note tests use the example keys and notes published with Go's golang.org/x/mod/sumdb/note, so they check
freeboard against an independent implementation, byte for byte."""

from __future__ import annotations

import threading

import pytest

from freeboard import c2sp, tlog
from freeboard.witness import HAVE_ED25519

pytestmark = pytest.mark.skipif(not HAVE_ED25519, reason="needs the cryptography package")

GO_VKEY = "PeterNeumann+c74f20a3+ARpc2QcUPDhMQegwxbzhKqiBfsVkmqq/LDE4izWy10TW"
GO_SKEY = "PRIVATE+KEY+PeterNeumann+c74f20a3+AYEKFALVFGyNhPJEMzD1QIDr+Y7hfZx09iUvxdXHKDFz"
GO_TEXT = "If you think cryptography is the answer to your problem,\nthen you don't know what your problem is.\n"
GO_SIG = ("— PeterNeumann x08go/ZJkuBS9UG/SffcvIAQxVBtiFupLLr8pAcElZInNIuGUgYN1FFYC2pZSNXgKvqfqdngotpRZb6KE6Ryy"
          "BwJnAM=\n")
GO_ENOCH = ("— EnochRoot rwz+eBzmZa0SO3NbfRGzPCpDckykFXSdeX+MNtCOXm2/5n2tiOHp+vAF1aGrQ5ovTG01oOTGwnWLox33WWd1RvMc+"
            "QQ=\n")
ORIGIN = "example.test/log"


def test_the_go_reference_key_and_signature_match_byte_for_byte():
    v = c2sp.NoteVerifier.from_vkey(GO_VKEY)
    assert v.vkey() == GO_VKEY and v.key_id.hex() == "c74f20a3"
    signer = c2sp.NoteSigner.from_private_key(GO_SKEY)
    assert signer.verifier() == v
    assert signer.sign(GO_TEXT) == GO_SIG  # Ed25519 is deterministic: the same bytes as Go
    text, good = c2sp.verify_note(GO_TEXT + "\n" + GO_SIG + GO_ENOCH, [v])  # an unknown signer is ignored
    assert text == GO_TEXT and [g[0].name for g in good] == ["PeterNeumann"]


def test_a_note_is_refused_when_a_known_signature_fails_or_none_is_known():
    v = c2sp.NoteVerifier.from_vkey(GO_VKEY)
    with pytest.raises(c2sp.NoteError, match="does not verify"):
        c2sp.verify_note(GO_TEXT.replace("problem,", "problem;") + "\n" + GO_SIG, [v])
    with pytest.raises(c2sp.NoteError, match="no signature by a known key"):
        c2sp.verify_note(GO_TEXT + "\n" + GO_ENOCH, [v])
    with pytest.raises(c2sp.NoteError, match="canonical"):
        c2sp.verify_note(GO_TEXT + "\n" + GO_SIG.replace("AM=", "AN="), [v])  # same bytes, other padding bits
    with pytest.raises(c2sp.NoteError):
        c2sp.NoteVerifier.from_vkey(GO_VKEY.replace("c74f20a3", "c74f20a4"))
    for _ in range(20):  # a key whose base64 holds '+' still round trips
        k = c2sp.NoteSigner.generate("plus.test/k").verifier()
        assert c2sp.NoteVerifier.from_vkey(k.vkey()) == k


def _log(n: int) -> tlog.TransparencyLog:
    log = tlog.TransparencyLog()
    for i in range(n):
        log.append(f"entry {i}".encode())
    return log


def test_a_checkpoint_note_and_its_cosignature():
    log, signer = _log(5), c2sp.NoteSigner.generate(ORIGIN)
    note = c2sp.log_note(log, ORIGIN, signer)
    text, _ = c2sp.verify_note(note, [signer.verifier()])
    assert c2sp.parse_checkpoint(text)[:3] == (ORIGIN, 5, log.root())
    assert text.splitlines()[1:] == ["5", __import__("base64").b64encode(log.root()).decode()]
    w = c2sp.CosignatureSigner.generate("witness.test/w1")
    line = w.sign(text, timestamp=1_793_522_179)
    _, good = c2sp.verify_note(note + line, [w.verifier()])
    assert good[0][1] == 1_793_522_179  # seconds, carried in the signature line
    with pytest.raises(c2sp.NoteError):
        c2sp.verify_note(note.replace("\n5\n", "\n6\n") + line, [w.verifier()])


def _witness(tmp_path, signer):
    w = c2sp.CosignatureSigner.generate("witness.test/w1")
    return w, c2sp.NoteWitness(w, {ORIGIN: signer.verifier()}, state_path=tmp_path / "witness.json",
                               clock=lambda: 1_793_522_179)


def _body(old: int, proof, note: str) -> bytes:
    return (f"old {old}\n" + "".join(c2sp._b64(p) + "\n" for p in proof) + "\n" + note).encode()


def test_the_witness_follows_the_protocol(tmp_path):
    signer = c2sp.NoteSigner.generate(ORIGIN)
    w, witness = _witness(tmp_path, signer)
    log = _log(3)
    status, out, _ = witness.add_checkpoint(_body(0, [], c2sp.log_note(log, ORIGIN, signer)))
    assert status == 200
    c2sp.verify_note(c2sp.log_note(log, ORIGIN, signer) + out.decode(), [w.verifier()])
    for i in range(3, 8):
        log.append(f"entry {i}".encode())
    note = c2sp.log_note(log, ORIGIN, signer)
    assert witness.add_checkpoint(_body(3, log.consistency_proof(3), note))[0] == 200
    # a stale old size: 409 with the size held, as text/x.tlog.size
    assert witness.add_checkpoint(_body(3, log.consistency_proof(3), note)) == (409, b"8\n", c2sp.SIZE_TYPE)
    # the state survives a restart
    again = c2sp.NoteWitness(w, {ORIGIN: signer.verifier()}, state_path=tmp_path / "witness.json")
    assert again.latest(ORIGIN) == (8, log.root())
    # unknown log, a note the log did not sign, an old size beyond the checkpoint, a proof from 0
    other = c2sp.NoteSigner.generate("other.test/log")
    assert witness.add_checkpoint(_body(0, [], c2sp.log_note(log, "other.test/log", other)))[0] == 404
    assert witness.add_checkpoint(_body(8, [], c2sp.log_note(log, ORIGIN, c2sp.NoteSigner.generate(ORIGIN))))[0] == 403
    assert witness.add_checkpoint(_body(9, [], note))[0] == 400
    assert witness.add_checkpoint(b"old 08\n\n" + note.encode())[0] == 400


def test_the_witness_refuses_a_rewritten_history(tmp_path):
    """The E1 attack in C2SP form: the operator drops an entry the witness has seen, regrows the tree and signs."""
    signer = c2sp.NoteSigner.generate(ORIGIN)
    _, witness = _witness(tmp_path, signer)
    honest = _log(6)
    assert witness.add_checkpoint(_body(0, [], c2sp.log_note(honest, ORIGIN, signer)))[0] == 200
    forged = tlog.TransparencyLog()
    for i in range(6):
        if i != 2:
            forged.append(f"entry {i}".encode())
    for i in range(6, 10):
        forged.append(f"entry {i}".encode())
    note = c2sp.log_note(forged, ORIGIN, signer)
    for proof in (forged.consistency_proof(6), []):
        assert witness.add_checkpoint(_body(6, proof, note))[0] == 422
    same_size = tlog.TransparencyLog()
    for i in range(6):
        same_size.append(f"other {i}".encode())
    assert witness.add_checkpoint(_body(6, [], c2sp.log_note(same_size, ORIGIN, signer)))[0] == 422


def test_the_witness_over_http(tmp_path):
    signer = c2sp.NoteSigner.generate(ORIGIN)
    w, witness = _witness(tmp_path, signer)
    server = c2sp.serve_witness(witness, prefix="/w")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/w"
    try:
        log = _log(4)
        note = c2sp.log_note(log, ORIGIN, signer)
        lines = c2sp.add_checkpoint(url, 0, [], note)
        c2sp.verify_note(note + lines, [w.verifier()])
        with pytest.raises(c2sp.NoteError) as refused:
            c2sp.add_checkpoint(url, 0, [], note)
        assert refused.value.status == 409 and refused.value.size == 4
    finally:
        server.shutdown()
