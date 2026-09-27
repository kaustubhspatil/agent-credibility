"""E1: can the operator of the log rewrite history without being caught?

The operator holds everything: the records, the log and the log's signing
key. It rewrites history, recomputes every hash so its own structures are
consistent again, signs the result, and then has to get past

    a plain linear hash chain that verifies itself      (chain.verify)
    a verifier holding one anchored linear chain head   (chain.verify, expected_head)
    a witness holding an earlier cosigned checkpoint    (Witness.cosign)

Attacks on the witnessed prefix: delete, modify, reorder, truncate, each
sometimes followed by appending fresh records so the log looks like it grew
normally. Then a fork (two histories shown to two witnesses), a split view
(honest log for the witness, rewritten log for an auditor), and a control
that rewrites only what no witness has seen yet, which nothing should catch.

    PYTHONPATH=src python experiments/witnessed_log/e1_rewrite.py --trials 500
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter

from common import OUT, chain, entries, make_bodies

from freeboard.chain import verify as chain_verify
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


class Clock:
    def __init__(self) -> None:
        self.t = 1_790_000_000_000

    def __call__(self) -> int:
        self.t += 1000
        return self.t


class World:
    """One operator, its key and clock, and helpers to build signed logs."""

    def __init__(self) -> None:
        self.key = HmacSigner.generate()
        self.clock = Clock()

    def build(self, bodies):
        chained = chain(bodies)
        log = TransparencyLog()
        for e in entries(chained):
            log.append(e)
        return chained, log

    def sign(self, log, size=None) -> SignedCheckpoint:
        size = log.size if size is None else size
        return sign_checkpoint(LogCheckpoint(ORIGIN, size, log.root(size), self.clock()), self.key)

    def witness(self, name="w1") -> Witness:
        return Witness(name, HmacSigner.generate(), {ORIGIN: self.key.verifier()}, clock=self.clock)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rewrite(kind: str, bodies: list, m: int, rng: random.Random) -> list:
    b = list(bodies)
    if kind == "delete":
        del b[rng.randrange(m)]
    elif kind == "modify":
        j = rng.randrange(m)
        b[j] = {**b[j], "resolved": not b[j]["resolved"]}
    elif kind == "reorder":
        j = rng.randrange(m - 1)
        b[j], b[j + 1] = b[j + 1], b[j]
    elif kind == "truncate":
        b = b[: rng.randrange(m)]
    elif kind == "unwitnessed_suffix":
        j = rng.randrange(m, len(b))
        if rng.random() < 0.5:
            del b[j]
        else:
            b[j] = {**b[j], "resolved": not b[j]["resolved"]}
    else:
        raise ValueError(kind)
    if rng.random() < 0.5:
        # grow again, sometimes past the witnessed size, to look like normal traffic
        b += make_bodies(rng.randint(1, 40) + max(0, m - len(b)) * rng.randint(0, 1), rng)
    return b


def trial_prefix(kind: str, rng: random.Random, n_max: int) -> dict:
    w = World()
    n = rng.randint(8, n_max)
    bodies = make_bodies(n, rng)
    if kind == "unwitnessed_suffix":
        m = rng.randint(2, n - 1)
    else:
        m = rng.randint(2, n)
    honest_chain, honest = w.build(bodies)
    witness = w.witness()
    witness.cosign(w.sign(honest, m))
    anchored_head = honest_chain[m - 1]["entry_hash"]

    new_chain, new = w.build(rewrite(kind, bodies, m, rng))
    signed = w.sign(new)

    linear_self_ok = bool(chain_verify(new_chain))
    anchored = chain_verify(new_chain[:m], expected_head=anchored_head)
    linear_anchor_detects = not anchored

    # the operator tries every proof it can make; the witness must refuse all
    attempts = [[]]
    if new.size >= m:
        attempts.append(new.consistency_proof(m))
    if m <= new.size <= honest.size:
        attempts.append(honest.consistency_proof(m, new.size))
    codes = []
    accepted = False
    for proof in attempts:
        try:
            witness.cosign(signed, proof)
            accepted = True
            break
        except WitnessRefusal as refusal:
            codes.append(refusal.code)
    return {
        "n": n, "m": m, "new_size": new.size,
        "changed": new_chain != honest_chain,
        "linear_self_verifies": linear_self_ok,
        "linear_anchor_detects": linear_anchor_detects,
        "witness_detects": not accepted,
        "codes": codes,
    }


def trial_fork(rng: random.Random, n_max: int, inside_witnessed: bool) -> dict:
    """Show branch A to witness 1 and branch B to witness 2."""
    w = World()
    n = rng.randint(16, n_max)
    bodies = make_bodies(n, rng)
    c = rng.randint(2, n // 2)
    _, a_log = w.build(bodies)
    w1, w2 = w.witness("w1"), w.witness("w2")
    common = w.sign(a_log, c)
    w1.cosign(common)
    w2.cosign(common)

    d = rng.randrange(0, c) if inside_witnessed else rng.randrange(c, n)
    b_bodies = list(bodies)
    b_bodies[d] = {**b_bodies[d], "resolved": not b_bodies[d]["resolved"]}
    if rng.random() < 0.5:
        b_bodies += make_bodies(rng.randint(1, 40), rng)
    _, b_log = w.build(b_bodies)

    a_signed, b_signed = w.sign(a_log), w.sign(b_log)
    out = {"n": n, "c": c, "d": d}
    cos = {}
    refused = []
    for name, wit, signed, log in (("w1", w1, a_signed, a_log), ("w2", w2, b_signed, b_log)):
        try:
            cos[name] = wit.cosign(signed, log.consistency_proof(c))
        except WitnessRefusal as refusal:
            refused.append(refusal.code)
    out["per_witness_detects"] = bool(refused)
    out["codes"] = refused
    if refused:
        return out

    # gossip: the two witnesses compare, the operator supplies whatever proof it can
    s1, s2 = w1.latest(ORIGIN), w2.latest(ORIGIN)
    lo, hi = sorted((s1.checkpoint.size, s2.checkpoint.size))
    proofs = [[]]
    for log in (a_log, b_log):
        if hi <= log.size:
            proofs.append(log.consistency_proof(lo, hi))
    statuses = {compare_checkpoints(s1, s2, w.key.verifier(), p).status for p in proofs}
    out["gossip_detects"] = statuses <= {"fork", "unproven"}
    out["gossip_statuses"] = sorted(statuses)

    verifiers = {"w1": w1.verifier(), "w2": w2.verifier()}
    a_full, b_full = a_signed.with_cosignature(cos["w1"]), b_signed.with_cosignature(cos["w2"])
    q1 = bool(verify_cosigned(a_full, w.key.verifier(), verifiers, 1)) and bool(
        verify_cosigned(b_full, w.key.verifier(), verifiers, 1)
    )
    q2 = bool(verify_cosigned(a_full, w.key.verifier(), verifiers, 2)) or bool(
        verify_cosigned(b_full, w.key.verifier(), verifiers, 2)
    )
    out["quorum1_split_accepted"] = q1
    out["quorum2_any_branch_accepted"] = q2
    return out


def trial_split_view(rng: random.Random, n_max: int) -> dict:
    """The witness keeps seeing the honest log; an auditor is shown a rewrite."""
    w = World()
    n = rng.randint(8, n_max)
    bodies = make_bodies(n, rng)
    _, honest = w.build(bodies)
    witness = w.witness()
    m = rng.randint(1, n)
    witness.cosign(w.sign(honest, m))
    honest_signed = w.sign(honest)
    cosig = witness.cosign(honest_signed, honest.consistency_proof(m))
    honest_signed = honest_signed.with_cosignature(cosig)

    kind = rng.choice(["delete", "modify", "reorder"]) if n >= 2 else "modify"
    fake_bodies = rewrite(kind, bodies, n, rng)
    _, fake = w.build(fake_bodies)
    fake_signed = w.sign(fake)
    # the operator even staples the real cosignature onto the fake checkpoint
    stapled = SignedCheckpoint(fake_signed.checkpoint, fake_signed.signature, (cosig,))
    leaves = [leaf_hash(e) for e in fake.entries()]
    replay = root_from_range(0, [], leaves)

    log_sig_only = verify_log_signature(stapled, w.key.verifier()) and replay == (
        stapled.checkpoint.size, stapled.checkpoint.root
    )
    quorum_ok = bool(verify_cosigned(stapled, w.key.verifier(), {"w1": witness.verifier()}, 1))
    against_witnessed = root_from_range(0, [], leaves[: honest_signed.checkpoint.size]) == (
        honest_signed.checkpoint.size, honest_signed.checkpoint.root
    )
    return {
        "n": n, "kind": kind,
        "log_signature_only_fooled": log_sig_only,
        "quorum_policy_fooled": quorum_ok,
        "fake_entries_match_witnessed_root": against_witnessed,
    }


def summarise(rows: list[dict], key: str) -> dict:
    k = sum(1 for r in rows if r.get(key))
    lo, hi = wilson(k, len(rows))
    return {"count": k, "trials": len(rows), "rate": k / len(rows) if rows else None,
            "wilson95": [round(lo, 4), round(hi, 4)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--n-max", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    t0 = time.perf_counter()
    report: dict = {"trials_per_scenario": args.trials, "n_range": [8, args.n_max], "seed": args.seed}

    for kind in ("delete", "modify", "reorder", "truncate", "unwitnessed_suffix"):
        rows = [trial_prefix(kind, rng, args.n_max) for _ in range(args.trials)]
        report[kind] = {
            "linear_self_verifies": summarise(rows, "linear_self_verifies"),
            "linear_anchor_detects": summarise(rows, "linear_anchor_detects"),
            "witness_detects": summarise(rows, "witness_detects"),
            "history_changed": summarise(rows, "changed"),
            "refusal_codes": dict(Counter(c for r in rows for c in r["codes"])),
            "mean_n": sum(r["n"] for r in rows) / len(rows),
        }
        print(kind, json.dumps(report[kind]["witness_detects"]), flush=True)

    for label, inside in (("fork_after_common_checkpoint", False), ("fork_inside_witnessed_prefix", True)):
        rows = [trial_fork(rng, args.n_max, inside) for _ in range(args.trials)]
        both = [r for r in rows if not r["per_witness_detects"]]
        report[label] = {
            "per_witness_detects": summarise(rows, "per_witness_detects"),
            "refusal_codes": dict(Counter(c for r in rows for c in r["codes"])),
        }
        if both:
            report[label].update({
                "gossip_detects": summarise(both, "gossip_detects"),
                "gossip_statuses": dict(Counter(s for r in both for s in r["gossip_statuses"])),
                "quorum1_split_accepted": summarise(both, "quorum1_split_accepted"),
                "quorum2_any_branch_accepted": summarise(both, "quorum2_any_branch_accepted"),
            })
        print(label, json.dumps(report[label]["per_witness_detects"]), flush=True)

    rows = [trial_split_view(rng, args.n_max) for _ in range(args.trials)]
    report["split_view_to_auditor"] = {
        "log_signature_only_fooled": summarise(rows, "log_signature_only_fooled"),
        "quorum_policy_fooled": summarise(rows, "quorum_policy_fooled"),
        "fake_entries_match_witnessed_root": summarise(rows, "fake_entries_match_witnessed_root"),
    }
    print("split_view", json.dumps(report["split_view_to_auditor"]), flush=True)

    report["seconds"] = round(time.perf_counter() - t0, 1)
    OUT.mkdir(exist_ok=True)
    path = OUT / "tlog_e1_rewrite.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} in {report['seconds']}s")


if __name__ == "__main__":
    main()
