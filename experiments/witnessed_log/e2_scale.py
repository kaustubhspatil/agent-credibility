"""E2: build time, proof size and verification time at 1e3, 1e5 and 1e6 entries.

Entries are real typed episode entries (validated and canonically encoded),
streamed into a MerkleTree so memory holds leaf and node hashes only. The
linear chain's costs are measured on the same shape of record for
comparison: a single record or an extension can only be checked there by
replaying every later record.

    PYTHONPATH=src python experiments/witnessed_log/e2_scale.py
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import time

from common import OUT, TEMPLATE

from freeboard.chain import ChainState
from freeboard.chain import verify as chain_verify
from freeboard.evidence import encode_entry, episode_entry
from freeboard.tlog import MerkleTree, root_from_range, verify_consistency, verify_inclusion


def body(i: int) -> dict:
    return {**TEMPLATE, "episode_id": "%016x" % (i * 2654435761 % 2**64),
            "resolved": i % 5 != 0, "n_actions": 1 + i % 11}


def build(n: int) -> tuple[MerkleTree, dict]:
    tree = MerkleTree()
    enc = app = 0.0
    total_bytes = 0
    zero = "0" * 64
    for i in range(n):
        t0 = time.perf_counter()
        data = encode_entry(episode_entry({**body(i), "seq": i, "prev_hash": zero, "entry_hash": zero},
                                          ts=1_790_000_000_000 + i))
        t1 = time.perf_counter()
        tree.append(data)
        t2 = time.perf_counter()
        enc += t1 - t0
        app += t2 - t1
        total_bytes += len(data)
    stored = sum(len(level) for level in tree._levels)
    return tree, {
        "encode_validate_s": round(enc, 3),
        "tree_append_s": round(app, 3),
        "append_us_per_entry": round(app / n * 1e6, 2),
        "mean_entry_bytes": round(total_bytes / n, 1),
        "tree_hash_bytes": stored,
    }


def timed(fn, items):
    out = []
    t0 = time.perf_counter()
    for it in items:
        out.append(fn(it))
    return out, (time.perf_counter() - t0) / max(1, len(items)) * 1e6


def proofs(tree: MerkleTree, n: int, rng: random.Random, samples: int) -> dict:
    root = tree.root()
    if n <= 2000:
        idx = list(range(n))
        olds = list(range(1, n))
    else:
        idx = [rng.randrange(n) for _ in range(samples)] + [0, n - 1, n // 2]
        olds = [rng.randint(1, n - 1) for _ in range(samples)] + [1, n - 1, n // 2, 2 ** (n.bit_length() - 1)]

    inc, inc_gen_us = timed(lambda i: tree.inclusion_proof(i), idx)
    leaves = [tree.leaf_hash(i) for i in idx]
    t0 = time.perf_counter()
    inc_ok = all(verify_inclusion(leaves[k], i, n, inc[k], root) for k, i in enumerate(idx))
    inc_ver_us = (time.perf_counter() - t0) / len(idx) * 1e6

    old_roots = {m: tree.root(m) for m in olds}
    con, con_gen_us = timed(lambda m: tree.consistency_proof(m), olds)
    t0 = time.perf_counter()
    con_ok = all(verify_consistency(m, n, old_roots[m], root, con[k]) for k, m in enumerate(olds))
    con_ver_us = (time.perf_counter() - t0) / len(olds) * 1e6

    inc_len = [len(p) for p in inc]
    con_len = [len(p) for p in con]
    log2n = math.ceil(math.log2(n))
    return {
        "inclusion": {
            "checked": len(idx), "all_verified": inc_ok,
            "len_min": min(inc_len), "len_mean": round(statistics.mean(inc_len), 2), "len_max": max(inc_len),
            "bytes_max": 32 * max(inc_len),
            "gen_us": round(inc_gen_us, 1), "verify_us": round(inc_ver_us, 1),
            "linear_chain_records_needed_mean": round(statistics.mean(n - 1 - i for i in idx), 1),
        },
        "consistency": {
            "checked": len(olds), "all_verified": con_ok,
            "len_min": min(con_len), "len_mean": round(statistics.mean(con_len), 2), "len_max": max(con_len),
            "bytes_max": 32 * max(con_len),
            "gen_us": round(con_gen_us, 1), "verify_us": round(con_ver_us, 1),
            "linear_chain_records_needed_mean": round(statistics.mean(n - m for m in olds), 1),
        },
        "ceil_log2_n": log2n,
    }


def full_audit(tree: MerkleTree, n: int) -> dict:
    """Recompute the root from every leaf hash, as an auditor holding all entries would."""
    leaves = (tree.leaf_hash(i) for i in range(n))
    t0 = time.perf_counter()
    got = root_from_range(0, [], leaves)
    dt = time.perf_counter() - t0
    assert got == (n, tree.root())
    return {"merkle_replay_from_leaf_hashes_s": round(dt, 3)}


def linear_chain(n: int, chunk: int = 100_000) -> dict:
    """Build then verify a linear chain of n records, in chunks so memory stays flat.
    Only the verify calls are timed."""
    state = ChainState()
    head, seq = "0" * 64, 0
    verify_s = 0.0
    done = 0
    while done < n:
        k = min(chunk, n - done)
        batch = []
        for i in range(done, done + k):
            b = body(i)
            s, prev, digest = state.advance(b)
            batch.append({**b, "seq": s, "prev_hash": prev, "entry_hash": digest})
        t0 = time.perf_counter()
        res = chain_verify(batch, start_hash=head, start_seq=seq)
        verify_s += time.perf_counter() - t0
        assert res.ok
        head, seq = batch[-1]["entry_hash"], seq + k
        done += k
    return {"linear_full_verify_s": round(verify_s, 3), "linear_us_per_record": round(verify_s / n * 1e6, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1000,100000,1000000")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    report = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "results": {},
    }
    for n in (int(s) for s in args.sizes.split(",")):
        tree, b = build(n)
        row = {"build": b}
        row.update(proofs(tree, n, rng, args.samples))
        row["full_audit"] = {**full_audit(tree, n), **linear_chain(n)}
        report["results"][str(n)] = row
        print(n, json.dumps(row), flush=True)
    OUT.mkdir(exist_ok=True)
    path = OUT / "tlog_e2_scale.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
