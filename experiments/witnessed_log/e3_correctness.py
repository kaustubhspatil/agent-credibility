"""E3: freeboard.tlog against a naive reference, for every size 1..300.

Every inclusion pair (m, n) with 0 <= m < n <= 300 and every consistency
pair 0 <= m <= n <= 300 is checked four ways: the proof equals the
reference proof, the RFC 9162 verifier accepts it, the reference's
recursive verifier accepts it, and one random mutation of it is rejected by
both. Sizes 301..3000 are then sampled. A disagreement anywhere is printed.

    PYTHONPATH=src python experiments/witnessed_log/e3_correctness.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

from common import OUT

from freeboard.tlog import MerkleTree, verify_consistency, verify_inclusion
from tlog_reference import RefTree, ref_verify_consistency, ref_verify_inclusion


def mutate_inclusion(rng, tree, leaf, m, n, proof, root):
    options = ["index", "leaf", "root"]
    if proof:
        options += ["flip", "drop", "extend"]
    kind = rng.choice(options)
    p = list(proof)
    if kind == "flip":
        i = rng.randrange(len(p))
        p[i] = bytes([p[i][0] ^ (1 << rng.randrange(8))]) + p[i][1:]
        return kind, (leaf, m, n, p, root)
    if kind == "drop":
        del p[rng.randrange(len(p))]
        return kind, (leaf, m, n, p, root)
    if kind == "extend":
        p.insert(rng.randrange(len(p) + 1), rng.randbytes(32))
        return kind, (leaf, m, n, p, root)
    if kind == "index":
        if n == 1:
            return kind, (leaf, 1, n, p, root)
        other = (m + rng.randrange(1, n)) % n
        return kind, (leaf, other, n, p, root)
    if kind == "leaf":
        return kind, (rng.randbytes(32), m, n, p, root)
    return kind, (leaf, m, n, p, rng.randbytes(32))


def mutate_consistency(rng, tree, m, n, proof, old_root, new_root):
    if m == 0:
        # every tree extends the empty tree, so the new root is not constrained
        options = ["old_root", "extend"]
    else:
        options = ["old_root", "new_root"]
    if proof:
        options += ["flip", "drop", "extend"]
    if 0 < m < n:
        options += ["old_size"]
    kind = rng.choice(options)
    p = list(proof)
    if kind == "flip":
        i = rng.randrange(len(p))
        p[i] = bytes([p[i][0] ^ (1 << rng.randrange(8))]) + p[i][1:]
        return kind, (m, n, old_root, new_root, p)
    if kind == "drop":
        del p[rng.randrange(len(p))]
        return kind, (m, n, old_root, new_root, p)
    if kind == "extend":
        p.insert(rng.randrange(len(p) + 1), rng.randbytes(32))
        return kind, (m, n, old_root, new_root, p)
    if kind == "old_size":
        other = rng.choice([x for x in (m - 1, m + 1) if 1 <= x < n] or [m])
        return kind, (other, n, tree.root(other), new_root, p)
    if kind == "old_root":
        return kind, (m, n, rng.randbytes(32), new_root, p)
    return kind, (m, n, old_root, rng.randbytes(32), p)


def _true_inclusion(tree, leaf, m, n, proof, root):
    """A mutant that states a true fact is not a failure to reject."""
    return (0 <= m < n and tree.leaf_hash(m) == leaf and root == tree.root(n)
            and tree.inclusion_proof(m, n) == list(proof))


def _true_consistency(tree, m, n, old_root, new_root, proof):
    return (0 <= m <= n and old_root == tree.root(m) and new_root == tree.root(n)
            and tree.consistency_proof(m, n) == list(proof))


def check_size(n, tree, ref, rng, stats, inc_pairs, con_pairs):
    root = tree.root(n)
    if root != ref.root(n):
        stats["root_mismatch"].append(n)
    for m in inc_pairs:
        proof = tree.inclusion_proof(m, n)
        leaf = tree.leaf_hash(m)
        stats["inclusion_checked"] += 1
        if proof != ref.inclusion(m, n):
            stats["inclusion_proof_mismatch"].append((m, n))
        if not verify_inclusion(leaf, m, n, proof, root):
            stats["inclusion_rfc_rejected_valid"].append((m, n))
        if not ref_verify_inclusion(leaf, m, n, proof, root):
            stats["inclusion_ref_rejected_valid"].append((m, n))
        kind, args = mutate_inclusion(rng, tree, leaf, m, n, proof, root)
        a, b = verify_inclusion(*args), ref_verify_inclusion(*args)
        if (a or b) and _true_inclusion(tree, *args):
            stats["mutants_stating_true_facts"] += 1
        elif a or b:
            stats["inclusion_mutant_accepted"].append((m, n, kind, a, b))
        stats["max_inclusion_excess"] = max(stats["max_inclusion_excess"], len(proof) - math.ceil(math.log2(n)) if n > 1 else 0)
    for m in con_pairs:
        proof = tree.consistency_proof(m, n)
        old_root = tree.root(m)
        stats["consistency_checked"] += 1
        if proof != ref.consistency(m, n):
            stats["consistency_proof_mismatch"].append((m, n))
        if not verify_consistency(m, n, old_root, root, proof):
            stats["consistency_rfc_rejected_valid"].append((m, n))
        if not ref_verify_consistency(m, n, old_root, root, proof):
            stats["consistency_ref_rejected_valid"].append((m, n))
        kind, args = mutate_consistency(rng, tree, m, n, proof, old_root, root)
        a, b = verify_consistency(*args), ref_verify_consistency(*args)
        if (a or b) and _true_consistency(tree, *args):
            stats["mutants_stating_true_facts"] += 1
        elif a or b:
            stats["consistency_mutant_accepted"].append((m, n, kind, a, b))
        if n > 1:
            stats["max_consistency_excess"] = max(stats["max_consistency_excess"], len(proof) - math.ceil(math.log2(n)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exhaustive-max", type=int, default=300)
    ap.add_argument("--sampled-max", type=int, default=3000)
    ap.add_argument("--sampled-sizes", type=int, default=200)
    ap.add_argument("--pairs-per-size", type=int, default=30)
    ap.add_argument("--seed", type=int, default=9162)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    entries = [rng.randbytes(rng.randint(8, 64)) for _ in range(args.sampled_max)]
    tree = MerkleTree()
    for e in entries:
        tree.append(e)
    ref = RefTree(entries)

    def fresh():
        return {k: [] for k in (
            "root_mismatch", "inclusion_proof_mismatch", "inclusion_rfc_rejected_valid",
            "inclusion_ref_rejected_valid", "inclusion_mutant_accepted",
            "consistency_proof_mismatch", "consistency_rfc_rejected_valid",
            "consistency_ref_rejected_valid", "consistency_mutant_accepted",
        )} | {"inclusion_checked": 0, "consistency_checked": 0, "mutants_stating_true_facts": 0,
              "max_inclusion_excess": 0, "max_consistency_excess": 0}

    t0 = time.perf_counter()
    exhaustive = fresh()
    for n in range(1, args.exhaustive_max + 1):
        check_size(n, tree, ref, rng, exhaustive, range(n), range(0, n + 1))

    sampled = fresh()
    sizes = sorted(rng.sample(range(args.exhaustive_max + 1, args.sampled_max + 1), args.sampled_sizes))
    for n in sizes:
        inc = {0, n - 1} | {rng.randrange(n) for _ in range(args.pairs_per_size)}
        con = {0, 1, n - 1, n} | {rng.randint(0, n) for _ in range(args.pairs_per_size)}
        check_size(n, tree, ref, rng, sampled, sorted(inc), sorted(con))

    def compact(s):
        return {k: (v if not isinstance(v, list) else {"count": len(v), "first": v[:5]}) for k, v in s.items()}

    report = {
        "exhaustive_sizes": [1, args.exhaustive_max],
        "exhaustive": compact(exhaustive),
        "sampled_sizes": {"count": len(sizes), "range": [sizes[0], sizes[-1]]},
        "sampled": compact(sampled),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    failures = sum(len(v) for s in (exhaustive, sampled) for v in s.values() if isinstance(v, list))
    report["disagreements"] = failures
    print(json.dumps(report, indent=2))
    OUT.mkdir(exist_ok=True)
    (OUT / "tlog_e3_correctness.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
