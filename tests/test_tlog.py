"""The Merkle log against published vectors and an independent reference.

Sizes that are not powers of two are where hand rolled Merkle trees usually
go wrong, so every size from 1 to 300 is checked, not only 2**k.
"""

import hashlib
import random

import pytest

from freeboard.tlog import (
    EMPTY_ROOT,
    MerkleTree,
    TransparencyLog,
    leaf_hash,
    node_hash,
    root_from_range,
    verify_consistency,
    verify_inclusion,
)
from tlog_reference import RefTree, ref_verify_consistency, ref_verify_inclusion

# The eight leaves and expected values used by the Certificate Transparency
# reference implementations' tests (RFC 6962 hashing, which RFC 9162 keeps).
CT_LEAVES = [
    b"", b"\x00", b"\x10", b"\x20\x21", b"\x30\x31", b"\x40\x41\x42\x43",
    bytes(range(0x50, 0x58)), bytes(range(0x60, 0x70)),
]
CT_ROOTS = [
    "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
    "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125",
    "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77",
    "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7",
    "4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4",
    "76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef",
    "ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c",
    "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328",
]
H96 = "96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7"
H5F = "5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e"
H6B = "6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4"
HBC = "bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b"
HCA = "ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0"
HD3 = CT_ROOTS[3]
H0E = "0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a"


def _entries(n, seed=0):
    rng = random.Random(seed)
    return [rng.randbytes(rng.randint(0, 40)) for _ in range(n)]


@pytest.fixture(scope="module")
def ct_tree():
    t = MerkleTree()
    for leaf in CT_LEAVES:
        t.append(leaf)
    return t


@pytest.fixture(scope="module")
def big():
    """Both implementations over the same 300 entries."""
    entries = _entries(300, seed=9162)
    tree = MerkleTree()
    for e in entries:
        tree.append(e)
    return tree, RefTree(entries), entries


# --- published vectors -------------------------------------------------------


def test_empty_tree_root_is_the_hash_of_nothing():
    assert MerkleTree().root() == EMPTY_ROOT == hashlib.sha256(b"").digest()


@pytest.mark.parametrize("n", range(1, 9))
def test_roots_match_the_ct_vectors(ct_tree, n):
    assert ct_tree.root(n).hex() == CT_ROOTS[n - 1]


@pytest.mark.parametrize(
    "index,size,expected",
    [(0, 8, [H96, H5F, H6B]), (5, 8, [HBC, HCA, HD3]), (0, 1, [])],
)
def test_inclusion_proofs_match_the_ct_vectors(ct_tree, index, size, expected):
    proof = ct_tree.inclusion_proof(index, size)
    assert [p.hex() for p in proof] == expected
    assert verify_inclusion(ct_tree.leaf_hash(index), index, size, proof, ct_tree.root(size))


@pytest.mark.parametrize(
    "old,new,expected",
    [(1, 8, [H96, H5F, H6B]), (6, 8, [H0E, HCA, HD3]), (2, 5, [H5F, HBC]), (4, 4, [])],
)
def test_consistency_proofs_match_the_ct_vectors(ct_tree, old, new, expected):
    proof = ct_tree.consistency_proof(old, new)
    assert [p.hex() for p in proof] == expected
    assert verify_consistency(old, new, ct_tree.root(old), ct_tree.root(new), proof)


# --- agreement with the reference, every size 1..300 ----------------------------


def test_every_root_from_1_to_300_matches_the_reference(big):
    tree, ref, _ = big
    for n in range(1, 301):
        assert tree.root(n) == ref.root(n), n


def test_a_tree_built_incrementally_agrees_with_one_built_at_once(big):
    tree, _, entries = big
    for n in (1, 2, 3, 7, 64, 255, 300):
        fresh = MerkleTree()
        for e in entries[:n]:
            fresh.append(e)
        assert fresh.root() == tree.root(n)


def _sample_pairs(rng, n, k):
    edge = {0, n - 1, n // 2, max(0, n - 2)}
    return sorted(edge | {rng.randrange(n) for _ in range(k)})


def test_inclusion_proofs_for_all_sizes_match_and_verify(big):
    tree, ref, _ = big
    rng = random.Random(1)
    checked = 0
    for n in range(1, 301):
        root = tree.root(n)
        for m in _sample_pairs(rng, n, 6):
            proof = tree.inclusion_proof(m, n)
            assert proof == ref.inclusion(m, n), (m, n)
            leaf = tree.leaf_hash(m)
            assert verify_inclusion(leaf, m, n, proof, root), (m, n)
            assert ref_verify_inclusion(leaf, m, n, proof, root), (m, n)
            assert len(proof) <= max(0, (n - 1).bit_length())
            checked += 1
    assert checked > 2000


def test_consistency_proofs_for_all_sizes_match_and_verify(big):
    tree, ref, _ = big
    rng = random.Random(2)
    checked = 0
    for n in range(1, 301):
        new_root = tree.root(n)
        for m in sorted({1, n} | {rng.randint(1, n) for _ in range(6)}):
            proof = tree.consistency_proof(m, n)
            assert proof == ref.consistency(m, n), (m, n)
            old_root = tree.root(m)
            assert verify_consistency(m, n, old_root, new_root, proof), (m, n)
            assert ref_verify_consistency(m, n, old_root, new_root, proof), (m, n)
            checked += 1
    assert checked > 1500


def test_every_pair_up_to_64_exhaustively(big):
    tree, ref, _ = big
    for n in range(1, 65):
        root = tree.root(n)
        for m in range(n):
            p = tree.inclusion_proof(m, n)
            assert p == ref.inclusion(m, n)
            assert verify_inclusion(tree.leaf_hash(m), m, n, p, root)
        for m in range(0, n + 1):
            c = tree.consistency_proof(m, n)
            assert c == ref.consistency(m, n)
            assert verify_consistency(m, n, tree.root(m), root, c)


# --- rejection -----------------------------------------------------------------


def _flip(proof, i):
    bad = list(proof)
    bad[i] = bytes([bad[i][0] ^ 1]) + bad[i][1:]
    return bad


def test_mutated_inclusion_proofs_are_rejected_by_both_verifiers(big):
    tree, _, _ = big
    rng = random.Random(3)
    rejected = 0
    for n in range(2, 301, 7):
        root = tree.root(n)
        for m in _sample_pairs(rng, n, 3):
            proof = tree.inclusion_proof(m, n)
            leaf = tree.leaf_hash(m)
            other = (m + 1 + rng.randrange(n - 1)) % n
            n2 = n + 1 if n < 300 else n - 1
            mutants = [
                (leaf, m, n, _flip(proof, rng.randrange(len(proof))), root),
                (leaf, m, n, proof[:-1], root),
                (leaf, m, n, proof + [proof[-1]], root),
                (leaf, other, n, proof, root),
                (leaf, m, n2, proof, tree.root(n2)) if m < n2 else None,
                (tree.leaf_hash(other), m, n, proof, root),
                (leaf, m, n, proof, tree.root(n - 1)),
            ]
            for args in filter(None, mutants):
                assert not verify_inclusion(*args), (m, n)
                assert not ref_verify_inclusion(*args), (m, n)
                rejected += 1
    assert rejected > 500


def test_mutated_consistency_proofs_are_rejected_by_both_verifiers(big):
    tree, _, _ = big
    rng = random.Random(4)
    rejected = 0
    for n in range(3, 301, 7):
        new_root = tree.root(n)
        for m in sorted({1, n - 1} | {rng.randint(1, n - 1) for _ in range(3)}):
            proof = tree.consistency_proof(m, n)
            old_root = tree.root(m)
            mutants = [
                (m, n, old_root, new_root, _flip(proof, rng.randrange(len(proof)))),
                (m, n, old_root, new_root, proof[:-1]),
                (m, n, old_root, new_root, proof + [proof[-1]]),
                (m, n, tree.root(m + 1), new_root, proof),
                (m, n, old_root, tree.root(n - 1), proof),
                (m, n, old_root, new_root, []),
            ]
            if m > 1:
                mutants.append((m - 1, n, tree.root(m - 1), new_root, proof))
            for args in mutants:
                assert not verify_consistency(*args), (m, n)
                assert not ref_verify_consistency(*args), (m, n)
                rejected += 1
    assert rejected > 300


def test_out_of_range_and_malformed_inputs_are_rejected_not_raised(ct_tree):
    root = ct_tree.root(8)
    leaf = ct_tree.leaf_hash(0)
    proof = ct_tree.inclusion_proof(0, 8)
    assert not verify_inclusion(leaf, 8, 8, proof, root)
    assert not verify_inclusion(leaf, -1, 8, proof, root)
    assert not verify_inclusion(leaf, 0, 0, [], root)
    assert not verify_inclusion(b"short", 0, 8, proof, root)
    assert not verify_inclusion(leaf, 0, 8, proof[:2] + [b"x" * 31], root)
    assert not verify_consistency(5, 4, root, root, [])
    assert not verify_consistency(4, 4, ct_tree.root(4), root, [])
    assert not verify_consistency(4, 4, root, root, [root])


def test_empty_tree_is_consistent_with_everything_and_only_with_an_empty_proof(ct_tree):
    root = ct_tree.root(8)
    assert verify_consistency(0, 8, EMPTY_ROOT, root, [])
    assert not verify_consistency(0, 8, EMPTY_ROOT, root, [root])
    assert not verify_consistency(0, 8, root, root, [])


def test_domain_separation_stops_an_interior_node_posing_as_a_leaf():
    """Without the 0x00 and 0x01 prefixes, the 64 bytes left||right would
    hash to the same value as their parent, so a two leaf tree could be
    passed off as a one leaf tree whose entry is that concatenation."""
    t = MerkleTree()
    t.append(b"a")
    t.append(b"b")
    forged_entry = leaf_hash(b"a") + leaf_hash(b"b")
    assert leaf_hash(forged_entry) != t.root()
    assert not verify_inclusion(leaf_hash(forged_entry), 0, 1, [], t.root())
    assert node_hash(leaf_hash(b"a"), leaf_hash(b"b")) == t.root()


# --- range proofs ----------------------------------------------------------------


def test_frontier_plus_later_entries_reproduces_every_later_root(big):
    tree, _, entries = big
    rng = random.Random(5)
    for _ in range(200):
        n = rng.randint(1, 300)
        start = rng.randint(0, n)
        got = root_from_range(start, tree.frontier(start), [leaf_hash(e) for e in entries[start:n]])
        assert got == (n, tree.root(n)), (start, n)


def test_a_range_with_anything_missing_or_swapped_does_not_reproduce_the_root(big):
    tree, _, entries = big
    start, n = 37, 211
    leaves = [leaf_hash(e) for e in entries[start:n]]
    good = root_from_range(start, tree.frontier(start), leaves)
    assert good == (n, tree.root(n))
    dropped = leaves[:50] + leaves[51:]
    swapped = leaves[:50] + [leaves[51], leaves[50]] + leaves[52:]
    for bad in (dropped, swapped, leaves[:-1]):
        assert root_from_range(start, tree.frontier(start), bad) != good
    assert root_from_range(start, tree.frontier(start)[:-1], leaves) is None
    # 38 has as many set bits as 37, so the shape passes and the root does not
    assert root_from_range(start, tree.frontier(start + 1), leaves) != good


def test_frontier_has_one_node_per_set_bit(big):
    tree, _, _ = big
    for n in range(0, 301):
        assert len(tree.frontier(n)) == bin(n).count("1")


# --- the log wrapper -----------------------------------------------------------------


def test_transparency_log_serves_the_bytes_its_leaves_commit_to():
    log = TransparencyLog()
    for e in _entries(20):
        log.append(e)
    for i in range(20):
        proof = log.inclusion_proof(i)
        assert verify_inclusion(leaf_hash(log.entry(i)), i, 20, proof, log.root())
    with pytest.raises(TypeError):
        log.append("text is not bytes")


def test_sizes_beyond_the_tree_are_refused():
    t = MerkleTree()
    t.append(b"x")
    with pytest.raises(ValueError):
        t.root(2)
    with pytest.raises(ValueError):
        t.inclusion_proof(1, 1)
    with pytest.raises(ValueError):
        t.consistency_proof(2, 1)
    with pytest.raises(ValueError):
        t.append_leaf_hash(b"short")
