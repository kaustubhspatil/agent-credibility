"""An append only Merkle transparency log, as specified in RFC 9162 section 2.1.

    leaf hash   sha256(0x00 || entry)
    node hash   sha256(0x01 || left || right)
    empty tree  sha256("")

For n > 1 leaves the tree splits at k, the largest power of two below n, so
sizes that are not powers of two are handled exactly as the RFC defines them,
not by padding.

Why this and not only the hash chain in chain.py: a chain needs every record
after position i to prove anything about record i, and every record between
two heads to prove one extends the other. Here both proofs are O(log n)
hashes, which is what makes it cheap for an outside witness to check each new
head before it cosigns it (see witness.py).

Everything a third party needs to check a proof offline is a plain function
over bytes that uses only hashlib: `verify_inclusion`, `verify_consistency`
and `root_from_range`. They take (size, root) as a pair and assume both came
from the same verified checkpoint.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Iterable, Sequence

__all__ = [
    "EMPTY_ROOT",
    "HASH_SIZE",
    "MerkleTree",
    "TransparencyLog",
    "leaf_hash",
    "node_hash",
    "root_from_range",
    "verify_consistency",
    "verify_inclusion",
]

HASH_SIZE = 32
EMPTY_ROOT = hashlib.sha256(b"").digest()


def leaf_hash(entry: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + entry).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _split(n: int) -> int:
    """Largest power of two strictly less than n, for n >= 2."""
    return 1 << ((n - 1).bit_length() - 1)


def _is_hash(value: object) -> bool:
    return isinstance(value, (bytes, bytearray)) and len(value) == HASH_SIZE


class MerkleTree:
    """Leaf hashes plus every completed perfect subtree, level by level.

    Level h holds the roots of the aligned subtrees of 2**h leaves, packed
    32 bytes each in a bytearray. Any node the RFC recursion asks for is
    either one of those or a right edge that folds from O(log n) of them, so
    roots of every past size and all proofs cost O(log n) to O(log**2 n)
    hashes without storing the entries.
    """

    def __init__(self) -> None:
        self._levels: list[bytearray] = [bytearray()]
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    def append(self, entry: bytes) -> int:
        return self.append_leaf_hash(leaf_hash(entry))

    def append_leaf_hash(self, digest: bytes) -> int:
        if not _is_hash(digest):
            raise ValueError("a leaf hash is 32 bytes")
        index = self._size
        self._levels[0] += digest
        self._size += 1
        level, count = 0, self._size
        while count % 2 == 0:
            row = self._levels[level]
            parent = node_hash(bytes(row[-64:-32]), bytes(row[-32:]))
            if level + 1 == len(self._levels):
                self._levels.append(bytearray())
            self._levels[level + 1] += parent
            count //= 2
            level += 1
        return index

    def leaf_hash(self, index: int) -> bytes:
        if not 0 <= index < self._size:
            raise IndexError(f"leaf {index} not in a tree of size {self._size}")
        return self._node(0, index)

    def _node(self, level: int, index: int) -> bytes:
        row = self._levels[level]
        return bytes(row[index * HASH_SIZE:(index + 1) * HASH_SIZE])

    def _perfect(self, start: int, end: int) -> list[bytes]:
        """Aligned perfect subtrees covering [start, end), largest first."""
        parts = []
        pos = start
        while pos < end:
            width = 1 << ((end - pos).bit_length() - 1)
            if pos % width:
                raise AssertionError(f"range [{start}, {end}) is not aligned")
            level = width.bit_length() - 1
            parts.append(self._node(level, pos >> level))
            pos += width
        return parts

    def _mth(self, start: int, end: int) -> bytes:
        """MTH(D[start:end]) for a range the RFC recursion can produce."""
        if start == end:
            return EMPTY_ROOT
        parts = self._perfect(start, end)
        acc = parts[-1]
        for part in reversed(parts[:-1]):
            acc = node_hash(part, acc)
        return acc

    def _check_size(self, size: int | None) -> int:
        if size is None:
            return self._size
        if not 0 <= size <= self._size:
            raise ValueError(f"size {size} outside [0, {self._size}]")
        return size

    def root(self, size: int | None = None) -> bytes:
        """Root of the tree as it stood at `size` leaves (default: now)."""
        return self._mth(0, self._check_size(size))

    def inclusion_proof(self, index: int, size: int | None = None) -> list[bytes]:
        """PATH(index, D[size]), RFC 9162 section 2.1.3.1, leaf upward."""
        size = self._check_size(size)
        if not 0 <= index < size:
            raise ValueError(f"leaf {index} not in a tree of size {size}")
        path = []
        start, end, m = 0, size, index
        while end - start > 1:
            k = _split(end - start)
            if m < k:
                path.append(self._mth(start + k, end))
                end = start + k
            else:
                path.append(self._mth(start, start + k))
                start += k
                m -= k
        path.reverse()
        return path

    def consistency_proof(self, old_size: int, new_size: int | None = None) -> list[bytes]:
        """PROOF(old_size, D[new_size]), RFC 9162 section 2.1.4.1."""
        new_size = self._check_size(new_size)
        if not 0 <= old_size <= new_size:
            raise ValueError(f"old size {old_size} outside [0, {new_size}]")
        if old_size in (0, new_size):
            return []
        proof = []
        start, end, m, complete = 0, new_size, old_size, True
        while True:
            n = end - start
            if m == n:
                if not complete:
                    proof.append(self._mth(start, end))
                break
            k = _split(n)
            if m <= k:
                proof.append(self._mth(start + k, end))
                end = start + k
            else:
                proof.append(self._mth(start, start + k))
                start += k
                m -= k
                complete = False
        proof.reverse()
        return proof

    def frontier(self, size: int | None = None) -> list[bytes]:
        """The perfect subtree roots covering [0, size), largest first.

        With these and the entries from `size` onward anyone can recompute a
        later root, which proves a range of entries is complete and in order
        (see `root_from_range`).
        """
        return self._perfect(0, self._check_size(size))


def verify_inclusion(
    leaf: bytes, index: int, size: int, proof: Sequence[bytes], root: bytes
) -> bool:
    """RFC 9162 section 2.1.3.2. `leaf` is the leaf hash, not the entry."""
    if not (_is_hash(leaf) and _is_hash(root)):
        return False
    if not 0 <= index < size:
        return False
    if not all(_is_hash(p) for p in proof):
        return False
    fn, sn = index, size - 1
    r = bytes(leaf)
    for p in proof:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            r = node_hash(p, r)
            while fn and not fn & 1:
                fn >>= 1
                sn >>= 1
        else:
            r = node_hash(r, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and hmac.compare_digest(r, root)


def verify_consistency(
    old_size: int,
    new_size: int,
    old_root: bytes,
    new_root: bytes,
    proof: Sequence[bytes],
) -> bool:
    """RFC 9162 section 2.1.4.2, plus the two edge cases it leaves out.

    Equal sizes are consistent only with an empty proof and equal roots. The
    empty tree is consistent with every later tree, with an empty proof.
    """
    if not (_is_hash(old_root) and _is_hash(new_root)):
        return False
    if not 0 <= old_size <= new_size:
        return False
    if not all(_is_hash(p) for p in proof):
        return False
    if old_size == new_size:
        return not proof and hmac.compare_digest(old_root, new_root)
    if old_size == 0:
        return not proof and hmac.compare_digest(old_root, EMPTY_ROOT)
    if not proof:
        return False

    path = [bytes(p) for p in proof]
    if old_size & (old_size - 1) == 0:
        path.insert(0, bytes(old_root))
    fn, sn = old_size - 1, new_size - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1
    fr = sr = path[0]
    for c in path[1:]:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            fr = node_hash(c, fr)
            sr = node_hash(c, sr)
            while fn and not fn & 1:
                fn >>= 1
                sn >>= 1
        else:
            sr = node_hash(sr, c)
        fn >>= 1
        sn >>= 1
    return (
        sn == 0
        and hmac.compare_digest(fr, old_root)
        and hmac.compare_digest(sr, new_root)
    )


def root_from_range(
    start_size: int, frontier: Sequence[bytes], leaves: Iterable[bytes]
) -> tuple[int, bytes] | None:
    """Recompute (size, root) from a frontier at `start_size` plus leaf hashes.

    If the result equals a verified checkpoint, the leaves are exactly the
    entries at positions start_size onward, none missing and none reordered,
    and the frontier was genuine. Returns None if the frontier has the wrong
    shape for `start_size`.
    """
    if start_size < 0:
        return None
    heights = [h for h in range(start_size.bit_length() - 1, -1, -1) if start_size >> h & 1]
    if len(frontier) != len(heights) or not all(_is_hash(f) for f in frontier):
        return None
    stack = [(h, bytes(f)) for h, f in zip(heights, frontier)]
    size = start_size
    for leaf in leaves:
        if not _is_hash(leaf):
            return None
        height, node = 0, bytes(leaf)
        while stack and stack[-1][0] == height:
            node = node_hash(stack.pop()[1], node)
            height += 1
        stack.append((height, node))
        size += 1
    if not stack:
        return size, EMPTY_ROOT
    acc = stack[-1][1]
    for _, node in reversed(stack[:-1]):
        acc = node_hash(node, acc)
    return size, acc


class TransparencyLog:
    """A MerkleTree that also keeps the entries, so it can serve them.

    Entries are opaque bytes here. Leaves always commit to the exact bytes
    stored, so a verifier hashes what it was served and never encodes again.
    """

    def __init__(self) -> None:
        self.tree = MerkleTree()
        self._entries: list[bytes] = []

    def __len__(self) -> int:
        return self.tree.size

    @property
    def size(self) -> int:
        return self.tree.size

    def append(self, entry: bytes) -> int:
        if not isinstance(entry, (bytes, bytearray)):
            raise TypeError("entries are bytes")
        entry = bytes(entry)
        index = self.tree.append(entry)
        self._entries.append(entry)
        return index

    def entry(self, index: int) -> bytes:
        return self._entries[index]

    def entries(self, start: int = 0, end: int | None = None) -> list[bytes]:
        return self._entries[start:end]

    def root(self, size: int | None = None) -> bytes:
        return self.tree.root(size)

    def inclusion_proof(self, index: int, size: int | None = None) -> list[bytes]:
        return self.tree.inclusion_proof(index, size)

    def consistency_proof(self, old_size: int, new_size: int | None = None) -> list[bytes]:
        return self.tree.consistency_proof(old_size, new_size)

    def frontier(self, size: int | None = None) -> list[bytes]:
        return self.tree.frontier(size)
