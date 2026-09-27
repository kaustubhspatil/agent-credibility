"""A naive RFC 9162 Merkle tree, written from the recursive definitions.

Shares no code with freeboard.tlog. Proof generation recomputes subtree
hashes from the leaves (memoised per range only for speed), and the
verifiers rebuild roots by walking the same recursion instead of using the
RFC's bit arithmetic. If the two implementations agree on every size and
pair, a mistake would have to be made identically twice.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache


def H(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def largest_pow2_below(n: int) -> int:
    k = 1
    while k * 2 < n:
        k *= 2
    return k


class RefTree:
    def __init__(self, entries: list[bytes]) -> None:
        self.entries = list(entries)
        self.mth = lru_cache(maxsize=None)(self._mth)

    def _mth(self, lo: int, hi: int) -> bytes:
        n = hi - lo
        if n == 0:
            return H(b"")
        if n == 1:
            return H(b"\x00" + self.entries[lo])
        k = largest_pow2_below(n)
        return H(b"\x01" + self.mth(lo, lo + k) + self.mth(lo + k, hi))

    def root(self, n: int) -> bytes:
        return self.mth(0, n)

    def path(self, m: int, lo: int, hi: int) -> list[bytes]:
        n = hi - lo
        if n == 1:
            return []
        k = largest_pow2_below(n)
        if m < k:
            return self.path(m, lo, lo + k) + [self.mth(lo + k, hi)]
        return self.path(m - k, lo + k, hi) + [self.mth(lo, lo + k)]

    def inclusion(self, m: int, n: int) -> list[bytes]:
        return self.path(m, 0, n)

    def subproof(self, m: int, lo: int, hi: int, b: bool) -> list[bytes]:
        n = hi - lo
        if m == n:
            return [] if b else [self.mth(lo, hi)]
        k = largest_pow2_below(n)
        if m <= k:
            return self.subproof(m, lo, lo + k, b) + [self.mth(lo + k, hi)]
        return self.subproof(m - k, lo + k, hi, False) + [self.mth(lo, lo + k)]

    def consistency(self, m: int, n: int) -> list[bytes]:
        if m == 0 or m == n:
            return []
        return self.subproof(m, 0, n, True)


def ref_verify_inclusion(leaf: bytes, m: int, n: int, proof: list[bytes], root: bytes) -> bool:
    """Rebuild the root by following the recursion that made the path."""
    if not 0 <= m < n:
        return False

    def rebuild(m: int, n: int, proof: list[bytes]) -> bytes | None:
        if n == 1:
            return leaf if not proof else None
        if not proof:
            return None
        k = largest_pow2_below(n)
        sibling, rest = proof[-1], proof[:-1]
        if m < k:
            left = rebuild(m, k, rest)
            return None if left is None else H(b"\x01" + left + sibling)
        right = rebuild(m - k, n - k, rest)
        return None if right is None else H(b"\x01" + sibling + right)

    return rebuild(m, n, list(proof)) == root


def ref_verify_consistency(
    m: int, n: int, old_root: bytes, new_root: bytes, proof: list[bytes]
) -> bool:
    """Rebuild both roots from the proof by walking SUBPROOF backwards."""
    if not 0 <= m <= n:
        return False
    if m == n:
        return not proof and old_root == new_root
    if m == 0:
        return not proof and old_root == H(b"")

    def rebuild(m: int, n: int, proof: list[bytes], b: bool):
        """Return (old subtree hash, new subtree hash) or None."""
        if m == n:
            if b:
                return (old_root, old_root) if not proof else None
            if len(proof) != 1:
                return None
            return proof[0], proof[0]
        if not proof:
            return None
        k = largest_pow2_below(n)
        sibling, rest = proof[-1], proof[:-1]
        if m <= k:
            got = rebuild(m, k, rest, b)
            if got is None:
                return None
            old, new = got
            return old, H(b"\x01" + new + sibling)
        got = rebuild(m - k, n - k, rest, False)
        if got is None:
            return None
        old, new = got
        return H(b"\x01" + sibling + old), H(b"\x01" + sibling + new)

    got = rebuild(m, n, list(proof), True)
    return got is not None and got[0] == old_root and got[1] == new_root
