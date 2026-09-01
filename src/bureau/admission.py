"""Who gets to move a published base rate.

Accountless ingest is right for adoption and privacy: no signup, no key to
manage, no PII. It is also the cheapest possible attack on the only asset that
is hard to copy. Anyone with `curl` could submit ten thousand fabricated
deployments and shift a class prior in whatever direction suits them -- and the
bureau, by design, cannot look at content to tell the difference.

So accepting a record and letting it move a number are separated. Anyone may
submit. Admission to the pool is earned, and the evidence is the chain.

Why the chain works as evidence: a record commits to its predecessor, and a
checkpoint commits the head at a point in time. Faking one long, continuous,
repeatedly-checkpointed chain is cheap. Faking hundreds of them, each with
independent timing, sustained over weeks, is expensive in exactly the way that
matters -- it costs real elapsed time, which is the one input an attacker
cannot parallelise.

None of this makes poisoning impossible. It makes it expensive, slow and
visible, and it keeps the quarantined data so a later audit can find it. The
honest framing for anyone who asks: this is Sybil *resistance*, not Sybil
proof, and a determined well-funded actor running genuine agents can still
contribute genuinely-generated but adversarially-selected traffic. Nothing
short of identity solves that, and identity is what we are deliberately not
asking for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# Thresholds. These are policy, not physics, and they are deliberately
# conservative to start: it is far cheaper to loosen an admission rule later
# than to explain why a published base rate moved because of a stranger.
MIN_EPISODES = 30          # enough that one deployment cannot dominate a thin class
MIN_AGE_SECONDS = 3 * 3600  # a chain that appeared fully formed is not evidence
MIN_CHECKPOINTS = 2         # a head committed more than once, over time


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reasons: tuple[str, ...]

    @property
    def summary(self) -> str:
        return "admitted" if self.admitted else "; ".join(self.reasons)


def evaluate(
    n_episodes: int,
    first_seen: float,
    n_checkpoints: int,
    chain_intact: bool,
    now: float | None = None,
) -> AdmissionDecision:
    """Decide whether a deployment's episodes may inform a published prior."""
    now = time.time() if now is None else now
    reasons: list[str] = []

    if not chain_intact:
        reasons.append("chain has a gap or a bad link")
    if n_episodes < MIN_EPISODES:
        reasons.append(f"{n_episodes}/{MIN_EPISODES} episodes")
    age = now - first_seen
    if age < MIN_AGE_SECONDS:
        reasons.append(f"{age / 3600:.1f}/{MIN_AGE_SECONDS / 3600:.0f} hours of history")
    if n_checkpoints < MIN_CHECKPOINTS:
        reasons.append(f"{n_checkpoints}/{MIN_CHECKPOINTS} checkpoints")

    return AdmissionDecision(admitted=not reasons, reasons=tuple(reasons))


def influence_cap(n_deployments_in_class: int) -> float:
    """Ceiling on the share of class exposure one deployment may hold.

    Buhlmann weights each risk by its exposure, so the cheapest poisoning
    attack is not many fake deployments but one enormous one -- a contributor
    with a hundred thousand episodes simply *is* the class prior.

    Two rules. The share falls as the class grows, so a mature class is hard to
    move. And once a class has three members no single contributor may hold a
    majority, however large it is, because a majority contributor is the class
    by another name.
    """
    if n_deployments_in_class < 3:
        return 1.0
    return min(0.5, 4.0 / n_deployments_in_class)


def capped_weights(exposures: list[float], max_share: float | None = None) -> list[float]:
    """Clip per-deployment exposure so no contributor exceeds `max_share`.

    Computed iteratively, and that detail is the whole point: capping against
    the *uncapped* total lets an attacker inflate the total and therefore its
    own ceiling. Clipping repeatedly until the shares stop moving converges on
    the guarantee actually wanted -- max(w)/sum(w) <= max_share.

    Excess weight is discarded rather than redistributed. Handing a capped
    contributor's surplus to everyone else would let a large attacker inflate
    whichever other deployments it also controls.
    """
    if not exposures:
        return []
    share = influence_cap(len(exposures)) if max_share is None else max_share
    if share >= 1.0:
        return list(exposures)

    capped = list(exposures)
    for _ in range(200):
        total = sum(capped)
        if total <= 0:
            return capped
        ceiling = share * total
        clipped = [min(e, ceiling) for e in capped]
        if all(abs(a - b) < 1e-12 for a, b in zip(clipped, capped)):
            break
        capped = clipped
    return capped
