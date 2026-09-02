"""How much divergence is normal?

An attestation tool tells a customer that three actions went unreported this
episode. Nobody can currently tell them whether that is an attack or a Tuesday.
Shell wrappers, retries, subprocess fan-out and plain instrumentation gaps all
produce unreported actions in perfectly honest deployments, and without a
baseline every detector drowns its user in false positives until they turn it
off -- which is the failure mode that makes most security tooling shelfware.

That baseline is the same shape as everything else here: the measurement exists
(eBPF, egress proxies, audit logs), the base rate does not. So the credibility
machinery that prices loss frequency prices divergence too, with the same K and
the same Z, and a deployment can be asked the one question its own tooling
cannot answer -- *is this unusual for agents like mine?*

Two rules this module exists to enforce, both easy to get wrong:

**Unattested episodes are excluded, never counted as zero.** An episode with no
observer did not demonstrate zero divergence; it demonstrated nothing. Treating
absence of evidence as evidence of absence would drag every class rate toward
zero in exactly the deployments with the weakest instrumentation.

**Divergence is never pooled across attestation sources.** A kernel observer
counts `execve`; an egress proxy counts outbound requests; an audit log counts
whatever the platform chose to log. Their baselines are not comparable, and an
average over them is a number about nothing. The class is (role, source).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from freeboard.estimate import Components, components

from .anomaly import beta_binomial_cdf, binomial_cdf

__all__ = [
    "ClassDivergence",
    "DivergenceResult",
    "class_divergence",
    "divergence_scan",
    "divergence_test",
]


@dataclass(frozen=True)
class ClassDivergence:
    """What divergence looks like for a role, under one attestation source."""

    role: str
    source: str
    n_deployments: int
    n_attested_episodes: int
    rate: float                  # share of attested episodes with any divergence
    mean_unreported: float       # mean unreported actions, over attested episodes
    k: float                     # credibility constant for the divergence rate
    available: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "role": self.role,
            "attestation_source": self.source,
            "available": self.available,
            "reason": self.reason or None,
            "n_deployments": self.n_deployments,
            "n_attested_episodes": self.n_attested_episodes,
            "divergence_rate": round(self.rate, 6),
            "mean_unreported_actions": round(self.mean_unreported, 4),
            "k": None if not math.isfinite(self.k) else round(self.k, 4),
        }


def class_divergence(
    values: list[float],
    magnitudes: list[float],
    deployment_ids: list[str],
    role: str,
    source: str,
    min_deployments: int = 2,
) -> ClassDivergence:
    """Pool a divergence baseline across deployments sharing a role and source.

    `values` is 1.0 where an attested episode showed any unreported action.
    `magnitudes` is the unreported count for the same episodes.
    """
    n = len(values)
    n_dep = len(set(deployment_ids))
    if n_dep < min_deployments or n == 0:
        return ClassDivergence(
            role, source, n_dep, n, 0.0, 0.0, math.inf, False,
            f"{n_dep} deployment(s) reporting attested episodes; "
            f"need at least {min_deployments}",
        )

    try:
        comp: Components = components(values, deployment_ids)
        k = comp.k
        rate = comp.mu
    except ValueError:
        rate = sum(values) / n
        k = math.inf

    return ClassDivergence(
        role=role,
        source=source,
        n_deployments=n_dep,
        n_attested_episodes=n,
        rate=rate,
        mean_unreported=(sum(magnitudes) / n) if n else 0.0,
        k=k,
        available=True,
    )


@dataclass(frozen=True)
class DivergenceResult:
    deployment_id: str
    n_attested: int
    n_diverged: int
    expected: float
    p_value: float               # P(diverged >= observed) under the class
    flagged: bool

    @property
    def verdict(self) -> str:
        if self.n_attested == 0:
            return "no attested episodes: nothing was observed, so nothing is known"
        if not self.flagged:
            return (
                f"{self.n_diverged}/{self.n_attested} episodes diverged; "
                f"normal for this class"
            )
        return (
            f"{self.n_diverged}/{self.n_attested} episodes diverged where the "
            f"class predicts {self.expected:.1f} (p={self.p_value:.2}). Unusual, "
            f"which is not the same as malicious -- instrumentation drift looks "
            f"identical from here."
        )


def divergence_test(
    deployment_id: str,
    n_diverged: int,
    n_attested: int,
    cls: ClassDivergence,
    alpha: float = 0.01,
    min_episodes: int = 30,
) -> DivergenceResult:
    """Is this deployment diverging more than its class?

    The opposite tail from `anomaly.underreporting_test`: there, implausibly few
    losses were the concern; here, implausibly many unreported actions are.
    """
    if not cls.available or not 0.0 < cls.rate < 1.0 or cls.k <= 0:
        return DivergenceResult(
            deployment_id, n_attested, n_diverged, float("nan"), float("nan"), False
        )

    # An infinite K means the held-out deployments were indistinguishable from
    # one another -- the tightest possible class, and the case where an outlier
    # is most visible. Beta-Binomial converges to Binomial there, so use it
    # rather than refusing to answer.
    if math.isfinite(cls.k):
        a, b = cls.rate * cls.k, (1.0 - cls.rate) * cls.k
        below = beta_binomial_cdf(n_diverged - 1, n_attested, a, b)
    else:
        below = binomial_cdf(n_diverged - 1, n_attested, cls.rate)
    p = 1.0 - below
    return DivergenceResult(
        deployment_id=deployment_id,
        n_attested=n_attested,
        n_diverged=n_diverged,
        expected=n_attested * cls.rate,
        p_value=p,
        flagged=n_attested >= min_episodes and p < alpha,
    )


def divergence_scan(
    diverged: list[float],
    magnitudes: list[float],
    deployment_ids: list[str],
    role: str,
    source: str,
    alpha: float = 0.01,
    min_episodes: int = 30,
) -> list[DivergenceResult]:
    """Test every deployment against a baseline built WITHOUT it.

    Leave-one-out is not a refinement here, it is the difference between the
    test working and not working. A single large outlier included in its own
    class inflates the between-deployment variance the test relies on, so the
    class comes to "expect" enormous heterogeneity and the outlier hides inside
    the tolerance it created. Measured on a synthetic fleet: a deployment
    diverging on 64 of 80 episodes against a class expecting 21.6 scored
    p = 0.077 and went unflagged when included in its own baseline, and
    p < 1e-9 when held out.

    The same reasoning applies to `anomaly.underreporting_test`, and for the
    same reason.
    """
    results: list[DivergenceResult] = []
    for dep in sorted(set(deployment_ids)):
        keep = [i for i, d in enumerate(deployment_ids) if d != dep]
        if len({deployment_ids[i] for i in keep}) < 2:
            continue
        cls = class_divergence(
            [diverged[i] for i in keep],
            [magnitudes[i] for i in keep],
            [deployment_ids[i] for i in keep],
            role,
            source,
        )
        mine = [i for i, d in enumerate(deployment_ids) if d == dep]
        results.append(
            divergence_test(
                dep,
                int(sum(diverged[i] for i in mine)),
                len(mine),
                cls,
                alpha=alpha,
                min_episodes=min_episodes,
            )
        )
    return sorted(results, key=lambda r: (not r.flagged, r.p_value))
