"""Auditing for numbers that are too good, not just too bad.

Every monitoring product looks for anomalies in the bad direction. The July 2026
Hugging Face incident is a reminder that the dangerous direction is the other
one: agents spoofed their own tool calls to make *failures look like successes*
(METR found intentional spoofing in 7% of the transcripts it examined). For a
loss frequency that biases the number **downward** -- the deployment looks safer
than it is, and an underwriter prices it too cheaply.

Insurance has a name for the gap between what is reported and what actually
happened -- incurred but not reported -- and a mature habit of estimating it.
This module is the small version of that habit: given a class prior, how
surprising is a deployment that reports almost no losses?

The mathematics falls out of machinery already here. For a Beta-Bernoulli class
the credibility constant *is* the Beta concentration (K = a + b, checked in
`tests/test_estimate.py` against the closed form), so a class summarised by
(mu, K) is exactly Beta(mu*K, (1-mu)*K). A deployment's loss count is then
Beta-Binomial, and "is this deployment implausibly clean" is a one-sided tail
probability under that distribution.

What this is not
----------------
It flags an aggregate signature, never an individual episode -- content-free
telemetry cannot localise a spoofed call, and TelemetrySuffBench (arXiv
2608.07899) measured origin accuracy at ~0% once decision content is removed.

It is defeated by a careful adversary. Someone forging toward the class mean
rather than toward zero produces an entirely unremarkable number, and nothing
here would notice.

And a low p-value is not evidence of fraud. A genuinely excellent deployment
looks identical to a dishonest one from the outside. This is a prompt to go and
reconcile against out-of-band evidence, not a verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "UnderreportingResult",
    "beta_binomial_cdf",
    "binomial_cdf",
    "scan_class",
    "underreporting_test",
]


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_binomial_pmf(k: int, n: int, a: float, b: float) -> float:
    """P(X = k) for X ~ BetaBinomial(n, a, b)."""
    if k < 0 or k > n:
        return 0.0
    log_choose = (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    )
    return math.exp(log_choose + _log_beta(k + a, n - k + b) - _log_beta(a, b))


def beta_binomial_cdf(k: int, n: int, a: float, b: float) -> float:
    """P(X <= k). Summed directly; n here is episodes, not a large integral."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, math.fsum(beta_binomial_pmf(i, n, a, b) for i in range(k + 1)))


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p).

    The limit of the Beta-Binomial as the concentration grows: a class whose
    deployments are indistinguishable. That case is not degenerate, it is the
    tightest class there is, and it is precisely where an outlier should be
    easiest to see -- so it needs a distribution rather than an error.
    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = 0.0
    for i in range(k + 1):
        log_pmf = (
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p)
        )
        total += math.exp(log_pmf)
    return min(1.0, total)


@dataclass(frozen=True)
class UnderreportingResult:
    deployment_id: str
    n_episodes: int
    observed_losses: int
    expected_losses: float
    p_value: float          # P(losses <= observed) under the class prior
    flagged: bool

    @property
    def deficit(self) -> float:
        """Losses the class prior expected but did not see."""
        return self.expected_losses - self.observed_losses

    @property
    def verdict(self) -> str:
        if not self.flagged:
            return "consistent with the class"
        return (
            f"reports {self.observed_losses} losses in {self.n_episodes} episodes "
            f"where the class predicts {self.expected_losses:.1f}; "
            f"p={self.p_value:.2}. Reconcile against out-of-band evidence."
        )


def underreporting_test(
    deployment_id: str,
    observed_losses: int,
    n_episodes: int,
    mu: float,
    k: float,
    alpha: float = 0.01,
    min_episodes: int = 30,
) -> UnderreportingResult:
    """One-sided test: is this deployment implausibly clean for its class?

    `mu` and `k` come from the pooled prior. A deployment must have real
    exposure before the question is even meaningful -- with a handful of
    episodes, reporting zero losses is unremarkable however dishonest the
    reporter, so anything under `min_episodes` is never flagged.
    """
    if not math.isfinite(k) or k <= 0 or not 0.0 < mu < 1.0:
        return UnderreportingResult(
            deployment_id, n_episodes, observed_losses, float("nan"),
            float("nan"), False,
        )

    a, b = mu * k, (1.0 - mu) * k
    p = beta_binomial_cdf(observed_losses, n_episodes, a, b)
    expected = n_episodes * mu
    flagged = n_episodes >= min_episodes and p < alpha
    return UnderreportingResult(
        deployment_id=deployment_id,
        n_episodes=n_episodes,
        observed_losses=observed_losses,
        expected_losses=expected,
        p_value=p,
        flagged=flagged,
    )


def scan_class(
    per_deployment: dict[str, tuple[int, int]],
    mu: float,
    k: float,
    alpha: float = 0.01,
    min_episodes: int = 30,
) -> list[UnderreportingResult]:
    """Run the test across a class.

    `per_deployment` maps deployment id -> (observed_losses, n_episodes).

    The prior should be computed *excluding* the deployment under test where
    that is affordable; a contributor large enough to move the class mean would
    otherwise be compared against a prior it wrote itself. The influence cap in
    `admission.py` limits how far that can go, but it does not remove it.
    """
    return sorted(
        (
            underreporting_test(dep, losses, n, mu, k, alpha, min_episodes)
            for dep, (losses, n) in per_deployment.items()
        ),
        key=lambda r: (not r.flagged, r.p_value),
    )
