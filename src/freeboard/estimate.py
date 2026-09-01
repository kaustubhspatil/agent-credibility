"""Bühlmann-Straub credibility, in the standard library.

Without this the package records but cannot measure, which does not match its
own description. The mathematics is weighted means and two variance components
-- arithmetic over lists -- so it does not need numpy, and keeping it
dependency-free matters more here than elsewhere: this code runs inside a
customer's process, next to their agent, and every dependency is something
their review has to clear.

    p̂ = Z · (this deployment's own experience) + (1 − Z) · (its class prior)

    Z = n / (n + K)          K = EPV / VHM

EPV is the expected process variance: how noisy one deployment is around its own
long-run rate. VHM is the variance of hypothetical means: how much deployments
genuinely differ. If deployments are noisy but alike, K explodes, Z stays near
zero, and everything is priced at the class average forever.

The estimators are the standard unbiased Bühlmann-Straub ones (Bühlmann &
Gisler, *A Course in Credibility Theory and its Applications*, 2005, ch. 4), and
`tests/test_estimate.py` checks them against both a closed form and the numpy
implementation used for the research, so the two cannot drift apart.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "Components",
    "bootstrap_k",
    "components",
    "credibility_estimate",
    "losses_from_records",
]


@dataclass(frozen=True)
class Components:
    """The variance decomposition for one risk class."""

    mu: float           # the class mean: what a brand-new deployment is priced at
    epv: float          # expected process variance, within a deployment
    vhm: float          # variance of hypothetical means, between deployments
    k: float            # EPV / VHM, in episodes. inf when deployments look alike
    n_total: float
    n_risks: int
    vhm_was_truncated: bool  # True when the debiased VHM estimate came out <= 0

    def z(self, n: float) -> float:
        """Credibility weight on own experience after n episodes."""
        if not math.isfinite(self.k):
            return 0.0
        return n / (n + self.k)

    def episodes_for_z(self, target: float) -> float:
        """How much exposure until own experience carries `target` weight."""
        if not 0.0 < target < 1.0:
            raise ValueError("target must be strictly between 0 and 1")
        if not math.isfinite(self.k):
            return math.inf
        return self.k * target / (1.0 - target)


def components(
    values: Sequence[float],
    risk_ids: Sequence[object],
    weights: Sequence[float] | None = None,
) -> Components:
    """Estimate (mu, EPV, VHM, K) from episode-level observations.

    `values` is one observation per episode -- a 0/1 loss indicator, or any
    real-valued behavioural signal. `risk_ids` says which deployment each
    belongs to.
    """
    if len(values) != len(risk_ids):
        raise ValueError("values and risk_ids must be the same length")
    if weights is None:
        weights = [1.0] * len(values)
    elif len(weights) != len(values):
        raise ValueError("weights must be the same length as values")

    # per-deployment exposure and weighted mean
    exposure: dict[object, float] = {}
    weighted_sum: dict[object, float] = {}
    for value, rid, weight in zip(values, risk_ids, weights):
        exposure[rid] = exposure.get(rid, 0.0) + weight
        weighted_sum[rid] = weighted_sum.get(rid, 0.0) + weight * value

    n_risks = len(exposure)
    if n_risks < 2:
        raise ValueError("need at least two deployments to decompose variance")

    means = {
        rid: (weighted_sum[rid] / exposure[rid] if exposure[rid] else 0.0)
        for rid in exposure
    }
    n_total = math.fsum(exposure.values())
    mu = math.fsum(weighted_sum.values()) / n_total

    # EPV: pooled within-deployment variance
    within_ss = math.fsum(
        weight * (value - means[rid]) ** 2
        for value, rid, weight in zip(values, risk_ids, weights)
    )
    dof = n_total - n_risks
    if dof <= 0:
        raise ValueError("no within-deployment degrees of freedom")
    epv = within_ss / dof

    # VHM: between-deployment variance, debiased for the sampling noise of the
    # deployment means themselves. Observed spread always overstates the true
    # spread, and by exactly (n_risks - 1) * EPV in expectation.
    between_ss = math.fsum(
        exposure[rid] * (means[rid] - mu) ** 2 for rid in exposure
    )
    scale = n_total - math.fsum(e * e for e in exposure.values()) / n_total
    raw_vhm = (between_ss - (n_risks - 1) * epv) / scale if scale > 0 else -1.0

    truncated = raw_vhm <= 0
    vhm = 0.0 if truncated else raw_vhm
    k = math.inf if vhm <= 0 else epv / vhm

    return Components(
        mu=mu,
        epv=epv,
        vhm=vhm,
        k=k,
        n_total=n_total,
        n_risks=n_risks,
        vhm_was_truncated=truncated,
    )


def credibility_estimate(own_mean: float, n: float, comp: Components) -> float:
    """Z · own experience + (1 − Z) · class prior."""
    z = comp.z(n)
    return z * own_mean + (1.0 - z) * comp.mu


def bootstrap_k(
    values: Sequence[float],
    risk_ids: Sequence[object],
    n_boot: int = 400,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval for K, resampling *deployments*, not episodes.

    Deployments are the right resampling unit: the uncertainty that matters is
    whether the deployments we happen to have represent the class, not whether
    their episodes do. K is a ratio whose denominator is a difference of large
    quantities, so a point estimate without an interval is misleading.
    """
    rng = random.Random(seed)
    index: dict[object, list[int]] = {}
    for i, rid in enumerate(risk_ids):
        index.setdefault(rid, []).append(i)
    unique = list(index)

    estimates: list[float] = []
    finite = 0
    for _ in range(n_boot):
        picked = [rng.choice(unique) for _ in unique]
        boot_values: list[float] = []
        boot_ids: list[object] = []
        for label, rid in enumerate(picked):
            for i in index[rid]:
                boot_values.append(values[i])
                boot_ids.append(label)  # relabel so duplicates stay distinct
        try:
            k = components(boot_values, boot_ids).k
        except ValueError:
            continue
        estimates.append(k)
        if math.isfinite(k):
            finite += 1

    usable = sorted(k for k in estimates if math.isfinite(k))
    if not estimates or finite < 0.5 * len(estimates):
        return math.inf, math.inf
    return _percentile(usable, 5.0), _percentile(usable, 95.0)


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile, matching numpy's default."""
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (pct / 100.0) * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[int(pos)]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def losses_from_records(
    records: Iterable[object],
    channel: str = "resolved",
) -> tuple[list[float], list[str]]:
    """Turn episode records into the (values, risk_ids) the estimator takes.

    Accepts `EpisodeRecord` objects or the dicts produced by `to_wire`, so it
    works equally on live recording and on a file read back later.

    `resolved` is inverted, because the estimator prices *loss* frequency and
    `resolved=True` is a success. Episodes with no outcome are skipped rather
    than counted as successes -- silently treating an unknown as a win biases
    the base rate downward, which is the direction that under-prices risk.
    """
    values: list[float] = []
    ids: list[str] = []
    for record in records:
        get = record.get if isinstance(record, dict) else lambda k: getattr(record, k)
        outcome = get(channel)
        if outcome is None:
            continue
        value = (0.0 if outcome else 1.0) if channel == "resolved" else float(outcome)
        values.append(value)
        ids.append(get("deployment_id"))
    return values, ids
