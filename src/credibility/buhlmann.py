"""Bühlmann-Straub credibility: variance components and the credibility weight.

The whole business question reduces to one estimated constant.

    Z_i = n_i / (n_i + K),      K = EPV / VHM

EPV is the expected process variance -- how noisy one deployment is around its
own long-run mean. VHM is the variance of the hypothetical means -- how much
deployments genuinely differ from each other. If deployments are noisy but
alike, K explodes, Z stays near zero, and every deployment is priced at the
class average forever. If deployments genuinely differ relative to their own
noise, K is small, Z rises quickly, and a deployment starts pricing off its own
experience within days.

Estimators are the standard unbiased Bühlmann-Straub ones; see Bühlmann &
Gisler, *A Course in Credibility Theory and its Applications* (2005), ch. 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Components:
    """Variance decomposition for one risk class."""

    mu: float          # collective (class) mean
    epv: float         # s^2 : expected process variance, within-deployment
    vhm: float         # a   : variance of hypothetical means, between-deployment
    k: float           # EPV / VHM -- the credibility constant, in episodes
    n_total: int
    n_risks: int
    vhm_was_truncated: bool  # True when the raw VHM estimate came out negative

    def z(self, n: np.ndarray | float) -> np.ndarray | float:
        """Credibility weight on own experience after n episodes."""
        if not np.isfinite(self.k):
            return np.zeros_like(np.asarray(n, dtype=float))
        return np.asarray(n, dtype=float) / (np.asarray(n, dtype=float) + self.k)

    def episodes_for_z(self, target: float) -> float:
        """How many episodes until own experience carries `target` weight."""
        if not np.isfinite(self.k):
            return float("inf")
        return self.k * target / (1.0 - target)


def components(
    values: np.ndarray,
    risk_ids: np.ndarray,
    weights: np.ndarray | None = None,
) -> Components:
    """Estimate (mu, EPV, VHM, K) from episode-level observations.

    Parameters
    ----------
    values
        One observation per episode. Binary (0/1 loss indicator) or continuous.
    risk_ids
        Deployment identifier for each episode.
    weights
        Optional per-episode exposure weight. Defaults to 1 per episode.
    """
    values = np.asarray(values, dtype=float)
    risk_ids = np.asarray(risk_ids)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=float)

    uniq, inv = np.unique(risk_ids, return_inverse=True)
    n_risks = len(uniq)
    if n_risks < 2:
        raise ValueError("need at least two deployments to decompose variance")

    # per-deployment exposure and weighted mean
    n_i = np.bincount(inv, weights=weights, minlength=n_risks)
    sum_i = np.bincount(inv, weights=weights * values, minlength=n_risks)
    x_i = np.divide(sum_i, n_i, out=np.zeros_like(sum_i), where=n_i > 0)

    n_total = n_i.sum()
    mu = float(sum_i.sum() / n_total)

    # ---- EPV: pooled within-deployment variance --------------------------
    resid = values - x_i[inv]
    within_ss = float(np.sum(weights * resid**2))
    dof = float(n_total - n_risks)
    if dof <= 0:
        raise ValueError("no within-deployment degrees of freedom")
    epv = within_ss / dof

    # ---- VHM: between-deployment variance, debiased ----------------------
    between_ss = float(np.sum(n_i * (x_i - mu) ** 2))
    # E[between_ss] = (n_risks - 1) * EPV + (n_total - sum n_i^2 / n_total) * VHM
    scale = float(n_total - np.sum(n_i**2) / n_total)
    raw_vhm = (between_ss - (n_risks - 1) * epv) / scale if scale > 0 else -1.0

    truncated = raw_vhm <= 0
    vhm = 0.0 if truncated else float(raw_vhm)
    k = float("inf") if vhm <= 0 else epv / vhm

    return Components(
        mu=mu,
        epv=float(epv),
        vhm=vhm,
        k=k,
        n_total=int(n_total),
        n_risks=n_risks,
        vhm_was_truncated=bool(truncated),
    )


def bootstrap_k(
    values: np.ndarray,
    risk_ids: np.ndarray,
    n_boot: int = 400,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI for K, resampling *deployments* (not episodes).

    Resampling whole deployments is the right unit: the uncertainty that
    matters is whether the set of deployments we happen to have is
    representative of the class.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    risk_ids = np.asarray(risk_ids)
    uniq = np.unique(risk_ids)
    index = {r: np.flatnonzero(risk_ids == r) for r in uniq}

    ks: list[float] = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        vals, ids = [], []
        for j, r in enumerate(picked):
            idx = index[r]
            vals.append(values[idx])
            ids.append(np.full(len(idx), j))  # relabel so duplicates stay distinct
        try:
            comp = components(np.concatenate(vals), np.concatenate(ids))
        except ValueError:
            continue
        ks.append(comp.k)

    finite = np.array([k for k in ks if np.isfinite(k)])
    if len(finite) < 0.5 * max(len(ks), 1):
        return float("inf"), float("inf")
    return float(np.percentile(finite, 5)), float(np.percentile(finite, 95))


def credibility_estimate(
    own_mean: np.ndarray | float,
    n: np.ndarray | float,
    comp: Components,
) -> np.ndarray:
    """Z * own experience + (1 - Z) * class prior."""
    z = np.asarray(comp.z(n), dtype=float)
    return z * np.asarray(own_mean, dtype=float) + (1.0 - z) * comp.mu
