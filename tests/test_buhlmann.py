"""The estimator has to be right before its verdict means anything.

Analytic anchor: if deployment failure rates are drawn theta_i ~ Beta(a, b) and
each episode is Bernoulli(theta_i), then

    VHM = Var(theta)          = m(1-m) / (a+b+1)
    EPV = E[theta(1-theta)]   = m(1-m) - Var(theta)
    K   = EPV / VHM           = a + b

So the credibility constant of a Beta-Bernoulli class is exactly the Beta
concentration. Any estimator that cannot recover it is not measuring what we
are about to make a decision with.
"""

import numpy as np
import pytest

from credibility.buhlmann import components, credibility_estimate


def _simulate(a, b, n_risks, n_per_risk, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.beta(a, b, size=n_risks)
    values, ids = [], []
    for i, t in enumerate(theta):
        n = n_per_risk if np.isscalar(n_per_risk) else n_per_risk[i]
        values.append(rng.binomial(1, t, size=n))
        ids.append(np.full(n, i))
    return np.concatenate(values), np.concatenate(ids), theta


@pytest.mark.parametrize("a,b", [(2.0, 8.0), (5.0, 5.0), (20.0, 30.0)])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_recovers_beta_concentration(a, b, seed):
    values, ids, _ = _simulate(a, b, n_risks=300, n_per_risk=200, seed=seed)
    comp = components(values, ids)
    # VHM is a difference of two large quantities and is the noisy component;
    # +/-30% on K with 300 deployments is sampling variation, not bias. This is
    # exactly why the real analysis reports a bootstrap interval on K rather
    # than a point estimate.
    assert comp.k == pytest.approx(a + b, rel=0.30)
    assert comp.mu == pytest.approx(a / (a + b), rel=0.10)
    assert not comp.vhm_was_truncated


def test_unequal_exposure_still_recovers_k():
    rng = np.random.default_rng(7)
    sizes = rng.integers(20, 600, size=250)
    values, ids, _ = _simulate(3.0, 12.0, n_risks=250, n_per_risk=sizes, seed=2)
    comp = components(values, ids)
    assert comp.k == pytest.approx(15.0, rel=0.25)


def test_identical_deployments_give_no_credibility():
    """The kill condition: deployments that do not differ can never earn Z."""
    rng = np.random.default_rng(3)
    n_risks, n_per = 200, 150
    values = rng.binomial(1, 0.3, size=n_risks * n_per).astype(float)
    ids = np.repeat(np.arange(n_risks), n_per)
    comp = components(values, ids)
    # True VHM is 0, so the debiased estimate lands on either side of zero by
    # chance; asserting the truncation flag would be a coin flip. Assert the
    # decision instead: no realistic amount of exposure earns credibility.
    assert comp.vhm < 0.01 * comp.epv
    assert comp.k > 1000
    assert comp.z(500) < 0.5
    # the estimate stays pinned near the class mean even with heavy experience
    assert credibility_estimate(0.9, 500, comp) == pytest.approx(comp.mu, abs=0.08)


def test_z_and_episodes_for_z_are_inverses():
    values, ids, _ = _simulate(4.0, 6.0, n_risks=200, n_per_risk=300, seed=5)
    comp = components(values, ids)
    for target in (0.25, 0.5, 0.9):
        n = comp.episodes_for_z(target)
        assert comp.z(n) == pytest.approx(target)


def test_continuous_values_supported():
    """Behavioural signals are continuous; the same decomposition applies."""
    rng = np.random.default_rng(11)
    n_risks, n_per = 200, 120
    means = rng.normal(10.0, 2.0, size=n_risks)          # VHM = 4
    values = np.concatenate(
        [rng.normal(m, 3.0, size=n_per) for m in means]  # EPV = 9
    )
    ids = np.repeat(np.arange(n_risks), n_per)
    comp = components(values, ids)
    assert comp.epv == pytest.approx(9.0, rel=0.15)
    assert comp.vhm == pytest.approx(4.0, rel=0.25)
    assert comp.k == pytest.approx(9.0 / 4.0, rel=0.30)
