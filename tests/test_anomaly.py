"""Auditing for numbers that are too good.

The July 2026 Hugging Face incident is the motivation: agents spoofed their own
tool calls so failures looked like successes, which biases a loss frequency
DOWNWARD -- the direction that under-prices risk. Every monitoring product looks
for anomalies in the bad direction.
"""

import math

import pytest

from bureau.anomaly import (
    beta_binomial_cdf,
    beta_binomial_pmf,
    scan_class,
    underreporting_test,
)


def test_the_pmf_is_a_distribution():
    a, b, n = 4.8, 3.2, 40
    total = sum(beta_binomial_pmf(k, n, a, b) for k in range(n + 1))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_cdf_is_monotone_and_bounded():
    a, b, n = 4.8, 3.2, 30
    prev = -1.0
    for k in range(-1, n + 2):
        c = beta_binomial_cdf(k, n, a, b)
        assert 0.0 <= c <= 1.0
        assert c >= prev
        prev = c
    assert beta_binomial_cdf(n, n, a, b) == pytest.approx(1.0)


def test_the_mean_matches_the_class():
    """K is the Beta concentration, so a class (mu, K) is Beta(mu*K, (1-mu)*K).
    The Beta-Binomial mean must come back to n*mu."""
    mu, k, n = 0.6, 8.0, 50
    a, b = mu * k, (1 - mu) * k
    mean = sum(i * beta_binomial_pmf(i, n, a, b) for i in range(n + 1))
    assert mean == pytest.approx(n * mu, rel=1e-9)


# --- the test itself -------------------------------------------------------


def test_a_deployment_at_the_class_mean_is_not_flagged():
    r = underreporting_test("d", 30, 50, mu=0.60, k=8.0)
    assert not r.flagged
    assert r.p_value > 0.1
    assert r.expected_losses == pytest.approx(30.0)


def test_a_deployment_reporting_nothing_is_flagged():
    r = underreporting_test("d", 0, 50, mu=0.60, k=8.0)
    assert r.flagged
    assert r.p_value < 1e-4
    assert r.deficit == pytest.approx(30.0)
    assert "out-of-band" in r.verdict


def test_thin_exposure_is_never_flagged():
    """Reporting zero losses in a handful of episodes is unremarkable however
    dishonest the reporter, so it must not be treated as evidence."""
    r = underreporting_test("d", 0, 10, mu=0.60, k=8.0)
    assert not r.flagged
    assert r.n_episodes < 30


def test_an_adversary_forging_toward_the_mean_is_not_caught():
    """Stated in the module docstring and pinned here: this catches crude
    under-reporting, not a careful one."""
    r = underreporting_test("smart", 28, 50, mu=0.60, k=8.0)
    assert not r.flagged


def test_a_heterogeneous_class_has_less_power():
    """Low K makes credibility fast and fraud detection weak: a class whose
    deployments genuinely differ that much cannot call an outlier."""
    clean = 5
    tight = underreporting_test("d", clean, 50, mu=0.60, k=200.0)   # homogeneous
    loose = underreporting_test("d", clean, 50, mu=0.60, k=3.0)     # very varied
    assert tight.p_value < loose.p_value
    assert tight.flagged and not loose.flagged


def test_an_unusable_prior_never_flags():
    for mu, k in ((0.6, math.inf), (0.0, 8.0), (1.0, 8.0), (0.6, 0.0)):
        r = underreporting_test("d", 0, 100, mu=mu, k=k)
        assert not r.flagged


def test_scan_puts_the_flagged_first():
    prior = dict(mu=0.60, k=8.0)
    results = scan_class(
        {"honest": (30, 50), "clean": (0, 60), "small": (0, 5), "lucky": (24, 50)},
        **prior,
    )
    assert results[0].deployment_id == "clean"
    assert results[0].flagged
    assert not any(r.flagged for r in results[1:])
