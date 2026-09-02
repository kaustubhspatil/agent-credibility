"""A baseline for attestation divergence.

The question an attestation tool cannot answer for its own user: three actions
went unreported this episode -- is that an attack, or is that Tuesday? Shell
wrappers, retries and subprocess fan-out all produce unreported actions in
honest deployments, and an uncalibrated detector drowns its user in false
positives until they switch it off.
"""

import math
import random

import pytest

from bureau.app import divergence_priors, ingest_episodes
from bureau.divergence import ClassDivergence, class_divergence, divergence_test
from bureau.store import Store
from freeboard import (
    Capability,
    Recorder,
    Reversibility,
    ToolSpec,
    to_wire,
)

TOOLS = [
    ToolSpec("read_case", capabilities=frozenset({Capability.PRIVATE_DATA}),
             reversibility=Reversibility.REVERSIBLE)
]


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "b.db")
    yield s
    s.close()


def fleet(tmp_path, store, name, n, divergence_rate, source="ebpf", seed=0,
          admit=True):
    rng = random.Random(seed)
    rec = Recorder(deployment_id=name, role="ops", tools=TOOLS,
                   state_path=str(tmp_path / f"{name}.state"))
    for i in range(n):
        with rec.episode(f"t{i}") as ep:
            ep.action("read_case")
            ep.action("read_case")
            if source != "none":
                extra = rng.randint(1, 3) if rng.random() < divergence_rate else 0
                ep.attest(actions_observed=2 + extra, source=source)
            ep.resolve(success=rng.random() > 0.3)
    ingest_episodes(store, [to_wire(r) for r in rec.records])
    if admit:
        store.set_admitted(rec.deployment_hash, True)
    return rec


# --- the two rules the module exists to enforce ----------------------------


def test_unattested_episodes_are_excluded_not_counted_as_zero(store, tmp_path):
    """An unobserved episode did not demonstrate zero divergence -- it
    demonstrated nothing. Counting it clean would drag every class rate toward
    zero in exactly the deployments instrumented worst."""
    fleet(tmp_path, store, "attested", 40, 0.5, source="ebpf", seed=1)
    fleet(tmp_path, store, "blind", 200, 0.0, source="none", seed=2)

    diverged, _, ids = store.divergence_observations("ops", "ebpf")
    assert len(diverged) == 40           # the 200 unattested episodes are absent
    assert set(ids) == {r for r in ids}
    assert 0.2 < sum(diverged) / len(diverged) < 0.8


def test_sources_are_never_pooled(store, tmp_path):
    """A kernel observer counts execve; a proxy counts requests. Averaging them
    is a number about nothing."""
    fleet(tmp_path, store, "a", 40, 0.9, source="ebpf", seed=3)
    fleet(tmp_path, store, "b", 40, 0.9, source="ebpf", seed=4)
    fleet(tmp_path, store, "c", 40, 0.05, source="proxy", seed=5)
    fleet(tmp_path, store, "d", 40, 0.05, source="proxy", seed=6)

    priors = {p["attestation_source"]: p for p in divergence_priors(store, "ops")}
    assert set(priors) == {"ebpf", "proxy"}
    assert priors["ebpf"]["divergence_rate"] > 0.6
    assert priors["proxy"]["divergence_rate"] < 0.3

    ebpf_rows, _, _ = store.divergence_observations("ops", "ebpf")
    assert len(ebpf_rows) == 80          # only the two ebpf deployments


# --- the baseline ----------------------------------------------------------


def test_a_baseline_needs_more_than_one_deployment(store, tmp_path):
    fleet(tmp_path, store, "solo", 60, 0.4, seed=7)
    p = divergence_priors(store, "ops")[0]
    assert p["available"] is False
    assert "need at least" in p["reason"]


def test_a_baseline_appears_with_two_deployments(store, tmp_path):
    for i in range(3):
        fleet(tmp_path, store, f"d{i}", 60, 0.35, seed=10 + i)
    p = divergence_priors(store, "ops")[0]
    assert p["available"] is True
    assert p["n_deployments"] == 3
    assert 0.15 < p["divergence_rate"] < 0.6
    assert p["mean_unreported_actions"] > 0


def test_only_admitted_deployments_inform_the_baseline(store, tmp_path):
    fleet(tmp_path, store, "in1", 60, 0.2, seed=20, admit=True)
    fleet(tmp_path, store, "in2", 60, 0.2, seed=21, admit=True)
    fleet(tmp_path, store, "out", 60, 0.99, seed=22, admit=False)
    p = divergence_priors(store, "ops")[0]
    assert p["n_deployments"] == 2
    assert p["divergence_rate"] < 0.5


# --- the test a customer actually asks -------------------------------------


def _cls(rate, k=6.0, n=200, deps=5):
    return ClassDivergence("ops", "ebpf", deps, n, rate, 1.2, k, True)


def test_a_normal_deployment_is_not_flagged():
    r = divergence_test("d", 12, 50, _cls(0.25))
    assert not r.flagged
    assert "normal for this class" in r.verdict


def test_a_deployment_diverging_far_above_its_class_is_flagged():
    r = divergence_test("d", 45, 50, _cls(0.10))
    assert r.flagged
    assert r.p_value < 0.01
    assert "not the same as malicious" in r.verdict


def test_thin_attestation_is_never_flagged():
    r = divergence_test("d", 8, 8, _cls(0.05))
    assert not r.flagged


def test_no_attested_episodes_says_so():
    r = divergence_test("d", 0, 0, _cls(0.2))
    assert not r.flagged
    assert "nothing is known" in r.verdict


def test_an_unusable_baseline_never_flags():
    unavailable = ClassDivergence("ops", "ebpf", 1, 10, 0.0, 0.0, float("inf"),
                                  False, "too few")
    assert not divergence_test("d", 50, 50, unavailable).flagged


def test_the_test_uses_the_upper_tail():
    """Opposite direction from underreporting: there, too few losses were the
    concern; here, too many unreported actions are."""
    low = divergence_test("d", 2, 50, _cls(0.30))
    high = divergence_test("d", 40, 50, _cls(0.30))
    assert low.p_value > high.p_value
    assert not low.flagged and high.flagged


# --- leave-one-out, and why it is not optional -----------------------------


def _fleet_arrays(honest_rate=0.18, outlier_rate=0.80, n=80, n_honest=4, seed=0):
    rng = random.Random(seed)
    diverged, mags, ids = [], [], []
    for d in range(n_honest):
        for _ in range(n):
            hit = rng.random() < honest_rate
            diverged.append(1.0 if hit else 0.0)
            mags.append(float(rng.randint(1, 3)) if hit else 0.0)
            ids.append(f"honest{d}")
    for _ in range(n):
        hit = rng.random() < outlier_rate
        diverged.append(1.0 if hit else 0.0)
        mags.append(float(rng.randint(1, 3)) if hit else 0.0)
        ids.append("outlier")
    return diverged, mags, ids


def test_an_outlier_hides_inside_a_class_it_is_part_of():
    """The reason leave-one-out is load-bearing rather than a refinement: a
    large outlier inflates the between-deployment variance the test relies on,
    so the class comes to expect the heterogeneity the outlier created."""
    from bureau.divergence import class_divergence, divergence_test

    diverged, mags, ids = _fleet_arrays(seed=99)
    cls_with = class_divergence(diverged, mags, ids, "ops", "ebpf")
    n = sum(1 for i in ids if i == "outlier")
    k = int(sum(v for v, i in zip(diverged, ids) if i == "outlier"))

    included = divergence_test("outlier", k, n, cls_with)
    assert not included.flagged          # hidden by the variance it created

    from bureau.divergence import divergence_scan

    held_out = {r.deployment_id: r for r in divergence_scan(diverged, mags, ids, "ops", "ebpf")}
    assert held_out["outlier"].flagged
    assert held_out["outlier"].p_value < included.p_value


def test_leave_one_out_clears_the_honest_deployments():
    from bureau.divergence import divergence_scan

    diverged, mags, ids = _fleet_arrays(seed=7)
    results = {r.deployment_id: r for r in divergence_scan(diverged, mags, ids, "ops", "ebpf")}
    assert results["outlier"].flagged
    assert not any(r.flagged for d, r in results.items() if d != "outlier")


def test_a_perfectly_tight_class_is_answerable_not_an_error():
    """Holding out the outlier can leave deployments that are indistinguishable,
    which sends K to infinity. That is the TIGHTEST class, and the case where an
    outlier is most visible -- so it needs the binomial limit, not a nan."""
    from bureau.anomaly import binomial_cdf
    from bureau.divergence import ClassDivergence, divergence_test

    assert binomial_cdf(-1, 10, 0.3) == 0.0
    assert binomial_cdf(10, 10, 0.3) == pytest.approx(1.0)
    assert binomial_cdf(5, 10, 0.5) == pytest.approx(0.623046875, rel=1e-9)

    tight = ClassDivergence("ops", "ebpf", 4, 320, 0.15, 0.4, float("inf"), True)
    r = divergence_test("d", 60, 80, tight)
    assert not math.isnan(r.p_value)
    assert r.flagged
    assert r.p_value < 1e-9


def test_a_single_other_deployment_is_not_enough_to_hold_out():
    from bureau.divergence import divergence_scan

    diverged = [0.0] * 40 + [1.0] * 40
    ids = ["a"] * 40 + ["b"] * 40
    mags = [0.0] * 40 + [1.0] * 40
    assert divergence_scan(diverged, mags, ids, "ops", "ebpf") == []
