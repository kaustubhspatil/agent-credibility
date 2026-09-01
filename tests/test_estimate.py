"""The stdlib estimator, checked two ways.

Against a closed form, because for a Beta(a,b)-Bernoulli class the credibility
constant is exactly `a + b`, so there is a known right answer.

And against the numpy implementation used for the published research, because
two implementations of the same formulas are only useful if they cannot drift.
If these ever disagree, the published numbers and the shipped SDK have stopped
describing the same estimator, and that is worse than having only one.
"""

import math
import random

import pytest

from freeboard import (
    bootstrap_k,
    components,
    credibility_estimate,
    losses_from_records,
)

np = pytest.importorskip("numpy", reason="reference implementation needs numpy")
from credibility import buhlmann as reference  # noqa: E402


def _beta_bernoulli(a, b, n_risks, n_per, seed):
    rng = random.Random(seed)
    values, ids = [], []
    for i in range(n_risks):
        theta = rng.betavariate(a, b)
        for _ in range(n_per):
            values.append(1.0 if rng.random() < theta else 0.0)
            ids.append(i)
    return values, ids


# --- closed form -----------------------------------------------------------


@pytest.mark.parametrize("a,b", [(2.0, 8.0), (5.0, 5.0), (20.0, 30.0)])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_recovers_the_beta_concentration(a, b, seed):
    values, ids = _beta_bernoulli(a, b, 300, 200, seed)
    comp = components(values, ids)
    # VHM is a difference of large quantities and is the noisy term; +/-30% at
    # 300 deployments is sampling variation, which is why K ships with an
    # interval rather than as a point estimate.
    assert comp.k == pytest.approx(a + b, rel=0.30)
    assert comp.mu == pytest.approx(a / (a + b), rel=0.10)
    assert not comp.vhm_was_truncated


def test_alike_deployments_never_earn_credibility():
    """The failure mode the whole study exists to detect."""
    rng = random.Random(3)
    values = [1.0 if rng.random() < 0.3 else 0.0 for _ in range(200 * 150)]
    ids = [i // 150 for i in range(200 * 150)]
    comp = components(values, ids)
    assert comp.vhm < 0.01 * comp.epv
    assert comp.k > 1000
    assert comp.z(500) < 0.5
    assert credibility_estimate(0.9, 500, comp) == pytest.approx(comp.mu, abs=0.08)


# --- agreement with the research implementation ----------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_the_numpy_implementation(seed):
    values, ids = _beta_bernoulli(3.0, 7.0, 150, 40, seed)
    mine = components(values, ids)
    theirs = reference.components(np.array(values), np.array(ids))

    assert mine.mu == pytest.approx(theirs.mu, rel=1e-12)
    assert mine.epv == pytest.approx(theirs.epv, rel=1e-12)
    assert mine.vhm == pytest.approx(theirs.vhm, rel=1e-9)
    assert mine.k == pytest.approx(theirs.k, rel=1e-9)
    assert mine.n_risks == theirs.n_risks
    assert mine.n_total == pytest.approx(theirs.n_total)


def test_matches_the_numpy_implementation_with_uneven_exposure():
    rng = random.Random(11)
    values, ids = [], []
    for i in range(120):
        theta = rng.betavariate(2.0, 6.0)
        for _ in range(rng.randint(5, 300)):
            values.append(1.0 if rng.random() < theta else 0.0)
            ids.append(i)
    mine = components(values, ids)
    theirs = reference.components(np.array(values), np.array(ids))
    assert mine.k == pytest.approx(theirs.k, rel=1e-9)
    assert mine.vhm == pytest.approx(theirs.vhm, rel=1e-9)


def test_matches_the_numpy_implementation_on_continuous_values():
    """Behavioural signals are continuous; the decomposition is the same."""
    rng = random.Random(5)
    values, ids = [], []
    for i in range(100):
        centre = rng.gauss(10.0, 2.0)
        for _ in range(50):
            values.append(rng.gauss(centre, 3.0))
            ids.append(i)
    mine = components(values, ids)
    theirs = reference.components(np.array(values), np.array(ids))
    assert mine.epv == pytest.approx(theirs.epv, rel=1e-10)
    assert mine.k == pytest.approx(theirs.k, rel=1e-9)


def test_matches_on_the_real_corpus_if_it_has_been_extracted():
    """The strongest agreement check: the actual published population."""
    pd = pytest.importorskip("pandas")
    import pathlib

    path = pathlib.Path("data/episodes.parquet")
    if not path.exists():
        pytest.skip("run `python -m credibility.extract` first")

    df = pd.read_parquet(path)
    df = df[df["scaffold"] == "tool"].drop_duplicates(subset=["traj_id"])
    counts = df["repo"].value_counts()
    df = df[df["repo"].isin(counts[counts >= 12].index)]

    values = [0.0 if r else 1.0 for r in df["resolved"]]
    ids = list(df["repo"])

    mine = components(values, ids)
    theirs = reference.components(np.array(values), np.array(ids))
    assert mine.k == pytest.approx(theirs.k, rel=1e-9)
    assert mine.k == pytest.approx(7.59, abs=0.05)  # the published headline


def test_bootstrap_interval_brackets_the_estimate():
    values, ids = _beta_bernoulli(3.0, 7.0, 120, 60, seed=7)
    comp = components(values, ids)
    lo, hi = bootstrap_k(values, ids, n_boot=200, seed=1)
    assert lo < comp.k < hi
    assert lo > 0


# --- the estimator's interface to the recorder -----------------------------


def test_z_and_episodes_for_z_are_inverses():
    values, ids = _beta_bernoulli(4.0, 6.0, 200, 100, seed=5)
    comp = components(values, ids)
    for target in (0.25, 0.5, 0.9):
        assert comp.z(comp.episodes_for_z(target)) == pytest.approx(target)


def test_infinite_k_gives_zero_weight_and_no_finite_exposure():
    comp = components([0.0, 1.0, 0.0, 1.0], [0, 0, 1, 1])
    if not math.isfinite(comp.k):
        assert comp.z(10_000) == 0.0
        assert comp.episodes_for_z(0.5) == math.inf


def test_losses_from_records_inverts_resolved_and_skips_unknown():
    from freeboard import Recorder, Reversibility, ToolSpec

    tools = [ToolSpec("t", capabilities=frozenset(),
                      reversibility=Reversibility.REVERSIBLE)]
    rec = Recorder(deployment_id="d", role="ops", tools=tools)
    with rec.episode("a") as ep:
        ep.resolve(success=True)
    with rec.episode("b") as ep:
        ep.resolve(success=False)
    with rec.episode("c") as ep:
        pass  # no outcome recorded at all

    values, ids = losses_from_records(rec.records)
    # success -> 0 loss, failure -> 1 loss, unknown -> dropped rather than
    # counted as a success, which would bias the base rate downward
    assert values == [0.0, 1.0]
    assert len(ids) == 2
    assert len(set(ids)) == 1


def test_losses_from_records_accepts_wire_dicts_too():
    from freeboard import Recorder, Reversibility, ToolSpec, to_wire

    tools = [ToolSpec("t", capabilities=frozenset(),
                      reversibility=Reversibility.REVERSIBLE)]
    rec = Recorder(deployment_id="d", role="ops", tools=tools)
    for ok in (True, False, True):
        with rec.episode("x") as ep:
            ep.resolve(success=ok)

    from_objects = losses_from_records(rec.records)
    from_dicts = losses_from_records([to_wire(r) for r in rec.records])
    assert from_objects == from_dicts


def test_mismatched_input_lengths_are_refused():
    with pytest.raises(ValueError):
        components([0.0, 1.0], [0])
    with pytest.raises(ValueError):
        components([0.0, 1.0], [0, 1], weights=[1.0])
    with pytest.raises(ValueError):
        components([0.0], [0])  # only one deployment
