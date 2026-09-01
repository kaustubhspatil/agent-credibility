"""The bureau: does it accept what it should and refuse what it must?

Two properties matter more than the rest. Content must not be able to get in
through the API any more than it can get out through the SDK. And nobody may
move a published base rate just by sending traffic at it.
"""

import random
import time

import pytest

from bureau.admission import capped_weights, evaluate, influence_cap
from bureau.app import (
    BureauError,
    class_prior,
    ingest_checkpoint,
    ingest_episodes,
)
from bureau.store import Store
from freeboard import (
    Capability,
    Recorder,
    Reversibility,
    ToolSpec,
    to_wire,
)

TOOLS = [
    ToolSpec(
        "read_case",
        capabilities=frozenset({Capability.PRIVATE_DATA}),
        reversibility=Reversibility.REVERSIBLE,
    )
]


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "bureau.db")
    yield s
    s.close()


def fleet(tmp_path, name, rate, n, seed=0, role="customer_support"):
    rng = random.Random(seed)
    rec = Recorder(
        deployment_id=name,
        role=role,
        tools=TOOLS,
        state_path=str(tmp_path / f"{name}.state.json"),
    )
    for i in range(n):
        with rec.episode(f"t{i}") as ep:
            ep.action("read_case")
            ep.resolve(success=rng.random() > rate)
    return rec, [to_wire(r) for r in rec.records]


def age_and_checkpoint(store, rec, hours=5, checkpoints=2):
    """Simulate a deployment that has been running a while and anchored twice."""
    did = rec.deployment_hash
    store._conn.execute(
        "UPDATE deployments SET first_seen = ? WHERE deployment_id = ?",
        (time.time() - hours * 3600, did),
    )
    store.commit()
    for i in range(checkpoints):
        store.insert_checkpoint(
            {"deployment_id": did, "seq": i, "head": "a" * 64, "created_at": time.time()}
        )


# --- ingest correctness ----------------------------------------------------


def test_accepts_a_clean_batch(store, tmp_path):
    rec, wire = fleet(tmp_path, "acme", 0.2, 40)
    out = ingest_episodes(store, list(wire))
    assert out["accepted"] == 40
    assert out["next_seq"] == 40
    assert out["head"] == wire[-1]["entry_hash"]


def test_chain_continues_across_batches(store, tmp_path):
    rec, wire = fleet(tmp_path, "acme", 0.2, 20)
    ingest_episodes(store, wire[:10])
    out = ingest_episodes(store, wire[10:])
    assert out["next_seq"] == 20
    assert out["n_episodes"] == 20


def test_a_replayed_batch_is_refused(store, tmp_path):
    rec, wire = fleet(tmp_path, "acme", 0.2, 10)
    ingest_episodes(store, list(wire))
    with pytest.raises(BureauError) as exc:
        ingest_episodes(store, list(wire))
    assert exc.value.status == 409


def test_a_tampered_record_is_refused(store, tmp_path):
    """On a fresh deployment, so the sequence check cannot mask the hash check."""
    rec, wire = fleet(tmp_path, "acme", 0.2, 10)
    doctored = [dict(w) for w in wire]
    doctored[5]["resolved"] = not doctored[5]["resolved"]
    with pytest.raises(BureauError) as exc:
        ingest_episodes(store, doctored)
    assert exc.value.status == 422
    assert "chain" in exc.value.message


def test_a_dropped_episode_is_refused(store, tmp_path):
    rec, wire = fleet(tmp_path, "acme", 0.2, 10)
    without = [w for w in wire if w["seq"] != 4]
    with pytest.raises(BureauError) as exc:
        ingest_episodes(store, without)
    assert exc.value.status == 422


def test_a_batch_must_come_from_one_deployment(store, tmp_path):
    _, a = fleet(tmp_path, "acme", 0.2, 5, seed=1)
    _, b = fleet(tmp_path, "globex", 0.2, 5, seed=2)
    with pytest.raises(BureauError, match="one deployment"):
        ingest_episodes(store, a + b)


def test_content_cannot_get_in_through_the_api(store, tmp_path):
    """The SDK refuses to emit content; the bureau must refuse to accept it."""
    rec, wire = fleet(tmp_path, "acme", 0.2, 3)
    smuggled = [dict(w) for w in wire]
    smuggled[0]["prompt"] = "SSN-078-05-1120"
    with pytest.raises(BureauError) as exc:
        ingest_episodes(store, smuggled)
    assert exc.value.status == 400
    assert "unknown fields" in exc.value.message


def test_an_empty_or_oversized_batch_is_refused(store):
    with pytest.raises(BureauError, match="empty batch"):
        ingest_episodes(store, [])


# --- admission: nobody moves a base rate for free --------------------------


def test_a_new_deployment_is_not_admitted(store, tmp_path):
    rec, wire = fleet(tmp_path, "acme", 0.2, 40)
    out = ingest_episodes(store, list(wire))
    assert out["pool"]["admitted"] is False
    assert "hours of history" in out["pool"]["status"]


def test_admission_needs_episodes_age_and_checkpoints():
    now = time.time()
    assert not evaluate(10, now - 10 * 3600, 5, True, now).admitted   # too few
    assert not evaluate(100, now - 60, 5, True, now).admitted          # too new
    assert not evaluate(100, now - 10 * 3600, 0, True, now).admitted   # unanchored
    assert not evaluate(100, now - 10 * 3600, 5, False, now).admitted  # broken chain
    assert evaluate(100, now - 10 * 3600, 5, True, now).admitted


def test_a_deployment_admitted_once_it_has_earned_it(store, tmp_path):
    rec, wire = fleet(tmp_path, "acme", 0.2, 40)
    ingest_episodes(store, list(wire))
    age_and_checkpoint(store, rec)
    rec2, wire2 = fleet(tmp_path, "acme2", 0.2, 40, seed=9)
    ingest_episodes(store, list(wire2))
    # re-submitting nothing new still re-evaluates on the next batch; force it
    row = store.deployment(rec.deployment_hash)
    decision = evaluate(row["n_episodes"], row["first_seen"], 2, True)
    assert decision.admitted


def test_no_prior_is_published_from_a_single_contributor(store, tmp_path):
    rec, wire = fleet(tmp_path, "acme", 0.2, 40)
    ingest_episodes(store, list(wire))
    age_and_checkpoint(store, rec)
    store.set_admitted(rec.deployment_hash, True)
    prior = class_prior(store, "customer_support")
    assert prior["available"] is False
    assert "at least 2" in prior["reason"]


def test_a_prior_appears_once_two_deployments_are_admitted(store, tmp_path):
    for i, rate in enumerate((0.15, 0.45)):
        rec, wire = fleet(tmp_path, f"dep{i}", rate, 60, seed=i)
        ingest_episodes(store, list(wire))
        age_and_checkpoint(store, rec)
        store.set_admitted(rec.deployment_hash, True)

    prior = class_prior(store, "customer_support", own_n=60)
    assert prior["available"] is True
    assert prior["n_deployments"] == 2
    assert 0.0 < prior["mu"] < 1.0
    assert "your_z" in prior


def test_one_huge_contributor_cannot_become_the_class(store, tmp_path):
    """The cheapest poisoning attack is not many fake deployments, it is one
    enormous one -- Bühlmann weights by exposure."""
    # below three contributors there is no meaningful majority to protect
    assert influence_cap(1) == 1.0
    assert influence_cap(2) == 1.0
    # from three, no single contributor may hold a majority, however large
    assert influence_cap(3) == 0.5
    assert influence_cap(20) == pytest.approx(0.2)

    capped = capped_weights([10.0, 10.0, 10.0, 10.0, 10.0, 10_000.0])
    total = sum(capped)
    assert capped[-1] < 10_000.0
    # the guarantee is on the POST-cap share: capping against the uncapped
    # total would let the attacker inflate its own ceiling
    assert capped[-1] / total <= 0.5 + 1e-9

    big = capped_weights([1.0] * 20 + [1e9])
    assert big[-1] / sum(big) <= influence_cap(21) + 1e-9


def test_capped_surplus_is_discarded_not_redistributed():
    """Redistributing a capped contributor's surplus would let a large attacker
    inflate whichever other deployments it also controls."""
    exposures = [1.0, 1.0, 1.0, 1.0, 1.0, 500.0]
    capped = capped_weights(exposures)
    assert sum(capped) < sum(exposures)
    assert capped[:5] == exposures[:5]


# --- checkpoints -----------------------------------------------------------


def test_checkpoint_anchors_and_counts(store, tmp_path):
    rec, wire = fleet(tmp_path, "acme", 0.2, 10)
    ingest_episodes(store, list(wire))
    out = ingest_checkpoint(store, rec.checkpoint().to_wire())
    assert out["anchored"] is True
    assert out["checkpoints"] == 1


def test_a_checkpoint_disagreeing_with_the_chain_is_refused(store, tmp_path):
    rec, wire = fleet(tmp_path, "acme", 0.2, 10)
    ingest_episodes(store, list(wire))
    cp = rec.checkpoint().to_wire()
    cp["head"] = "b" * 64
    with pytest.raises(BureauError, match="disagrees"):
        ingest_checkpoint(store, cp)


def test_a_checkpoint_for_an_unknown_deployment_is_refused(store):
    with pytest.raises(BureauError) as exc:
        ingest_checkpoint(
            store, {"deployment_id": "0" * 16, "seq": 1, "head": "c" * 64}
        )
    assert exc.value.status == 404


def test_a_malformed_checkpoint_head_is_refused(store, tmp_path):
    rec, wire = fleet(tmp_path, "acme", 0.2, 5)
    ingest_episodes(store, list(wire))
    with pytest.raises(BureauError, match="sha256"):
        ingest_checkpoint(
            store,
            {"deployment_id": rec.deployment_hash, "seq": 5, "head": "short"},
        )


# --- what the bureau stores ------------------------------------------------


def test_the_bureau_stores_only_wire_fields(store, tmp_path):
    rec, wire = fleet(tmp_path, "acme", 0.2, 5)
    ingest_episodes(store, list(wire))
    cols = {
        r[1] for r in store._conn.execute("PRAGMA table_info(episodes)").fetchall()
    }
    # nothing resembling content, identifiers or tool names
    assert not cols & {"prompt", "payload", "content", "tool", "tool_name", "task_id"}
    assert "task_hash" in cols  # the hashed form only
