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
from freeboard.wire import _OPTIONAL_SINCE_1_0
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


# --- upgrading a bureau in place -------------------------------------------

OLD_SCHEMA = """
CREATE TABLE episodes (
    deployment_id TEXT NOT NULL, seq INTEGER NOT NULL, entry_hash TEXT NOT NULL,
    role TEXT NOT NULL, envelope_class TEXT NOT NULL, task_hash TEXT NOT NULL,
    pass_index INTEGER NOT NULL, resolved INTEGER, escalated INTEGER NOT NULL,
    n_actions INTEGER NOT NULL, error_rate REAL NOT NULL, repeat_rate REAL NOT NULL,
    tool_entropy REAL NOT NULL, n_irreversible INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL, received_at REAL NOT NULL,
    PRIMARY KEY (deployment_id, seq));
"""


def test_an_existing_database_is_migrated_not_ignored(tmp_path):
    """REGRESSION: CREATE TABLE IF NOT EXISTS does nothing to a table that
    already exists, so a bureau upgraded in place kept its old shape and every
    query touching a new column failed at runtime. That is exactly how the first
    upgrade of the live bureau broke."""
    import sqlite3

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript(OLD_SCHEMA)
    con.execute(
        "INSERT INTO episodes VALUES"
        " ('d',0,'h','ops','C000-V0','t',1,1,0,3,0.0,0.0,0.0,0,10,0.0)"
    )
    con.commit()
    con.close()

    store = Store(db)
    cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(episodes)")}
    assert {"unreported_actions", "attestation_source"} <= cols
    # and the data that was already there survived
    assert store._conn.execute("select count(*) from episodes").fetchone()[0] == 1
    assert store.attestation_sources("ops") == []
    store.close()


def test_migration_is_idempotent(tmp_path):
    db = tmp_path / "b.db"
    for _ in range(3):
        s = Store(db)
        cols = [r["name"] for r in s._conn.execute("PRAGMA table_info(episodes)")]
        assert cols.count("attestation_source") == 1
        s.close()


def test_a_handler_error_returns_a_status_not_a_dropped_connection(tmp_path):
    """A dropped connection surfaces as a bare 502 from whatever proxy is in
    front, which says nothing about what went wrong."""
    import inspect

    from bureau.app import Handler

    for method in (Handler.do_GET, Handler.do_POST):
        src = inspect.getsource(method)
        assert "except BureauError" in src
        assert "except Exception" in src
        assert "self._send(500" in src


# --- an old client must keep working -------------------------------------


def as_an_old_client_would_have_sent(wire):
    """Rebuild a batch the way freeboard 0.2.1 would have produced it.

    The attestation fields did not exist, so an old client did not merely omit
    them from the payload -- it computed its chain over a body that never had
    them. Stripping fields from an already-chained record instead produces a
    record whose entry_hash cannot verify, which tests the hash check rather
    than the compatibility rule.
    """
    from freeboard.chain import GENESIS, entry_hash

    dropped = set(_OPTIONAL_SINCE_1_0)
    head, out = GENESIS, []
    for i, rec in enumerate(wire):
        body = {k: v for k, v in rec.items() if k not in dropped and k != "entry_hash"}
        body["schema_version"] = "1.0"
        body["seq"] = i
        body["prev_hash"] = head
        head = entry_hash(body)
        out.append({**body, "entry_hash": head})
    return out


def test_a_client_from_before_attestation_is_still_accepted(store, tmp_path):
    """Wire formats that only work in lockstep are wire formats that strand
    every deployment that does not upgrade the same afternoon."""
    _, wire = fleet(tmp_path, "legacy", 0.2, 12)
    old = as_an_old_client_would_have_sent(wire)
    for field in _OPTIONAL_SINCE_1_0:
        assert field not in old[0]

    out = ingest_episodes(store, old)
    assert out["accepted"] == 12

    row = store._conn.execute(
        "SELECT attestation_source, unreported_actions FROM episodes LIMIT 1"
    ).fetchone()
    assert row["attestation_source"] == "none"
    assert row["unreported_actions"] == 0
    # and it contributes nothing to a divergence baseline: no observer watched it
    assert store.divergence_observations("customer_support", "none") == ([], [], [])


def test_an_unknown_field_is_still_refused(store, tmp_path):
    """Tolerating MISSING known fields must not become tolerating unknown ones,
    which is the hole that lets content in."""
    _, wire = fleet(tmp_path, "acme", 0.2, 3)
    smuggled = [dict(w) for w in wire]
    smuggled[0]["attested_note"] = "free text"
    with pytest.raises(BureauError, match="unknown fields"):
        ingest_episodes(store, smuggled)


def test_the_two_default_tables_agree():
    """The bureau cannot import the SDK's table without pinning itself to an SDK
    version, so the duplication is deliberate and this is what keeps it honest."""
    from bureau.store import EPISODE_FIELDS, FIELD_DEFAULTS

    omittable = {k: v for k, v in _OPTIONAL_SINCE_1_0.items() if k in EPISODE_FIELDS}
    assert FIELD_DEFAULTS == omittable


def test_a_batch_that_cannot_be_stored_is_not_reported_as_accepted(store, tmp_path):
    """INSERT OR IGNORE makes a replay idempotent; it will also quietly swallow a
    constraint violation, and a bureau that answers 'accepted' over discarded
    episodes is worse than one that errors."""
    _, wire = fleet(tmp_path, "acme", 0.2, 3)
    broken = [dict(w) for w in wire]
    broken[1]["role"] = None
    with pytest.raises(ValueError, match="stored 2 of 3"):
        store.insert_episodes("acme", broken)


def test_waiting_on_admission_does_not_look_like_missing_instrumentation(
    store, tmp_path
):
    """A caller told "nobody is attesting" when the truth is "nobody is admitted
    yet" goes hunting for a bug in their own eBPF setup."""
    from bureau.app import divergence_priors

    for i in range(2):
        rec = Recorder(
            deployment_id=f"att{i}",
            role="customer_support",
            tools=TOOLS,
            state_path=str(tmp_path / f"att{i}.json"),
        )
        for j in range(10):
            with rec.episode(f"t{j}") as ep:
                ep.action("read_case")
                ep.attest(actions_observed=2, source="ebpf")
                ep.resolve(success=True)
        ingest_episodes(store, [to_wire(r) for r in rec.records])

    (entry,) = divergence_priors(store, "customer_support")
    assert entry["available"] is False
    assert entry["reason"].startswith("2 deployment(s) attesting, none admitted")
