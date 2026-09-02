"""Tests for the recorder, led by the one a privacy review would ask for.

The leak test is the point of the whole package: canary strings are pushed into
every surface that accepts a string, and the assertion is that none of them can
be found anywhere in the outbound payload -- not in a value, not in a key, not
in the JSON encoding.
"""

import json

import pytest

from freeboard import (
    GENESIS,
    Capability,
    Recorder,
    Reversibility,
    ToolSpec,
    WireViolation,
    canonical,
    compare,
    derive_envelope,
    manifest_hash,
    to_json,
    to_otel_attributes,
    to_wire,
    validate,
    verify,
)

CANARY = "SSN-078-05-1120-CANARY"


def _tools():
    return [
        ToolSpec(
            "read_case",
            capabilities=frozenset({Capability.PRIVATE_DATA}),
            reversibility=Reversibility.REVERSIBLE,
        ),
        ToolSpec(
            "send_email",
            capabilities=frozenset({Capability.EXTERNAL_EFFECT}),
            reversibility=Reversibility.IRREVERSIBLE,
        ),
    ]


# --- the leak test ---------------------------------------------------------


def test_canary_never_reaches_the_wire():
    """Content pushed into every accepting surface must not come out."""
    rec = Recorder(
        deployment_id=f"tenant-{CANARY}",
        role="customer_support",
        tools=[
            ToolSpec(
                f"tool_{CANARY}",
                description=f"handles {CANARY} records",
                capabilities=frozenset({Capability.PRIVATE_DATA}),
                reversibility=Reversibility.REVERSIBLE,
            )
        ],
    )
    with rec.episode(task_id=f"ticket-{CANARY}") as ep:
        ep.action(f"tool_{CANARY}", output_chars=len(CANARY))
        ep.resolve(success=False)

    record = rec.records[-1]
    blob = to_json(record)
    assert CANARY not in blob
    assert "SSN" not in blob
    # and not via the OTel path either
    assert CANARY not in json.dumps(to_otel_attributes(record))
    # nor hiding in a key
    assert not any(CANARY in k for k in to_wire(record))


def test_output_chars_takes_a_length_not_the_output():
    """The API has no parameter that accepts tool output."""
    import inspect

    from freeboard.episode import Episode

    sig = inspect.signature(Episode.action)
    assert set(sig.parameters) == {"self", "tool", "error", "output_chars"}
    assert sig.parameters["output_chars"].annotation == "int"


def test_free_text_in_role_is_refused_not_silently_passed():
    rec = Recorder(deployment_id="d", role=f"notes: {CANARY}", tools=_tools())
    with rec.episode("t") as ep:
        ep.resolve(success=True)
    with pytest.raises(WireViolation):
        to_wire(rec.records[-1])


def test_validator_rejects_smuggled_fields_and_values():
    rec = Recorder(deployment_id="d", role="ops", tools=_tools())
    with rec.episode("t") as ep:
        ep.resolve(success=True)
    good = to_wire(rec.records[-1])

    smuggled = dict(good, note=CANARY)
    with pytest.raises(WireViolation, match="unknown fields"):
        validate(smuggled)

    swapped = dict(good, task_hash=CANARY)
    with pytest.raises(WireViolation):
        validate(swapped)

    truncated = {k: v for k, v in good.items() if k != "n_actions"}
    with pytest.raises(WireViolation, match="missing fields"):
        validate(truncated)

    with pytest.raises(WireViolation):
        validate(dict(good, n_actions=-1))
    with pytest.raises(WireViolation):
        validate(dict(good, max_reversibility="mostly_fine"))
    with pytest.raises(WireViolation):
        validate(dict(good, observed_capabilities=[CANARY]))


def test_ids_are_salted_so_they_are_not_guessable_across_deployments():
    a = Recorder(deployment_id="acme", role="ops", tools=_tools())
    b = Recorder(deployment_id="acme", role="ops", tools=_tools())
    for rec in (a, b):
        with rec.episode("ticket-1") as ep:
            ep.resolve(success=True)
    # same task id, different deployments -> different hashes
    assert a.records[0].task_hash != b.records[0].task_hash
    # but stable within a deployment, which is what credibility needs
    with a.episode("ticket-1") as ep:
        ep.resolve(success=True)
    assert a.records[0].task_hash == a.records[1].task_hash


# --- the derived envelope --------------------------------------------------


def test_rule_of_two_fires_only_on_all_three_capabilities():
    two = derive_envelope(_tools())
    assert not two.rule_of_two_violated

    three = derive_envelope(
        _tools()
        + [
            ToolSpec(
                "fetch_web",
                capabilities=frozenset({Capability.UNTRUSTED_INPUT}),
                reversibility=Reversibility.REVERSIBLE,
            )
        ]
    )
    assert three.rule_of_two_violated
    assert three.class_code.startswith("C111")


def test_class_code_ignores_names_but_not_powers():
    """Two customers with the same powers are the same risk class."""
    a = derive_envelope(_tools())
    b = derive_envelope(
        [
            ToolSpec(
                "lookup_matter",
                capabilities=frozenset({Capability.PRIVATE_DATA}),
                reversibility=Reversibility.REVERSIBLE,
            ),
            ToolSpec(
                "dispatch_notice",
                capabilities=frozenset({Capability.EXTERNAL_EFFECT}),
                reversibility=Reversibility.IRREVERSIBLE,
            ),
        ]
    )
    assert a.class_code == b.class_code
    assert a.manifest_hash != b.manifest_hash  # names still distinguish them


def test_reversible_and_recoverable_do_not_collide_in_the_class_code():
    """Both start with 'r'; an undoable write must not price as a read."""
    read_only = derive_envelope(
        [ToolSpec("r", capabilities=frozenset(), reversibility=Reversibility.REVERSIBLE)]
    )
    recoverable = derive_envelope(
        [ToolSpec("w", capabilities=frozenset(), reversibility=Reversibility.RECOVERABLE)]
    )
    assert read_only.class_code != recoverable.class_code


def test_manifest_hash_tracks_powers_not_prose():
    base = ToolSpec(
        "act",
        description="does a thing",
        capabilities=frozenset({Capability.PRIVATE_DATA}),
        reversibility=Reversibility.REVERSIBLE,
    )
    reworded = ToolSpec(
        "act",
        description="performs an operation, rewritten for clarity",
        capabilities=frozenset({Capability.PRIVATE_DATA}),
        reversibility=Reversibility.REVERSIBLE,
    )
    escalated = ToolSpec(
        "act",
        description="does a thing",
        capabilities=frozenset({Capability.PRIVATE_DATA, Capability.EXTERNAL_EFFECT}),
        reversibility=Reversibility.REVERSIBLE,
    )
    assert manifest_hash([base]) == manifest_hash([reworded])
    assert manifest_hash([base]) != manifest_hash([escalated])


def test_manifest_hash_is_order_insensitive():
    tools = _tools()
    assert manifest_hash(tools) == manifest_hash(list(reversed(tools)))


def test_keyword_priors_are_marked_inferred():
    env = derive_envelope([ToolSpec("send_invoice_email")])
    assert env.inferred_tools == 1
    assert Capability.EXTERNAL_EFFECT in env.capabilities
    assert env.max_reversibility is Reversibility.IRREVERSIBLE

    explicit = derive_envelope(
        [
            ToolSpec(
                "send_invoice_email",
                capabilities=frozenset({Capability.EXTERNAL_EFFECT}),
                reversibility=Reversibility.IRREVERSIBLE,
            )
        ]
    )
    assert explicit.inferred_tools == 0


# --- divergence ------------------------------------------------------------


def test_understating_scope_scores_higher_than_overstating():
    env = derive_envelope(_tools())  # grants PRIVATE_DATA + EXTERNAL_EFFECT

    understated = compare(env, declared={Capability.PRIVATE_DATA})
    overstated = compare(
        env,
        declared={
            Capability.PRIVATE_DATA,
            Capability.EXTERNAL_EFFECT,
            Capability.UNTRUSTED_INPUT,
        },
    )
    assert understated.understated
    assert not overstated.understated
    assert understated.score > overstated.score


def test_granted_but_unused_is_reported_not_resolved():
    rec = Recorder(
        deployment_id="d",
        role="ops",
        tools=_tools(),
        declared_capabilities={Capability.PRIVATE_DATA, Capability.EXTERNAL_EFFECT},
    )
    with rec.episode("t") as ep:
        ep.action("read_case")
        ep.resolve(success=True)

    div = rec.divergence()
    assert Capability.EXTERNAL_EFFECT in div.granted_but_unused
    assert not div.understated
    assert div.unused_tool_share == pytest.approx(0.5)


# --- episode statistics ----------------------------------------------------


def test_moments_are_what_the_estimator_expects():
    rec = Recorder(deployment_id="d", role="ops", tools=_tools())
    with rec.episode("t") as ep:
        ep.action("read_case")
        ep.action("read_case", error=True)   # consecutive repeat
        ep.action("send_email", output_chars=100)
        ep.resolve(success=False)

    r = rec.records[-1]
    assert r.n_actions == 3
    assert r.n_distinct_tools == 2
    assert r.repeat_rate == pytest.approx(0.5)      # 1 repeat over 2 gaps
    assert r.error_rate == pytest.approx(1 / 3)
    assert r.n_irreversible == 1
    assert r.max_reversibility == "irreversible"
    assert r.output_chars == 100
    assert r.resolved is False
    assert set(r.observed_capabilities) == {"private_data", "external_effect"}


def test_entropy_is_zero_for_one_tool_and_maximal_when_uniform():
    rec = Recorder(deployment_id="d", role="ops", tools=_tools())
    with rec.episode("a") as ep:
        ep.action("read_case")
        ep.action("read_case")
        ep.resolve(success=True)
    assert rec.records[-1].tool_entropy == pytest.approx(0.0)

    with rec.episode("b") as ep:
        ep.action("read_case")
        ep.action("send_email")
        ep.resolve(success=True)
    assert rec.records[-1].tool_entropy == pytest.approx(1.0)


def test_a_raising_episode_is_recorded_as_a_failure_not_dropped():
    """Silently losing failed episodes would bias the base rate downward."""
    rec = Recorder(deployment_id="d", role="ops", tools=_tools())
    with pytest.raises(RuntimeError):
        with rec.episode("t") as ep:
            ep.action("read_case")
            raise RuntimeError("agent crashed")

    assert len(rec.records) == 1
    assert rec.records[0].resolved is False


def test_episode_cannot_be_finished_twice_or_used_after():
    rec = Recorder(deployment_id="d", role="ops", tools=_tools())
    ep = rec.episode("t")
    ep.resolve(success=True)
    ep.finish()
    with pytest.raises(RuntimeError):
        ep.finish()
    with pytest.raises(RuntimeError):
        ep.action("read_case")


def test_negative_output_chars_is_rejected_at_the_call_site():
    rec = Recorder(deployment_id="d", role="ops", tools=_tools())
    with rec.episode("t") as ep:
        with pytest.raises(ValueError):
            ep.action("read_case", output_chars=-1)
        ep.resolve(success=True)


def test_sink_receives_every_record():
    seen = []

    class Sink:
        def write(self, record):
            seen.append(record)

    rec = Recorder(deployment_id="d", role="ops", tools=_tools(), sink=Sink())
    for i in range(3):
        with rec.episode(f"t{i}") as ep:
            ep.resolve(success=True)
    assert len(seen) == 3


def test_otel_attributes_are_flat_and_validated():
    rec = Recorder(deployment_id="d", role="ops", tools=_tools())
    with rec.episode("t") as ep:
        ep.action("read_case")
        ep.resolve(success=True)

    attrs = to_otel_attributes(rec.records[-1])
    assert attrs["gen_ai.agent.name"] == "ops"
    assert attrs["agent.credibility.actions"] == 1
    for key, value in attrs.items():
        assert key.startswith(("gen_ai.", "agent.credibility."))
        assert isinstance(value, (str, int, float, bool, list))


@pytest.mark.parametrize(
    "name,expect_caps,expect_rev",
    [
        # separator-joined names are the normal case and must not silently
        # fall through to "no capabilities at all"
        ("read_file", {Capability.PRIVATE_DATA}, Reversibility.REVERSIBLE),
        ("http_get", {Capability.UNTRUSTED_INPUT}, Reversibility.REVERSIBLE),
        ("query_customer_db", {Capability.PRIVATE_DATA}, Reversibility.REVERSIBLE),
        ("fetch-web-page", {Capability.UNTRUSTED_INPUT}, Reversibility.REVERSIBLE),
        ("send_invoice_email", {Capability.EXTERNAL_EFFECT}, Reversibility.IRREVERSIBLE),
        ("git_commit", {Capability.EXTERNAL_EFFECT}, Reversibility.RECOVERABLE),
        (
            "delete_record",
            {Capability.PRIVATE_DATA, Capability.EXTERNAL_EFFECT},
            Reversibility.IRREVERSIBLE,
        ),
        (
            "run_sql",
            {Capability.PRIVATE_DATA, Capability.EXTERNAL_EFFECT},
            Reversibility.IRREVERSIBLE,
        ),
    ],
)
def test_priors_survive_separator_tokenisation(name, expect_caps, expect_rev):
    """A false negative here understates an agent's powers, which is the
    dangerous direction: a database read tool classified as touching no
    private data would price as a read-only agent."""
    env = derive_envelope([ToolSpec(name)])
    assert expect_caps <= env.capabilities, f"{name}: got {env.capabilities}"
    assert env.max_reversibility is expect_rev


def test_no_prior_matches_is_visible_rather_than_silent():
    """An unrecognised tool must not look like a safe one."""
    env = derive_envelope([ToolSpec("zzz_opaque_widget")])
    assert env.capabilities == frozenset()
    assert env.inferred_tools == 1  # flagged as guessed, so it can be loaded


# --- tamper-evident sequencing --------------------------------------------


def _chained(n=5):
    rec = Recorder(deployment_id="acme", role="ops", tools=_tools())
    for i in range(n):
        with rec.episode(f"t{i}") as ep:
            ep.action("read_case")
            ep.resolve(success=(i % 2 == 0))
    return rec, [to_wire(r) for r in rec.records]


def test_intact_chain_verifies_against_its_checkpoint():
    rec, wire = _chained()
    cp = rec.checkpoint()
    assert verify(wire, expected_head=cp.head)
    assert verify(wire, expected_head=cp.head).checked == 5
    assert cp.seq == 5


def test_dropping_a_failed_episode_is_detected():
    """The Article 12 failure mode: selective deletion before an audit."""
    rec, wire = _chained()
    cp = rec.checkpoint()
    survivors = [w for w in wire if w["resolved"] is not False]
    assert len(survivors) < len(wire)

    result = verify(survivors, expected_head=cp.head)
    assert not result
    assert "sequence gap" in result.reason


def test_editing_a_record_is_detected():
    rec, wire = _chained()
    cp = rec.checkpoint()
    wire[2] = dict(wire[2], resolved=True, n_actions=99)
    result = verify(wire, expected_head=cp.head)
    assert not result
    assert result.first_bad_seq == 2
    assert "entry_hash" in result.reason


def test_reordering_is_detected():
    rec, wire = _chained()
    cp = rec.checkpoint()
    swapped = [wire[0], wire[2], wire[1], wire[3], wire[4]]
    assert not verify(swapped, expected_head=cp.head)


def test_truncating_the_end_verifies_alone_but_fails_the_checkpoint():
    """Trimming the tail leaves a self-consistent chain -- the checkpoint is
    what catches it, which is exactly why the head has to be anchored."""
    rec, wire = _chained()
    cp = rec.checkpoint()
    trimmed = wire[:3]
    assert verify(trimmed)                      # self-consistent
    assert not verify(trimmed, expected_head=cp.head)   # but short of the head


def test_a_rewritten_chain_self_verifies_and_only_the_checkpoint_catches_it():
    """The honest limitation, pinned as a test.

    A vendor who drops an episode and recomputes every hash after it produces
    a chain that verifies perfectly on its own. Chaining is tamper-*evident*
    only once the head has been committed somewhere they cannot rewrite.
    """
    rec, wire = _chained()
    real_head = rec.checkpoint().head

    honest = Recorder(deployment_id="acme", role="ops", tools=_tools())
    for i in range(5):
        if i == 1:
            continue  # the episode being disappeared
        with honest.episode(f"t{i}") as ep:
            ep.action("read_case")
            ep.resolve(success=(i % 2 == 0))
    rewritten = [to_wire(r) for r in honest.records]

    assert verify(rewritten)                       # indistinguishable alone
    assert not verify(rewritten, expected_head=real_head)


def test_chain_fields_are_hex64_and_validated():
    rec, wire = _chained(2)
    assert wire[0]["prev_hash"] == GENESIS
    assert wire[1]["prev_hash"] == wire[0]["entry_hash"]
    for w in wire:
        assert len(w["entry_hash"]) == 64
    with pytest.raises(WireViolation):
        validate(dict(wire[0], entry_hash="deadbeef"))


def test_canonical_encoding_is_stable_across_key_order():
    rec, wire = _chained(1)
    shuffled = dict(reversed(list(wire[0].items())))
    assert canonical(shuffled) == canonical(wire[0])


def test_chaining_does_not_reintroduce_a_leak_path():
    rec = Recorder(
        deployment_id=f"tenant-{CANARY}",
        role="ops",
        tools=[ToolSpec("read_case", capabilities=frozenset(),
                        reversibility=Reversibility.REVERSIBLE)],
    )
    with rec.episode(task_id=f"ticket-{CANARY}") as ep:
        ep.action("read_case")
        ep.resolve(success=True)
    blob = to_json(rec.records[-1]) + json.dumps(rec.checkpoint().to_wire())
    assert CANARY not in blob


# --- out-of-band attestation -----------------------------------------------


def test_attestation_records_the_gap_the_environment_saw():
    """The July 2026 Hugging Face incident: agents paused the tool-call handler
    and substituted an impersonator process, so the recorder was told a story.
    An in-process recorder cannot detect that. An independent observer can, and
    the gap is the signal."""
    rec = Recorder(deployment_id="d", role="ops", tools=_tools())
    with rec.episode("t") as ep:
        ep.action("read_case")
        ep.action("read_case")
        ep.attest(actions_observed=5, source="ebpf")   # kernel saw five
        ep.resolve(success=True)

    w = to_wire(rec.records[-1])
    assert w["n_actions"] == 2
    assert w["attested_actions"] == 5
    assert w["unreported_actions"] == 3
    assert w["attestation_source"] == "ebpf"


def test_only_the_positive_gap_counts():
    """Fewer observed than reported is an instrumentation gap. MORE observed
    than reported is the signature of a recorder being lied to."""
    rec = Recorder(deployment_id="d", role="ops", tools=_tools())
    with rec.episode("a") as ep:
        for _ in range(6):
            ep.action("read_case")
        ep.attest(actions_observed=2, source="proxy")
        ep.resolve(success=True)
    assert rec.records[-1].unreported_actions == 0


def test_no_attestation_is_the_honest_default():
    rec = Recorder(deployment_id="d", role="ops", tools=_tools())
    with rec.episode("t") as ep:
        ep.action("read_case")
        ep.resolve(success=True)
    w = to_wire(rec.records[-1])
    assert w["attested_actions"] is None
    assert w["attestation_source"] == "none"
    assert w["unreported_actions"] == 0


def test_attestation_is_covered_by_the_chain():
    rec = Recorder(deployment_id="d", role="ops", tools=_tools())
    with rec.episode("t") as ep:
        ep.action("read_case")
        ep.attest(actions_observed=9, source="audit_log")
        ep.resolve(success=True)
    wire = [to_wire(r) for r in rec.records]
    assert verify(wire, expected_head=rec.checkpoint().head)
    # editing the attestation away breaks the chain like any other field
    doctored = [dict(wire[0], unreported_actions=0, attested_actions=1)]
    assert not verify(doctored)


def test_a_bogus_attestation_source_is_refused():
    rec = Recorder(deployment_id="d", role="ops", tools=_tools())
    with rec.episode("t") as ep:
        with pytest.raises(ValueError):
            ep.attest(3, source="an SSN could go here")
        with pytest.raises(ValueError):
            ep.attest(-1, source="ebpf")
        ep.resolve(success=True)


def test_a_record_from_an_older_client_is_still_accepted():
    """Bureau and clients do not upgrade in lockstep. A record predating the
    attestation fields must still validate, or every bureau upgrade strands
    every client that has not upgraded yet."""
    rec = Recorder(deployment_id="d", role="ops", tools=_tools())
    with rec.episode("t") as ep:
        ep.action("read_case")
        ep.resolve(success=True)
    old = {
        k: v for k, v in to_wire(rec.records[-1]).items()
        if k not in {"attested_actions", "unreported_actions", "attestation_source"}
    }
    validate(old)          # must not raise


def test_unknown_fields_are_still_refused():
    """The tolerance is only for MISSING known fields. Accepting unknown ones
    would give up the allow-list, which is the entire privacy guarantee."""
    rec = Recorder(deployment_id="d", role="ops", tools=_tools())
    with rec.episode("t") as ep:
        ep.resolve(success=True)
    with pytest.raises(WireViolation, match="unknown fields"):
        validate(dict(to_wire(rec.records[-1]), prompt="SSN-078-05-1120"))


def test_schema_version_moved_with_the_fields():
    from freeboard.wire import SCHEMA_VERSION

    assert SCHEMA_VERSION == "1.1"
