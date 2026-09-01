"""Tests for the recorder, led by the one a privacy review would ask for.

The leak test is the point of the whole package: canary strings are pushed into
every surface that accepts a string, and the assertion is that none of them can
be found anywhere in the outbound payload -- not in a value, not in a key, not
in the JSON encoding.
"""

import json

import pytest

from credibility.recorder import (
    Capability,
    Recorder,
    Reversibility,
    ToolSpec,
    WireViolation,
    compare,
    derive_envelope,
    manifest_hash,
    to_json,
    to_otel_attributes,
    to_wire,
    validate,
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

    from credibility.recorder.episode import Episode

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
