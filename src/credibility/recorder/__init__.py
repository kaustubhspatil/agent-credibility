"""Recorder: turn an agent loop into sufficient statistics, and nothing else.

    from credibility.recorder import Recorder, ToolSpec, Capability, Reversibility

    rec = Recorder(
        deployment_id="acme-prod",
        role="customer_support",
        tools=[
            ToolSpec("search_orders",
                     capabilities=frozenset({Capability.PRIVATE_DATA}),
                     reversibility=Reversibility.REVERSIBLE),
            ToolSpec("issue_refund",
                     capabilities=frozenset({Capability.EXTERNAL_EFFECT}),
                     reversibility=Reversibility.IRREVERSIBLE),
        ],
    )

    with rec.episode(task_id="ticket-8821") as ep:
        ep.action("search_orders", output_chars=412)
        ep.action("issue_refund")
        ep.resolve(success=True)

    payload = to_wire(rec.records[-1])   # validated; refuses to carry content
"""

from .chain import (
    GENESIS,
    ChainState,
    Checkpoint,
    VerificationResult,
    canonical,
    checkpoint,
    verify,
)
from .envelope import (
    Capability,
    Divergence,
    Reversibility,
    TaskEnvelope,
    ToolSpec,
    compare,
    derive_envelope,
    manifest_hash,
)
from .episode import Episode, EpisodeRecord, Recorder
from .wire import (
    SCHEMA_VERSION,
    SPAN_NAME,
    WireViolation,
    to_json,
    to_otel_attributes,
    to_wire,
    validate,
)

__all__ = [
    "ChainState",
    "Checkpoint",
    "GENESIS",
    "VerificationResult",
    "canonical",
    "checkpoint",
    "verify",
    "Capability",
    "Divergence",
    "Episode",
    "EpisodeRecord",
    "Recorder",
    "Reversibility",
    "SCHEMA_VERSION",
    "SPAN_NAME",
    "TaskEnvelope",
    "ToolSpec",
    "WireViolation",
    "compare",
    "derive_envelope",
    "manifest_hash",
    "to_json",
    "to_otel_attributes",
    "to_wire",
    "validate",
]
