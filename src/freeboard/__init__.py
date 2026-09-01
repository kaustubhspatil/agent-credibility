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

__version__ = "0.2.0"

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
from .client import BureauClient, BureauError, BureauSink
from .episode import Episode, EpisodeRecord, Recorder
from .estimate import (
    Components,
    bootstrap_k,
    components,
    credibility_estimate,
    losses_from_records,
)
from .sinks import JsonlSink, read_checkpoint, read_jsonl, write_checkpoint
from .state import RecorderState, StateError
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
    "BureauSink",
    "BureauError",
    "BureauClient",
    "write_checkpoint",
    "read_jsonl",
    "read_checkpoint",
    "losses_from_records",
    "credibility_estimate",
    "components",
    "bootstrap_k",
    "JsonlSink",
    "Components",
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
    "RecorderState",
    "StateError",
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
