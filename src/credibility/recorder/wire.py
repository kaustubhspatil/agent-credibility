"""The wire format, and the validator that makes it checkable.

Everything above this module is a promise about behaviour. This module turns
the promise into something a reviewer can verify: a strict allowlist over the
outbound payload, where every field must be a number, a bool, a null, a value
from a fixed enum, or a hex digest. Any string that is none of those is a
violation and the payload is refused.

That inverts the usual arrangement. Redaction pipelines are deny-lists -- they
strip the sensitive things someone thought of. This is an allow-list: content
does not leak because there is no shape it could take that would pass.

`validate` is intended to be run in the customer's process, immediately before
transmission, and to be pointed at during a privacy review.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from .envelope import Capability, Reversibility
from .episode import EpisodeRecord

SCHEMA_VERSION = "1.0"

_HEX16 = re.compile(r"\A[0-9a-f]{16}\Z")
_SLUG = re.compile(r"\A[a-z0-9][a-z0-9_.\-]{0,63}\Z")
_CLASS_CODE = re.compile(r"\AC[01]{3}-[VCI][0-4]\Z")
_VERSION = re.compile(r"\A[0-9]+\.[0-9]+(\.[0-9]+)?\Z")

_REVERSIBILITY = {r.value for r in Reversibility}
_CAPABILITIES = {c.value for c in Capability}


class WireViolation(ValueError):
    """A field carried something the wire format does not permit."""


def _check_str(field: str, value: Any, rule: Any) -> None:
    if not isinstance(value, str):
        raise WireViolation(f"{field}: expected string, got {type(value).__name__}")
    if isinstance(rule, set):
        if value not in rule:
            raise WireViolation(f"{field}: {value!r} not in permitted values")
    elif not rule.match(value):
        raise WireViolation(f"{field}: {value!r} does not match {rule.pattern}")


# field -> ("num" | "bool" | "int" | rule), and whether None is allowed
_RULES: dict[str, tuple[Any, bool]] = {
    "episode_id": (_HEX16, False),
    "deployment_id": (_HEX16, False),
    "role": (_SLUG, False),
    "task_hash": (_HEX16, False),
    "pass_index": ("int", False),
    "envelope_class": (_CLASS_CODE, False),
    "manifest_hash": (_HEX16, False),
    "resolved": ("bool", True),
    "reward": ("num", True),
    "escalated": ("bool", False),
    "n_actions": ("int", False),
    "n_distinct_tools": ("int", False),
    "tool_entropy": ("num", False),
    "repeat_rate": ("num", False),
    "error_rate": ("num", False),
    "n_mutating": ("int", False),
    "n_irreversible": ("int", False),
    "max_reversibility": (_REVERSIBILITY, False),
    "output_chars": ("int", False),
    "duration_ms": ("int", False),
    "observed_capabilities": ("caps", False),
    "schema_version": (_VERSION, False),
    "recorder_version": (_VERSION, False),
}


def to_wire(record: EpisodeRecord) -> dict[str, Any]:
    """Serialise an episode to the outbound payload, then validate it.

    Validation is not optional and not a separate step a caller can forget:
    nothing leaves this function without having passed.
    """
    payload = asdict(record)
    payload["observed_capabilities"] = list(record.observed_capabilities)
    validate(payload)
    return payload


def validate(payload: dict[str, Any]) -> None:
    """Refuse anything the wire format does not explicitly permit."""
    unknown = set(payload) - set(_RULES)
    if unknown:
        raise WireViolation(f"unknown fields: {sorted(unknown)}")
    missing = set(_RULES) - set(payload)
    if missing:
        raise WireViolation(f"missing fields: {sorted(missing)}")

    for field, (rule, nullable) in _RULES.items():
        value = payload[field]
        if value is None:
            if not nullable:
                raise WireViolation(f"{field}: null not permitted")
            continue

        if rule == "bool":
            if not isinstance(value, bool):
                raise WireViolation(f"{field}: expected bool")
        elif rule == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise WireViolation(f"{field}: expected int")
            if value < 0:
                raise WireViolation(f"{field}: expected non-negative")
        elif rule == "num":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise WireViolation(f"{field}: expected number")
        elif rule == "caps":
            if not isinstance(value, (list, tuple)):
                raise WireViolation(f"{field}: expected list")
            for item in value:
                _check_str(field, item, _CAPABILITIES)
        else:
            _check_str(field, value, rule)


def to_json(record: EpisodeRecord) -> str:
    return json.dumps(to_wire(record), separators=(",", ":"), sort_keys=True)


# --- OpenTelemetry ---------------------------------------------------------
# Attribute names follow the OTel gen_ai conventions where one exists, and a
# reserved `agent.credibility.*` namespace where none does. Emitting attributes
# rather than depending on the OTel SDK keeps this package dependency-free;
# an integrator hands these to whatever tracer they already run.

_OTEL_MAP = {
    "role": "gen_ai.agent.name",
    "episode_id": "gen_ai.conversation.id",
    "n_actions": "agent.credibility.actions",
    "n_distinct_tools": "agent.credibility.distinct_tools",
    "tool_entropy": "agent.credibility.tool_entropy",
    "repeat_rate": "agent.credibility.repeat_rate",
    "error_rate": "agent.credibility.error_rate",
    "n_mutating": "agent.credibility.mutating_actions",
    "n_irreversible": "agent.credibility.irreversible_actions",
    "max_reversibility": "agent.credibility.max_reversibility",
    "output_chars": "agent.credibility.output_chars",
    "duration_ms": "agent.credibility.duration_ms",
    "resolved": "agent.credibility.resolved",
    "reward": "agent.credibility.reward",
    "escalated": "agent.credibility.escalated",
    "envelope_class": "agent.credibility.envelope_class",
    "manifest_hash": "agent.credibility.manifest_hash",
    "deployment_id": "agent.credibility.deployment",
    "task_hash": "agent.credibility.task",
    "pass_index": "agent.credibility.pass",
    "schema_version": "agent.credibility.schema",
}

SPAN_NAME = "agent.episode"


def to_otel_attributes(record: EpisodeRecord) -> dict[str, Any]:
    """OTel span attributes for one episode. Validated on the way out."""
    payload = to_wire(record)
    attrs = {
        otel: payload[field]
        for field, otel in _OTEL_MAP.items()
        if payload.get(field) is not None
    }
    if payload["observed_capabilities"]:
        attrs["agent.credibility.observed_capabilities"] = list(
            payload["observed_capabilities"]
        )
    return attrs
