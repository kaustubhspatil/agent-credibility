"""The task envelope, derived from what an agent *can* do -- not what it claims.

A declared envelope is an incentive to lie. Premiums fall as scope narrows, so
a customer writing their own label will write the narrowest one they can
defend, and the measured cost of a wrong class label is 2.6-2.9x in day-zero
pricing error. So the envelope is derived from the tool manifest the agent is
actually wired to, and the declared label is kept only as a cross-check.

Three signals come out of this module:

    derived      what the granted tools make possible
    observed     what the agent actually did, once episodes have run
    declared     what the customer said (optional, never trusted)

Divergence between any two of them is itself a risk signal, and is reported
rather than resolved -- an agent granted a database write tool it never uses is
a different risk from one that uses it constantly, and both differ from one
that was declared read-only.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum


class Capability(str, Enum):
    """The three capabilities whose combination the 'rule of two' bounds.

    From the AI insurance blueprint: an agent should hold at most two of these
    three at once, because holding all three completes an attack chain --
    untrusted input can reach private data and then leave the building.
    """

    UNTRUSTED_INPUT = "untrusted_input"
    PRIVATE_DATA = "private_data"
    EXTERNAL_EFFECT = "external_effect"


class Reversibility(str, Enum):
    """How bad the worst outcome of a tool call is to undo."""

    REVERSIBLE = "reversible"      # read-only, or trivially undone
    RECOVERABLE = "recoverable"    # undoable with effort: a write with history
    IRREVERSIBLE = "irreversible"  # sent email, payment, hard delete


_ORDER = {
    Reversibility.REVERSIBLE: 0,
    Reversibility.RECOVERABLE: 1,
    Reversibility.IRREVERSIBLE: 2,
}

# Keyword priors. These are a *fallback* for tools that declare nothing, and
# they are deliberately treated as low-confidence: a name-based guess is not
# evidence, it is a prompt to go and ask. Anything inferred this way is
# reported with `inferred=True` so a downstream premium can be loaded for it.
# NOTE: matched against a haystack already normalised to space-separated
# tokens, so multi-word entries are written with spaces, never underscores.
_PRIORS: list[tuple[re.Pattern[str], set[Capability], Reversibility]] = [
    (re.compile(r"\b(fetch|browse|crawl|search|scrape|rss|inbox|receive|"
                r"webhook|listen|read url|http get|get url|open page)\b", re.I),
     {Capability.UNTRUSTED_INPUT}, Reversibility.REVERSIBLE),
    (re.compile(r"\b(query|select|lookup|retrieve|secret|credential|vault|"
                r"customer|patient|payroll|read file|get record|read record|"
                r"get file|load file|read db|run sql|execute sql)\b", re.I),
     {Capability.PRIVATE_DATA}, Reversibility.REVERSIBLE),
    (re.compile(r"\b(send|email|post|publish|tweet|slack|notify|sms|dispatch)\b",
                re.I),
     {Capability.EXTERNAL_EFFECT}, Reversibility.IRREVERSIBLE),
    (re.compile(r"\b(write|update|insert|upsert|delete|drop|truncate|execute|"
                r"migrate|deploy|refund|charge|transfer|pay|run sql|"
                r"execute sql)\b", re.I),
     {Capability.PRIVATE_DATA, Capability.EXTERNAL_EFFECT},
     Reversibility.IRREVERSIBLE),
    (re.compile(r"\b(patch|commit|push|merge|rename|move|chmod)\b", re.I),
     {Capability.EXTERNAL_EFFECT}, Reversibility.RECOVERABLE),
]


@dataclass(frozen=True)
class ToolSpec:
    """One tool as the agent is wired to it.

    `capabilities` and `reversibility` are what an integrator should set
    explicitly. When they are left unset the keyword priors fill in, and the
    result is marked inferred.
    """

    name: str
    description: str = ""
    capabilities: frozenset[Capability] | None = None
    reversibility: Reversibility | None = None

    def resolve(self) -> tuple[frozenset[Capability], Reversibility, bool]:
        if self.capabilities is not None and self.reversibility is not None:
            return self.capabilities, self.reversibility, False

        caps: set[Capability] = set()
        worst = Reversibility.REVERSIBLE
        # Split on separators first: underscore is a word character, so a bare
        # \bsend\b never matches inside `send_invoice_email` -- which is
        # precisely how tools are named.
        haystack = re.sub(r"[_\-./]+", " ", f"{self.name} {self.description}")
        for pattern, pattern_caps, rev in _PRIORS:
            if pattern.search(haystack):
                caps |= pattern_caps
                if _ORDER[rev] > _ORDER[worst]:
                    worst = rev

        caps = set(self.capabilities) if self.capabilities is not None else caps
        rev = self.reversibility if self.reversibility is not None else worst
        return frozenset(caps), rev, True


@dataclass(frozen=True)
class TaskEnvelope:
    """What this deployment is structurally able to do."""

    manifest_hash: str
    n_tools: int
    capabilities: frozenset[Capability]
    max_reversibility: Reversibility
    n_irreversible_tools: int
    inferred_tools: int

    @property
    def rule_of_two_violated(self) -> bool:
        """True when all three capabilities are held at once."""
        return len(self.capabilities) == 3

    @property
    def class_code(self) -> str:
        """A short, stable, sortable code for this structural risk shape.

        This is what a role registry entry keys on. It deliberately excludes
        tool *names* -- two customers with differently named tools that grant
        the same powers are the same risk class, and should price the same.
        """
        bits = "".join(
            "1" if c in self.capabilities else "0"
            for c in (
                Capability.UNTRUSTED_INPUT,
                Capability.PRIVATE_DATA,
                Capability.EXTERNAL_EFFECT,
            )
        )
        # Not first letters: "reversible" and "recoverable" both start with r,
        # and a class code that silently conflates them would price an undoable
        # write the same as a read.
        rev = {
            Reversibility.REVERSIBLE: "V",
            Reversibility.RECOVERABLE: "C",
            Reversibility.IRREVERSIBLE: "I",
        }[self.max_reversibility]
        bucket = min(self.n_tools, 20) // 5  # 0-4, 5-9, 10-14, 15-19, 20+
        return f"C{bits}-{rev}{bucket}"


def derive_envelope(tools: list[ToolSpec]) -> TaskEnvelope:
    """Derive the envelope from the manifest the agent is actually wired to."""
    caps: set[Capability] = set()
    worst = Reversibility.REVERSIBLE
    n_irreversible = 0
    inferred = 0

    for tool in tools:
        tool_caps, rev, was_inferred = tool.resolve()
        caps |= set(tool_caps)
        if _ORDER[rev] > _ORDER[worst]:
            worst = rev
        if rev is Reversibility.IRREVERSIBLE:
            n_irreversible += 1
        if was_inferred:
            inferred += 1

    return TaskEnvelope(
        manifest_hash=manifest_hash(tools),
        n_tools=len(tools),
        capabilities=frozenset(caps),
        max_reversibility=worst,
        n_irreversible_tools=n_irreversible,
        inferred_tools=inferred,
    )


def manifest_hash(tools: list[ToolSpec]) -> str:
    """Stable hash of the granted powers, insensitive to ordering.

    Hashes capabilities rather than descriptions: re-wording a tool's docstring
    must not silently look like a new risk class, but granting it a new power
    must.
    """
    rows = []
    for tool in sorted(tools, key=lambda t: t.name):
        tool_caps, rev, _ = tool.resolve()
        rows.append(
            [tool.name, sorted(c.value for c in tool_caps), rev.value]
        )
    payload = json.dumps(rows, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class Divergence:
    """Where the three views of scope disagree.

    Nothing here is resolved into a single verdict on purpose. An agent holding
    a power it never exercises, and one exercising a power it was never
    declared to hold, are different risks, and an underwriter should see both.
    """

    declared_but_not_granted: frozenset[Capability] = field(
        default_factory=frozenset
    )
    granted_but_not_declared: frozenset[Capability] = field(
        default_factory=frozenset
    )
    granted_but_unused: frozenset[Capability] = field(default_factory=frozenset)
    unused_tool_share: float = 0.0

    @property
    def understated(self) -> bool:
        """The customer claimed less scope than the wiring grants."""
        return bool(self.granted_but_not_declared)

    @property
    def score(self) -> float:
        """0 = the three views agree; higher = they do not.

        Understatement is weighted double: a power granted but not declared is
        the direction a premium-minimising customer errs in, and it is the one
        that produced the 2.6-2.9x pricing penalty when the class was wrong.
        """
        return (
            2.0 * len(self.granted_but_not_declared)
            + 1.0 * len(self.declared_but_not_granted)
            + 0.5 * len(self.granted_but_unused)
        )


def compare(
    derived: TaskEnvelope,
    declared: set[Capability] | None = None,
    observed_capabilities: set[Capability] | None = None,
    unused_tool_share: float = 0.0,
) -> Divergence:
    """Three-way comparison of declared, derived and observed scope."""
    granted = set(derived.capabilities)
    declared = set(declared) if declared is not None else granted
    observed = (
        set(observed_capabilities) if observed_capabilities is not None else granted
    )
    return Divergence(
        declared_but_not_granted=frozenset(declared - granted),
        granted_but_not_declared=frozenset(granted - declared),
        granted_but_unused=frozenset(granted - observed),
        unused_tool_share=unused_tool_share,
    )
