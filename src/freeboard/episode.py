"""Recording an agent episode as sufficient statistics.

The estimator needs a failure indicator and a handful of moments per episode.
It does not need, and must never receive, the prompt, the tool arguments, the
tool output or the model's generation. So the recording API is built so that
those cannot be handed to it in the first place: there is no parameter anywhere
in this module that takes free text other than a tool name and an identifier,
and both of those are hashed before they reach the wire.

That is a deliberately stronger guarantee than "we redact before sending".
Redaction is a promise about behaviour and has to be audited. A surface that
cannot accept content is a property of the type signature, and a privacy review
can check it in a minute.

    rec = Recorder(deployment_id="acme-prod", role="customer_support",
                   tools=[...])

    with rec.episode(task_id="ticket-8821") as ep:
        ep.action("search_orders")
        ep.action("search_orders", error=True)   # a retry
        ep.action("issue_refund", output_chars=180)
        ep.resolve(success=False)
"""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field

from . import __version__ as _VERSION
from .chain import ChainState, Checkpoint, checkpoint
from .state import RecorderState
from .envelope import Capability, Reversibility, TaskEnvelope, ToolSpec, derive_envelope


def _hash_id(value: str, salt: str) -> str:
    """Stable pseudonymous id.

    Identifiers are customer-controlled and routinely carry meaning -- a task id
    can be an email address or an account number. They are salted per
    deployment and truncated, so records stay linkable within a deployment
    (which is what credibility needs) without the raw value travelling.
    """
    digest = hashlib.sha256(f"{salt}\x00{value}".encode()).hexdigest()
    return digest[:16]


@dataclass
class EpisodeRecord:
    """One finished episode, as sufficient statistics only.

    Every field is a number, a bool, an enum value or a hex hash. There is
    nowhere here for content to hide, which is what `wire.validate` enforces.
    """

    episode_id: str
    deployment_id: str
    role: str
    task_hash: str
    pass_index: int
    envelope_class: str
    manifest_hash: str

    # outcome -- the loss channel the estimator consumes
    resolved: bool | None = None
    reward: float | None = None
    escalated: bool = False

    # behavioural moments
    n_actions: int = 0
    n_distinct_tools: int = 0
    tool_entropy: float = 0.0
    repeat_rate: float = 0.0
    error_rate: float = 0.0
    n_mutating: int = 0
    n_irreversible: int = 0
    max_reversibility: str = Reversibility.REVERSIBLE.value
    output_chars: int = 0
    duration_ms: int = 0

    # scope actually exercised, for the divergence signal
    observed_capabilities: tuple[str, ...] = ()

    # tamper-evident sequencing. seq and prev_hash are set as the record is
    # appended to the deployment's chain; entry_hash commits to everything else.
    seq: int = 0
    prev_hash: str = "0" * 64
    entry_hash: str = ""

    # provenance
    schema_version: str = "1.0"
    recorder_version: str = _VERSION


class Episode:
    """A single in-flight episode. Not thread-safe by design; one per task."""

    def __init__(
        self,
        recorder: Recorder,
        task_id: str,
        pass_index: int,
    ) -> None:
        self._rec = recorder
        self._start = time.monotonic()
        self._tools: list[str] = []
        self._errors = 0
        self._observations = 0
        self._output_chars = 0
        self._escalated = False
        self._resolved: bool | None = None
        self._reward: float | None = None
        self._finished = False

        self.record_id = uuid.uuid4().hex[:16]
        self._task_hash = _hash_id(task_id, recorder.salt)
        self._pass_index = pass_index

    # -- the recording surface. note what is absent: no text parameters. ----

    def action(
        self,
        tool: str,
        *,
        error: bool = False,
        output_chars: int = 0,
    ) -> None:
        """Record one tool invocation.

        `output_chars` is a length, never the output. Callers pass
        `len(result)`, and the result itself stays in their process.
        """
        if self._finished:
            raise RuntimeError("episode already finished")
        if output_chars < 0:
            raise ValueError("output_chars must be non-negative")
        self._tools.append(tool)
        self._observations += 1
        if error:
            self._errors += 1
        self._output_chars += output_chars

    def escalate(self) -> None:
        """The agent handed off to a human."""
        self._escalated = True

    def resolve(
        self, *, success: bool | None = None, reward: float | None = None
    ) -> None:
        """Close out the episode's outcome."""
        self._resolved = success
        self._reward = reward

    # -- finishing ---------------------------------------------------------

    def finish(self) -> EpisodeRecord:
        if self._finished:
            raise RuntimeError("episode already finished")
        self._finished = True

        counts = Counter(self._tools)
        n = len(self._tools)
        total = sum(counts.values())
        entropy = 0.0
        if total > 1:
            entropy = -sum(
                (c / total) * math.log2(c / total) for c in counts.values() if c
            )

        repeats = sum(
            1 for a, b in zip(self._tools, self._tools[1:]) if a == b
        )

        caps: set[Capability] = set()
        n_mutating = 0
        n_irreversible = 0
        worst = Reversibility.REVERSIBLE
        order = {
            Reversibility.REVERSIBLE: 0,
            Reversibility.RECOVERABLE: 1,
            Reversibility.IRREVERSIBLE: 2,
        }
        for name in self._tools:
            spec = self._rec.tool_index.get(name)
            if spec is None:
                continue
            tool_caps, rev, _ = spec.resolve()
            caps |= set(tool_caps)
            if rev is not Reversibility.REVERSIBLE:
                n_mutating += 1
            if rev is Reversibility.IRREVERSIBLE:
                n_irreversible += 1
            if order[rev] > order[worst]:
                worst = rev

        record = EpisodeRecord(
            episode_id=self.record_id,
            deployment_id=self._rec.deployment_hash,
            role=self._rec.role,
            task_hash=self._task_hash,
            pass_index=self._pass_index,
            envelope_class=self._rec.envelope.class_code,
            manifest_hash=self._rec.envelope.manifest_hash,
            resolved=self._resolved,
            reward=self._reward,
            escalated=self._escalated,
            n_actions=n,
            n_distinct_tools=len(counts),
            tool_entropy=round(entropy, 6),
            repeat_rate=round(repeats / max(n - 1, 1), 6) if n > 1 else 0.0,
            error_rate=round(self._errors / max(self._observations, 1), 6),
            n_mutating=n_mutating,
            n_irreversible=n_irreversible,
            max_reversibility=worst.value,
            output_chars=self._output_chars,
            duration_ms=int((time.monotonic() - self._start) * 1000),
            observed_capabilities=tuple(sorted(c.value for c in caps)),
        )
        # Tool names are handed to the recorder, which stays in the customer's
        # process, and never to the record, which leaves it.
        self._rec._note_tools(set(self._tools))
        self._rec._emit(record)
        return record

    def __enter__(self) -> Episode:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._finished:
            if exc_type is not None and self._resolved is None:
                # an episode that raised did not succeed; record that rather
                # than dropping it, since silent loss biases the base rate
                self._resolved = False
            self.finish()


@dataclass
class Recorder:
    """Per-deployment recorder. One per (customer, role, agent configuration)."""

    deployment_id: str
    role: str
    tools: list[ToolSpec] = field(default_factory=list)
    declared_capabilities: set[Capability] | None = None
    salt: str = ""
    sink: object | None = None  # anything with .write(EpisodeRecord)
    state_path: str | None = None  # persist salt + chain across restarts

    def __post_init__(self) -> None:
        # A per-deployment salt keeps hashed ids from being reversible via a
        # rainbow table of likely task ids. Without `state_path` it lives only
        # for this process, which breaks the chain and re-keys every task hash
        # on restart -- fine for a notebook, wrong for anything containerised.
        self.state = RecorderState.load_or_create(
            self.state_path, self.deployment_id, self.salt
        )
        self.salt = self.state.salt
        self.envelope: TaskEnvelope = derive_envelope(self.tools)
        self.tool_index: dict[str, ToolSpec] = {t.name: t for t in self.tools}
        self.deployment_hash = _hash_id(self.deployment_id, self.salt)
        self.chain = self.state.chain_state()
        self._records: list[EpisodeRecord] = []
        self._observed: set[Capability] = set()
        self._used_tools: set[str] = set()

    def episode(self, task_id: str, pass_index: int = 1) -> Episode:
        return Episode(self, task_id, pass_index)

    def checkpoint(self) -> Checkpoint:
        """Commitment to the current chain head.

        Only meaningful once it has left this process -- see chain.py.
        """
        return checkpoint(self.deployment_hash, self.chain)

    def _note_tools(self, names: set[str]) -> None:
        """In-process only: which granted tools have actually been exercised."""
        self._used_tools |= names

    def _emit(self, record: EpisodeRecord) -> None:
        # Chain before the record can reach a sink: anything that leaves the
        # process must already be committed to the sequence.
        from dataclasses import asdict

        body = asdict(record)
        body["observed_capabilities"] = list(record.observed_capabilities)
        body.pop("seq", None)
        body.pop("prev_hash", None)
        body.pop("entry_hash", None)
        seq, prev, digest = self.chain.advance(body)
        record.seq = seq
        record.prev_hash = prev
        record.entry_hash = digest

        # Persist the advanced head before the record can reach a sink. If the
        # process dies between the two, the state is ahead of what was sent,
        # which verification reports as a gap -- the safe direction. The
        # reverse would silently re-issue a sequence number.
        if self.state_path is not None:
            self.state.absorb(self.chain)
            self.state.save(self.state_path)

        self._records.append(record)
        self._observed |= {Capability(c) for c in record.observed_capabilities}
        if self.sink is not None:
            self.sink.write(record)  # type: ignore[attr-defined]

    # -- reporting ---------------------------------------------------------

    @property
    def records(self) -> list[EpisodeRecord]:
        return list(self._records)

    def divergence(self):
        """Three-way scope comparison, using everything observed so far."""
        from .envelope import compare

        # Tool-level usage is tracked on the recorder rather than the record,
        # because the record must not carry tool names. The recorder stays in
        # the customer's process, so it may hold them; the wire may not.
        unused_share = 0.0
        if self.tools:
            unused_share = 1.0 - len(self._used_tools) / len(self.tools)

        return compare(
            self.envelope,
            declared=self.declared_capabilities,
            observed_capabilities=self._observed,
            unused_tool_share=round(unused_share, 6),
        )
