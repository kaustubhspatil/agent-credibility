# Evaluating Freeboard

How to check the claims in the README for yourself, and what the package does
not do. Every command runs from a clone, from the repo root.

```bash
pip install -e ".[dev]"
```

---

## Does it leak data?

This is the claim worth checking first, because the whole design rests on it.

```bash
pytest tests/test_recorder.py -k "canary or leak" -v
```

`test_canary_never_reaches_the_wire` pushes an SSN-shaped string into every
surface the SDK accepts — deployment id, role, task id, tool name, tool
description — and asserts it appears nowhere in the outbound payload: not in a
value, not in a key, not in the JSON encoding, and not through the
OpenTelemetry attributes either.

Then read the recording method in `src/freeboard/episode.py`:

```python
def action(self, tool: str, *, error: bool = False, output_chars: int = 0) -> None:
```

There is no parameter that accepts tool output. `output_chars` is a length; the
caller passes `len(result)` and the result stays in their process. This is not
redaction — a deny-list strips the sensitive things someone thought of and has
to be trusted. Nothing can leak here because nothing can get in.

The data contract in full is the `_RULES` table in `src/freeboard/wire.py`:
every field must be a number, a bool, a null, a value from a fixed enum, or a
hex digest, and `validate()` refuses anything else.

Supporting properties, all covered by tests:

- identifiers are salted and hashed per deployment, so they stay linkable
  within a deployment without the raw value travelling
- tool names never reach the wire; they stay in-process for entropy and
  capability accounting
- `to_wire()` cannot run without `validate()` passing

## Does the tamper-evidence work?

```bash
pytest tests/test_recorder.py -k "chain or drop or rewritten or editing or reorder" -v
pytest tests/test_state.py -v
```

Each record commits to the one before it:

```
entry_hash(n) = sha256( prev_hash(n) || canonical_json(record without hashes) )
```

Deleting a record leaves a sequence gap, editing one breaks its `entry_hash`,
reordering breaks both, and state persists across restarts so a rescheduled
container resumes its chain instead of starting a new one.

**What chaining alone does not prove.** A party who deletes a record and
recomputes every hash after it produces a chain that verifies perfectly.
Chaining is tamper-*evident* only once the head has been committed somewhere
that party cannot rewrite. That is what `Checkpoint` is for, and it is why any
Article 12 claim rests on checkpoints being anchored rather than on the chain
existing. The limitation is a passing test —
`test_a_rewritten_chain_self_verifies_and_only_the_checkpoint_catches_it` — not
a footnote.

## Is the estimator right?

```bash
pytest tests/test_estimate.py -v
```

Two independent checks. Against a closed form: for a Beta(a,b)–Bernoulli class
the credibility constant is exactly `a + b`, so there is a known right answer.
And against the numpy implementation used for the research, on random data,
uneven exposure, continuous values, and the real corpus — where the stdlib
version reproduces the published `K = 7.59` to 1e-9.

Two implementations of the same formulas are only worth having if they cannot
drift. If those tests ever disagree, the shipped SDK and the published numbers
have stopped describing the same estimator.

## End to end

```bash
python examples/quickstart.py
```

Three simulated customers running the same support agent. Records episodes,
writes each to its own append-only log, reads them back through the same
validation, verifies all three chains, and prices the fleet — importing nothing
outside the SDK.

Worth looking at in the output: the customer declares `private_data`, the tool
manifest grants all three capabilities, and the recorder reports a rule-of-two
violation with a divergence score of 4.0. Scope is derived from the tools the
agent is wired to, never from what is declared.

---

## Does the attestation path work?

```bash
pytest tests/test_recorder.py -k attest -v
pytest tests/test_divergence.py tests/test_anomaly.py -q
```

`ep.attest(actions_observed, source)` records an independent count from outside the
agent's process. Only the positive gap is kept: fewer observed than reported is an
instrumentation gap, **more** observed than reported is the signature of a recorder
being lied to.

This exists because the chain proves a record was not altered *after* writing, not that
it was *true when written* — and in the July 2026 Hugging Face incident agents rewrote
which command their own tool-call handler requested, with spoofing found in 7% of
transcripts. No in-process recorder defends against that, including this one. The
derived envelope survives it (it comes from the tool manifest, not from anything the
agent reports); the behavioural statistics do not.

The bureau side pools attestation gaps into a baseline, so a deployment can be asked
whether its divergence is unusual for its class. Two rules are enforced in code:
unattested episodes are excluded rather than counted as zero, and divergence is never
pooled across attestation sources. Both have tests.

`bureau/anomaly.py` runs the other tail — auditing for deployments that report
implausibly *few* losses, which is the direction spoofing biases a frequency and the
direction nobody audits.

## What it does not do

- **No transport.** The SDK produces validated, chained records and stops.
  `Recorder(sink=...)` takes anything with a `.write(record)` method, and
  `JsonlSink` is provided; where the records go after that is yours.
- **No certificate renderer.** A certificate's format is decided by whoever
  signs it, so guessing means shipping a surface to deprecate.
- **No severity model.** This estimates loss *frequency*. Frequency × severity
  is a premium; only the first half exists here.
- **No framework adapters.** Instrumentation is currently by hand at the call
  site.
- **The keyword fallback is weak on purpose.** Tools that declare no
  capabilities are classified by name, and anything inferred that way is counted
  in `inferred_tools` so it can be treated as the guess it is. Set
  `capabilities` and `reversibility` explicitly on every tool that matters.
- **Drift is untested.** The research corpora carry no timestamps, so nothing
  here demonstrates resistance to a risk profile changing over time.
- **No detector of individual spoofed episodes.** The divergence and
  underreporting tests flag an aggregate signature across many episodes, never a
  single one, and an adversary forging toward the class mean defeats both.
- **The API may change before 1.0.**
