# The witnessed log

freeboard's hash chain proves that nothing changed relative to a head you
already hold. It does not stop the party holding the chain: delete a record,
recompute every hash after it, and the result verifies perfectly. Until now the
answer was to send the head to the bureau. This document describes what
replaces that single anchor, what it defends against, what it does not, and
what was measured.

Three modules, standard library only:

| file | what |
|---|---|
| `src/freeboard/tlog.py` | append only Merkle log, RFC 9162 section 2.1: roots for every past size, inclusion, consistency and range proofs, and verifiers a third party runs offline |
| `src/freeboard/witness.py` | signed checkpoints, a pluggable `Signer`, witnesses that cosign only consistent growth, quorum and freshness checks, fork comparison |
| `src/freeboard/evidence.py` | typed content free entries, `EvidenceLog` (usable as a `Recorder` sink), and `verify_timeline` |

## How it works

**The tree.** Each entry is canonical JSON. Its leaf hash is
`sha256(0x00 || entry)`, an interior node is `sha256(0x01 || left || right)`,
and a tree of n leaves splits at the largest power of two below n, exactly as
RFC 9162 defines it, so sizes that are not powers of two need no padding. The
two prefixes stop an interior node from being passed off as a leaf.

**Proofs.** An inclusion proof shows one entry sits at index i of the tree of
size n. A consistency proof shows the tree of size m is a prefix of the tree of
size n. Both are O(log n) hashes. A range proof (a frontier of perfect subtree
roots plus the entries from index s onward) shows that a run of entries is
complete and in order up to a checkpoint. The chain can only prove any of these
by replaying every later record.

**Checkpoints.** A checkpoint is (origin, size, root, timestamp), signed by the
log as a domain separated text body. It proves only that the operator said so:
an operator who rewrites history signs the new root just as readily.

**Witnesses.** A witness is a separate party that keeps the last checkpoint it
cosigned for each log. It cosigns a new checkpoint only if the log signature
verifies, the timestamp does not run backwards or far ahead, the size does not
shrink, an equal size carries an equal root, and a consistency proof links the
stored checkpoint to the new one. The old size is always the witness's own,
never one the caller names. State is written to disk before the cosignature
exists, and a damaged state file raises rather than resetting, because a
witness that forgets can be rolled back.

**Relying parties.** An auditor, insurer or bureau accepts a checkpoint only
through `verify_cosigned`: the log signed it and a quorum of known witnesses
cosigned it. The quorum defaults to a strict majority of the witnesses the
relying party knows. Each honest witness cosigns one linear history, so two
histories that fork cannot both reach a majority. `max_age_ms` discounts old
cosignatures, so a log that stops asking its witnesses shows up as stale.

**Entries.** Four types share one log:

| type | carries |
|---|---|
| `episode` | the validated wire record, exactly as `to_wire` emits it |
| `eval_commitment` | a hiding commitment to an evaluation held elsewhere, its verdict, and the manifest hash and class code it applies to |
| `evidence_pack` | the digest of an evidence bundle held elsewhere, its kind, the commitment it supports |
| `revalidation` | reason codes only, from a fixed list, and optionally the commitment they call into question |

The format is an allow list in the style of `wire.validate`: numbers, bools,
values from fixed enums and hex digests, nothing else, and only the canonical
encoding decodes. `commitment_hash` takes a nonce, because a pass or fail
report from a known template can otherwise be confirmed by guessing.

**The timeline.** `verify_timeline` takes a cosigned checkpoint, the index of a
commitment, the frontier at that index and the entries from there to the
checkpoint. If they reproduce the checkpointed root, the auditor knows that
configuration passed commitment X at time T and holds every entry logged since:
nothing dropped, inserted, altered or reordered. It then reports the episodes
for that deployment, episodes whose manifest no longer matches the one
evaluated, revalidation triggers, evidence packs and later commitments. Inside
the range it also follows each recorder's own hash chain: chains started,
records whose predecessor is missing, two records claiming one predecessor, and
records whose own hash is wrong.

```python
log = EvidenceLog("log.example/acme", log_key)              # a Recorder sink
rec = Recorder(deployment_id="acme-prod", role="support", tools=tools, sink=log)
c = commitment_hash(report, nonce); log.commit_evaluation(rec, c)   # then run episodes
cp = log.cosign_with(witness_a, witness_b, witness_c)       # each checks consistency
assert verify_cosigned(cp, log_key.verifier(), witnesses)   # majority by default
t = verify_timeline(cp, *log.export_timeline(c), commitment=c)
```

`examples/witnessed_log.py` runs the whole path, including a rewrite the witness
refuses.

## Threat model

**Parties.** The *operator* runs the agent deployment and the log, and holds
the records, the log and the log signing key. *Witnesses* are independent of
the operator and hold only their own keys and state. *Relying parties* (an
auditor, an insurer, the bureau) pin the log's origin and key and the
witnesses' keys. The *recorder* runs inside the operator's process.

**Assumptions.** sha256 is collision resistant. Witness keys are not held by
the operator. More than half of the witnesses a relying party counts are
honest; with f colluding witnesses out of W the quorum has to exceed (W + f) / 2.
Witnesses are enrolled when the log is created. Signatures are Ed25519, not the
HMAC stand in.

### What the operator cannot do without being caught

- Delete, modify, reorder or truncate any entry covered by a checkpoint a
  witness has cosigned, even after recomputing every hash and signing the
  result. The witness refuses (E1).
- Show an auditor a rewritten log, if the auditor accepts only cosigned
  checkpoints. The rewritten checkpoint has no valid cosignature, and a real
  cosignature stapled onto it does not verify (E1).
- Show two histories to relying parties that require a majority quorum.
  Neither branch reaches it (E1).
- Sign two roots for one size. The pair is proof of misbehaviour, and with
  Ed25519 anyone can check it; the witness returns both in `WitnessRefusal.evidence`.
- Serve a timeline with an episode dropped, added, altered or reordered
  (`verify_timeline` fails).
- Hide a record the recorder chained but never logged, if later records from
  the same recorder were logged: it shows as an unlinked record.

### What the operator can still do

- **Rewrite what no witness has seen yet.** Entries appended since the last
  cosigned checkpoint are unprotected (E1 control: 0 of 1000 caught). The window
  is exactly as long as the interval between witness checkpoints.
- **Rewrite before first contact.** A witness accepts its first checkpoint for a
  log without a proof. Enrol witnesses at creation.
- **Fork, against weak relying parties.** Show branch A to one witness and
  branch B to another after a common checkpoint, and each witness sees a
  consistent history (E1: 0 of 1000 caught by any single witness). Gossip
  catches it, and so does a majority quorum; a quorum of one does not.
- **Stop asking the witnesses.** Nothing forces a log to publish. Relying
  parties see it only if they set `max_age_ms`.
- **Never log an episode.** The log proves what was submitted and its order,
  not that the recorder submitted everything. A withheld episode in the middle
  of a chain shows as unlinked; a withheld last episode is invisible.
- **Log fiction.** The log proves a record was not changed after it was
  written, not that it was true when written. An agent that intercepts its own
  tool call path feeds the recorder whatever it likes; out of band attestation
  is the only partial answer (see the README).
- **Choose timestamps.** Entry and checkpoint timestamps are the operator's
  claims. The only external time bound is a witness cosignature: the entries
  existed by the time it was made. RFC 3161 would add a trusted one.
- **Pick which occurrence of a repeated commitment starts a timeline.** The
  range proof covers everything after the start, not the absence of an earlier
  occurrence.

**Witnesses** can refuse to cosign, which stops progress but forges nothing.
A witness colluding with the operator weakens the quorum as described above.

**HMAC.** `HmacSigner` exists so the tests need no dependency. It is a shared
secret: whoever can verify it can forge it, so it proves nothing to a third
party and a fork proof made of HMAC signatures convinces only the key holder.
`Ed25519Signer` and `Ed25519Verifier` use the optional `cryptography` package
behind an import guard; without it they raise, and nothing else changes.

## Serverless: one chain per container

In the Lambda deployment every container starts its own chain, and without a
shared salt every container also gets its own pseudonymous deployment id. The
chain alone cannot say which containers existed or whether one of them lost
its tail.

With the witnessed log, every container's sink writes to one `EvidenceLog`, so
records from all containers get a single order under one witnessed root. Two
changes make that useful:

- Give every container the same salt from the deployment's secret
  configuration, so they share one deployment id and one timeline. The chains
  still differ per container.
- Read the chain counts in the timeline. `chains_started` is the number of
  fresh containers, which can be reconciled with the platform's own count of
  cold starts. `chain_forks` catches two containers restored from one snapshot
  of state (for example SnapStart with a state file baked in). `unlinked` shows
  a record missing from the middle of a container's chain.

This change ships the log, not the service in front of it. A shared log
reachable from many containers is a small service wrapping `EvidenceLog`, with
one writer, and it is not built here.

## Anchoring later: RFC 3161 and Rekor

Design only. Nothing in this change calls either.

**RFC 3161 timestamps.** Send `sha256(checkpoint.body())` in a `TimeStampReq` to
a timestamp authority and keep the returned token beside the checkpoint. The
token binds "this checkpoint existed by time T" with a time the operator does
not control, which bounds every entry in it, and in the EU a qualified
timestamp carries legal weight. It adds nothing to consistency: a timestamp
authority stamps anything it is sent, including both branches of a fork. It is
a notary, not a witness. Verifying a token needs ASN.1 parsing and the
authority's certificate chain, so it would sit behind the same optional import
as Ed25519.

**Rekor.** Publish each signed checkpoint to Sigstore's public transparency log
as a signed artifact under the log's Ed25519 key, and keep the returned entry
and inclusion proof. Rekor is itself a Merkle log with its own witnesses and
monitors, so a split view would have to fork Rekor as well, and anyone can list
every checkpoint published for an origin and look for two roots at one size.
Only checkpoint hashes leave the operator, never entries, but origin, sizes and
timing become public, which matters for a deployment that does not want its
volume known.

A shared shape for both: an `Anchor` protocol with
`anchor(signed) -> receipt` and an offline `verify(signed, receipt)`, receipts
stored next to checkpoints, and relying parties free to require one, both or
neither. The order that follows from the threat model is witnesses first
(consistency is what catches rewrites), Rekor for public discoverability, and a
timestamp authority where someone needs a trusted time.

## Experiments

Reproduce with `PYTHONPATH=src python experiments/witnessed_log/e1_rewrite.py`
(and `e2_scale.py`, `e3_correctness.py`); results land in
`out/tlog_e1_rewrite.json`, `out/tlog_e2_scale.json` and
`out/tlog_e3_correctness.json`.

### E1: the rewrite attack

1000 randomised trials per scenario, logs of 8 to 600 real episode records
(mean about 300), seed 20260927. The operator holds the log key, recomputes
everything after the rewrite, signs the result, and offers the witness every
proof it can build (from the rewritten tree, from the honest tree, and none).
Half the trials keep appending after the rewrite so the log looks like normal
growth.

| attack on the witnessed prefix | plain chain verifies itself | anchored chain head catches it | witness refuses |
|---|---:|---:|---:|
| delete one record | 1000 / 1000 | 1000 / 1000 | 1000 / 1000 |
| flip one outcome | 1000 / 1000 | 1000 / 1000 | 1000 / 1000 |
| swap two neighbours | 1000 / 1000 | 1000 / 1000 | 1000 / 1000 |
| truncate | 1000 / 1000 | 1000 / 1000 | 1000 / 1000 |
| control: rewrite only unwitnessed entries | 1000 / 1000 | 0 / 1000 | 0 / 1000 |

With 1000 of 1000 the 95% Wilson interval is [0.996, 1]; with 0 of 1000 it is
[0, 0.004]. The first column is the known weakness, reproduced: after every
rewrite the linear chain verified clean. The second column is worth being
straight about: a verifier holding one anchored chain head and replaying the
whole prefix catches the same rewrites. What the witnessed log adds is that the
check costs O(log n) instead of a replay, that it runs before anything is
cosigned rather than at audit time, that many independent parties can hold
it, and the fork and split view results below.

| fork: branch A to witness 1, branch B to witness 2 | result |
|---|---:|
| divergence after the common checkpoint, caught by either witness alone | 0 / 1000 |
| same, caught by the witnesses comparing checkpoints | 1000 / 1000 |
| same, both branches accepted by relying parties with quorum 1 | 1000 / 1000 |
| same, either branch accepted with a majority quorum (2 of 2) | 0 / 1000 |
| divergence inside the witnessed prefix, refused by a witness | 1000 / 1000 |

| split view: witness sees the honest log, auditor sees a rewrite | fooled |
|---|---:|
| auditor who checks only the log signature and replays the entries | 1000 / 1000 |
| auditor who requires a witness cosignature (a real one stapled on) | 0 / 1000 |
| rewritten entries checked against the witnessed checkpoint | 0 / 1000 |

### E2: scale

Python 3.11.9 on one core of a Windows machine, real typed episode entries of
about 865 bytes each, 1000 sampled proofs per size plus the edges (every index
and every old size at n = 1e3).

| entries | append | tree hashes in memory | inclusion proof, max | consistency proof, max | verify inclusion | verify consistency |
|---:|---:|---:|---:|---:|---:|---:|
| 1e3 | 0.004 s | 64 KB | 10 hashes (320 B) | 11 hashes | 13 µs | 17 µs |
| 1e5 | 0.43 s | 6.4 MB | 17 hashes (544 B) | 18 hashes | 19 µs | 26 µs |
| 1e6 | 4.2 s | 64 MB | 20 hashes (640 B) | 21 hashes | 21 µs | 30 µs |

Proof generation took 14 to 25 µs. Inclusion proofs never exceeded
ceil(log2 n) hashes and consistency proofs never exceeded ceil(log2 n) + 1:
O(log n), as expected, and the cost per append stayed at 4.1 to 4.3 µs from
1e3 to 1e6. Validating and encoding the entries costs far more than the tree
(49 s for 1e6 entries against 4.2 s of hashing into the tree).

For comparison, checking one record or one extension with the linear chain
means replaying every later record, about n / 2 records on average (492,496
for an inclusion check at 1e6). A full replay of the chain took 13.7 s at 1e6
(13.7 µs per record); recomputing the Merkle root from all 1e6 leaf hashes
took 1.6 s.

### E3: correctness against a reference

`tests/tlog_reference.py` is a second implementation written straight from
the RFC 9162 recursion, sharing no code with `tlog.py`, with verifiers that
rebuild roots by walking the recursion instead of the RFC's bit arithmetic.
Over the same entries, every size from 1 to 300 and every pair:

| check | pairs | disagreements |
|---|---:|---:|
| inclusion, all 0 <= m < n <= 300 | 45,150 | 0 |
| consistency, all 0 <= m <= n <= 300 | 45,450 | 0 |
| sampled sizes 318 to 2996 (200 sizes), inclusion and consistency | 13,035 | 0 |

Each pair checks four things: the proof equals the reference proof, the RFC
verifier accepts it, the reference verifier accepts it, and one random
mutation (flipped bit, dropped or extra hash, wrong index, leaf, size or root)
is rejected by both. Roots and proofs also match the Certificate Transparency
test vectors for the first eight leaves.

### What the experiments changed

- **E1 changed the default quorum.** `verify_cosigned` first required the
  caller to pick a quorum. The fork trials show why that is unsafe: with a
  quorum of one, two relying parties accepted two different histories in 1000
  of 1000 trials. The quorum now defaults to a strict majority of the known
  witnesses, which accepted neither branch in any trial.
- **E1 also added freshness.** Nothing forces an operator to keep asking its
  witnesses, so a log could stop at the last honest checkpoint and keep
  serving it. `max_age_ms` lets a relying party discount stale cosignatures.
  This one follows from the threat model rather than from a measured failure.
- **E3's first run flagged 253 accepted mutants, and the harness was wrong.**
  All 253 were consistency proofs from size 0 with a changed new root. Every
  tree extends the empty tree, so those claims are true and the verifiers were
  right to accept them. That mutation is no longer applied at size 0; nothing
  in `tlog.py` changed.

## What this does not do

- No network anything: no witness daemon, no transport for proofs, no gossip
  protocol. `compare_checkpoints` is the check two parties run once they have
  exchanged checkpoints.
- Not wire compatible with the C2SP checkpoint and cosignature formats, though
  close in shape. Adopting them would let existing witness networks cosign.
- `EvidenceLog` keeps entries in memory and replays its file on open; a very
  large log would want the tree's hash levels persisted instead.
- No key rotation, no witness enrolment protocol, no log shutdown record.
- The Ed25519 path was exercised against `cryptography` 50.0.1 in a separate
  local environment; the main test environment does not have it and skips that
  test.
