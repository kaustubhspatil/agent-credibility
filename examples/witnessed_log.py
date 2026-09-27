"""A witnessed evidence log, end to end, in one process.

Run:  PYTHONPATH=src python examples/witnessed_log.py

A deployment passes an evaluation, the commitment goes into the log, every
episode after it goes into the same log, and a witness cosigns checkpoints
as the log grows. An auditor then checks, offline, that the configuration
passed commitment X at time T and sees every monitored episode since. Last,
the operator deletes a failed episode and rebuilds the log, and the witness
refuses to cosign it.

HmacSigner keys stand in for real ones so this runs with no dependencies.
They are not public key signatures; use Ed25519Signer for anything real.
"""

from __future__ import annotations

import os
import random

from freeboard import (
    Capability,
    EvidenceLog,
    HmacSigner,
    Recorder,
    Reversibility,
    ToolSpec,
    Witness,
    WitnessRefusal,
    commitment_hash,
    verify_cosigned,
    verify_timeline,
)

TOOLS = [
    ToolSpec("search_orders", capabilities=frozenset({Capability.PRIVATE_DATA}),
             reversibility=Reversibility.REVERSIBLE),
    ToolSpec("issue_refund", capabilities=frozenset({Capability.EXTERNAL_EFFECT}),
             reversibility=Reversibility.IRREVERSIBLE),
]


def run(episodes: int = 40, seed: int = 1) -> dict:
    rng = random.Random(seed)
    log_key, witness_key = HmacSigner.generate(), HmacSigner.generate()
    log = EvidenceLog("log.example/acme", log_key)
    witness = Witness("witness.example", witness_key, {log.origin: log_key.verifier()})
    rec = Recorder(deployment_id="acme-prod", role="customer_support", tools=TOOLS, sink=log)

    report = b"evaluation report, held by the deployer, never logged"
    commitment = commitment_hash(report, os.urandom(16))
    log.commit_evaluation(rec, commitment)

    for i in range(episodes):
        with rec.episode(f"ticket-{i}") as ep:
            ep.action("search_orders", output_chars=rng.randint(50, 900))
            if rng.random() < 0.3:
                ep.action("issue_refund")
            ep.resolve(success=rng.random() < 0.85)
        if i % 10 == 9:
            signed = log.cosign_with(witness)

    # the auditor's side: nothing here needs the operator's cooperation
    trusted = verify_cosigned(signed, log_key.verifier(), {witness.name: witness.verifier()})
    timeline = verify_timeline(signed, *log.export_timeline(commitment), commitment=commitment)

    # the operator drops a failed episode, rebuilds the whole log, and keeps
    # appending so it looks like normal growth
    rewritten = EvidenceLog("log.example/acme", log_key)
    dropped = False
    for data in log.entries():
        if not dropped and b'"resolved":false' in data:
            dropped = True
            continue
        rewritten.log.append(data)
    later = Recorder(deployment_id="acme-prod", role="customer_support", tools=TOOLS, sink=rewritten)
    for i in range(3):
        with later.episode(f"ticket-late-{i}") as ep:
            ep.resolve(success=True)
    try:
        rewritten.cosign_with(witness)
        refusal = None
    except WitnessRefusal as exc:
        refusal = exc.code

    return {"trusted": trusted.ok, "timeline": timeline, "rewrite_refused": refusal}


def main() -> None:
    out = run()
    t = out["timeline"]
    print(f"checkpoint cosigned by a majority of witnesses: {out['trusted']}")
    print(f"timeline verifies: {t.ok}  verdict: {t.verdict}  committed at: {t.commitment_ts}")
    print(f"episodes since the commitment: {t.episodes}  "
          f"(chains started {t.chains_started}, unlinked {t.unlinked}, forks {t.chain_forks})")
    print(f"operator deletes a failed episode and rebuilds: witness says {out['rewrite_refused']!r}")


if __name__ == "__main__":
    main()
