# Helix Harness Adaptive Intelligence Cycle

Date: 2026-09-16

## Purpose

Helix Harness adds an observable-artifact adaptive loop around ordinary chat/tool execution. It records what the runtime can legitimately observe, evaluates claims against backend-owned evidence, asks the same loaded model for a post-task self-audit, and produces an advisory adaptation decision. The loop is not itself proof of model improvement and does not make one anomalous turn a training target.

## Non-negotiable invariants

1. Ordinary chat and tool execution are authoritative. Helix capture, audit, evidence, ledger, or controller failures are fail-open and must not change a successful tool return.
2. Original tool exceptions and cancellation semantics are preserved.
3. No hidden/private chain of thought is captured. Post-task audit receives only observable artifacts: objective, tool calls/results, public final output, cache/runtime telemetry, and explicit verification records.
4. A model-authored command/tool name cannot mint verifier authority. Verification authority enters at the backend tool-capture boundary as typed metadata.
5. Authority and relevance are separate. A passed typed verifier supports a claim only when the backend receipt binds to that exact normalized claim.
6. Raw tool results and model-reported support are context for the auditor, not objective proof by themselves.
7. Model self-audit cannot become training truth. Verified training targets come only from `human_correction` or `objective_correction`; objective correction requires backend verifier provenance.
8. One anomalous trajectory cannot trigger QLoRA. Recurrence, evidence, behavioral-pattern, regression, and verified-target gates remain authoritative.
9. Cache/speculative telemetry must be request-scoped. Global GGUF speculative counters are not presented as per-trajectory evidence when safe attribution is unavailable.
10. No `>=5x` or equivalent performance claim is accepted without a real decode benchmark; speculative accepted-draft evidence is also required where the claim depends on speculation.

## Data flow

```text
foreground chat/tool loop
    |
    +--> bounded tool capture (`execute_tool`)
    |       - turn/thread identity
    |       - arguments/result/error/retry
    |       - backend-only typed verification receipt
    |
    +--> request-scoped runtime telemetry
            - token/cache/context timing
            - compaction/reconstruction/schema changes
            - speculative requested/resolved/engaged evidence

post-task
    |
    +--> Trajectory
    +--> CacheIntegrityReport
    +--> same-model SelfAuditReport (observable artifacts only)
    +--> EvidenceClaim adjudication
    +--> QualityVector (Q/E/C/S)
    +--> deterministic Hermes adaptation gate
    +--> advisory DecisionRecords
    +--> provenance/recurrence ledgers
    +--> optional VerifiedTrainingTargetReceipt gate
            |
            +--> QLoRA candidate only after all independent gates pass
```

## Implementation map

- `core/helix_engine/capture.py` — bounded per-turn capture sessions and failure/cancellation-safe capture plumbing.
- `core/inference/tools.py` — real tool execution boundary; preserves exact tool semantics and attaches backend-owned verification provenance.
- `core/helix_engine/trajectory.py` — trajectory, tool-step, verification enums/receipt.
- `core/helix_engine/cache_integrity.py` — request telemetry normalization and disruption attribution.
- `core/helix_engine/audit.py` — machine-readable self-audit parsing.
- `core/helix_engine/evidence.py` — deterministic evidence resolution and claim adjudication.
- `core/helix_engine/quality.py` — Q/E/C/S quality vector.
- `core/helix_engine/decision_controller.py` — typed advisory shadow decisions with probabilities/confidence/outcome fields.
- `core/helix_engine/hermes.py` — authoritative adaptation gate.
- `core/helix_engine/training_targets.py` — signed/bound verified-target receipt contract.
- `core/helix_engine/ledger.py` — recurrence and provenance ledger.
- `core/helix_engine/controller.py` — closed-loop orchestration; schema `helix.closed-loop.v1`.
- `core/helix_engine/ingest.py` — post-turn integration and downstream recommendation staging.
- `routes/self_training.py` — autonomous training admission/reconciliation with policy and provenance safeguards.
- `frontend/src/features/chat/lib/hermes-learning.ts` — self-audit artifact construction/parsing and legacy-path isolation.

## Cache/context interpretation

`CacheIntegrityReport` records prompt/cached/stable-prefix/newly-evaluated tokens, KV resets, compactions, prompt reconstructions, system/tool-schema changes, insertions, reorder events, prefill/decode/TTFT, speculative status, accepted/rejected drafts, runtime configuration, and attributed disruptions.

`search_memory` and `search_conversation` are public aliases over the same thread-memory retrieval path. Calling both with the same arguments and the same leading result is classified as duplicate retrieval. Diagnostics describe this as an **equivalent retrieval path**, not literally the same tool.

Efficiency is secondary to result quality and evidentiary completeness. A mechanically avoidable disruption routes first to a runtime/policy fix, not weight training.

## Packaging durability contract

The managed backend compatibility marker is:

```text
HELIX_HARNESS_BACKEND_CONTRACT = "helix.adaptive.backend.v1"
```

The macOS bundle stages a dependency-closed runtime overlay rather than trusting the upstream package version. Current staging contains 640 runtime files and excludes tests, bytecode/cache directories, and backend `requirements/`. Preflight fingerprints Helix-critical files and probes the contract, `require_tool_access`, and `run_closed_loop`; a same-version/upstream overwrite cannot remain hidden behind a cached Ready verdict.

The updater and installer both have explicit bundled-overlay reapplication paths. See the dated acceptance document for what has and has not been exercised live.
