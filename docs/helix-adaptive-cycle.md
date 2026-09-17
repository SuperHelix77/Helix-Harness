# Helix Harness Adaptive Intelligence Cycle

Date: 2026-09-17

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

post-task (ordinary answer already complete)
    |
    +--> versioned observable TrajectoryRecord
    +--> CacheIntegrityReport + EvidenceClaim graph
    +--> cheap typed DEEP_SELF_AUDIT advisory
    |       - deterministic reliability conditions can force TAKE
    |       - controller/preparation failure falls back to TAKE
    +--> same-model SelfAuditReport only when admitted (observable artifacts only)
    +--> EvidenceClaim adjudication
    +--> QualityVector (Q/E/C/S)
    +--> CounterfactualTrajectoryCandidate (UNVERIFIED unless separately replay-verified)
    +--> deterministic Hermes adaptation gate
    +--> advisory DecisionRecords + objective retrospective labels
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
- `core/helix_engine/audit.py` — backend-resolved observable audit bundle and pre-audit cheap decision.
- `core/helix_engine/compress.py` — first-class, provenance-linked counterfactual compression candidates.
- `core/helix_engine/evidence.py` — deterministic important-claim discovery, evidence resolution and claim adjudication.
- `core/helix_engine/quality.py` — Q/E/C/S quality vector.
- `core/helix_engine/decision_controller.py` — typed advisory decisions, fail-open fallback, probabilities/confidence, objective outcome labels, Brier/bin measurement.
- `core/helix_engine/hermes.py` — authoritative adaptation gate.
- `core/helix_engine/training_targets.py` — signed/bound verified-target receipt contract.
- `core/helix_engine/ledger.py` — recurrence and provenance ledger.
- `core/helix_engine/controller.py` — closed-loop orchestration; schema `helix.closed-loop.v1`.
- `core/helix_engine/ingest.py` — post-turn integration and downstream recommendation staging.
- `routes/self_training.py` — autonomous training admission/reconciliation with policy and provenance safeguards.
- `routes/helix_engine.py` — fail-open `/prepare-audit`, `/ingest-turn`, typed decision/outcome and calibration control-plane routes.
- `routes/inference.py` — backend-owned `X-Helix-Background-Audit` boundary that makes server/client/MCP/deep-research/checkpoint-memory tool paths unreachable for the internal audit even when the launcher globally forces tools on.
- `frontend/src/features/chat/lib/helix-context-telemetry.ts` — bounded hash/counter-only live context provenance and cache-disruption observations.
- `frontend/src/features/chat/lib/hermes-learning.ts` — bounded backend-resolved self-audit artifact transport/parsing and legacy-path isolation. The parser accepts canonical tagged JSON plus the narrowly bounded local-model case where only the closing tag is omitted; partial JSON/trailing prose are still rejected.

## Cache/context interpretation

`CacheIntegrityReport` records prompt/cached/stable-prefix/newly-evaluated tokens, KV resets, compactions, prompt reconstructions, system/tool-schema changes, insertions, reorder events, prefill/decode/TTFT, speculative status, accepted/rejected drafts, runtime configuration, and attributed disruptions.

Every live field now carries provenance or an explicit unavailable marker. The collector stores hashes/counters rather than duplicating context. KV reset attribution is intentionally `UNKNOWN` when the runtime has no request-scoped event. Tool-schema fingerprint changes likewise remain `UNKNOWN` unless the source is actually known. A cache miss is never promoted into a correctness failure.

`search_memory` and `search_conversation` are public aliases over the same thread-memory retrieval path. Calling both with the same arguments and the same leading result is classified as duplicate retrieval. Diagnostics describe this as an **equivalent retrieval path**, not literally the same tool.

Efficiency is secondary to result quality and evidentiary completeness. A mechanically avoidable disruption routes first to a runtime/policy fix, not weight training.

## Cheap decision controller and measured calibration

The controller is a typed, advisory allocation layer. The first production cost-saving decision is `DEEP_SELF_AUDIT`: it executes **before** the expensive same-model audit. Trivial low-risk completed turns may use the bounded fallback audit instead; failed tools, explicit acceptance criteria, high-impact claims, and typed verifier evidence force a deep audit. Any controller exception preserves the pre-existing full-audit behavior.

Objective post-task labels are currently available for evidence sufficiency, actual QLoRA qualification, and whether observable audit-risk signals existed. These labels feed Brier score/probability bins. Measured labels do not imply a scientifically calibrated controller; the status remains `calibrated=false` until held-out calibration evidence exists.

The typed enum supports the broader requested decision family, but only `DEEP_SELF_AUDIT` is currently a live pre-generative resource-allocation point in the normal path. Existing deterministic memory/search/skill/model-routing/test/safety behavior remains authoritative elsewhere. This implementation therefore does not claim that every supported decision kind has been probabilistically wired into runtime control.

## Same-model audit isolation and claim discovery

Before asking the model to audit, `/prepare-audit` reconstructs the observable trajectory from backend capture, builds cache integrity, discovers bounded high-impact claims from the visible final answer, resolves evidence, and decides whether deep audit work is justified. Claim discovery does not create proof; it ensures that an obvious final-answer assertion such as a performance claim cannot bypass evidence merely because the client did not predeclare a `claims` array.

The generative audit then runs without thread/session recovery and with rolling context, tools/thinking disabled at the request layer, and `X-Helix-Background-Audit: 1`. The backend treats that header as an internal no-tools execution boundary, so a process-level `--enable-tools` override cannot inflate the audit prompt with tool schemas or cause memory/tool re-retrieval.

Hermes compares the model recommendation with objective evidence/cache state. If Hermes selects an adaptation different from the model's recommendation, `self_assessment_disagreement=true` records that disagreement; the model recommendation remains advisory.

## Counterfactual compression

Helix stores a versioned `CounterfactualTrajectoryCandidate` beside the actual trajectory. It records kept actions, estimated savings, source evidence and source quality, but shortness does not establish equivalent outcome quality. No client/model field can mark a candidate equivalent. Until a backend-owned replay/equivalence verifier is implemented, all such candidates remain `UNVERIFIED` and cannot become efficiency-training pairs.

The versioned trajectory envelope also separates legacy `verified` (the existing pipeline's completed/full tool-loop bit) from `objective_verified` (backend-resolved objective outcome proof). A completed turn therefore cannot be mistaken for objectively verified success merely because its tool loop finished cleanly.

## Packaging durability contract

The managed backend compatibility marker is:

```text
HELIX_HARNESS_BACKEND_CONTRACT = "helix.adaptive.backend.v1"
```

The macOS bundle stages a dependency-closed runtime overlay rather than trusting the upstream package version. The last packaged/installed overlay baseline contained 640 runtime files with SHA-256 `09b8dd171c857bea0184b22ef663cba21655101b6fc44e53ccdcd8009e787f28` and excluded tests, bytecode/cache directories, and backend `requirements/`. Preflight fingerprints Helix-critical files and probes the contract, `require_tool_access`, and `run_closed_loop`; a same-version/upstream overwrite cannot remain hidden behind a cached Ready verdict.

The updater and installer both have explicit bundled-overlay reapplication paths. See the dated acceptance document for what has and has not been exercised live.

The 2026-09-17 adaptive-cycle completion pass was validated from repository source against the managed Python environment. The existing installed `Helix Harness v2.app` and managed overlay were **not rebuilt from this completion source** during that pass. Source/GitHub conformance must therefore not be described as packaged-app parity until a later packaging rebuild re-stages and fingerprints the new backend/frontend files.
