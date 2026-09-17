# Helix Harness — Adaptive Intelligence Cycle Conformance Audit

Date: 2026-09-17
Audited HEAD: `fe105e5` (`fix: restore Helix engine UI and glass branding`)
Adaptive-cycle implementation commits: `174e429` baseline, `158d6bf` hardening, `1523ed4` completed self-evolution loop

> **Post-audit completion update — 2026-09-17:** the nine concrete gaps identified by this audit have now been implemented in the completion pass described by `docs/HELIX_ADAPTIVE_CYCLE_COMPLETION_CHECKPOINT_20260917.md`. The historical audit below is preserved as the evidence that drove the work; it is not rewritten to pretend those features existed at the audited HEAD. Current completion evidence is in `docs/helix-adaptive-cycle-acceptance-20260917.md`.
>
> Closure highlights: live bounded context fingerprints/provenance; pre-generative `DEEP_SELF_AUDIT` allocation with fail-open old-behavior fallback; objective retrospective decision labels plus measured Brier/reliability bins; backend-resolved audit artifacts; `helix.trajectory.v1`; `helix.counterfactual.v1`; append-only trajectory/counterfactual provenance; a committed SSE acceptance collector; matched hot-path/control-plane overhead benchmarks; and a dedicated injected decision-controller failure test. The controller is **measured but not declared calibrated**. Counterfactuals are **candidates, not verified equivalents**; no client/model field can authorize equivalence or efficiency training. No 27B/general model improvement is claimed.

## Executive verdict

**The architectural core is implemented and substantially matches the original plan, but the specification is not fully fulfilled.**

The strongest parts are the fail-open control plane, typed evidence/self-audit/adaptation records, same-model post-task audit, Hermes adjudication, provenance-preserving QLoRA admission, recurrence gates, objective claim checking, and the real controlled-discrepancy acceptance run.

The remaining gaps are not cosmetic:

1. **Cache/context observation is only partially wired to live telemetry.** The schema accepts most requested fields, but several are not actually emitted by the normal chat path.
2. **The Jev-inspired decision controller is mostly a shadow/post-task API, not yet the cheap pre-action allocator described in the target architecture.**
3. **Decision calibration is not empirically measured in current data.** The controller can store outcomes and compute Brier score, but the current ledger has zero labelled decisions and explicitly reports `calibrated=false`.
4. **The model self-audit input is narrower than the requested artifact set.** The live frontend path sends objective/final result/telemetry/tool steps, but not structured diffs, edits, tests, benchmarks, presented context, or acceptance criteria.
5. **Counterfactual compression is implemented but not yet a first-class versioned candidate with confidence/provenance/equivalent-quality validation.**
6. **Performance acceptance is incomplete.** Self-audit cost, controller cost, wall times, and ledger growth were measured, but matched-workload telemetry overhead, total-token capture for the real acceptance run, and full cache timing metrics remain unmeasured.
7. **`Trajectory` itself is not a versioned wire record**, even though the deliverable asked for versioned trajectory/self-audit/evidence/adaptation records.
8. The required focused test for **decision-controller failure falling back to existing behavior** is only covered indirectly by the outer closed-loop fail-open test; there is no dedicated `shadow_decision` failure-injection test.

No evidence was found that these gaps compromise ordinary chat/tool reliability. The adaptive/QLOra safety boundary is materially stronger than the original baseline.

## Evidence base

Current-tree validation performed for this audit:

- Backend adaptive/ingest/pipeline/engine/self-training set: **74/74 passed**.
- Frontend Hermes/learning/Engine/theme focused set: **25/25 passed**.
- Frontend TypeScript typecheck at current HEAD: **PASS**.
- Current default Helix ledger inspection:
  - decision records: 6
  - labelled decisions: **0**
  - empirical Brier score: `None`
  - `calibrated: false`
  - current ledger storage inspected from CLI context: 9,300 bytes

Historical real-run evidence remains in `docs/helix-adaptive-cycle-acceptance-20260916.md`.

## Requirement-by-requirement audit

| Section | Requirement | Status | Current evidence | Remaining gap |
|---|---|---|---|---|
| 0 | Fail-open invariant | **PASS** | `core/helix_engine/pipeline.py`, `core/inference/tools.py`; failure/cancellation tests | None material found |
| 1 | Ingest current state / checkpoint first | **PASS** | `docs/HELIX_ADAPTIVE_CYCLE_CHECKPOINT.md`; baseline `174e429` | Historical checkpoint remains valid |
| 2 | Empryo-inspired cache integrity controller | **PARTIAL** | `cache_integrity.py`, `CacheIntegrityReport`, live prompt/cached/prefill/decode/TTFT/spec telemetry | Many requested observables are schema-only/not emitted live |
| 3 | Jev-inspired typed decision controller | **PARTIAL** | `decision_controller.py`, typed enums, advisory API, probability/confidence/outcome fields | Not broadly wired into pre-action resource allocation; no empirical calibration |
| 4 | Evidence / grounding controller | **PASS** | `evidence.py`; typed verifier receipts; `SUPPORTED/UNVERIFIED/...` | Speed telemetry trust is specialized, but unsupported claims fail closed |
| 5 | Same-model self-audit | **PARTIAL** | `scheduleSelfReflect`, `helixSelfAuditPrompt`, `SelfAuditReport`, real Qwen audit | Live audit artifact set is narrower than requested |
| 6 | Hermes adaptation controller | **PASS** | `hermes.py`, `controller.py`, `ingest.py` | No direct weight control by model audit |
| 7 | Adaptation hierarchy | **PASS** | `AdaptationKind`; Hermes runtime/memory/skill/QLoRA/capability routing | Manual legacy self-QLoRA remains separate by explicit user action |
| 8 | Q/E/C/S quality vector | **PASS** | `quality.py`; independent dimensions + raw metrics | No aggregate longitudinal self-audit calibration summary |
| 9 | Counterfactual trajectory compression | **PARTIAL** | `compress.py`; `SelfAuditReport.better_trajectory` | No versioned candidate record/confidence/equivalent-quality validation |
| 10 | Closed loop | **PASS** | `controller.py`: observe → audit → evidence → Hermes → adaptation | Decision controller still post-task/shadow for most choices |
| 11 | Self-poisoning safeguards | **PASS** | recurrence ledger, typed verifier, verified training-target receipt, immutable dataset snapshot | No autonomous single-event training path found |
| 12 | Performance requirements / overhead | **PARTIAL** | bounded capture, fire-and-forget audit, measured controller/audit/storage costs | Matched telemetry overhead and full token/cache metrics unmeasured |
| 13 | Required focused tests | **MOSTLY PASS** | 17 requirements have direct/near-direct tests | Dedicated decision-controller crash/fallback test missing |
| 14 | Real acceptance experiment | **PARTIAL** | real Qwen task, deliberate unsupported 5x claim, duplicate retrieval, Hermes rejection | Required total tokens/cache metrics/telemetry overhead were not retained |
| 15 | Do not overclaim | **PASS** | acceptance docs explicitly separate implemented/tested/measured/real-run/improvement | Continue preserving this discipline |
| 16 | Deliverables | **PARTIAL** | implementation, tests, docs, schemas, acceptance doc, commits, handoff exist | trajectory schema versioning + complete overhead evidence + committed collector missing |

## 0. Non-negotiable fail-open invariant

This is fulfilled.

`studio/backend/core/inference/tools.py` wraps the real tool execution boundary. Capture is best effort; it catches its own failures and re-raises the exact original exception object. Successful return objects are returned unchanged. `asyncio.CancelledError` remains cancellation.

`studio/backend/core/helix_engine/pipeline.py` contains the adaptive loop behind a `try/except`; if the controller fails, the established trajectory pipeline returns a fail-open adaptive result rather than breaking the completed task.

Direct tests include:

- `test_closed_loop_failure_cannot_break_legacy_pipeline`
- `test_cache_controller_failure_is_fail_open`
- `test_hermes_failure_cannot_alter_completed_result`
- `test_failed_tool_call_is_recorded_and_same_exception_is_reraised`
- `test_successful_tool_call_preserves_exact_return_object`
- `test_cancellation_remains_cancellation_and_is_observed`
- `test_capture_failure_cannot_change_tool_return`

## 1. State ingestion / checkpoint

Fulfilled. `docs/HELIX_ADAPTIVE_CYCLE_CHECKPOINT.md` was written before the large implementation edits and records the authoritative baseline and pre-change regression floor. The baseline was then frozen as `174e429`.

## 2. Cache/context integrity controller

### Implemented

`CacheIntegrityReport` includes:

- prompt tokens
- stable-prefix tokens
- newly evaluated tokens
- cached tokens / reuse ratio
- KV resets
- compactions
- prompt reconstructions
- system-prompt changes
- tool-schema changes
- context insertions / repeated insertions
- prefill/decode/TTFT
- context reorder count
- speculative requested/engaged state
- accepted/rejected drafts
- runtime config
- typed disruption attribution (`NECESSARY`, `USER_CAUSED`, `TOOL_CAUSED`, `HARNESS_CAUSED`, `MODEL_CAUSED`, `UNKNOWN`)
- `observable_fields` so an unobserved zero is not automatically treated as a measurement

The controller also detects repeated identical tool reads and treats `search_memory` + `search_conversation` with identical arguments/results as an equivalent retrieval path.

### Live wiring gap

The normal chat path in `chat-adapter.ts` currently emits:

- prompt/completion tokens
- cached tokens
- prefill/decode/TTFT
- wall latency
- context compaction/truncation
- speculative mode/requested/engaged/accepted/rejected/counter scope
- model/backend/speculative runtime config

It **does not currently emit** most of the following requested observations:

- explicit stable-prefix length (the report falls back to cached tokens)
- explicit newly-prefilled tokens (derived as prompt minus cached)
- KV reset/invalidation events
- prompt reconstruction events
- system-prompt change events
- tool-schema change events
- memory/skill/context insertion events
- context reordering/movement events
- explicit repeated file-output reinsertion events beyond duplicate tool-step inference

Thus the report schema is broader than the real instrumentation. This section is **PARTIAL**, not complete.

## 3. Typed calibrated decision controller

### Implemented

`DecisionKind` covers the planned decision family, including memory, conversation search, skill retrieval/creation, context compaction/preservation, repository search, evidence sufficiency, further tests, independent verification, retry, model escalation, local/Astra, speculative decoding, deep self-audit, reusable procedure, QLoRA candidacy, and noise.

`DecisionRecord` carries:

- predicted probability
- confidence
- typed choice
- advisory-only flag
- policy version
- evidence/features
- eventual outcome
- retrospective usefulness

Malformed decision names are rejected. The controller remains advisory.

Routes exist for:

- `/api/helix-engine/decision`
- `/api/helix-engine/decision-outcome`
- `/api/helix-engine/decision-calibration`

### What is not fulfilled

The closed loop currently creates only three automatic shadow decisions after a task:

- `EVIDENCE_SUFFICIENT`
- `DEEP_SELF_AUDIT`
- `QLORA_CANDIDATE`

The controller is **not yet the pre-action allocator** deciding whether to retrieve memory, search the repository, use a skill, run another test, escalate model, use Astra, or enable speculation. Those choices remain driven by existing deterministic/chat logic.

The `DEEP_SELF_AUDIT` decision is also computed **after** the same-model self-audit has already run, so it does not yet save generative audit work.

### Calibration status

The code can compute Brier score when outcomes are labelled. Current inspected ledger state reports:

```text
labelled_decisions = 0
brier_score = None
calibrated = false
```

Therefore the controller must **not** be described as calibrated. It is a typed probabilistic shadow controller with calibration infrastructure.

## 4. Evidence / grounding controller

Fulfilled to the intended safety standard.

`EvidenceClaim` represents support, contradictions, missing evidence, confidence, and status. Model-reported support cannot promote a claim unless it cites backend-resolved evidence. Backend tool-verification authority is typed and claim-bound. Free-form command names cannot mint verifier authority. Raw tool results are context, not proof.

The controlled speed-claim path deliberately distinguishes wiring from measured performance. A 5x claim remains unverified without a real speed ratio and accepted-draft evidence.

## 5. Model self-audit

### Strongly implemented

The normal local-chat path schedules a post-task fire-and-forget audit. The request uses the task's checkpoint and preserves adapter state. The prompt explicitly forbids hidden chain-of-thought reconstruction and asks for the planned structured audit questions. The parser returns a versioned `SelfAuditReport`.

The acceptance document records a real Qwen3.8-27B run in which the same loaded model produced the self-audit and identified the unsupported performance claim.

### Artifact coverage gap

`audit.py` defines a richer `observable_audit_payload`, including presented context, acceptance criteria, cache integrity, and evidence. However that helper is currently unused by the production frontend audit path.

The live `scheduleSelfReflect` payload contains only:

- objective/user text
- final assistant result
- telemetry
- up to 16 tool steps

The requested explicit structured artifacts are not all supplied:

- presented prompts/context
- edits/diffs
- tests as a distinct artifact type
- benchmarks as a distinct artifact type
- acceptance criteria
- evidence claims before the model audit

Some of these may appear indirectly inside tool output, but that is weaker than the specification. Therefore this section is **PARTIAL**.

## 6. Hermes adjudication

Fulfilled.

Hermes compares audit recommendations to objective evidence and quality, records self-assessment disagreement, and keeps adaptation advisory. Unsupported or contradicted claims are retained in `rejected_claim_ids`.

A QLoRA recommendation requires recurrence, a behavioral-pattern flag, adequate evidence, and a verified corrected-target receipt. The model audit cannot authorize training.

## 7. Adaptation hierarchy

Fulfilled for the adaptive loop.

Implemented actions are exactly the intended hierarchy:

- `IGNORE`
- `RUNTIME_POLICY`
- `MEMORY`
- `SKILL`
- `QLORA_CANDIDATE`
- `CAPABILITY_GAP`

Mechanical cache waste is routed to runtime policy before QLoRA. Reusable procedures are preferred as skills. QLoRA recurrence threshold is currently 3 objective occurrences plus independent target verification.

Legacy manual Self-QLoRA remains independently available by explicit user action and can train on the ordinary bounded example collection. Autonomous adaptive training uses only `eligibleForTraining=true` Hermes-qualified examples. This preserves the existing product workflow while preventing the adaptive loop from silently training on raw self-criticism.

## 8. Q / E / C / S quality vector

Fulfilled.

`quality.py` keeps four separate dimensions and raw metrics. Cache misses affect computational efficiency but do not lower task quality or evidentiary completeness. The code does not collapse Q/E/C/S into one opaque score.

Self-assessment calibration is computed per trajectory when an objective outcome label exists. There is no dedicated longitudinal aggregate for model self-assessment calibration yet; quality records make such an aggregate possible later.

## 9. Counterfactual trajectory compression

Partially fulfilled.

There are two mechanisms:

1. deterministic `compress_counterfactual()` retaining credited steps;
2. model-authored `SelfAuditReport.better_trajectory`.

What is missing relative to the plan:

- no dedicated versioned `CounterfactualTrajectoryCandidate` record
- no candidate confidence
- no explicit evidence/provenance linkage for the compressed sequence
- no verifier that the compressed sequence is likely to preserve equivalent verified quality
- no ledger comparing actual vs proposed vs objective evidence

The current output is useful analysis, but not yet a training-grade counterfactual pair.

## 10. Closed loop

The core closed loop exists:

```text
trajectory
  → cache report
  → evidence claims
  → self-audit
  → Q/E/C/S
  → Hermes
  → shadow decisions
  → provenance ledgers
  → optional verified QLoRA admission
```

The adaptation side is closed-loop. The cheap decision controller is not yet fully placed alongside the generative model as an online allocator, so the complete target diagram is only partially realized.

## 11. Safety against self-poisoning

This is one of the strongest areas and is fulfilled.

Protections include:

- self-audit prose does not create recurrence by itself
- re-running the same trajectory cannot manufacture recurrence
- unrelated self-reported patterns do not collide into recurrence
- public/client example fields cannot spoof training qualification
- training target sources are limited to human/objective correction
- objective correction requires backend verifier provenance
- training-target receipts are hash-bound and tamper-evident
- QLoRA admission is provenance-logged
- admitted training snapshots are immutable
- autonomous policy is rechecked at queue, execution, benchmark, and recovery boundaries
- candidate promotion requires held-out score improvement and measured throughput preservation

## 12. Performance / overhead

### Implemented architecture

- hot-path tool capture is bounded
- capture failures are swallowed
- post-task audit is fire-and-forget
- audit waits for inference idle before consuming the same model
- controller records its own elapsed time
- ledgers are bounded per line and fail-open

### Measurement status

The real acceptance run measured:

- model load wall time
- baseline request wall time
- instrumented task wall time
- same-model self-audit wall time
- ingest wall time
- closed-loop controller overhead
- ledger storage growth
- unload wall time

It explicitly did **not** establish a matched baseline/instrumented overhead ratio. The temporary acceptance collector failed to retain final usage/timing metadata, so the following required measurements remained unmeasured in that run:

- prompt/completion/cached token totals
- prefill/decode/TTFT for the acceptance trajectory
- hot-path telemetry overhead
- matched equivalent-quality cache improvement

The acceptance document correctly refuses to call the difference between a 32 s baseline and a 2,439 s multi-tool task “overhead.”

## 13. Required test mapping

| # | Required proof | Current test evidence | Status |
|---|---|---|---|
| 1 | Helix failure cannot break ordinary task path | `test_closed_loop_failure_cannot_break_legacy_pipeline` | PASS |
| 2 | Recorder failure cannot break tool execution | `test_capture_failure_cannot_change_tool_return` | PASS |
| 3 | Cache telemetry failure cannot break inference/task | `test_cache_controller_failure_is_fail_open` | PASS at post-task boundary |
| 4 | Self-audit failure cannot alter result | `test_invalid_model_audit_uses_observable_fallback_without_changing_task_result` | PASS |
| 5 | Hermes failure cannot alter result | `test_hermes_failure_cannot_alter_completed_result` | PASS |
| 6 | Decision-controller failure falls back | outer closed-loop fail-open would catch it; no direct injected `shadow_decision` failure test | **PARTIAL** |
| 7 | Typed decisions reject malformed output | `test_typed_decisions_reject_unknown_values_and_are_advisory` | PASS |
| 8 | Evidence distinguishes supported/unverified | `test_evidence_distinguishes_supported_from_unverified_speed_claim` + typed-verifier tests | PASS |
| 9 | Self-audit cannot directly enqueue QLoRA | `test_single_self_audit_cannot_make_qlora_eligible` | PASS |
| 10 | One anomaly cannot trigger training | `test_ordinary_examples_do_not_autostart_qlora`, recurrence tests | PASS |
| 11 | Quality protected from cache misses | `test_cache_miss_does_not_reduce_task_quality_or_evidence_quality` | PASS |
| 12 | Failed calls recorded, original exception preserved | `test_failed_tool_call_is_recorded_and_same_exception_is_reraised` | PASS |
| 13 | Successful calls preserve exact return | `test_successful_tool_call_preserves_exact_return_object` | PASS |
| 14 | Cancellation remains cancellation | `test_cancellation_remains_cancellation_and_is_observed` | PASS |
| 15 | Cache/context telemetry provenance | `test_cache_report_records_provenance_and_avoidable_repeat` | PASS |
| 16 | Self-audit machine-readable/versioned | `test_self_audit_is_machine_readable_and_versioned` | PASS |
| 17 | Adaptation provenance retained | `test_adaptation_retains_trajectory_and_evidence_provenance` | PASS |
| 18 | Existing regressions pass | 74 backend tests + 25 focused frontend tests + current TS typecheck in this audit; broader dated suites documented | PASS for audited floor |

## 14. Acceptance experiment

A real acceptance task was run with Qwen3.8-27B. It deliberately:

- retrieved a known memory
- called both equivalent retrieval paths
- emitted an unsupported `>=5x` performance sentence
- produced a same-model structured self-audit
- ran evidence adjudication
- ran Hermes
- demonstrated duplicate retrieval as an efficiency finding
- kept QLoRA ineligible

The evidence controller marked the speed claim `UNVERIFIED`; Hermes did not accept unsupported success; no QLoRA candidate was staged.

This satisfies the **behavioral controlled-discrepancy** portion of the acceptance plan.

It does **not** completely satisfy the acceptance data-capture list because total tokens, full cache/context timing metrics, and telemetry overhead were not retained by the temporary collector.

## 15. Overclaim discipline

Fulfilled. Existing acceptance and v2 validation documents explicitly distinguish:

- implemented
- tested
- measured
- observed in a real agent run
- training candidate generated
- actual model improvement demonstrated

The current evidence still says:

- real self-audit/closed loop: **yes**
- real autonomous QLoRA mechanics: **yes**, using the 0.5B mechanism smoke
- Qwen3.8-27B held-out model improvement: **no**
- general intelligence improvement: **no**
- decision controller empirically calibrated: **no**
- cache optimization objectively improved equivalent-quality work: **no**

## 16. Deliverables audit

### Present

- implementation
- focused tests
- regression evidence
- architecture documentation
- versioned self-audit/evidence/decision/adaptation/cache/quality schemas
- measured controller/self-audit/storage costs
- real acceptance trajectory
- unresolved-blocker list
- Git commits
- restart/handoff document
- modular generative/decision separation suitable for later Q38 replacement

### Missing or incomplete

1. `Trajectory` is still a plain dataclass without an explicit schema version.
2. Counterfactual compression has no first-class versioned candidate schema.
3. The real acceptance collector was temporary; the acceptance doc explicitly calls for a committed collector retaining final usage/timing control frames.
4. Matched telemetry-overhead measurement remains absent.
5. Empirical decision calibration has not begun because no outcomes are labelled.
6. The decision controller does not yet allocate most online resources/actions.
7. Live self-audit artifact construction does not yet use the richer backend `observable_audit_payload`.

## Priority closure plan

The safest completion order is:

1. **Telemetry provenance completion** — wire the currently schema-only cache/context events from the authoritative runtime boundary.
2. **Committed acceptance collector** — retain usage/timing/spec frames and run matched workloads.
3. **Pre-action decision shadowing** — call the typed decision controller alongside existing deterministic decisions without changing behavior.
4. **Outcome labelling/calibration** — attach retrospective outcomes and begin Brier/reliability-bin measurement; keep `calibrated=false` until evidence supports otherwise.
5. **Audit artifact unification** — make the production same-model audit consume the backend-defined observable audit payload, including acceptance criteria/evidence and structured test/benchmark/diff references where available.
6. **Counterfactual candidate schema** — version, provenance-link, and verify equivalent-quality compressed trajectories before any training use.
7. **Trajectory wire versioning + dedicated decision-failure test**.

None of these should weaken the current fail-open or QLoRA provenance gates.

## Final architectural assessment

The intended principle is already visible in code:

> THE GENERATIVE MODEL PROPOSES.  
> TELEMETRY OBSERVES.  
> SELF-AUDIT INTERPRETS.  
> EVIDENCE CONSTRAINS.  
> HERMES DECIDES WHAT IS LEARNED.

The remaining work is chiefly to make **TELEMETRY OBSERVES** broader and empirically measured, and to make the cheaper **DECISION CONTROLLER** actually sit in front of routine resource-allocation decisions rather than merely shadow three post-task decisions.

Until those gaps are closed, the correct description is:

**Helix Harness has a functioning, fail-open adaptive intelligence control loop with real same-model audit and guarded adaptation, but it has not yet completed the full calibrated cheap-decision/cache-efficiency architecture specified in the original plan.**

---

## Completion closure — 2026-09-17

The verdict above is the historical verdict for the audited HEAD and is intentionally unchanged. The subsequent completion pass implemented the concrete gaps that audit identified. This section records the later state; it does not retroactively alter what existed at the audited commit.

| Former gap | Completion state | Completion evidence / boundary |
|---|---|---|
| Live cache/context observation breadth | **CLOSED FOR OBSERVABLE SIGNALS** | `helix-context-telemetry.ts` adds bounded fingerprints/counters for stable-prefix derivation, prompt/tool-catalog change, insertion/reinsertion and reorder observations; backend capture contributes tool-output insertions. Runtime facts that remain unavailable are explicitly marked unavailable rather than synthesized. |
| Cheap typed controller as a live allocator | **CLOSED TO A DELIBERATELY LIMITED LIVE SCOPE** | `DEEP_SELF_AUDIT` is now a real pre-generative allocation point. The broader typed decision family remains available/shadowed, while existing deterministic memory/search/skill/model/test/safety logic remains authoritative. No claim is made that every supported decision kind is probabilistically wired online. |
| No empirical decision outcome/calibration data | **MEASUREMENT CLOSED; CALIBRATION CLAIM OPEN BY DESIGN** | Objective retrospective labels feed `helix.decision-calibration.v1`, Brier score and reliability bins. The final isolated acceptance replay produced 3 labels and Brier `0.011133333333333335`; `measurement_mature=false`, minimum 50, and `calibrated=false`. |
| `DEEP_SELF_AUDIT` decision happened after generative audit | **CLOSED** | `/api/helix-engine/prepare-audit` builds cache/evidence first and makes the typed audit decision before any same-model audit. Failure conservatively falls back to deep audit. Failed tools, acceptance criteria, high-impact claims and typed verification can deterministically force it. |
| Self-audit input too narrow | **CLOSED** | Same-model audit consumes backend-resolved `helix.audit-input.v1`: versioned trajectory, objective/context, bounded tool steps, edits, tests, benchmarks, cache telemetry, evidence, final result, acceptance criteria and objective outcome evidence. Hidden reasoning is excluded. |
| Counterfactual compression not first-class/versioned/equivalence-gated | **CLOSED** | `helix.counterfactual.v1` is provenance-linked and stores actual/proposed actions, estimated savings, source quality/evidence and equivalence state. Current candidates remain `UNVERIFIED`, `equivalence_verified=false`, `training_pair_eligible=false` without backend-owned equivalence verification. |
| Matched telemetry/performance evidence incomplete | **CLOSED FOR IMPLEMENTATION OVERHEAD MEASUREMENT** | `docs/helix-adaptive-cycle-overhead-20260917.json` records 20,000-iteration hot-path telemetry overhead and 100-trajectory post-task control-plane/storage measurements. The real final foreground/audit collector retains token/cache/timing frames. These measurements do not prove an equivalent-quality cache optimization speedup. |
| Trajectory not a versioned wire record | **CLOSED** | `trajectory_record()` persists `helix.trajectory.v1` without hidden/private reasoning while preserving the existing internal dataclass for compatibility. |
| Dedicated decision-controller crash/fallback test absent | **CLOSED** | Focused failure injection verifies controller failure produces `helix-fallback-v1`, confidence 0 and the conservative pre-existing TAKE/deep-audit behavior without suppressing task reliability. |

Additional production defects exposed by the final acceptance were also closed rather than hidden:

1. checkpoint/thread recovery could auto-admit memory search into a background self-audit; the audit now omits thread/session recovery and forces rolling context;
2. process-level `--enable-tools` could still inject the full tool catalog despite request-level `enable_tools=false`; `X-Helix-Background-Audit: 1` is now a backend-owned no-tools boundary and the post-fix same-model audit emitted zero tool events;
3. Qwythos reproducibly returned a complete audit JSON object while omitting only `</helix-self-audit>`; the production parser now accepts only that narrowly bounded shape and still rejects partial JSON or trailing prose;
4. Hermes could override a model recommendation because of objective cache/evidence state without marking the disagreement flag; any Hermes-vs-self-audit adaptation mismatch now records `self_assessment_disagreement=true`.

Final source validation, exact real-run metrics and unresolved evidence boundaries are recorded in `docs/helix-adaptive-cycle-acceptance-20260917.md`. The completion pass does **not** establish a held-out model-quality improvement, a mature empirically calibrated decision controller, independently verified counterfactual equivalence, or a measured equivalent-quality cache-optimization speedup. At the end of the source-completion pass the pre-existing installed macOS app/managed overlay had not yet been rebuilt; a later same-day packaging follow-up rebuilt `Helix Harness v2.app` from commit `5e8af2a`, staged/applied the 639-file completion overlay, and passed packaged-app launch/health/quit/relaunch smoke. The historical statement is preserved here with that subsequent closure rather than rewritten as though packaging occurred before the source acceptance.
