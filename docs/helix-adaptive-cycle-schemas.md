# Helix Adaptive Cycle Schemas and Trust Boundaries

Date: 2026-09-16

Primary adaptive schema version: `helix.adaptive.v1`
Closed-loop response: `helix.closed-loop.v1`
Verified training target: `helix.training-target-receipt.v1`
Managed backend contract: `helix.adaptive.backend.v1`

## Core records

### `ToolVerificationReceipt`

Fields: `kind`, `status`, `provenance`, `detail`, `claim`, `subject`.

Admissibility rules:

- must be an actual typed `ToolVerificationReceipt` produced at backend capture;
- `provenance` must be `backend_tool_capture`;
- status must be typed `PASSED`;
- claim support requires a non-empty backend-bound `claim` and exact normalized claim equality;
- model/tool arguments and free-form command names are never authority.

### `ToolStep`

Fields: `name`, `arguments`, `result`, `useful_hint`, `error`, `retry`, `verification`.

A result is observable context. It is not automatically objective verification.

### `Trajectory`

Fields: `prompt_state`, `retrieved_context`, `reasoning`, `steps`, `user_corrections`, `final_result`, `latency_ms`, `prompt_tokens`, `completion_tokens`, `verified`, `semantic_turns`, `holdout_passed`, `allow_qlora`, `frequent_behavior`, `one_off_fact`, `regression_passed`, `dataset_hash`, `adapter_version`, `extras`.

The `reasoning` field must contain only explicitly supplied/observable reasoning artifacts. Helix does not capture hidden chain of thought.

### `CacheDisruption`

Fields: `cause`, `action`, `necessary`, `estimated_cost_tokens`, `evidence`.

Causes are typed; the report distinguishes necessary, user-caused, tool-caused, harness-caused, and model-caused effects.

### `CacheIntegrityReport`

Fields:

`schema_version`, `prompt_tokens`, `stable_prefix_tokens`, `newly_evaluated_tokens`, `cached_tokens`, `cache_reuse_ratio`, `kv_cache_resets`, `context_compactions`, `prompt_reconstructions`, `system_prompt_changes`, `tool_schema_changes`, `context_insertions`, `repeated_context_insertions`, `prefill_ms`, `decode_ms`, `ttft_ms`, `context_reorders`, `speculative_requested`, `speculative_engaged`, `accepted_drafts`, `rejected_drafts`, `runtime_config`, `disruptions`, `observable_fields`.

A field value of zero is not treated as a measurement unless the runtime actually exposed that field; `observable_fields` records the distinction.

### `EvidenceClaim`

Fields: `schema_version`, `claim_id`, `claim`, `supporting_evidence`, `reported_supporting_evidence`, `evidence_refs`, `contradicting_evidence`, `missing_evidence`, `confidence`, `status`.

Status is recomputed from backend-resolved evidence. Model-supplied `status` cannot promote unsupported evidence. Relevant states include `SUPPORTED`, `PARTIALLY_SUPPORTED`, `CONTRADICTED`, `UNVERIFIED`, and `NOT_APPLICABLE`.

Evidence reference grammar is intentionally bounded. Frontend self-audit parsing discards evidence references the backend cannot resolve.

### `SelfAuditReport`

Fields:

`schema_version`, `objective`, `achieved`, `contributing_actions`, `unnecessary_actions`, `failures`, `retries`, `rediscovered_information`, `excess_retrieval`, `avoidable_cache_disruption`, `tool_selection_correct`, `expensive_resource_misuse`, `overclaimed_claims`, `stopped_too_early`, `continued_too_long`, `better_trajectory`, `reusable_lessons`, `likely_behavioral_pattern`, `recommendation`, `recommendation_reason`, `self_assessment_confidence`, `model_id`, `source`.

The report is advisory model output, not training truth.

### `QualityVector`

Fields: `schema_version`, `task_quality`, `evidentiary_completeness`, `computational_efficiency`, `self_assessment_calibration`, `raw_metrics`.

Q/E/C/S must not be interpreted as a model-improvement score. It is per-trajectory controller input.

### `DecisionRecord`

Fields: `schema_version`, `decision_id`, `trajectory_id`, `decision`, `probability`, `confidence`, `choice`, `advisory_only`, `policy_version`, `evidence_features`, `eventual_outcome`, `retrospective_usefulness`.

Probabilistic decisions remain advisory. Deterministic reliability/safety gates win.

### `AdaptationDecision`

Fields: `schema_version`, `action`, `reason`, `qlora_eligible`, `recurrence_count`, `pattern_fingerprint`, `source_trajectory_ids`, `evidence_ids`, `rejected_claim_ids`, `self_assessment_disagreement`, `advisory_only`.

`qlora_eligible=true` is impossible without recurrence/evidence/behavior gates and a valid verified-target receipt.

### `VerifiedTrainingTargetReceipt`

Fields: `receipt_id`, `trajectory_id`, `target`, `source`, `source_ref`, `evidence_refs`, `verifier_ref`, `target_sha256`, `receipt_sha256`, `created_at_ms`, `source_thread_id`, `schema_version`, `provenance`, `_issuer_token`.

Allowed target sources are `human_correction` and `objective_correction`. Model self-audit is not an allowed source. Objective correction requires backend verifier evidence. Receipt hashes bind the target and provenance to prevent later substitution.

## Trust hierarchy

Highest authority is explicit human correction and backend-owned objective verification. Runtime request-scoped measurements follow. Ordinary tool outputs are observable but not automatically proof. Same-model self-audit is advisory. Free-form model assertions are untrusted until independently resolved.

No layer below another can promote itself by changing a string field, tool name, claimed status, recommendation, or confidence value.
