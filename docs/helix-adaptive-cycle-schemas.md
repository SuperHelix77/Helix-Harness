# Helix Adaptive Cycle Schemas and Trust Boundaries

Date: 2026-09-17

Primary adaptive schema version: `helix.adaptive.v1`
Closed-loop response: `helix.closed-loop.v1`
Verified training target: `helix.training-target-receipt.v1`
Managed backend contract: `helix.adaptive.backend.v1`
Trajectory wire record: `helix.trajectory.v1`
Counterfactual candidate: `helix.counterfactual.v1`
Tool-control event: `helix.tool-control.v1`
Audit preparation: `helix.audit-preparation.v1`
Audit input: `helix.audit-input.v1`
Decision calibration: `helix.decision-calibration.v1`

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

### `TrajectoryRecord`

Version: `helix.trajectory.v1`.

Fields: `trajectory_id`, `objective`, `presented_context`, `tool_steps`, `control_events`, `final_result`, `acceptance_criteria`, `telemetry`, `verified`, `objective_verified`, `latency_ms`, `prompt_tokens`, `completion_tokens`, `model_id`.

This is the persistence/audit wire envelope for an internal `Trajectory`. It intentionally has **no reasoning field**. `verified` preserves the legacy pipeline meaning that the tool-loop/critic completed fully; it is not objective proof. `objective_verified` is the separate backend-resolved outcome-verification bit used by evidence/quality logic. Internal execution code keeps using the established dataclass so the wire-versioning requirement does not introduce a hot-path compatibility dependency.

### `ToolControlEvent`

Version: `helix.tool-control.v1`.

Fields: `action`, `tool_name`, `arguments`, `reason`, `equivalent_to`, `failed_attempts`, `progress`, `provenance`.

This record means the runtime observed a model-requested action and prevented it **before execution**. It is deliberately separate from `ToolStep`: an exact/semantic duplicate, repeated unchanged failure, disabled call, forced-choice mismatch, or spent one-shot tool must never be persisted as though the tool actually ran. Arguments/reason/progress are bounded, and observer failure is swallowed so this telemetry cannot become an execution dependency.

### `CacheDisruption`

Fields: `cause`, `action`, `necessary`, `estimated_cost_tokens`, `evidence`, `provenance`.

Causes are typed; the report distinguishes necessary, user-caused, tool-caused, harness-caused, and model-caused effects.

### `CacheIntegrityReport`

Fields:

`schema_version`, `prompt_tokens`, `stable_prefix_tokens`, `newly_evaluated_tokens`, `cached_tokens`, `cache_reuse_ratio`, `kv_cache_resets`, `context_compactions`, `prompt_reconstructions`, `system_prompt_changes`, `tool_schema_changes`, `context_insertions`, `repeated_context_insertions`, `prefill_ms`, `decode_ms`, `ttft_ms`, `context_reorders`, `speculative_requested`, `speculative_engaged`, `accepted_drafts`, `rejected_drafts`, `runtime_config`, `disruptions`, `observable_fields`, `telemetry_provenance`, `unavailable_fields`.

A field value of zero is not treated as a measurement unless the runtime actually exposed that field; `observable_fields`, `unavailable_fields`, and per-field `telemetry_provenance` record the distinction. The frontend uses server `cached_tokens` as the stable-prefix reuse observation when a separate runtime stable-prefix counter is unavailable; provenance keeps that distinction explicit.

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

`DEEP_SELF_AUDIT` is evaluated before the generative audit. Existing reliability behavior is the fallback: any controller/preparation failure means **run the deep audit**. Failed tools, explicit acceptance criteria, high-impact claims, and typed verification evidence can deterministically force a deep audit even when the advisory probability says skip.

Retrospective outcomes are persisted only when there is an objective observable label. Calibration reporting includes labelled sample count, Brier score, probability bins, and a minimum-sample maturity indicator. `calibrated` remains false until a separate held-out calibration study justifies that claim.

### `DecisionCalibrationSummary`

Version: `helix.decision-calibration.v1`.

Fields: `labelled_decisions`, `brier_score`, `calibration_measured`, `measurement_mature`, `minimum_labels_for_maturity`, `bins`, `calibrated`. Each populated reliability bin contains `lower`, `upper`, `count`, `mean_probability`, and `observed_frequency`.

The current maturity threshold is 50 objective labels. A Brier score or reliability bin is a measurement, not a calibration guarantee: `calibrated` remains false until a dedicated held-out calibration study exists.

### `AuditPreparation`

Version: `helix.audit-preparation.v1`.

Fields: `available`, `perform_deep_audit`, `decision`, `artifacts`; fail-open preparation errors additionally return `fail_open=true` and a bounded `error`. `decision` is the pre-generative typed `DEEP_SELF_AUDIT` decision. Deterministic reliability conditions may force `perform_deep_audit=true` regardless of the advisory probability.

### `AuditInput`

Version: `helix.audit-input.v1`.

The backend-resolved observable bundle contains the versioned trajectory envelope, objective, presented context, bounded tool steps, bounded tool-control events, edits, typed tests and benchmarks, final result, acceptance criteria, cache-integrity report, backend-resolved evidence, objective outcome evidence, prompt/completion/latency values, the pre-audit decision, and deterministic deep-audit force reasons. Hidden/private reasoning is intentionally absent.

The frontend self-audit parser accepts the canonical closed tag. It also tolerates one narrow local-model failure mode: an opening `<helix-self-audit>` followed by exactly one complete JSON object and whitespace with no closing tag. Partial JSON and trailing prose remain invalid. Model-authored top-level string arrays are capped at four items, claims at eight objects, and each claim evidence array at four strings; non-string entries are dropped rather than being promoted into typed evidence.

### `CounterfactualTrajectoryCandidate`

Version: `helix.counterfactual.v1`.

Fields: `candidate_id`, `source_trajectory_id`, `provenance`, `proposed_actions`, `kept_indices`, `same_result_target`, `actual_tool_calls`, `proposed_tool_calls`, `estimated_tool_calls_saved`, `estimated_tokens_saved`, `confidence`, `equivalence_status`, `equivalence_verified`, `evidence_ids`, `source_quality`, `training_pair_eligible`.

The candidate is a compression hypothesis, not ground truth. There is currently no backend replay/equivalence receipt, therefore candidates remain `UNVERIFIED` and `training_pair_eligible=false` even if model/client telemetry attempts to assert equivalence. A future verifier must introduce backend-owned typed provenance before Hermes may admit efficiency-training pairs.

### `AdaptationDecision`

Fields: `schema_version`, `action`, `reason`, `qlora_eligible`, `recurrence_count`, `pattern_fingerprint`, `source_trajectory_ids`, `evidence_ids`, `rejected_claim_ids`, `self_assessment_disagreement`, `advisory_only`, `counterfactual_candidate_id`, `counterfactual_equivalence_status`, `efficiency_training_eligible`.

`qlora_eligible=true` is impossible without recurrence/evidence/behavior gates and a valid verified-target receipt.

### `VerifiedTrainingTargetReceipt`

Fields: `receipt_id`, `trajectory_id`, `target`, `source`, `source_ref`, `evidence_refs`, `verifier_ref`, `target_sha256`, `receipt_sha256`, `created_at_ms`, `source_thread_id`, `schema_version`, `provenance`, `_issuer_token`.

Allowed target sources are `human_correction` and `objective_correction`. Model self-audit is not an allowed source. Objective correction requires backend verifier evidence. Receipt hashes bind the target and provenance to prevent later substitution.

### `ClosedLoopResult`

Version: `helix.closed-loop.v1`.

Fields: `trajectory_id`, `cache_integrity`, `evidence`, `self_audit`, `quality`, `adaptation`, `trajectory`, `counterfactual_candidate`, `shadow_decisions`, `decision_outcomes`, `decision_calibration`, `pattern_fingerprint`, `controller_overhead_ms`, `ledger_storage_bytes`.

The result is post-task control-plane output. None of its advisory fields can retroactively change the already completed user-facing result. QLoRA staging remains gated by the separately verified training-target and recurrence policy.

## Trust hierarchy

Highest authority is explicit human correction and backend-owned objective verification. Runtime request-scoped measurements follow. Ordinary tool outputs are observable but not automatically proof. Same-model self-audit is advisory. Free-form model assertions are untrusted until independently resolved.

No layer below another can promote itself by changing a string field, tool name, claimed status, recommendation, or confidence value.
