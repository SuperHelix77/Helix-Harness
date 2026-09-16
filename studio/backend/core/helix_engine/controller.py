# SPDX-License-Identifier: AGPL-3.0-only
"""Post-task closed loop: observe → audit → verify → Hermes → adaptation candidate."""

from __future__ import annotations

import time
import hashlib
import json
from typing import Any

from .audit import fallback_self_audit, parse_self_audit
from .cache_integrity import build_cache_integrity_report
from .deficits import classify_step_deficits
from .decision_controller import shadow_decision
from .evidence import build_evidence_claims
from .hermes import adjudicate
from .ledger import append_record, pattern_fingerprint, recurrence_sources, storage_bytes
from .quality import build_quality_vector
from .schemas import DecisionKind
from .training_targets import validated_training_target_receipt
from .trajectory import Trajectory


def run_closed_loop(
    traj: Trajectory,
    *,
    self_audit: dict[str, Any] | None = None,
    claims: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    telemetry = traj.extras.get("telemetry", {}) if isinstance(traj.extras, dict) else {}
    model_id = str(traj.extras.get("model_id") or "") if isinstance(traj.extras, dict) else ""
    cache = build_cache_integrity_report(telemetry if isinstance(telemetry, dict) else {}, traj.steps)
    evidence = build_evidence_claims(traj, cache, claims)
    audit = parse_self_audit(self_audit, model_id=model_id)
    if audit is None:
        audit = fallback_self_audit(traj, cache, evidence, model_id=model_id)
    quality = build_quality_vector(traj, cache, evidence, audit)

    labels = classify_step_deficits(traj)
    concrete_pattern: list[str] = []
    for step in traj.steps:
        if step.useful_hint == "redundant":
            concrete_pattern.append(f"redundant_tool:{step.name}")
        if step.error:
            error_family = step.error.split(":", 1)[0].strip().lower()
            concrete_pattern.append(f"tool_failure:{step.name}:{error_family}")
    for item in cache.disruptions:
        if not item.necessary:
            concrete_pattern.append(f"cache:{item.cause.value}:{item.action}")
    # Self-audit and legacy correction prose annotate the record but never create
    # recurrence by themselves.  The legacy Correction schema has no independent
    # verifier bit, so counting it here would turn model/user prose into objective
    # repeated-behavior evidence.
    if concrete_pattern and model_id:
        concrete_pattern.append(f"model:{model_id}")
    fingerprint = pattern_fingerprint(concrete_pattern)
    trajectory_id = str(traj.extras.get("trajectory_id") or "") if isinstance(traj.extras, dict) else ""
    if fingerprint != "none":
        recurrence, recurrence_ids = recurrence_sources(fingerprint, trajectory_id)
    else:
        recurrence, recurrence_ids = 1, [trajectory_id] if trajectory_id else []
    adaptation = adjudicate(traj, audit, evidence, cache, quality, recurrence=recurrence)
    adaptation.pattern_fingerprint = fingerprint
    adaptation.source_trajectory_ids = recurrence_ids

    extras = traj.extras if isinstance(traj.extras, dict) else {}
    target_receipt = validated_training_target_receipt(
        extras.get("verified_training_target_receipt"),
        trajectory_id=trajectory_id,
    )
    if adaptation.qlora_eligible and target_receipt is None:
        adaptation.qlora_eligible = False
        adaptation.reason += "; autonomous training blocked because the corrected-target receipt is invalid"

    if adaptation.qlora_eligible and target_receipt is not None:
        evidence_payload = [item.to_dict() for item in evidence]
        evidence_digest = hashlib.sha256(
            json.dumps(evidence_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        admission_receipt = {
            "schema_version": "helix.qlora-admission.v1",
            "trajectory_id": trajectory_id,
            "source_trajectory_ids": recurrence_ids,
            "model_id": model_id,
            "pattern_fingerprint": fingerprint,
            "recurrence_count": recurrence,
            "evidence_ids": [item.claim_id for item in evidence],
            "evidence_sha256": evidence_digest,
            "training_target_verified": True,
            "training_target_receipt": target_receipt.metadata(),
        }
        if not append_record("qlora-admissions", admission_receipt):
            adaptation.qlora_eligible = False
            adaptation.reason += "; autonomous training blocked because provenance receipt could not be persisted"

    decision_features = {
        "missing_evidence_count": sum(len(item.missing_evidence) for item in evidence),
        "contradiction_count": sum(bool(item.contradicting_evidence) for item in evidence),
        "has_evidence": any(item.supporting_evidence for item in evidence),
        "meaningful_task": bool(traj.steps or traj.final_result),
        "high_impact_claim": any("speed" in item.claim.lower() or "5x" in item.claim.lower() for item in evidence),
        "recurrence_count": recurrence,
        "redundant_tool_calls": sum(step.useful_hint == "redundant" for step in traj.steps),
        "cache_reuse_ratio": cache.cache_reuse_ratio,
        "behavior_gap": bool(labels),
        "evidence_supported": not adaptation.rejected_claim_ids,
        "accepted_drafts": cache.accepted_drafts,
        "sidecar_ok": bool(telemetry.get("speculative_sidecar_ok")) if isinstance(telemetry, dict) else False,
    }
    shadow = {}
    for kind in (
        DecisionKind.EVIDENCE_SUFFICIENT,
        DecisionKind.DEEP_SELF_AUDIT,
        DecisionKind.QLORA_CANDIDATE,
    ):
        decision = shadow_decision(
            kind,
            decision_features,
            decision_id=f"{trajectory_id}:{kind.value}",
            trajectory_id=trajectory_id,
        )
        shadow[kind.value] = decision.to_dict()
        append_record("decisions", decision.to_dict())

    append_record("cache-integrity", {"trajectory_id": trajectory_id, **cache.to_dict()})
    append_record("evidence", {"trajectory_id": trajectory_id, "claims": [item.to_dict() for item in evidence]})
    append_record("self-audits", {"trajectory_id": trajectory_id, **audit.to_dict()})
    append_record("adaptations", {"trajectory_id": trajectory_id, **adaptation.to_dict()})
    append_record("quality", {"trajectory_id": trajectory_id, **quality.to_dict()})
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    return {
        "schema_version": "helix.closed-loop.v1",
        "trajectory_id": trajectory_id,
        "cache_integrity": cache.to_dict(),
        "evidence": [item.to_dict() for item in evidence],
        "self_audit": audit.to_dict(),
        "quality": quality.to_dict(),
        "adaptation": adaptation.to_dict(),
        "shadow_decisions": shadow,
        "pattern_fingerprint": fingerprint,
        "controller_overhead_ms": elapsed_ms,
        "ledger_storage_bytes": storage_bytes(),
    }
