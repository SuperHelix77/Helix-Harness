# SPDX-License-Identifier: AGPL-3.0-only
"""Typed model self-audit over observable artifacts only. No hidden reasoning capture."""

from __future__ import annotations

from typing import Any

from .schemas import AdaptationKind, CacheIntegrityReport, EvidenceClaim, SelfAuditReport
from .trajectory import Trajectory


def _strings(value: Any, limit: int = 32, width: int = 1_000) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item)[:width] for item in value[:limit] if str(item).strip()]


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def parse_self_audit(raw: dict[str, Any] | None, *, model_id: str = "") -> SelfAuditReport | None:
    if not isinstance(raw, dict):
        return None
    rec_raw = str(raw.get("recommendation") or "IGNORE").strip().upper().replace("HERMES_SKILL", "SKILL")
    try:
        recommendation = AdaptationKind(rec_raw)
    except ValueError:
        return None
    achieved = _optional_bool(raw.get("achieved"))
    try:
        confidence = max(0.0, min(1.0, float(raw.get("self_assessment_confidence", raw.get("confidence", 0.5)))))
    except (TypeError, ValueError):
        confidence = 0.5
    return SelfAuditReport(
        objective=str(raw.get("objective") or "")[:4_000],
        achieved=achieved,
        contributing_actions=_strings(raw.get("contributing_actions")),
        unnecessary_actions=_strings(raw.get("unnecessary_actions")),
        failures=_strings(raw.get("failures")),
        retries=_strings(raw.get("retries")),
        rediscovered_information=_strings(raw.get("rediscovered_information")),
        excess_retrieval=_strings(raw.get("excess_retrieval")),
        avoidable_cache_disruption=_strings(raw.get("avoidable_cache_disruption")),
        tool_selection_correct=_optional_bool(raw.get("tool_selection_correct")),
        expensive_resource_misuse=_strings(raw.get("expensive_resource_misuse")),
        overclaimed_claims=_strings(raw.get("overclaimed_claims")),
        stopped_too_early=bool(raw.get("stopped_too_early", False)),
        continued_too_long=bool(raw.get("continued_too_long", False)),
        better_trajectory=_strings(raw.get("better_trajectory")),
        reusable_lessons=_strings(raw.get("reusable_lessons")),
        likely_behavioral_pattern=bool(raw.get("likely_behavioral_pattern", False)),
        recommendation=recommendation,
        recommendation_reason=str(raw.get("recommendation_reason") or raw.get("reason") or "")[:2_000],
        self_assessment_confidence=confidence,
        model_id=model_id or str(raw.get("model_id") or "")[:500],
        source="model",
    )


def observable_audit_payload(
    traj: Trajectory,
    cache: CacheIntegrityReport,
    evidence: list[EvidenceClaim],
) -> dict[str, Any]:
    """Artifacts the model may audit. Intentionally excludes ``Trajectory.reasoning``."""
    return {
        "schema_version": "helix.audit-input.v1",
        "objective": traj.prompt_state[:4_000],
        "presented_context": traj.retrieved_context[:8_000],
        "tool_steps": [
            {
                "name": step.name,
                "arguments": step.arguments[:2_000],
                "result": step.result[:4_000],
                "error": step.error,
                "retry": step.retry,
                "useful_hint": step.useful_hint,
            }
            for step in traj.steps[-100:]
        ],
        "final_result": traj.final_result[:8_000],
        "acceptance_criteria": (traj.extras.get("acceptance_criteria") or []) if isinstance(traj.extras, dict) else [],
        "cache_integrity": cache.to_dict(),
        "evidence": [item.to_dict() for item in evidence],
        "verified": bool(traj.verified),
        "prompt_tokens": traj.prompt_tokens,
        "completion_tokens": traj.completion_tokens,
        "latency_ms": traj.latency_ms,
    }


def fallback_self_audit(
    traj: Trajectory,
    cache: CacheIntegrityReport,
    evidence: list[EvidenceClaim],
    *,
    model_id: str = "",
) -> SelfAuditReport:
    redundant = [f"{step.name}: repeated/redundant call" for step in traj.steps if step.useful_hint == "redundant"]
    failures = [f"{step.name}: {step.error}" for step in traj.steps if step.error]
    unsupported = [item.claim for item in evidence if item.status.value in {"UNVERIFIED", "CONTRADICTED"}]
    return SelfAuditReport(
        objective=traj.prompt_state[:4_000],
        achieved=(
            True
            if isinstance(traj.extras, dict) and traj.extras.get("objective_verified") is True
            else None
        ),
        contributing_actions=[step.name for step in traj.steps if step.useful_hint in {"useful", "evidence"}][:32],
        unnecessary_actions=redundant[:32],
        failures=failures[:32],
        excess_retrieval=[item for item in redundant if "read" in item.lower() or "search" in item.lower()][:32],
        avoidable_cache_disruption=[item.action for item in cache.disruptions if not item.necessary][:32],
        tool_selection_correct=False if failures else None,
        overclaimed_claims=unsupported[:32],
        continued_too_long=bool(redundant),
        better_trajectory=["retain only credited non-redundant actions"] if redundant else [],
        recommendation=AdaptationKind.RUNTIME_POLICY if redundant else AdaptationKind.IGNORE,
        recommendation_reason="deterministic fallback derived from observable trajectory; model audit unavailable",
        self_assessment_confidence=0.0,
        model_id=model_id,
        source="deterministic_fallback",
    )
