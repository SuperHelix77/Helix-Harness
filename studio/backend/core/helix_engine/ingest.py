# SPDX-License-Identifier: AGPL-3.0-only
"""Turn ingest: critic + captured tools → Helix route → Hermes / self-QLoRA actions."""

from __future__ import annotations

from typing import Any
import hashlib
import json
from uuid import uuid4

from .capture import archive_session, session_steps, trajectory_from_session
from .critic import SelfCritic, critic_from_steps, parse_self_critic
from .pipeline import run_adaptation_pipeline
from .training_targets import (
    VerifiedTrainingTargetReceipt,
    validated_training_target_receipt,
)


def _stage_hermes(
    critic: SelfCritic,
    *,
    prompt: str,
    thread_id: str | None,
    subject: str,
    adaptation_action: str = "",
    audit: dict[str, Any] | None = None,
) -> str | None:
    from routes.learning import LearningProposalRequest, create_learning_proposal

    audit = audit if isinstance(audit, dict) else {}
    reusable = audit.get("reusable_lessons") or []
    reusable_text = "\n".join(str(item) for item in reusable[:8]) if isinstance(reusable, list) else ""
    title = critic.skill_title.strip() or "Helix post-task adaptation"
    content = reusable_text.strip() or critic.skill_content.strip() or critic.notes.strip() or critic.reason.strip()
    if not content:
        content = "The self-critic found a reusable gap. Prefer an existing skill before creating a new one."
    gap = (
        critic.too_many_tools
        or not critic.right_skills
        or not critic.right_tools
        or critic.finished != "full"
        or critic.recommendation == "skill"
    )
    if adaptation_action == "MEMORY":
        kind = "memory"
    elif adaptation_action == "SKILL":
        kind = "skill"
    else:
        kind = "skill" if gap else "memory"
    name = None
    if kind == "skill":
        slug = "".join(ch if ch.isalnum() else "-" for ch in title.lower()).strip("-")[:64]
        name = slug or "turn-critic-skill"
    proposal = LearningProposalRequest(
        kind=kind,  # type: ignore[arg-type]
        title=title[:240],
        content=content[:8_000],
        reason=(critic.reason or "Helix Engine staged this from the per-turn self-critic.")[:2_000],
        name=name,
        sourceThreadId=thread_id,
        recommendationAction=critic.recommendation if critic.recommendation != "none" else "skill",
        recommendationReason=critic.reason[:2_000],
    )
    create_learning_proposal(proposal, current_subject=subject)
    return "stage_skill" if kind == "skill" else "stage_memory"


def _advise_self_training(action: str, reason: str) -> None:
    import time

    from routes.self_training import _STATE_LOCK, _read_state, _write_state

    with _STATE_LOCK:
        state = _read_state()
        state["lastRecommendation"] = {
            "action": action,
            "reason": reason.strip()[:2_000],
            "createdAt": int(time.time() * 1_000),
            "advisoryOnly": True,
        }
        _write_state(state)


def apply_engine_actions(
    analysis: dict[str, Any],
    critic: SelfCritic,
    *,
    prompt: str,
    thread_id: str | None,
    subject: str,
) -> list[str]:
    actions: list[str] = []
    if analysis.get("success_authorizes_weight_update"):
        return ["reject_weight_update"]

    decision = analysis.get("decision")
    adaptive = (
        analysis.get("adaptive_cycle")
        if isinstance(analysis.get("adaptive_cycle"), dict)
        else {}
    )
    adaptation = (
        adaptive.get("adaptation")
        if isinstance(adaptive.get("adaptation"), dict)
        else {}
    )
    adaptive_available = bool(adaptation) and adaptive.get("available") is not False

    if adaptive_available:
        # Hermes is authoritative whenever the closed loop completed.  The legacy
        # critic is retained only as compatibility input; it cannot override a
        # new-Hermes rejection or independently nominate QLoRA.
        adaptation_action = str(adaptation.get("action") or "")
        audit = adaptive.get("self_audit") if isinstance(adaptive.get("self_audit"), dict) else {}
        if adaptation_action in {"SKILL", "MEMORY"}:
            try:
                staged = _stage_hermes(
                    critic,
                    prompt=prompt,
                    thread_id=thread_id,
                    subject=subject,
                    adaptation_action=adaptation_action,
                    audit=audit,
                )
            except Exception:
                staged = None
            if staged:
                actions.append(staged)
        elif adaptation_action == "RUNTIME_POLICY":
            try:
                _advise_self_training(
                    "runtime-fix",
                    str(adaptation.get("reason") or "Hermes identified a deterministic runtime-policy candidate."),
                )
                actions.append("advise_runtime_fix")
            except Exception:
                pass
        elif adaptation_action == "QLORA_CANDIDATE" and adaptation.get("qlora_eligible") is True:
            try:
                _advise_self_training(
                    "qlora",
                    str(adaptation.get("reason") or "Hermes admitted a repeated evidence-backed QLoRA candidate."),
                )
                actions.append("advise_qlora_candidate")
            except Exception:
                pass
    else:
        # Compatibility fail-open: if the optional adaptive controller is absent
        # or crashed, preserve the pre-cycle behavior instead of breaking learning.
        if decision == "promote_hermes":
            try:
                staged = _stage_hermes(
                    critic,
                    prompt=prompt,
                    thread_id=thread_id,
                    subject=subject,
                )
            except Exception:
                staged = None
            if staged:
                actions.append(staged)
        if critic.recommendation == "qlora":
            try:
                _advise_self_training(
                    "qlora",
                    critic.reason or "Legacy self-critic recommended QLoRA; advisory only.",
                )
                actions.append("advise_qlora")
            except Exception:
                pass
        elif critic.recommendation == "runtime-fix":
            try:
                _advise_self_training(
                    "runtime-fix",
                    critic.reason or "Legacy self-critic asked for a reversible runtime repair.",
                )
                actions.append("advise_runtime_fix")
            except Exception:
                pass

    return list(dict.fromkeys(action for action in actions if action != "start_qlora"))


def ingest_turn(
    *,
    session_id: str,
    prompt: str,
    final_result: str = "",
    critic: dict[str, Any] | None = None,
    self_audit: dict[str, Any] | None = None,
    claims: list[dict[str, Any]] | None = None,
    telemetry: dict[str, Any] | None = None,
    acceptance_criteria: list[str] | None = None,
    model_id: str = "",
    effective_model_id: str = "",
    adapter_state: bool | None = None,
    training_target_receipt: VerifiedTrainingTargetReceipt | None = None,
    thread_id: str | None = None,
    turn_id: str | None = None,
    subject: str = "local",
) -> dict[str, Any]:
    steps = session_steps(session_id)
    parsed = parse_self_critic(critic) if critic else critic_from_steps(
        steps,
        finished="full" if final_result and not any(step.error for step in steps) else "partial",
    )
    if not parsed.tool_count:
        parsed.tool_count = len(steps)
    telemetry = telemetry if isinstance(telemetry, dict) else {}
    trajectory_id = str(turn_id or telemetry.get("trajectory_id") or uuid4().hex)
    behavioral_model_id = str(effective_model_id or model_id or "").strip()
    target_receipt = validated_training_target_receipt(
        training_target_receipt,
        trajectory_id=trajectory_id,
    )
    traj = trajectory_from_session(
        session_id,
        prompt_state=prompt,
        final_result=final_result,
        latency_ms=float(telemetry.get("latency_ms") or 0),
        prompt_tokens=int(telemetry.get("prompt_tokens") or telemetry.get("promptTokens") or 0),
        completion_tokens=int(telemetry.get("completion_tokens") or telemetry.get("completionTokens") or 0),
        verified=parsed.finished == "full",
        extras={
            "critic": parsed.as_dict(),
            "telemetry": telemetry,
            "acceptance_criteria": list(acceptance_criteria or []),
            "model_id": behavioral_model_id,
            "base_model_id": model_id,
            "adapter_state": adapter_state,
            "objective_verified": telemetry.get("objective_verified") is True,
            "trajectory_id": trajectory_id,
            "thread_id": thread_id,
            "capture_session_id": session_id,
            "verified_training_target_receipt": target_receipt,
        },
    )
    # Turn capture is a consumable snapshot. Archive only for the bounded inspector;
    # exact future turns always use their own generation key.
    archive_session(session_id)
    analysis = run_adaptation_pipeline(traj, self_audit=self_audit, claims=claims)
    actions = apply_engine_actions(
        analysis,
        parsed,
        prompt=prompt,
        thread_id=thread_id,
        subject=subject,
    )
    adaptive = analysis.get("adaptive_cycle") if isinstance(analysis.get("adaptive_cycle"), dict) else {}
    adaptation = adaptive.get("adaptation") if isinstance(adaptive.get("adaptation"), dict) else {}
    qualified_target_receipt = validated_training_target_receipt(
        traj.extras.get("verified_training_target_receipt"),
        trajectory_id=trajectory_id,
    )
    if (
        adaptation.get("action") == "QLORA_CANDIDATE"
        and adaptation.get("qlora_eligible") is True
        and qualified_target_receipt is not None
    ):
        try:
            from routes.self_training import record_hermes_qualified_candidate

            if record_hermes_qualified_candidate(
                model_id=model_id,
                prompt=prompt,
                completion=qualified_target_receipt.target,
                source_trajectory_id=trajectory_id,
                evidence_ids=[str(item) for item in adaptation.get("evidence_ids", [])],
                source_trajectory_ids=[str(item) for item in adaptation.get("source_trajectory_ids", [])],
                evidence_sha256=hashlib.sha256(
                    json.dumps(
                        (analysis.get("adaptive_cycle") or {}).get("evidence", []),
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
                source_thread_id=thread_id,
            ):
                actions.append("stage_qlora_candidate")
        except Exception:
            pass
    actions = list(dict.fromkeys(actions))
    return {
        **analysis,
        "critic": parsed.as_dict(),
        "actions": actions,
        "session_id": session_id,
    }
