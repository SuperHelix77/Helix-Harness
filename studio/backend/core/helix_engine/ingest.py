# SPDX-License-Identifier: AGPL-3.0-only
"""Turn ingest: critic + captured tools → Helix route → Hermes / self-QLoRA actions."""

from __future__ import annotations

from typing import Any

from .capture import session_steps, trajectory_from_session
from .critic import SelfCritic, critic_from_steps, parse_self_critic
from .pipeline import run_adaptation_pipeline


def _stage_hermes(critic: SelfCritic, *, prompt: str, thread_id: str | None, subject: str) -> str | None:
    from routes.learning import LearningProposalRequest, create_learning_proposal

    title = critic.skill_title.strip() or "Turn self-critic"
    content = critic.skill_content.strip() or critic.notes.strip() or critic.reason.strip()
    if not content:
        content = "The self-critic found a reusable gap. Prefer an existing skill before creating a new one."
    gap = (
        critic.too_many_tools
        or not critic.right_skills
        or not critic.right_tools
        or critic.finished != "full"
        or critic.recommendation == "skill"
    )
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
    if decision == "promote_hermes":
        try:
            staged = _stage_hermes(critic, prompt=prompt, thread_id=thread_id, subject=subject)
        except Exception:
            staged = None
        if staged:
            actions.append(staged)
    if critic.recommendation == "qlora":
        try:
            _advise_self_training(
                "qlora",
                critic.reason or "Self-critic recommended QLoRA; Helix keeps it advisory until holdout gates pass.",
            )
            actions.append("advise_qlora")
        except Exception:
            pass
    elif critic.recommendation == "runtime-fix":
        try:
            _advise_self_training(
                "runtime-fix",
                critic.reason or "Self-critic asked for a reversible runtime repair.",
            )
            actions.append("advise_runtime_fix")
        except Exception:
            pass
    if "start_qlora" in actions:
        actions.remove("start_qlora")
    return actions


def ingest_turn(
    *,
    session_id: str,
    prompt: str,
    final_result: str = "",
    critic: dict[str, Any] | None = None,
    thread_id: str | None = None,
    subject: str = "local",
) -> dict[str, Any]:
    steps = session_steps(session_id)
    parsed = parse_self_critic(critic) if critic else critic_from_steps(
        steps,
        finished="full" if final_result and not any(step.error for step in steps) else "partial",
    )
    if not parsed.tool_count:
        parsed.tool_count = len(steps)
    traj = trajectory_from_session(
        session_id,
        prompt_state=prompt,
        final_result=final_result,
        verified=parsed.finished == "full",
    )
    traj.extras["critic"] = parsed.as_dict()
    analysis = run_adaptation_pipeline(traj)
    actions = apply_engine_actions(
        analysis,
        parsed,
        prompt=prompt,
        thread_id=thread_id,
        subject=subject,
    )
    return {
        **analysis,
        "critic": parsed.as_dict(),
        "actions": actions,
        "session_id": session_id,
    }
