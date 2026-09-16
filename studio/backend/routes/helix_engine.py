# SPDX-License-Identifier: AGPL-3.0-only
"""In-app Helix Engine API — not a separate webapp."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject

from core.helix_engine import (
    Correction,
    ToolStep,
    Trajectory,
    cluster_deficits,
    ingest_turn,
    monitor_semantic_turns,
    run_adaptation_pipeline,
)
from core.inference.computer_browse import feed_session_id, list_live_feed
from core.hub.simple_hub import download_local_model, list_local_gguf, load_local_model, select_local_model

router = APIRouter()


class ToolStepIn(BaseModel):
    name: str
    arguments: str = ""
    result: str = ""
    useful_hint: str = ""
    error: str | None = None


class CorrectionIn(BaseModel):
    state: str
    bad_action: str
    failure_evidence: str
    correct_action: str
    why: str


class TrajectoryIn(BaseModel):
    prompt_state: str
    retrieved_context: str = ""
    reasoning: str = ""
    steps: list[ToolStepIn] = Field(default_factory=list)
    user_corrections: list[CorrectionIn] = Field(default_factory=list)
    final_result: str = ""
    latency_ms: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    verified: bool = False
    semantic_turns: int = 0
    holdout_passed: bool = False
    allow_qlora: bool = False
    frequent_behavior: bool = False
    one_off_fact: bool = False
    regression_passed: bool = False
    dataset_hash: str = ""


def _traj(payload: TrajectoryIn) -> Trajectory:
    return Trajectory(
        prompt_state=payload.prompt_state,
        retrieved_context=payload.retrieved_context,
        reasoning=payload.reasoning,
        steps=[ToolStep(**item.model_dump()) for item in payload.steps],
        user_corrections=[Correction(**item.model_dump()) for item in payload.user_corrections],
        final_result=payload.final_result,
        latency_ms=payload.latency_ms,
        prompt_tokens=payload.prompt_tokens,
        completion_tokens=payload.completion_tokens,
        verified=payload.verified,
        semantic_turns=payload.semantic_turns,
        holdout_passed=payload.holdout_passed,
        allow_qlora=payload.allow_qlora,
        frequent_behavior=payload.frequent_behavior,
        one_off_fact=payload.one_off_fact,
        regression_passed=payload.regression_passed,
        dataset_hash=payload.dataset_hash,
    )


class IngestTurnIn(BaseModel):
    session_id: str = ""
    thread_id: str | None = None
    prompt: str = ""
    final_result: str = ""
    critic: dict[str, Any] | None = None


@router.post("/analyze")
def analyze_trajectory(payload: TrajectoryIn) -> dict[str, Any]:
    return run_adaptation_pipeline(_traj(payload))


@router.post("/ingest-turn")
def ingest_completed_turn(
    payload: IngestTurnIn,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    key = feed_session_id(payload.session_id, payload.thread_id)
    return ingest_turn(
        session_id=key,
        prompt=payload.prompt,
        final_result=payload.final_result,
        critic=payload.critic,
        thread_id=payload.thread_id,
        subject=current_subject,
    )


@router.post("/semantic-turns")
def semantic_turns(payload: dict[str, list[str]]) -> dict[str, Any]:
    return monitor_semantic_turns(payload.get("turns") or [])


@router.get("/live-feed")
def live_feed(session_id: str | None = None, thread_id: str | None = None) -> dict[str, Any]:
    key = feed_session_id(session_id, thread_id)
    return {"session_id": key, "events": list_live_feed(key)}


@router.get("/session/{session_id}")
def session_trace(session_id: str) -> dict[str, Any]:
    from core.helix_engine.capture import session_steps

    return {
        "session_id": session_id,
        "steps": [
            {
                "name": step.name,
                "arguments": step.arguments,
                "result": step.result[:500],
                "useful_hint": step.useful_hint,
            }
            for step in session_steps(session_id)
        ],
    }


@router.get("/hub/local")
def hub_local(root: str = "") -> dict[str, Any]:
    from pathlib import Path

    models = list_local_gguf(Path(root).expanduser()) if root else []
    return {"models": models}


@router.post("/hub/select")
def hub_select(payload: dict[str, str]) -> dict[str, Any]:
    return select_local_model(payload.get("path") or "")


@router.post("/hub/load")
async def hub_load(
    payload: dict[str, str],
    fastapi_request: Request,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    from models.inference import LoadRequest
    from routes.inference import load_model_gated

    selected = select_local_model(payload.get("path") or "")
    if not selected.get("ok"):
        return selected

    async def _load(path: str):
        return await load_model_gated(
            LoadRequest(model_path=path),
            fastapi_request,
            current_subject,
            user_initiated=True,
        )

    try:
        loaded = await _load(selected["path"])
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:800], "path": selected["path"], "loaded": False}
    return load_local_model(selected["path"], load_fn=lambda path: loaded)


@router.post("/hub/download")
def hub_download(payload: dict[str, str]) -> dict[str, Any]:
    from pathlib import Path

    return download_local_model(payload.get("source") or "", Path(payload.get("dest") or "").expanduser())


@router.post("/deficits")
def deficits(payload: list[TrajectoryIn]) -> dict[str, Any]:
    return {"clusters": cluster_deficits(_traj(item) for item in payload)}
