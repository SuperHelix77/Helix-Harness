# SPDX-License-Identifier: AGPL-3.0-only
"""Live tool-loop capture. Pure in-process store; llama I/O stays out."""

from __future__ import annotations

import json
import threading
from typing import Any

from .pipeline import run_adaptation_pipeline
from .trajectory import ToolStep, Trajectory

_LOCK = threading.Lock()
_STEPS: dict[str, list[ToolStep]] = {}


def _args_text(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(arguments)


def record_tool_execution(
    session_id: str,
    name: str,
    arguments: Any,
    result: str,
) -> ToolStep:
    text = str(result or "")
    args = _args_text(arguments)
    hint = "wrong" if text.lower().startswith("error:") else "useful"
    error = text[:400] if hint == "wrong" else None
    key = session_id or "default"
    with _LOCK:
        prior = _STEPS.setdefault(key, [])
        if any(step.name == name and step.arguments == args and step.result[:200] == text[:200] for step in prior):
            hint = "redundant"
        step = ToolStep(name=str(name), arguments=args, result=text[:8_000], useful_hint=hint, error=error)
        prior.append(step)
        _STEPS[key] = prior[-200:]
        return step


def session_steps(session_id: str) -> list[ToolStep]:
    with _LOCK:
        return list(_STEPS.get(session_id or "default", []))


def clear_session(session_id: str) -> None:
    with _LOCK:
        _STEPS.pop(session_id or "default", None)


def trajectory_from_session(
    session_id: str,
    *,
    prompt_state: str = "",
    retrieved_context: str = "",
    reasoning: str = "",
    final_result: str = "",
    user_corrections: list | None = None,
    latency_ms: float = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    verified: bool = False,
    **kwargs: Any,
) -> Trajectory:
    return Trajectory(
        prompt_state=prompt_state,
        retrieved_context=retrieved_context,
        reasoning=reasoning,
        steps=session_steps(session_id),
        user_corrections=list(user_corrections or []),
        final_result=final_result,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        verified=verified,
        **kwargs,
    )


def finalize_session(session_id: str, **kwargs: Any) -> dict[str, Any]:
    traj = trajectory_from_session(session_id, **kwargs)
    return run_adaptation_pipeline(traj)
