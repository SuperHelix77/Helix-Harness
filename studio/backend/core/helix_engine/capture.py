# SPDX-License-Identifier: AGPL-3.0-only
"""Live tool-loop capture. Pure in-process store; llama I/O stays out."""

from __future__ import annotations

import json
import re
import threading
from typing import Any

from .pipeline import run_adaptation_pipeline
from .trajectory import (
    ToolStep,
    ToolVerificationReceipt,
    Trajectory,
    VerificationKind,
    VerificationStatus,
)

_LOCK = threading.Lock()
_STEPS: dict[str, list[ToolStep]] = {}
_RECENT_COMPLETED: dict[str, list[ToolStep]] = {}
_MAX_CAPTURE_SESSIONS = 512


def capture_session_key(
    session_id: str | None = None,
    thread_id: str | None = None,
    turn_id: str | None = None,
) -> str:
    """Return the capture scope without changing the filesystem sandbox id.

    Project chats intentionally share a filesystem sandbox, but their behavioral
    trajectories must not share tool history.  Thread scope is therefore appended
    only for Helix capture.  Non-project callers that already use the thread id as
    session id retain the historical key.
    """
    session = str(session_id or "").strip()
    thread = str(thread_id or "").strip()
    turn = str(turn_id or "").strip()
    base = (
        f"{session}::thread::{thread}"
        if session and thread and session != thread
        else session or thread or "default"
    )
    return f"{base}::turn::{turn}" if turn else base


def _args_text(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(arguments)


_EXIT_RE = re.compile(r"^exit code\s+(-?\d+)\s*:?", re.IGNORECASE)


def _receipt_error(text: str) -> str | None:
    stripped = text.lstrip()
    lowered = stripped.lower()
    if lowered.startswith("error:"):
        return stripped[:400]
    match = _EXIT_RE.match(stripped)
    if match and int(match.group(1)) != 0:
        return stripped[:400]
    if lowered.startswith((
        "cancelled",
        "canceled",
        "timed out",
        "timeout:",
        "execution cancelled",
        "execution canceled",
        "execution timed out",
    )):
        return stripped[:400]
    return None


def _verification_kind(value: VerificationKind | str | None) -> VerificationKind | None:
    if isinstance(value, VerificationKind):
        return value
    if value is None:
        return None
    try:
        return VerificationKind(str(value).strip().lower())
    except Exception:  # noqa: BLE001 -- optional capture metadata must fail closed
        return None


def record_tool_execution(
    session_id: str,
    name: str,
    arguments: Any,
    result: str,
    *,
    verification_kind: VerificationKind | str | None = None,
    verification_claim: str | None = None,
    verification_subject: str | None = None,
) -> ToolStep:
    text = str(result or "")
    args = _args_text(arguments)
    error = _receipt_error(text)
    hint = "wrong" if error else "useful"
    typed_kind = _verification_kind(verification_kind)
    bound_claim = str(verification_claim or "").strip()[:2_000]
    bound_subject = str(verification_subject or "").strip()[:500]
    verification = (
        ToolVerificationReceipt(
            kind=typed_kind,
            status=VerificationStatus.FAILED if error else VerificationStatus.PASSED,
            detail=(error or text)[:500],
            claim=bound_claim,
            subject=bound_subject,
        )
        if typed_kind is not None
        else None
    )
    key = session_id or "default"
    with _LOCK:
        if key not in _STEPS and len(_STEPS) >= _MAX_CAPTURE_SESSIONS:
            # Global bound for abandoned/cancelled generations that never reach
            # post-task ingestion. Dict insertion order gives deterministic FIFO
            # eviction without adding hot-path bookkeeping.
            _STEPS.pop(next(iter(_STEPS)), None)
        prior = _STEPS.setdefault(key, [])
        if any(step.name == name and step.arguments == args and step.result[:200] == text[:200] for step in prior):
            hint = "redundant"
        step = ToolStep(
            name=str(name),
            arguments=args,
            result=text[:8_000],
            useful_hint=hint,
            error=error,
            verification=verification,
        )
        prior.append(step)
        _STEPS[key] = prior[-200:]
        return step


def session_steps(session_id: str) -> list[ToolStep]:
    with _LOCK:
        return list(_STEPS.get(session_id or "default", []))


def _capture_base(key: str) -> str:
    return key.split("::turn::", 1)[0]


def archive_session(session_id: str) -> None:
    """Consume one exact turn into a bounded latest-completed inspector snapshot."""
    key = session_id or "default"
    with _LOCK:
        steps = _STEPS.pop(key, [])
        if not steps:
            return
        base = _capture_base(key)
        if base not in _RECENT_COMPLETED and len(_RECENT_COMPLETED) >= _MAX_CAPTURE_SESSIONS:
            _RECENT_COMPLETED.pop(next(iter(_RECENT_COMPLETED)), None)
        # Reinsert to make dict order reflect recency without an OrderedDict.
        _RECENT_COMPLETED.pop(base, None)
        _RECENT_COMPLETED[base] = list(steps)


def latest_session_steps(
    session_id: str | None = None, thread_id: str | None = None
) -> list[ToolStep]:
    """Latest completed turn, falling back to the newest still-live exact turn."""
    base = capture_session_key(session_id, thread_id)
    with _LOCK:
        archived = _RECENT_COMPLETED.get(base)
        if archived is not None:
            return list(archived)
        prefix = base + "::turn::"
        for key in reversed(_STEPS):
            if key.startswith(prefix):
                return list(_STEPS[key])
        return list(_STEPS.get(base, []))


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
