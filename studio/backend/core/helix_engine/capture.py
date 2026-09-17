# SPDX-License-Identifier: AGPL-3.0-only
"""Live tool-loop capture. Pure in-process store; llama I/O stays out."""

from __future__ import annotations

import json
import re
import threading
from typing import Any

from .pipeline import run_adaptation_pipeline
from .schemas import ToolControlEvent
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
_CONTROL_EVENTS: dict[str, list[ToolControlEvent]] = {}
_RECENT_CONTROL_EVENTS: dict[str, list[ToolControlEvent]] = {}
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
        # Retry is objective execution history, not a model-authored label.  The
        # first execution is 0; each later execution of the same normalized tool
        # + arguments increments it regardless of whether the earlier attempt
        # succeeded.  Successful duplicates should normally be stopped by the
        # live ToolLoopController, while failed calls are allowed a bounded retry.
        retry = sum(1 for step in prior if step.name == name and step.arguments == args)
        if any(step.name == name and step.arguments == args and step.result[:200] == text[:200] for step in prior):
            hint = "redundant"
        step = ToolStep(
            name=str(name),
            arguments=args,
            result=text[:8_000],
            useful_hint=hint,
            error=error,
            retry=retry,
            verification=verification,
        )
        prior.append(step)
        _STEPS[key] = prior[-200:]
        return step


def record_tool_control_event(
    session_id: str,
    *,
    action: str,
    tool_name: str,
    arguments: Any,
    reason: str = "",
    equivalent_to: str = "",
    failed_attempts: int = 0,
    progress: dict[str, Any] | None = None,
) -> ToolControlEvent:
    """Record a bounded pre-execution controller no-op separately from real tools."""
    args = _args_text(arguments)[:4_000]
    bounded_progress: dict[str, Any] = {}
    for key, value in dict(progress or {}).items():
        if isinstance(value, (bool, int, float)) or value is None:
            bounded_progress[str(key)[:120]] = value
    event = ToolControlEvent(
        action=str(action)[:120],
        tool_name=str(tool_name)[:200],
        arguments=args,
        reason=str(reason or "")[:2_000],
        equivalent_to=str(equivalent_to or "")[:200],
        failed_attempts=max(0, int(failed_attempts or 0)),
        progress=bounded_progress,
    )
    key = session_id or "default"
    with _LOCK:
        if key not in _CONTROL_EVENTS and len(_CONTROL_EVENTS) >= _MAX_CAPTURE_SESSIONS:
            _CONTROL_EVENTS.pop(next(iter(_CONTROL_EVENTS)), None)
        prior = _CONTROL_EVENTS.setdefault(key, [])
        prior.append(event)
        _CONTROL_EVENTS[key] = prior[-200:]
    return event


def make_tool_control_observer(
    session_id: str | None = None,
    thread_id: str | None = None,
    turn_id: str | None = None,
):
    """Return a fail-open observer for ToolLoopController no-op decisions."""
    if not str(turn_id or "").strip():
        return None
    key = capture_session_key(session_id, thread_id, turn_id)

    def _observe(decision, progress) -> None:
        try:
            record_tool_control_event(
                key,
                action=getattr(decision, "action", ""),
                tool_name=getattr(decision, "tool_name", ""),
                arguments=getattr(decision, "arguments", {}),
                reason=getattr(decision, "noop_result", ""),
                equivalent_to=getattr(decision, "equivalent_to", ""),
                failed_attempts=getattr(decision, "failed_attempts", 0),
                progress=dict(progress or {}),
            )
        except BaseException:
            # Optional Helix observation must never alter the foreground loop.
            pass

    return _observe


def session_steps(session_id: str) -> list[ToolStep]:
    with _LOCK:
        return list(_STEPS.get(session_id or "default", []))


def session_control_events(session_id: str) -> list[ToolControlEvent]:
    with _LOCK:
        return list(_CONTROL_EVENTS.get(session_id or "default", []))


def _capture_base(key: str) -> str:
    return key.split("::turn::", 1)[0]


def archive_session(session_id: str) -> None:
    """Consume one exact turn into a bounded latest-completed inspector snapshot."""
    key = session_id or "default"
    with _LOCK:
        steps = _STEPS.pop(key, [])
        control_events = _CONTROL_EVENTS.pop(key, [])
        if not steps and not control_events:
            return
        base = _capture_base(key)
        if steps:
            if base not in _RECENT_COMPLETED and len(_RECENT_COMPLETED) >= _MAX_CAPTURE_SESSIONS:
                _RECENT_COMPLETED.pop(next(iter(_RECENT_COMPLETED)), None)
            # Reinsert to make dict order reflect recency without an OrderedDict.
            _RECENT_COMPLETED.pop(base, None)
            _RECENT_COMPLETED[base] = list(steps)
        if control_events:
            if base not in _RECENT_CONTROL_EVENTS and len(_RECENT_CONTROL_EVENTS) >= _MAX_CAPTURE_SESSIONS:
                _RECENT_CONTROL_EVENTS.pop(next(iter(_RECENT_CONTROL_EVENTS)), None)
            _RECENT_CONTROL_EVENTS.pop(base, None)
            _RECENT_CONTROL_EVENTS[base] = list(control_events)


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


def latest_session_control_events(
    session_id: str | None = None, thread_id: str | None = None
) -> list[ToolControlEvent]:
    base = capture_session_key(session_id, thread_id)
    with _LOCK:
        archived = _RECENT_CONTROL_EVENTS.get(base)
        if archived is not None:
            return list(archived)
        prefix = base + "::turn::"
        for key in reversed(_CONTROL_EVENTS):
            if key.startswith(prefix):
                return list(_CONTROL_EVENTS[key])
        return list(_CONTROL_EVENTS.get(base, []))


def clear_session(session_id: str) -> None:
    with _LOCK:
        _STEPS.pop(session_id or "default", None)
        _CONTROL_EVENTS.pop(session_id or "default", None)


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
    extras = kwargs.pop("extras", {})
    extras = dict(extras) if isinstance(extras, dict) else {}
    control_events = session_control_events(session_id)
    if control_events:
        extras["tool_control_events"] = [event.to_dict() for event in control_events]
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
        extras=extras,
        **kwargs,
    )


def finalize_session(session_id: str, **kwargs: Any) -> dict[str, Any]:
    traj = trajectory_from_session(session_id, **kwargs)
    return run_adaptation_pipeline(traj)
