# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Governed Hermes-style self-improvement for the local agent.

The agent may propose a compact lesson or a plain-text skill, but proposals stay
pending until the user approves them. Approval never executes code: skill writes
contain only ``SKILL.md`` and memory is bounded before it is injected into a
future prompt.
"""

from __future__ import annotations

import json
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from utils.paths import account_path, ensure_dir

router = APIRouter()

LearningKind = Literal["memory", "user", "skill"]
LearningTarget = Literal["codex", "claude", "both"]
LearningRecommendationAction = Literal["skill", "qlora", "runtime-fix", "none"]
LearningDecisionMode = Literal["ask", "autonomous"]

_STATE_VERSION = 1
_MEMORY_MAX_CHARS = 2_200
_USER_MAX_CHARS = 1_375
_PENDING_MAX = 50
_SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_LEARNING_LOCK = threading.RLock()


class LearningProposalRequest(BaseModel):
    kind: LearningKind
    title: str = Field(min_length = 1, max_length = 240)
    content: str = Field(min_length = 1, max_length = 100_000)
    reason: str = Field(default = "", max_length = 2_000)
    name: str | None = Field(default = None, max_length = 64)
    target: LearningTarget = "codex"
    sourceThreadId: str | None = Field(default = None, max_length = 200)
    recommendationAction: LearningRecommendationAction | None = None
    recommendationReason: str = Field(default = "", max_length = 2_000)


class LearningConfigRequest(BaseModel):
    enabled: bool | None = None
    mem0Enabled: bool | None = None
    onTheFlySkills: bool | None = None
    decisionMode: LearningDecisionMode | None = None
    allowSkillCreation: bool | None = None
    allowQloraTraining: bool | None = None
    allowRuntimeFix: bool | None = None


def _state_path() -> Path:
    return account_path("learning/state.json")


def _empty_state() -> dict[str, Any]:
    return {
        "version": _STATE_VERSION,
        "enabled": True,
        "mem0Enabled": True,
        "onTheFlySkills": True,
        "decisionMode": "ask",
        "allowSkillCreation": False,
        "allowQloraTraining": False,
        "allowRuntimeFix": False,
        "memory": [],
        "user": [],
        "pending": [],
        "lastRecommendation": None,
    }


def _read_state() -> dict[str, Any]:
    path = _state_path()
    try:
        raw = json.loads(path.read_text(encoding = "utf-8"))
    except FileNotFoundError:
        return _empty_state()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HTTPException(status_code = 500, detail = "The local learning store could not be read.") from error
    if not isinstance(raw, dict):
        raise HTTPException(status_code = 500, detail = "The local learning store is invalid.")
    state = _empty_state()
    state["enabled"] = raw.get("enabled") is not False
    state["mem0Enabled"] = raw.get("mem0Enabled") is not False
    state["onTheFlySkills"] = raw.get("onTheFlySkills") is not False
    state["decisionMode"] = raw.get("decisionMode") if raw.get("decisionMode") in {"ask", "autonomous"} else "ask"
    for key in ("allowSkillCreation", "allowQloraTraining", "allowRuntimeFix"):
        state[key] = raw.get(key) is True
    for key in ("memory", "user", "pending"):
        value = raw.get(key)
        if isinstance(value, list):
            state[key] = [item for item in value if isinstance(item, dict)]
    if isinstance(raw.get("lastRecommendation"), dict):
        state["lastRecommendation"] = raw["lastRecommendation"]
    return state


def _write_state(state: dict[str, Any]) -> None:
    path = _state_path()
    ensure_dir(path.parent)
    # Replace in the same directory so a crash cannot leave a half-written JSON store.
    with tempfile.NamedTemporaryFile(
        mode = "w",
        encoding = "utf-8",
        dir = path.parent,
        prefix = ".learning-",
        suffix = ".tmp",
        delete = False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(state, handle, ensure_ascii = False, indent = 2)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _entry_text(entry: dict[str, Any]) -> str:
    title = str(entry.get("title") or "Lesson").strip()
    content = str(entry.get("content") or "").strip()
    return f"{title}: {content}" if title else content


def _fit_entries(entries: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep newest entries while enforcing the prompt budget in characters."""
    kept: list[dict[str, Any]] = []
    remaining = limit
    for original in reversed(entries):
        entry = dict(original)
        title = str(entry.get("title") or "Lesson").strip()[:240]
        content = str(entry.get("content") or "").strip()
        overhead = len(title) + 2
        available = max(0, remaining - overhead)
        if available <= 0:
            break
        entry["title"] = title
        entry["content"] = content[:available]
        used = len(_entry_text(entry)) + 2
        remaining -= used
        kept.append(entry)
    return list(reversed(kept))


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(state.get("enabled", True)),
        "mem0Enabled": bool(state.get("mem0Enabled", True)),
        "onTheFlySkills": bool(state.get("onTheFlySkills", True)),
        "decisionMode": state.get("decisionMode", "ask"),
        "allowSkillCreation": bool(state.get("allowSkillCreation", False)),
        "allowQloraTraining": bool(state.get("allowQloraTraining", False)),
        "allowRuntimeFix": bool(state.get("allowRuntimeFix", False)),
        "memory": state.get("memory", []),
        "user": state.get("user", []),
        "pending": state.get("pending", []),
        "limits": {"memoryChars": _MEMORY_MAX_CHARS, "userChars": _USER_MAX_CHARS},
        "context": _context_instruction(state),
        "lastRecommendation": state.get("lastRecommendation"),
    }


def _context_instruction(state: dict[str, Any]) -> str:
    if state.get("enabled") is False:
        return ""
    memory = [_entry_text(entry) for entry in state.get("memory", []) if _entry_text(entry)]
    user = [_entry_text(entry) for entry in state.get("user", []) if _entry_text(entry)]
    if not memory and not user and not state.get("onTheFlySkills", True):
        return ""
    lines = [
        "<hermes_memory>",
        "These are user-approved local lessons. Treat them as hints, verify them against current evidence, and do not mention this block unless relevant.",
        "Learning ladder: record the completed experience first; use Mem0 to find recurrence; turn a repeated procedural gap into a Hermes skill; recommend QLoRA only after a deterministic held-out benchmark proves a model behavior deficiency. Never use a self-written score as proof.",
        "Durable facts live in Mem0. Call search_memory to recall them. Do not assume the full lesson text is in this block.",
    ]
    if user:
        lines.append("User preferences:")
        lines.extend(f"- {item}" for item in user)
    if memory:
        lines.append("Verified lesson index (call search_memory for the body):")
        for entry in state.get("memory", []):
            title = str(entry.get("title") or "Lesson").strip()[:240]
            if title:
                lines.append(f"- {title}")
    if state.get("onTheFlySkills", True):
        lines.extend([
            "On-the-fly skill lane: look at existing skills first and reuse them. Prefer a plain-text skill when it can solve the gap without changing model weights. At the end of a completed local task, you may emit one <unsloth-skill-draft> JSON block with name, title, content, and reason. The app stages it only after Mem0 finds a prior similar experience; do not claim it was saved and never include executable code.",
        ])
    lines.append("</hermes_memory>")
    return "\n".join(lines)


def _valid_skill_name(value: str) -> str:
    normalized = value.strip().lower()
    if not _SKILL_NAME.fullmatch(normalized):
        raise HTTPException(status_code = 400, detail = "Skill proposals need a lowercase name using letters, numbers, dots, underscores, or hyphens.")
    return normalized


def _skill_targets(target: LearningTarget) -> list[Path]:
    roots = {
        "codex": Path.home() / ".codex" / "skills",
        "claude": Path.home() / ".claude" / "skills",
    }
    return [roots[name] for name in ("codex", "claude") if target == "both" or target == name]


def _write_plain_skill(proposal: dict[str, Any]) -> str:
    name = _valid_skill_name(str(proposal.get("name") or proposal.get("title") or ""))
    description = str(proposal.get("title") or name).strip()[:500]
    content = str(proposal.get("content") or "").strip()
    body = f"---\nname: {name}\ndescription: {description}\n---\n\n{content}\n"
    targets = _skill_targets(proposal.get("target", "codex"))
    destinations = [root / name for root in targets]
    for root, destination in zip(targets, destinations):
        if root.is_symlink() or destination.is_symlink() or destination.exists():
            raise HTTPException(status_code = 409, detail = f"Skill destination already exists: {name}")
    for root, destination in zip(targets, destinations):
        root.mkdir(parents = True, exist_ok = True)
        destination.mkdir()
        (destination / "SKILL.md").write_text(body, encoding = "utf-8")
    return name


@router.get("")
def get_learning(current_subject: str = Depends(get_current_subject)):
    with _LEARNING_LOCK:
        return _public_state(_read_state())


@router.get("/context")
def get_learning_context(current_subject: str = Depends(get_current_subject)):
    with _LEARNING_LOCK:
        state = _read_state()
        return {"enabled": bool(state.get("enabled", True)), "instruction": _context_instruction(state)}


@router.post("/config")
def set_learning_config(payload: LearningConfigRequest, current_subject: str = Depends(get_current_subject)):
    with _LEARNING_LOCK:
        state = _read_state()
        if payload.enabled is not None:
            state["enabled"] = payload.enabled
        if payload.mem0Enabled is not None:
            state["mem0Enabled"] = payload.mem0Enabled
        if payload.onTheFlySkills is not None:
            state["onTheFlySkills"] = payload.onTheFlySkills
        for field in ("decisionMode", "allowSkillCreation", "allowQloraTraining", "allowRuntimeFix"):
            value = getattr(payload, field)
            if value is not None:
                state[field] = value
        _write_state(state)
        return _public_state(state)


@router.post("/proposals")
def create_learning_proposal(payload: LearningProposalRequest, current_subject: str = Depends(get_current_subject)):
    title = payload.title.strip()
    content = payload.content.strip()
    if not title or not content:
        raise HTTPException(status_code = 400, detail = "A learning proposal needs a title and content.")
    if payload.kind == "skill":
        _valid_skill_name(payload.name or title)
    proposal = {
        "id": uuid.uuid4().hex,
        "kind": payload.kind,
        "title": title,
        "content": content,
        "reason": payload.reason.strip(),
        "name": payload.name.strip().lower() if payload.name else None,
        "target": payload.target,
        "sourceThreadId": payload.sourceThreadId,
        "createdAt": int(time.time() * 1_000),
        "recommendationAction": payload.recommendationAction,
        "recommendationReason": payload.recommendationReason.strip(),
    }
    with _LEARNING_LOCK:
        state = _read_state()
        if not state.get("enabled", True):
            raise HTTPException(status_code = 409, detail = "Hermes learning is disabled in the sidebar.")
        autonomous_skill = (
            proposal["kind"] == "skill"
            and state.get("decisionMode") == "autonomous"
            and state.get("allowSkillCreation") is True
        )
        if autonomous_skill:
            _write_plain_skill(proposal)
        else:
            pending = state.setdefault("pending", [])
            pending.append(proposal)
            state["pending"] = pending[-_PENDING_MAX:]
        if payload.recommendationAction:
            state["lastRecommendation"] = {
                "action": payload.recommendationAction,
                "reason": payload.recommendationReason.strip(),
                "createdAt": proposal["createdAt"],
            }
        _write_state(state)
        return {"proposal": proposal, "autoApproved": autonomous_skill, **_public_state(state)}


@router.post("/proposals/{proposal_id}/approve")
def approve_learning_proposal(proposal_id: str, current_subject: str = Depends(get_current_subject)):
    with _LEARNING_LOCK:
        state = _read_state()
        pending = state.get("pending", [])
        proposal = next((item for item in pending if item.get("id") == proposal_id), None)
        if proposal is None:
            raise HTTPException(status_code = 404, detail = "Learning proposal not found.")
        if proposal.get("kind") == "skill":
            _write_plain_skill(proposal)
        else:
            kind = "user" if proposal.get("kind") == "user" else "memory"
            state[kind] = _fit_entries(
                [*state.get(kind, []), proposal],
                _USER_MAX_CHARS if kind == "user" else _MEMORY_MAX_CHARS,
            )
        state["pending"] = [item for item in pending if item.get("id") != proposal_id]
        _write_state(state)
        return _public_state(state)


@router.post("/proposals/{proposal_id}/reject")
def reject_learning_proposal(proposal_id: str, current_subject: str = Depends(get_current_subject)):
    with _LEARNING_LOCK:
        state = _read_state()
        pending = state.get("pending", [])
        if not any(item.get("id") == proposal_id for item in pending):
            raise HTTPException(status_code = 404, detail = "Learning proposal not found.")
        state["pending"] = [item for item in pending if item.get("id") != proposal_id]
        _write_state(state)
        return _public_state(state)
