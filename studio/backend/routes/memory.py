# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read-only Mem0 status and retrieval endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject

router = APIRouter()


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=5, ge=1, le=10)


class MemoryExperienceRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8_000)
    threadId: str | None = Field(default=None, max_length=200)
    kind: str = Field(default="experience", max_length=40)
    title: str | None = Field(default=None, max_length=240)


@router.get("")
def get_memory_status(current_subject: str = Depends(get_current_subject)):
    from core.memory.mem0_store import status

    return status()


@router.post("/search")
def search_memory(payload: MemorySearchRequest, current_subject: str = Depends(get_current_subject)):
    from routes.learning import _read_state

    if _read_state().get("mem0Enabled") is False:
        return {"results": [], "available": False, "reason": "disabled"}
    from core.memory.mem0_store import search

    return search(current_subject, payload.query, payload.limit)


@router.post("/experiences")
def add_memory_experience(payload: MemoryExperienceRequest, current_subject: str = Depends(get_current_subject)):
    from routes.learning import _read_state

    if _read_state().get("mem0Enabled") is False:
        return {"stored": False, "reason": "disabled"}
    from core.memory.mem0_store import add_experience

    return add_experience(
        current_subject,
        payload.text,
        thread_id=payload.threadId,
        kind=payload.kind,
        title=payload.title,
    )


@router.get("/graph")
def get_memory_graph(current_subject: str = Depends(get_current_subject)):
    from routes.learning import _read_state

    if _read_state().get("mem0Enabled") is False:
        return {"nodes": [], "edges": [], "available": False, "reason": "disabled"}
    from core.memory.mem0_store import graph_snapshot

    return graph_snapshot(current_subject)
