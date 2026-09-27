# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read-only Mem0 status and retrieval endpoints."""

import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from auth.authentication import get_current_subject
from core.memory.experience_idempotency import (
    MemoryExperienceIdempotencyConflict,
    MemoryExperienceReplayUnavailable,
    run_idempotent_memory_experience,
)

router = APIRouter()


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=5, ge=1, le=10)


class TypedMemorySearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=5, ge=1, le=10)
    threadId: str | None = Field(default=None, max_length=200)


class MemoryExperienceRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8_000)
    threadId: str | None = Field(default=None, max_length=200)
    kind: str = Field(default="experience", max_length=40)
    title: str | None = Field(default=None, max_length=240)
    idempotencyKey: str | None = Field(default=None, min_length=1, max_length=240)


def persist_memory_experience(
    subject: str | None,
    text: str,
    *,
    thread_id: str | None = None,
    kind: str = "experience",
    title: str | None = None,
    enabled: bool = True,
    idempotency_key: str | None = None,
) -> dict:
    """Persist one bounded experience without making callers depend on Mem0 health."""
    def _persist() -> dict:
        if not enabled:
            return {"stored": False, "reason": "disabled"}
        try:
            from core.memory.mem0_store import add_experience

            return add_experience(
                subject,
                text,
                thread_id=thread_id,
                kind=kind,
                title=title,
                idempotency_key=idempotency_key,
            )
        except Exception as error:  # noqa: BLE001 -- learning stays durable in its local ledger
            return {"stored": False, "reason": str(error)[:1_000] or "memory-unavailable"}

    if not idempotency_key:
        return _persist()
    return run_idempotent_memory_experience(
        idempotency_key=idempotency_key,
        thread_id=thread_id,
        payload={
            "text": text,
            "threadId": thread_id,
            "kind": kind,
            "title": title,
        },
        operation=_persist,
        recovery_probe=(
            lambda: __import__(
                "core.memory.mem0_store",
                fromlist=["experience_receipt_for_idempotency"],
            ).experience_receipt_for_idempotency(
                idempotency_key,
                thread_id=thread_id,
            )
        ),
    )


def persist_completed_turn_memory(
    subject: str | None,
    text: str,
    *,
    thread_id: str,
    idempotency_key: str,
    title: str | None = None,
    enabled: bool = True,
) -> dict:
    """Persist finalized assistant text as a typed, non-authoritative model claim."""

    if not enabled:
        return {"stored": False, "reason": "disabled"}
    from core.memory.mem0_store import admit_completed_turn_memory

    return run_idempotent_memory_experience(
        idempotency_key=idempotency_key,
        thread_id=thread_id,
        payload={
            "text": text,
            "threadId": thread_id,
            "kind": "completed-turn-model-claim",
            "title": title,
            "epistemicClass": "MODEL_CLAIM",
        },
        operation=lambda: admit_completed_turn_memory(
            subject,
            text,
            thread_id=thread_id,
            idempotency_key=idempotency_key,
            title=title,
        ),
        recovery_probe=lambda: __import__(
            "core.memory.mem0_store",
            fromlist=["completed_turn_memory_receipt_for_idempotency"],
        ).completed_turn_memory_receipt_for_idempotency(
            idempotency_key,
            thread_id=thread_id,
        ),
    )


def persist_durable_tool_observations(
    subject: str | None,
    receipts: list[dict] | tuple[dict, ...] | None,
) -> dict:
    """Admit eligible, already-finished backend tool receipts as observations."""

    from core.memory.mem0_store import admit_durable_observation

    allowed = {"web_search", "search_knowledge_base", "read_observation"}
    admitted: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for receipt in receipts or []:
        if not isinstance(receipt, dict):
            rejected.append({"reason": "receipt-not-object"})
            continue
        tool_name = str(
            receipt.get("tool_name") or receipt.get("toolName") or ""
        ).strip()
        if tool_name not in allowed:
            rejected.append({"toolName": tool_name[:120], "reason": "tool-not-observation-source"})
            continue
        execution_id = str(
            receipt.get("execution_id") or receipt.get("executionId") or ""
        ).strip()
        if execution_id and execution_id in seen:
            continue
        if execution_id:
            seen.add(execution_id)
        result = admit_durable_observation(subject, receipt)
        if result.get("stored") is True:
            terminal_seq = result.get("terminal_seq")
            projected = {
                "runId": result.get("run_id"),
                "threadId": result.get("thread_id"),
                "executionId": result.get("execution_id"),
                "toolName": result.get("tool_name"),
                "toolCallId": result.get("tool_call_id"),
                "receiptRef": result.get("receipt_ref"),
                "receiptDigest": result.get("receipt_digest"),
                "terminalSeq": terminal_seq,
            }
            source = {
                "runId": str(receipt.get("runId") or receipt.get("run_id") or "").strip(),
                "threadId": str(receipt.get("threadId") or receipt.get("thread_id") or "").strip(),
                "executionId": str(
                    receipt.get("executionId") or receipt.get("execution_id") or ""
                ).strip(),
                "toolName": str(receipt.get("toolName") or receipt.get("tool_name") or "").strip(),
                "toolCallId": str(
                    receipt.get("toolCallId") or receipt.get("tool_call_id") or ""
                ).strip(),
                "receiptRef": str(
                    receipt.get("receiptRef") or receipt.get("receipt_ref") or ""
                ).strip(),
                "receiptDigest": str(
                    receipt.get("receiptDigest") or receipt.get("receipt_digest") or ""
                ).strip(),
                "terminalSeq": receipt.get("terminalSeq", receipt.get("terminal_seq")),
            }
            if (
                result.get("epistemic_class") != "OBSERVED"
                or result.get("model_facing") is not True
                or not result.get("node_id")
                or not result.get("record_id")
                or not projected["runId"]
                or not projected["threadId"]
                or not projected["executionId"]
                or not projected["toolName"]
                or not projected["toolCallId"]
                or not projected["receiptRef"]
                or not re.fullmatch(r"[0-9a-f]{64}", str(projected["receiptDigest"] or ""))
                or isinstance(terminal_seq, bool)
                or not isinstance(terminal_seq, int)
                or terminal_seq <= 0
                or projected != source
            ):
                rejected.append(
                    {
                        "executionId": execution_id or None,
                        "toolName": tool_name[:120],
                        "reason": "observation-receipt-incomplete",
                    }
                )
                continue
            admitted.append(
                {
                    **projected,
                    "nodeId": result.get("node_id"),
                    "recordId": result.get("record_id"),
                    "epistemicClass": "OBSERVED",
                    "modelFacing": True,
                    "idempotent": result.get("idempotent") is True,
                }
            )
        else:
            rejected.append(
                {
                    "executionId": execution_id or None,
                    "toolName": tool_name[:120],
                    "reason": str(result.get("reason") or "receipt-rejected")[:200],
                }
            )
    return {
        "attempted": len(admitted) + len(rejected),
        "admitted": admitted,
        "rejected": rejected,
        "eligibleToolNames": sorted(allowed),
    }


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


@router.post("/search/typed")
def search_typed_memory(
    payload: TypedMemorySearchRequest,
    current_subject: str = Depends(get_current_subject),
):
    from routes.learning import _read_state

    if _read_state().get("mem0Enabled") is False:
        return {
            "available": False,
            "status": "disabled",
            "items": [],
            "results": [],
            "exclusions": [],
        }
    from core.memory.mem0_store import search_typed

    return search_typed(
        current_subject,
        payload.query,
        payload.limit,
        thread_id=payload.threadId,
    )


@router.post("/experiences")
def add_memory_experience(payload: MemoryExperienceRequest, current_subject: str = Depends(get_current_subject)):
    from routes.learning import _read_state

    enabled = _read_state().get("mem0Enabled") is not False
    try:
        return persist_memory_experience(
            current_subject,
            payload.text,
            thread_id=payload.threadId,
            kind=payload.kind,
            title=payload.title,
            enabled=enabled,
            idempotency_key=payload.idempotencyKey,
        )
    except MemoryExperienceIdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MemoryExperienceReplayUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/graph")
def get_memory_graph(current_subject: str = Depends(get_current_subject)):
    from routes.learning import _read_state

    if _read_state().get("mem0Enabled") is False:
        return {"nodes": [], "edges": [], "available": False, "reason": "disabled"}
    from core.memory.mem0_store import graph_snapshot

    return graph_snapshot(current_subject)
