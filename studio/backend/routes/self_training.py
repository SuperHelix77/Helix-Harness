# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Safe self-QLoRA orchestration.

Completed local chat turns can be collected as ChatML examples. The original
model identity is retained as the baseline and adapter promotion is gated by an
explicit score comparison. This route never overwrites the base model and does
not auto-promote a newly trained adapter.
"""

from __future__ import annotations

import json
import hashlib
import asyncio
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from models.training import TrainingStartRequest
from utils.paths import account_path, ensure_dir

router = APIRouter()

_STATE_LOCK = threading.RLock()
_MAX_EXAMPLES = 256
_MIN_EXAMPLES = 4
_MAX_MIN_EXAMPLES = 64
_PROMOTION_MARGIN = 0.01
_ADAPTER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class _BenchmarkTask(BaseModel):
    id: str
    prompt: str
    expected: str
    kind: Literal["contains", "json", "ordered"]

# Fixed held-out checks are intentionally outside the model. The candidate cannot
# award itself a score or write the rubric; the harness scores both variants using
# the same deterministic tasks.
_HOLDOUT_BENCHMARK: tuple[_BenchmarkTask, ...] = (
    _BenchmarkTask(
        id = "arithmetic",
        prompt = "Compute 17 * 23. Return only the integer.",
        expected = "391",
        kind = "contains",
    ),
    _BenchmarkTask(
        id = "json-contract",
        prompt = "Return exactly one JSON object with keys name and count. Use name=mlx and count=64.",
        expected = '{"name":"mlx","count":64}',
        kind = "json",
    ),
    _BenchmarkTask(
        id = "ordered-instructions",
        prompt = "Write these three tokens in this exact order, separated by commas: alpha, beta, gamma. Return nothing else.",
        expected = "alpha,beta,gamma",
        kind = "ordered",
    ),
    _BenchmarkTask(
        id = "context-policy",
        prompt = "What is the safe default for a candidate adapter when its benchmark score falls? Answer in one short sentence and include the words revert and retrain.",
        expected = "revert retrain",
        kind = "contains",
    ),
)


class SelfTrainingConfigRequest(BaseModel):
    enabled: bool | None = None
    autoTrain: bool | None = None
    minExamples: int | None = Field(default = None, ge = _MIN_EXAMPLES, le = _MAX_MIN_EXAMPLES)
    maxSeqLength: int | None = Field(default = None, ge = 512, le = 65_536)


class BaselineRequest(BaseModel):
    modelId: str = Field(min_length = 1, max_length = 500)
    snapshotPath: str | None = Field(default = None, max_length = 4_096)
    contextLength: int | None = Field(default = None, ge = 512, le = 65_536)


class SelfTrainingExampleRequest(BaseModel):
    modelId: str = Field(min_length = 1, max_length = 500)
    prompt: str = Field(min_length = 1, max_length = 20_000)
    completion: str = Field(min_length = 1, max_length = 24_000)
    sourceThreadId: str | None = Field(default = None, max_length = 200)
    score: float | None = Field(default = None, ge = 0, le = 1)
    critique: str | None = Field(default = None, max_length = 4_000)
    eligibleForTraining: bool = False
    sourceTrajectoryId: str | None = Field(default = None, max_length = 200)
    evidenceIds: list[str] = Field(default_factory = list, max_length = 64)


class SelfTrainingEvaluationRequest(BaseModel):
    candidateAdapterPath: str = Field(min_length = 1, max_length = 4_096)
    baseScore: float = Field(ge = 0, le = 1)
    candidateScore: float = Field(ge = 0, le = 1)
    baseTokPerSec: float = Field(ge = 0)
    candidateTokPerSec: float = Field(ge = 0)
    evaluator: Literal["human", "holdout"] = "human"
    rubric: str = Field(default = "", max_length = 2_000)


class SelfTrainingBenchmarkRequest(BaseModel):
    candidateAdapterPath: str = Field(min_length = 1, max_length = 4_096)
    adapterName: str | None = Field(default = None, max_length = 64)


class AdapterHotSwapRequest(BaseModel):
    adapterPath: str = Field(min_length = 1, max_length = 4_096)
    adapterName: str | None = Field(default = None, max_length = 64)


class SelfTrainingRecommendationRequest(BaseModel):
    action: Literal["skill", "qlora", "runtime-fix", "none"]
    reason: str = Field(default = "", max_length = 2_000)


def _state_path() -> Path:
    return account_path("learning/self_qlora/state.json")


def _dataset_path() -> Path:
    return account_path("learning/self_qlora/task_examples.jsonl")


def _empty_state() -> dict[str, Any]:
    return {
        "version": 1,
        # Collection is local and bounded; costly training is separately opt-in.
        "enabled": True,
        "autoTrain": False,
        "minExamples": 8,
        "maxSeqLength": 8_192,
        "baseModelId": None,
        "baseSnapshotPath": None,
        "baseContextLength": None,
        "activeAdapterPath": None,
        "examples": [],
        "status": "idle",
        "lastJobId": None,
        "lastEvaluation": None,
        "lastRecommendation": None,
        "trainingQualifiedOnly": False,
        "activeDatasetSnapshot": None,
        "activeDatasetSha256": None,
        "activeDatasetExampleCount": 0,
        "lastError": None,
        "lastRecovery": None,
        "updatedAt": int(time.time() * 1_000),
    }


def _read_state() -> dict[str, Any]:
    try:
        raw = json.loads(_state_path().read_text(encoding = "utf-8"))
    except FileNotFoundError:
        return _empty_state()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HTTPException(status_code = 500, detail = "The self-QLoRA state could not be read.") from error
    if not isinstance(raw, dict):
        raise HTTPException(status_code = 500, detail = "The self-QLoRA state is invalid.")
    state = _empty_state()
    for key in state:
        if key in raw:
            state[key] = raw[key]
    state["examples"] = [item for item in state.get("examples", []) if isinstance(item, dict)][-_MAX_EXAMPLES:]
    return state


def _write_state(state: dict[str, Any]) -> None:
    path = _state_path()
    ensure_dir(path.parent)
    state["updatedAt"] = int(time.time() * 1_000)
    with tempfile.NamedTemporaryFile(
        mode = "w", encoding = "utf-8", dir = path.parent,
        prefix = ".self-qlora-", suffix = ".tmp", delete = False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(state, handle, ensure_ascii = False, indent = 2)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _qualified_examples(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in state.get("examples", []) if item.get("eligibleForTraining") is True]


def _autonomous_qlora_policy_allows(state: dict[str, Any]) -> bool:
    if not state.get("enabled", True) or not state.get("autoTrain"):
        return False
    if len(_qualified_examples(state)) < int(state.get("minExamples", 8)):
        return False
    try:
        from routes.learning import _read_state as _read_learning_state

        learning = _read_learning_state()
    except Exception:
        return False
    return (
        learning.get("decisionMode") == "autonomous"
        and learning.get("allowQloraTraining") is True
    )


def _autonomous_qlora_admission(state: dict[str, Any]) -> bool:
    """Single admission check for any autonomous training queue transition."""
    return (
        state.get("status") in {"idle", "error", "rejected", "needs-more-data"}
        and _autonomous_qlora_policy_allows(state)
    )


def queue_hermes_training_if_allowed() -> bool:
    """Atomically queue a qualified Hermes dataset when all user policy gates allow it."""
    with _STATE_LOCK:
        state = _read_state()
        if not _autonomous_qlora_admission(state):
            return False
        state["status"] = "queued"
        state["trainingQualifiedOnly"] = True
        _write_state(state)
        return True


def _training_runtime_snapshot() -> tuple[str | None, bool | None]:
    """Return the process-local training job identity without guessing on failure."""
    try:
        from core.training import get_training_backend

        backend = get_training_backend()
        job_id = str(getattr(backend, "current_job_id", "") or "")
        return job_id, bool(backend.is_training_active())
    except Exception:
        # Reconciliation is safety logic, not a reason to break the learning UI.
        # Unknown runtime state means "do not mutate persisted state".
        return None, None


def _reconcile_persisted_training_state(state: dict[str, Any]) -> tuple[bool, bool]:
    """Reconcile crash-persisted queued/training states with the live worker.

    Returns ``(changed, reschedule_autonomous)``.  A stale autonomous queue may
    be re-run only when the CURRENT policy still allows it.  A manual queue is
    never silently replayed after restart.  A stale ``training`` marker becomes
    an explicit error with provenance instead of permanently blocking /start.
    """
    status = str(state.get("status") or "")
    if status not in {"queued", "training"}:
        return False, False
    live_job_id, live_active = _training_runtime_snapshot()
    if live_active is None:
        return False, False

    persisted_job_id = str(state.get("lastJobId") or "")
    now = int(time.time() * 1_000)
    if status == "training":
        if live_active and persisted_job_id and live_job_id == persisted_job_id:
            return False, False
        state["status"] = "error"
        state["lastError"] = (
            "Recovered stale self-QLoRA training state: the persisted job is not active "
            "in this process. Start a new run after reviewing the preserved dataset provenance."
        )
        state["lastRecovery"] = {
            "kind": "stale-training",
            "persistedJobId": persisted_job_id or None,
            "runtimeJobId": live_job_id or None,
            "runtimeActive": bool(live_active),
            "recoveredAt": now,
        }
        state["trainingQualifiedOnly"] = False
        return True, False

    # queued: a live job belonging to this persisted run means the state write
    # lagged the worker start; promote it to training rather than spawning twice.
    if live_active:
        if persisted_job_id and live_job_id == persisted_job_id:
            state["status"] = "training"
            state["lastRecovery"] = {
                "kind": "queued-job-observed-running",
                "persistedJobId": persisted_job_id,
                "runtimeJobId": live_job_id,
                "runtimeActive": True,
                "recoveredAt": now,
            }
            return True, False
        # Some other training owns the backend.  Preserve the queue as an error
        # instead of attaching Helix provenance to a foreign job.
        state["status"] = "error"
        state["lastError"] = "Recovered queued self-QLoRA state while another training job is active."
        state["lastRecovery"] = {
            "kind": "queued-runtime-conflict",
            "persistedJobId": persisted_job_id or None,
            "runtimeJobId": live_job_id or None,
            "runtimeActive": True,
            "recoveredAt": now,
        }
        return True, False

    if state.get("trainingQualifiedOnly") is True and _autonomous_qlora_policy_allows(state):
        state["lastRecovery"] = {
            "kind": "stale-autonomous-queue-rescheduled",
            "persistedJobId": persisted_job_id or None,
            "runtimeJobId": None,
            "runtimeActive": False,
            "recoveredAt": now,
        }
        # Keep status=queued; the caller schedules exactly one new background task.
        return True, True

    state["status"] = "idle"
    state["trainingQualifiedOnly"] = False
    state["lastError"] = None
    state["lastRecovery"] = {
        "kind": "stale-queue-cleared",
        "persistedJobId": persisted_job_id or None,
        "runtimeJobId": None,
        "runtimeActive": False,
        "recoveredAt": now,
        "reason": "manual queue is not silently replayed, or autonomous policy is no longer allowed",
    }
    return True, False


def _training_row(example: dict[str, Any]) -> str:
    # Evaluator metadata and self-critique never become target text.
    row = {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    }
    return json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"


def _write_dataset(state: dict[str, Any], *, qualified_only: bool = False) -> None:
    path = _dataset_path()
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(
        mode = "w", encoding = "utf-8", dir = path.parent,
        prefix = ".task-examples-", suffix = ".jsonl", delete = False,
    ) as handle:
        temporary = Path(handle.name)
        examples = _qualified_examples(state) if qualified_only else state.get("examples", [])
        for example in examples:
            handle.write(_training_row(example))
        handle.flush()
    temporary.replace(path)


def _snapshot_training_dataset(
    state: dict[str, Any], *, qualified_only: bool
) -> tuple[Path, str, int]:
    """Write an immutable-by-convention per-run dataset and return path/hash/count.

    The training worker gets this unique path, never the mutable collection view.
    Later chat turns therefore cannot change what an already admitted run trains on.
    """
    examples = _qualified_examples(state) if qualified_only else list(state.get("examples", []))
    payload = "".join(_training_row(example) for example in examples).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    run_dir = account_path("learning/self_qlora/runs")
    ensure_dir(run_dir)
    path = run_dir / f"dataset-{int(time.time() * 1_000)}-{uuid.uuid4().hex[:12]}-{digest[:12]}.jsonl"
    # Exclusive create: no later collection path ever opens this filename for writing.
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    return path, digest, len(examples)


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        **state,
        "datasetPath": str(_dataset_path()),
        "exampleCount": len(state.get("examples", [])),
        "eligibleExampleCount": len(_qualified_examples(state)),
        "promotionMargin": _PROMOTION_MARGIN,
        "acceptanceCriteria": [
            "candidate intelligence score >= base score + 0.01",
            "candidate throughput >= base throughput (no speed regression)",
            "original base model remains resident/available for rollback",
        ],
    }


def _validated_adapter_path(raw_path: str) -> Path:
    """Validate a local PEFT adapter directory before it reaches the worker."""
    path = Path(raw_path).expanduser()
    try:
        if path.is_symlink():
            raise HTTPException(status_code = 400, detail = "Adapter symlinks are not accepted.")
        resolved = path.resolve(strict = True)
    except FileNotFoundError as error:
        raise HTTPException(status_code = 404, detail = "The adapter directory does not exist.") from error
    except OSError as error:
        raise HTTPException(status_code = 400, detail = "The adapter path could not be inspected.") from error
    if not resolved.is_dir():
        raise HTTPException(status_code = 400, detail = "The adapter path must be a directory.")
    config = resolved / "adapter_config.json"
    if not config.is_file() or config.is_symlink():
        raise HTTPException(status_code = 400, detail = "The adapter must contain a local adapter_config.json.")
    try:
        json.loads(config.read_text(encoding = "utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HTTPException(status_code = 400, detail = "The adapter_config.json is invalid.") from error
    return resolved


def _adapter_name(raw_name: str | None, path: Path) -> str:
    name = (raw_name or path.name).strip().replace(".", "_")
    if not _ADAPTER_NAME_RE.fullmatch(name):
        raise HTTPException(status_code = 400, detail = "Adapter name must contain only letters, numbers, '_' or '-'.")
    return name


def _record_mem0_experience(subject: str, payload: SelfTrainingExampleRequest) -> None:
    """Persist a bounded task experience for recurrence search.

    This is deliberately separate from QLoRA: every experience may inform a
    future skill, but it never becomes a training target or a promotion score by
    itself. Mem0 is supplementary; the local JSON ledger remains authoritative.
    """
    try:
        from routes.learning import _read_state

        if _read_state().get("mem0Enabled") is False:
            return
        from core.memory.mem0_store import add_experience

        text = (
            "User task:\n"
            f"{payload.prompt.strip()}\n\n"
            "Assistant result:\n"
            f"{payload.completion.strip()}\n\n"
            f"Critique: {(payload.critique or '').strip()}"
        )
        add_experience(subject, text, thread_id=payload.sourceThreadId, kind="task-experience")
    except Exception:
        # Memory must never make a successful chat or dataset write fail.
        return


def _score_holdout(task: _BenchmarkTask, output: str) -> float:
    """Score a held-out answer without asking either model to judge itself."""
    answer = output.strip()
    if task.kind == "contains":
        wanted = task.expected.lower().split()
        return 1.0 if all(token in answer.lower() for token in wanted) else 0.0
    if task.kind == "ordered":
        compact = re.sub(r"\s+", "", answer).lower()
        return 1.0 if compact == task.expected.lower() else 0.0
    # JSON checks compare parsed values, so whitespace/key order cannot affect the score.
    try:
        start = answer.find("{")
        end = answer.rfind("}")
        parsed = json.loads(answer[start : end + 1]) if start >= 0 and end >= start else None
        expected = json.loads(task.expected)
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0
    return 1.0 if parsed == expected else 0.0


def _run_holdout(backend: Any, use_adapter: str | bool) -> tuple[float, float, bool, list[dict[str, Any]]]:
    """Run the same fixed benchmark and return score plus genuinely measured speed.

    A missing backend timing is not estimated from wall-clock words. Estimates can
    make a candidate look like it preserved throughput when the backend did not
    actually report token accounting, so the promotion gate treats them as an
    unmeasured failure.
    """
    receipts: list[dict[str, Any]] = []
    rates: list[float] = []
    speed_measured = True
    for task in _HOLDOUT_BENCHMARK:
        stats_holder: dict[str, Any] = {}
        chunks: list[str] = []
        for chunk in backend.generate_with_adapter_control(
            use_adapter = use_adapter,
            messages = [{"role": "user", "content": task.prompt}],
            system_prompt = "Answer the held-out task directly. Do not describe your confidence or score.",
            temperature = 0.0,
            top_p = 1.0,
            max_new_tokens = 128,
            stats_holder = stats_holder,
        ):
            chunks.append(str(chunk))
        output = "".join(chunks)
        score = _score_holdout(task, output)
        stats = stats_holder.get("stats")
        timings = stats.get("timings") if isinstance(stats, dict) else None
        tok_per_sec = timings.get("predicted_per_second") if isinstance(timings, dict) else None
        if not isinstance(tok_per_sec, (int, float)) or tok_per_sec <= 0:
            speed_measured = False
        else:
            rates.append(float(tok_per_sec))
        receipts.append({"id": task.id, "score": score, "output": output[:2_000], "tokPerSec": tok_per_sec})
    return (
        sum(item["score"] for item in receipts) / len(receipts),
        sum(rates) / len(rates) if speed_measured and rates else 0.0,
        speed_measured and len(rates) == len(receipts),
        receipts,
    )


async def _start_training_for_state(subject: str) -> None:
    """Start one bounded LoRA/QLoRA run after the HTTP response has returned."""
    with _STATE_LOCK:
        state = _read_state()
        if state.get("status") != "queued":
            return
        model_id = str(state.get("baseModelId") or "").strip()
        qualified_only = state.get("trainingQualifiedOnly") is True
        candidate_examples = _qualified_examples(state) if qualified_only else state.get("examples", [])
        if qualified_only and not _autonomous_qlora_policy_allows(state):
            # Re-check at execution time: a user can revoke autonomous/QLoRA
            # permission after queueing but before the background worker runs.
            state["status"] = "idle"
            state["trainingQualifiedOnly"] = False
            _write_state(state)
            return
        if not model_id or len(candidate_examples) < int(state.get("minExamples", 8)):
            state["status"] = "idle"
            state["trainingQualifiedOnly"] = False
            _write_state(state)
            return
        dataset_snapshot, dataset_sha256, dataset_count = _snapshot_training_dataset(
            state, qualified_only=qualified_only
        )
        state["activeDatasetSnapshot"] = str(dataset_snapshot)
        state["activeDatasetSha256"] = dataset_sha256
        state["activeDatasetExampleCount"] = dataset_count
        _write_state(state)
        snapshot = state.get("baseSnapshotPath")
        request_data: dict[str, Any] = {
            "model_name": model_id,
            "training_type": "LoRA/QLoRA",
            "load_in_4bit": True,
            "max_seq_length": int(state.get("maxSeqLength", 8_192)),
            "local_datasets": [str(dataset_snapshot)],
            "format_type": "chatml",
            "num_epochs": 1,
            "max_steps": 32,
            "save_steps": 16,
            "project_name": "hermes-self-qlora",
        }
        if isinstance(snapshot, str) and snapshot.strip():
            request_data.update({
                "model_local_path": snapshot,
                "model_snapshot_path": snapshot,
                "model_known_cached": True,
            })
        state["status"] = "training"
        state["lastError"] = None
        _write_state(state)
    try:
        # Reuse the existing training admission, VRAM coordination, MLX/CUDA
        # selection, and provenance checks instead of creating a second trainer.
        from routes.training import start_training

        result = await start_training(
            TrainingStartRequest(**request_data),
            current_subject = subject,
            via_api_key = False,
        )
        job_id = getattr(result, "job_id", None) or (
            result.get("job_id") if isinstance(result, dict) else None
        )
        with _STATE_LOCK:
            state = _read_state()
            state["lastJobId"] = job_id
            if getattr(result, "status", None) == "error" or (
                isinstance(result, dict) and result.get("status") == "error"
            ):
                state["status"] = "error"
                state["lastError"] = getattr(result, "message", None) or (
                    result.get("message") if isinstance(result, dict) else "Self-QLoRA could not start."
                )
            _write_state(state)
    except Exception as error:  # noqa: BLE001
        with _STATE_LOCK:
            state = _read_state()
            state["status"] = "error"
            state["lastError"] = str(error)[:2_000]
            _write_state(state)


async def _apply_runtime_fix_for_state() -> None:
    """Safest automatic runtime repair: return to the retained base adapter.

    This does not delete an adapter from disk. It only disables the active
    candidate so a later benchmark or hotswap can restore it, which gives the
    autonomous runtime-fix permission a bounded, reversible meaning.
    """
    with _STATE_LOCK:
        state = _read_state()
        base_model = str(state.get("baseModelId") or "").strip()
    if not base_model:
        with _STATE_LOCK:
            state = _read_state()
            state["status"] = "runtime-fix-needs-baseline"
            _write_state(state)
        return
    try:
        from core.inference import get_inference_backend

        await asyncio.to_thread(get_inference_backend().revert_to_base_model, base_model)
        with _STATE_LOCK:
            state = _read_state()
            state["activeAdapterPath"] = None
            state["status"] = "runtime-fixed"
            state["lastError"] = None
            _write_state(state)
    except Exception as error:  # noqa: BLE001 -- keep the recommendation visible
        with _STATE_LOCK:
            state = _read_state()
            state["status"] = "runtime-fix-error"
            state["lastError"] = str(error)[:2_000]
            _write_state(state)


@router.get("")
def get_self_training(
    background_tasks: BackgroundTasks,
    current_subject: str = Depends(get_current_subject),
):
    reschedule = False
    with _STATE_LOCK:
        state = _read_state()
        changed, reschedule = _reconcile_persisted_training_state(state)
        if changed:
            _write_state(state)
        result = _public_state(state)
    if reschedule:
        background_tasks.add_task(_start_training_for_state, current_subject)
    return result


@router.post("/config")
def set_self_training_config(
    payload: SelfTrainingConfigRequest,
    current_subject: str = Depends(get_current_subject),
):
    with _STATE_LOCK:
        state = _read_state()
        for field, key in (("enabled", "enabled"), ("autoTrain", "autoTrain"), ("minExamples", "minExamples"), ("maxSeqLength", "maxSeqLength")):
            value = getattr(payload, field)
            if value is not None:
                state[key] = value
        if state.get("autoTrain") and not state.get("enabled"):
            state["autoTrain"] = False
        _write_state(state)
        return _public_state(state)


@router.post("/recommendation")
async def set_self_training_recommendation(
    payload: SelfTrainingRecommendationRequest,
    background_tasks: BackgroundTasks,
    current_subject: str = Depends(get_current_subject),
):
    """Store a model's training-vs-skill suggestion as advisory metadata only."""
    schedule = False
    schedule_runtime_fix = False
    with _STATE_LOCK:
        state = _read_state()
        state["lastRecommendation"] = {
            "action": payload.action,
            "reason": payload.reason.strip(),
            "createdAt": int(time.time() * 1_000),
            "advisoryOnly": True,
        }
        if payload.action == "qlora":
            if _autonomous_qlora_admission(state):
                state["status"] = "queued"
                state["trainingQualifiedOnly"] = True
                schedule = True
        elif payload.action == "runtime-fix":
            state["status"] = "runtime-fix-recommended"
            from routes.learning import _read_state as _read_learning_state

            learning_state = _read_learning_state()
            schedule_runtime_fix = (
                learning_state.get("decisionMode") == "autonomous"
                and learning_state.get("allowRuntimeFix") is True
                and bool(state.get("baseModelId"))
            )
            if schedule_runtime_fix:
                state["status"] = "runtime-fix-queued"
        _write_state(state)
        result = _public_state(state)
    if schedule:
        background_tasks.add_task(_start_training_for_state, current_subject)
    if schedule_runtime_fix:
        background_tasks.add_task(_apply_runtime_fix_for_state)
    background_tasks.add_task(_record_mem0_experience, current_subject, payload)
    return result


@router.post("/baseline")
def set_self_training_baseline(
    payload: BaselineRequest,
    current_subject: str = Depends(get_current_subject),
):
    with _STATE_LOCK:
        state = _read_state()
        state["baseModelId"] = payload.modelId.strip()
        state["baseSnapshotPath"] = payload.snapshotPath.strip() if payload.snapshotPath else None
        state["baseContextLength"] = payload.contextLength
        # A new baseline must never reuse examples from a different model.
        state["examples"] = [item for item in state.get("examples", []) if item.get("modelId") == state["baseModelId"]]
        _write_dataset(state)
        _write_state(state)
        return _public_state(state)


def record_hermes_qualified_candidate(
    *,
    model_id: str,
    prompt: str,
    completion: str,
    source_trajectory_id: str,
    evidence_ids: list[str],
    source_trajectory_ids: list[str] | None = None,
    evidence_sha256: str | None = None,
    source_thread_id: str | None = None,
) -> bool:
    """Stage an evidence-linked candidate. Hermes is the only internal caller."""
    if not model_id.strip() or not prompt.strip() or not completion.strip() or not source_trajectory_id.strip():
        return False
    with _STATE_LOCK:
        state = _read_state()
        if not state.get("enabled", True):
            return False
        baseline = str(state.get("baseModelId") or "").strip()
        if baseline and baseline != model_id.strip():
            return False
        if not baseline:
            state["baseModelId"] = model_id.strip()
        if any(item.get("sourceTrajectoryId") == source_trajectory_id for item in state.get("examples", [])):
            return False
        example = {
            "id": uuid.uuid4().hex,
            "modelId": model_id.strip(),
            "prompt": prompt.strip()[:20_000],
            "completion": completion.strip()[:24_000],
            "sourceThreadId": source_thread_id,
            "score": None,
            "critique": "Hermes-qualified repeated behavioral candidate",
            "eligibleForTraining": True,
            "sourceTrajectoryId": source_trajectory_id.strip()[:200],
            "sourceTrajectoryIds": [str(item)[:200] for item in (source_trajectory_ids or [source_trajectory_id])[:64]],
            "evidenceIds": [str(item)[:200] for item in evidence_ids[:64]],
            "evidenceSha256": str(evidence_sha256 or "")[:128] or None,
            "createdAt": int(time.time() * 1_000),
        }
        state["examples"] = [*state.get("examples", []), example][-_MAX_EXAMPLES:]
        _write_state(state)
        return True


@router.post("/examples")
async def record_self_training_example(
    payload: SelfTrainingExampleRequest,
    background_tasks: BackgroundTasks,
    current_subject: str = Depends(get_current_subject),
):
    with _STATE_LOCK:
        state = _read_state()
        if not state.get("enabled", True):
            return {"recorded": False, "reason": "disabled", **_public_state(state)}
        baseline = str(state.get("baseModelId") or "").strip()
        if baseline and baseline != payload.modelId.strip():
            return {"recorded": False, "reason": "model differs from baseline", **_public_state(state)}
        if not baseline:
            state["baseModelId"] = payload.modelId.strip()
        example = {
            "id": uuid.uuid4().hex,
            "modelId": payload.modelId.strip(),
            "prompt": payload.prompt.strip(),
            "completion": payload.completion.strip(),
            "sourceThreadId": payload.sourceThreadId,
            "score": payload.score,
            "critique": payload.critique,
            # Qualification is an internal Hermes capability, never a client assertion.
            "eligibleForTraining": False,
            "sourceTrajectoryId": None,
            "evidenceIds": [],
            "createdAt": int(time.time() * 1_000),
        }
        state["examples"] = [*state.get("examples", []), example][-_MAX_EXAMPLES:]
        _write_dataset(state)
        _write_state(state)
        result = {"recorded": True, **_public_state(state)}
    return result


@router.post("/start")
async def start_self_training(
    background_tasks: BackgroundTasks,
    current_subject: str = Depends(get_current_subject),
):
    reschedule = False
    with _STATE_LOCK:
        state = _read_state()
        changed, reschedule = _reconcile_persisted_training_state(state)
        if changed:
            _write_state(state)
        if reschedule:
            result = {"queued": True, "recovered": True, **_public_state(state)}
        else:
            result = None
        if result is not None:
            pass
        elif not state.get("enabled", True):
            raise HTTPException(status_code = 409, detail = "Self-QLoRA collection is disabled in the sidebar.")
        elif not state.get("baseModelId"):
            raise HTTPException(status_code = 400, detail = "Load a model and record a task before starting self-QLoRA.")
        elif len(state.get("examples", [])) < int(state.get("minExamples", 8)):
            raise HTTPException(status_code = 400, detail = f"Self-QLoRA needs at least {state.get('minExamples', 8)} bounded task examples.")
        elif state.get("status") in {"queued", "training"}:
            raise HTTPException(status_code = 409, detail = "Self-QLoRA is already queued or training.")
        elif result is None:
            state["status"] = "queued"
            state["trainingQualifiedOnly"] = False
            _write_dataset(state, qualified_only=False)
            _write_state(state)
            result = {"queued": True, **_public_state(state)}
    background_tasks.add_task(_start_training_for_state, current_subject)
    return result


@router.post("/evaluate")
def evaluate_self_training_candidate(
    payload: SelfTrainingEvaluationRequest,
    current_subject: str = Depends(get_current_subject),
):
    with _STATE_LOCK:
        state = _read_state()
        try:
            _validated_adapter_path(payload.candidateAdapterPath)
            adapter_exists = True
        except HTTPException:
            adapter_exists = False
        intelligence_improved = payload.candidateScore >= payload.baseScore + _PROMOTION_MARGIN
        speed_preserved = payload.candidateTokPerSec >= payload.baseTokPerSec
        speed_measured = payload.baseTokPerSec > 0 and payload.candidateTokPerSec > 0
        evaluation = {
            "candidateAdapterPath": payload.candidateAdapterPath,
            "baseScore": payload.baseScore,
            "candidateScore": payload.candidateScore,
            "baseTokPerSec": payload.baseTokPerSec,
            "candidateTokPerSec": payload.candidateTokPerSec,
            "evaluator": payload.evaluator,
            "rubric": payload.rubric.strip(),
            "adapterExists": adapter_exists,
            "intelligenceImproved": intelligence_improved,
            "speedPreserved": speed_preserved,
            "speedMeasured": speed_measured,
            "promoted": intelligence_improved and speed_preserved and speed_measured and adapter_exists,
            "createdAt": int(time.time() * 1_000),
        }
        state["lastEvaluation"] = evaluation
        if evaluation["promoted"]:
            # Promotion changes only the active adapter pointer. The base model
            # identity/snapshot stays permanently available for comparison/rollback.
            state["activeAdapterPath"] = payload.candidateAdapterPath
            state["status"] = "promoted"
        else:
            # A failed candidate is never made active. A later accepted task example
            # can trigger another bounded run when autoTrain is enabled.
            state["status"] = "needs-more-data"
        _write_state(state)
        return _public_state(state)


@router.post("/benchmark")
async def benchmark_self_training_candidate(
    payload: SelfTrainingBenchmarkRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Benchmark base and candidate with fixed held-out checks.

    The model outputs are scored by this process, not by an assistant-generated
    critique. The candidate is evaluated first, then the worker is reverted to
    the untouched base before the baseline run.
    """
    path = _validated_adapter_path(payload.candidateAdapterPath)
    name = _adapter_name(payload.adapterName, path)
    with _STATE_LOCK:
        state = _read_state()
        base_model = str(state.get("baseModelId") or "").strip()
        if not base_model:
            raise HTTPException(status_code = 400, detail = "Set a self-QLoRA base model before benchmarking.")
        state["status"] = "benchmarking"
        state["lastError"] = None
        _write_state(state)

    candidate_loaded = False
    try:
        from core.inference import get_inference_backend

        backend = get_inference_backend()
        await asyncio.to_thread(
            backend.hot_swap_adapter,
            str(path),
            name,
            base_model,
        )
        candidate_loaded = True
        candidate_score, candidate_speed, candidate_speed_measured, candidate_tasks = await asyncio.to_thread(
            _run_holdout,
            backend,
            name,
        )
        await asyncio.to_thread(backend.revert_to_base_model, base_model)
        candidate_loaded = False
        base_score, base_speed, base_speed_measured, base_tasks = await asyncio.to_thread(
            _run_holdout,
            backend,
            False,
        )
    except Exception as error:  # noqa: BLE001
        if candidate_loaded:
            try:
                from core.inference import get_inference_backend

                await asyncio.to_thread(get_inference_backend().revert_to_base_model, base_model)
            except Exception:
                pass
        with _STATE_LOCK:
            state = _read_state()
            state["status"] = "error"
            state["lastError"] = str(error)[:2_000]
            _write_state(state)
        raise HTTPException(status_code = 409, detail = str(error)) from error

    intelligence_improved = candidate_score >= base_score + _PROMOTION_MARGIN
    speed_preserved = candidate_speed >= base_speed
    speed_measured = candidate_speed_measured and base_speed_measured
    promoted = intelligence_improved and speed_preserved and speed_measured
    promotion_error = None
    if promoted:
        try:
            # Leave the untouched base resident during comparison, then make an
            # accepted adapter active without reloading the base weights.
            from core.inference import get_inference_backend

            await asyncio.to_thread(get_inference_backend().hot_swap_adapter, str(path), name, base_model)
        except Exception as error:  # noqa: BLE001 -- acceptance must reflect activation too
            promoted = False
            promotion_error = str(error)[:2_000]
    evaluation = {
        "candidateAdapterPath": str(path),
        "baseScore": base_score,
        "candidateScore": candidate_score,
        "baseTokPerSec": base_speed,
        "candidateTokPerSec": candidate_speed,
        "evaluator": "holdout",
        "rubric": "fixed held-out harness; identical prompts and deterministic checks",
        "adapterExists": True,
        "intelligenceImproved": intelligence_improved,
        "speedPreserved": speed_preserved,
        "speedMeasured": speed_measured,
        "baseTasks": base_tasks,
        "candidateTasks": candidate_tasks,
        "promoted": promoted,
        "promotionError": promotion_error,
        "createdAt": int(time.time() * 1_000),
    }
    with _STATE_LOCK:
        state = _read_state()
        state["lastEvaluation"] = evaluation
        if evaluation["promoted"]:
            state["activeAdapterPath"] = str(path)
            state["status"] = "promoted"
        else:
            state["status"] = "needs-more-data"
        _write_state(state)
        return _public_state(state)


@router.post("/hot-swap")
async def hot_swap_self_training_adapter(
    payload: AdapterHotSwapRequest,
    current_subject: str = Depends(get_current_subject),
):
    path = _validated_adapter_path(payload.adapterPath)
    name = _adapter_name(payload.adapterName, path)
    with _STATE_LOCK:
        state = _read_state()
        base_model = str(state.get("baseModelId") or "").strip()
    if not base_model:
        raise HTTPException(status_code = 400, detail = "Set a self-QLoRA base model before hotswapping an adapter.")
    try:
        from core.inference import get_inference_backend

        backend = get_inference_backend()
        await asyncio.to_thread(
            backend.hot_swap_adapter,
            str(path),
            name,
            base_model,
        )
    except Exception as error:  # noqa: BLE001
        with _STATE_LOCK:
            state = _read_state()
            state["lastError"] = str(error)[:2_000]
            state["status"] = "error"
            _write_state(state)
        raise HTTPException(status_code = 409, detail = str(error)) from error
    with _STATE_LOCK:
        state = _read_state()
        state["activeAdapterPath"] = str(path)
        state["status"] = "hot-swapped"
        state["lastError"] = None
        _write_state(state)
        return _public_state(state)


@router.post("/revert")
async def revert_self_training_adapter(
    current_subject: str = Depends(get_current_subject),
):
    with _STATE_LOCK:
        state = _read_state()
        base_model = str(state.get("baseModelId") or "").strip()
    if not base_model:
        raise HTTPException(status_code = 400, detail = "There is no self-QLoRA base model to revert.")
    try:
        from core.inference import get_inference_backend

        backend = get_inference_backend()
        await asyncio.to_thread(backend.revert_to_base_model, base_model)
    except Exception as error:  # noqa: BLE001
        with _STATE_LOCK:
            state = _read_state()
            state["lastError"] = str(error)[:2_000]
            state["status"] = "error"
            _write_state(state)
        raise HTTPException(status_code = 409, detail = str(error)) from error
    with _STATE_LOCK:
        state = _read_state()
        state["activeAdapterPath"] = None
        state["status"] = "reverted"
        state["lastError"] = None
        _write_state(state)
        return _public_state(state)
