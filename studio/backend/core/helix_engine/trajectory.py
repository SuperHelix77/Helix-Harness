# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VerificationKind(str, Enum):
    TEST = "test"
    BENCHMARK = "benchmark"
    VERIFIER = "verifier"


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class ToolVerificationReceipt:
    kind: VerificationKind
    status: VerificationStatus
    provenance: str = "backend_tool_capture"
    detail: str = ""
    # Backend-owned semantic binding. A passed verifier is not evidence for an
    # arbitrary model-authored claim merely because the model cites its tool id.
    # ``claim`` names the exact assertion the verifier established; ``subject``
    # binds the run to the artifact/object it verified (for example a corrected
    # training target digest). Model tool arguments never populate either field.
    claim: str = ""
    subject: str = ""


@dataclass
class ToolStep:
    name: str
    arguments: str
    result: str
    useful_hint: str = ""
    error: Optional[str] = None
    retry: int = 0
    verification: Optional[ToolVerificationReceipt] = None


@dataclass
class Correction:
    state: str
    bad_action: str
    failure_evidence: str
    correct_action: str
    why: str


@dataclass
class Trajectory:
    prompt_state: str
    retrieved_context: str
    reasoning: str
    steps: list[ToolStep]
    user_corrections: list[Correction]
    final_result: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    verified: bool
    semantic_turns: int = 0
    holdout_passed: bool = False
    allow_qlora: bool = False
    frequent_behavior: bool = False
    one_off_fact: bool = False
    regression_passed: bool = False
    dataset_hash: str = ""
    adapter_version: str = ""
    extras: dict = field(default_factory=dict)
