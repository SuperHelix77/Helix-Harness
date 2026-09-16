# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolStep:
    name: str
    arguments: str
    result: str
    useful_hint: str = ""
    error: Optional[str] = None
    retry: int = 0


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
