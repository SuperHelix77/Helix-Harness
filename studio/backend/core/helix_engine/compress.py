# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from .credit import assign_credit
from .trajectory import Trajectory


def _tokens(text: str, cap: int) -> int:
    return min(cap, max(1, len(text) // 4))


def compress_counterfactual(traj: Trajectory) -> dict:
    kept = assign_credit(traj)
    steps = [traj.steps[i] for i in kept]
    state = traj.prompt_state.strip()
    evidence = "\n".join(step.result for step in steps if step.useful_hint == "evidence")
    decision = traj.final_result.strip()
    return {
        "tool_calls": len(steps),
        "state_summary_tokens": _tokens(state, 1_800),
        "necessary_evidence_tokens": _tokens(evidence or traj.retrieved_context, 3_200),
        "final_decision": decision[:700],
        "kept_indices": kept,
    }
