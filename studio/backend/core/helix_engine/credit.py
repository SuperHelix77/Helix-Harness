# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from .trajectory import Trajectory


def assign_credit(traj: Trajectory) -> list[int]:
    """Keep the useful subsequence. Success of the whole trajectory is not enough."""
    seen: set[tuple[str, str, str]] = set()
    kept: list[int] = []
    for index, step in enumerate(traj.steps):
        hint = (step.useful_hint or "").lower()
        key = (step.name, step.arguments, step.result[:200])
        if hint == "wrong" or step.error:
            continue
        if hint == "redundant" or key in seen:
            continue
        seen.add(key)
        if hint in {"evidence", "useful", ""}:
            kept.append(index)
    return kept
