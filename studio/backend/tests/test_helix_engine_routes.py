# SPDX-License-Identifier: AGPL-3.0-only
from routes.helix_engine import TrajectoryIn, ToolStepIn, CorrectionIn, analyze_trajectory


def test_analyze_route_does_not_promote_qlora_on_success_alone():
    payload = TrajectoryIn(
        prompt_state="fix helper",
        retrieved_context="tests",
        steps=[
            ToolStepIn(name="read_file", arguments="a.py", result="ok", useful_hint="evidence"),
            ToolStepIn(name="read_file", arguments="a.py", result="ok", useful_hint="redundant"),
            ToolStepIn(name="search_repo", arguments="a.py", result="same", useful_hint="redundant"),
            ToolStepIn(name="edit_file", arguments="bad", result="broke", useful_hint="wrong", error="fail"),
            ToolStepIn(name="grep", arguments="x", result="hit", useful_hint="useful"),
            ToolStepIn(name="run_tests", arguments="pytest", result="FAIL", useful_hint="useful"),
            ToolStepIn(name="read_file", arguments="b.py", result="ok", useful_hint="evidence"),
            ToolStepIn(name="edit_file", arguments="good", result="pass", useful_hint="useful"),
        ],
        user_corrections=[
            CorrectionIn(
                state="failed",
                bad_action="edit_file bad",
                failure_evidence="FAIL",
                correct_action="edit_file good",
                why="user correction",
            )
        ],
        final_result="tests pass",
        verified=True,
        allow_qlora=False,
    )
    result = analyze_trajectory(payload)
    assert result["success_authorizes_weight_update"] is False
    assert result["route"] == "hermes"
    assert result["compressed"]["tool_calls"] < 8
    assert result["corrections"][0]["correct_action"] == "edit_file good"
    assert result["product"] == "Helix Harness"
    assert result["decision"] == "promote_hermes"
    assert result["gates"][-1] == "promote"
