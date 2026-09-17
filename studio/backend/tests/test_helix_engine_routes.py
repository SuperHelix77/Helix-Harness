# SPDX-License-Identifier: AGPL-3.0-only
from routes.helix_engine import (
    CorrectionIn,
    IngestTurnIn,
    ToolStepIn,
    TrajectoryIn,
    analyze_trajectory,
    prepare_completed_turn_audit,
)


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


def test_prepare_audit_route_fails_open_to_historical_deep_audit(monkeypatch):
    from core.helix_engine import audit

    monkeypatch.setattr(
        audit,
        "prepare_observable_self_audit",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("optional audit preparation died")),
    )
    result = prepare_completed_turn_audit(
        IngestTurnIn(
            session_id="s",
            turn_id="t",
            prompt="hello",
            final_result="hi",
            telemetry={"prompt_tokens": 3},
        )
    )
    assert result["available"] is False
    assert result["perform_deep_audit"] is True
    assert result["fail_open"] is True


def test_prepare_audit_discovers_visible_high_impact_claim_without_client_claims(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    result = prepare_completed_turn_audit(
        IngestTurnIn(
            session_id="claim-session",
            turn_id="claim-turn",
            prompt="report the result",
            final_result="Helix is at least 5x faster.",
            telemetry={"prompt_tokens": 10, "cached_tokens": 4},
        )
    )
    assert result["available"] is True
    assert result["perform_deep_audit"] is True
    assert "high_impact_claim" in result["artifacts"]["deep_audit_forced_reasons"]
    speed = next(item for item in result["artifacts"]["evidence"] if "5x" in item["claim"])
    assert speed["status"] == "UNVERIFIED"
    assert any("benchmark" in item for item in speed["missing_evidence"])
