# SPDX-License-Identifier: AGPL-3.0-only

from core.helix_engine.capture import clear_session, record_tool_execution
from core.helix_engine.critic import critic_from_steps, parse_self_critic
from core.helix_engine.ingest import ingest_turn
from core.helix_engine.trajectory import ToolStep, Trajectory
from core.helix_engine.routing import route_adaptation


def test_parse_self_critic_reads_finished_and_tool_judgement():
    critic = parse_self_critic(
        {
            "finished": "partial",
            "rightTools": False,
            "rightSkills": False,
            "tooManyTools": True,
            "toolCount": 9,
            "usedSkills": [],
            "notes": "reread the same file",
            "recommendation": "skill",
            "skillTitle": "Use the existing json skill",
            "skillContent": "Call read_skill before inventing a workflow.",
            "reason": "The task already has a skill.",
        }
    )
    assert critic.finished == "partial"
    assert critic.right_tools is False
    assert critic.too_many_tools is True
    assert critic.recommendation == "skill"


def test_unfinished_or_wrong_tools_route_hermes_not_qlora():
    traj = Trajectory(
        prompt_state="fix helper",
        retrieved_context="",
        reasoning="",
        steps=[ToolStep(name="read_file", arguments="a.py", result="ok", useful_hint="useful")],
        user_corrections=[],
        final_result="still broken",
        latency_ms=1,
        prompt_tokens=1,
        completion_tokens=1,
        verified=False,
        extras={
            "critic": {
                "finished": "partial",
                "right_tools": False,
                "right_skills": False,
                "too_many_tools": True,
                "recommendation": "qlora",
            }
        },
    )
    assert route_adaptation(traj) == "hermes"


def test_full_finish_with_right_tools_does_not_train():
    traj = Trajectory(
        prompt_state="what time is it",
        retrieved_context="",
        reasoning="",
        steps=[],
        user_corrections=[],
        final_result="afternoon",
        latency_ms=1,
        prompt_tokens=1,
        completion_tokens=1,
        verified=True,
        extras={
            "critic": {
                "finished": "full",
                "right_tools": True,
                "right_skills": True,
                "too_many_tools": False,
                "recommendation": "none",
            }
        },
    )
    assert route_adaptation(traj) == "discard"


def test_deterministic_critic_flags_redundant_tool_spam():
    steps = [
        ToolStep(name="read_file", arguments="a.py", result="ok", useful_hint="useful"),
        ToolStep(name="read_file", arguments="a.py", result="ok", useful_hint="redundant"),
        ToolStep(name="search_repo", arguments="a.py", result="same", useful_hint="redundant"),
        ToolStep(name="grep", arguments="a", result="same", useful_hint="redundant"),
    ]
    critic = critic_from_steps(steps, finished="partial")
    assert critic.too_many_tools is True
    assert critic.recommendation == "skill"


def test_ingest_turn_stages_hermes_skill_and_never_auto_qlora(tmp_path, monkeypatch):
    from routes import learning, self_training

    monkeypatch.setattr(learning, "_state_path", lambda: tmp_path / "learning.json")
    monkeypatch.setattr(self_training, "_state_path", lambda: tmp_path / "qlora.json")
    monkeypatch.setattr(self_training, "_dataset_path", lambda: tmp_path / "examples.jsonl")

    clear_session("critic-session")
    record_tool_execution("critic-session", "read_file", {"path": "a.py"}, "ok")
    record_tool_execution("critic-session", "read_file", {"path": "a.py"}, "ok")
    result = ingest_turn(
        session_id="critic-session",
        prompt="verify the json contract",
        final_result="still missing a check",
        thread_id="thread-1",
        subject="tester",
        critic={
            "finished": "partial",
            "rightTools": False,
            "rightSkills": False,
            "tooManyTools": True,
            "toolCount": 2,
            "usedSkills": [],
            "recommendation": "qlora",
            "skillTitle": "json-checks",
            "skillContent": "Read the json-checks skill and follow it.",
            "reason": "Existing skill covers this.",
        },
    )
    assert result["success_authorizes_weight_update"] is False
    assert result["route"] == "hermes"
    assert result["decision"] == "promote_hermes"
    assert "stage_skill" in result["actions"]
    assert "start_qlora" not in result["actions"]
    state = learning._read_state()
    assert state["pending"]
    assert state["pending"][-1]["kind"] == "skill"
    qlora = self_training._read_state()
    assert (qlora.get("lastRecommendation") or {}).get("action") == "qlora"
    assert (qlora.get("lastRecommendation") or {}).get("advisoryOnly") is True
    assert qlora.get("status") != "queued"
