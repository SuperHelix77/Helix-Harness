# SPDX-License-Identifier: AGPL-3.0-only

from routes import learning


def test_learning_entries_keep_newest_content_inside_budget():
    entries = [
        {"title": "old", "content": "x" * 100},
        {"title": "new", "content": "y" * 100},
    ]
    kept = learning._fit_entries(entries, 120)
    assert kept[-1]["title"] == "new"
    assert len(learning._entry_text(kept[-1])) <= 120


def test_learning_context_is_bounded_and_marked_as_approved():
    state = learning._empty_state()
    state["memory"] = [{"title": "Loader", "content": "Verify the active backend."}]
    state["user"] = [{"title": "Style", "content": "Prefer concise status updates."}]
    context = learning._context_instruction(state)
    assert context.startswith("<hermes_memory>")
    assert "user-approved" in context
    assert "Loader" in context
    assert "search_memory" in context
    assert "Verify the active backend" not in context
    assert "Prefer concise status updates." in context


def test_learning_state_defaults_to_the_guarded_experience_ladder():
    state = learning._empty_state()
    assert state["mem0Enabled"] is True
    assert state["onTheFlySkills"] is True
    assert state["decisionMode"] == "ask"
    assert state["allowQloraTraining"] is False
    context = learning._context_instruction(state)
    assert "record the completed experience first" in context
    assert "use Mem0 to find recurrence" in context
    assert "deterministic held-out benchmark" in context
