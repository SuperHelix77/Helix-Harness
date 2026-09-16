# SPDX-License-Identifier: AGPL-3.0-only

from core.memory import mem0_store


class _FakeMemory:
    def __init__(self):
        self.add_calls = []
        self.search_calls = []

    def add(self, *args, **kwargs):
        self.add_calls.append((args, kwargs))
        return {"id": "memory-1"}

    def search(self, *args, **kwargs):
        self.search_calls.append((args, kwargs))
        return [{"id": "memory-1", "memory": "repeatable procedure"}]


def test_mem0_is_account_scoped_and_supplementary(monkeypatch):
    fake = _FakeMemory()
    monkeypatch.setattr(mem0_store, "_instance", lambda: fake)
    added = mem0_store.add_experience("alice@example.test", "Task experience", thread_id="thread-1")
    found = mem0_store.search("alice@example.test", "similar task", limit=99)

    assert added["stored"] is True
    assert fake.add_calls[0][1]["infer"] is False
    assert fake.add_calls[0][1]["user_id"].startswith("unsloth-")
    assert "alice@example.test" not in fake.add_calls[0][1]["user_id"]
    assert found["available"] is True
    assert found["results"][0]["memory"] == "repeatable procedure"
    assert fake.search_calls[0][1]["limit"] == 10
    assert fake.search_calls[0][1]["filters"]["user_id"].startswith("unsloth-")


def test_mem0_graph_persists_nodes_and_links_without_dumping_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(mem0_store, "_root", lambda: tmp_path)
    monkeypatch.setattr(mem0_store, "_instance", lambda: (_ for _ in ()).throw(RuntimeError("no mem0")))
    added = mem0_store.add_experience(
        "alice@example.test",
        "Prefer q4_0 KV on Qwen3.8 Mac loads.",
        thread_id = "thread-9",
        kind = "context-handoff",
        title = "Q38 KV default",
        entities = ["Qwen3.8", "q4_0"],
        links = [{"from": "Qwen3.8", "to": "q4_0", "relation": "uses"}],
    )
    assert added["stored"] is True
    snap = mem0_store.graph_snapshot("alice@example.test")
    titles = [node["title"] for node in snap["nodes"]]
    assert "Q38 KV default" in titles
    assert snap["edges"]
    found = mem0_store.search("alice@example.test", "Qwen3.8 KV", limit = 5)
    assert found["available"] is True
    assert any("q4_0" in str(item).lower() for item in found["results"])


class _FakeMemoryV2:
    def search(self, _query, **kwargs):
        assert "user_id" not in kwargs
        assert kwargs["filters"]["user_id"].startswith("unsloth-")
        return {
            "results": [
                {
                    "id": "v2-memory",
                    "memory": "vector-backed result",
                    "metadata": {"thread_id": "thread-v2"},
                    "score": 0.9,
                }
            ]
        }


def test_mem0_v2_structured_search_results_are_used(monkeypatch):
    monkeypatch.setattr(mem0_store, "_instance", lambda: _FakeMemoryV2())
    monkeypatch.setattr(mem0_store, "_graph_search", lambda *_args, **_kwargs: [])
    found = mem0_store.search("alice@example.test", "vector result", limit=3)
    assert found["available"] is True
    assert found["results"][0]["id"] == "v2-memory"
    assert found["results"][0]["memory"] == "vector-backed result"


def test_vector_and_graph_copies_of_same_memory_consume_one_slot(monkeypatch):
    class _Vector:
        def search(self, _query, **_kwargs):
            return {"results": [{"id": "vector-1", "memory": "Same durable memory", "score": 0.8}]}

    monkeypatch.setattr(mem0_store, "_instance", lambda: _Vector())
    monkeypatch.setattr(
        mem0_store,
        "_graph_search",
        lambda *_args, **_kwargs: [
            {"id": "graph-1", "memory": "Same durable memory", "source": "mem0-graph"},
            {"id": "graph-2", "memory": "Different fallback memory", "source": "mem0-graph"},
        ],
    )
    found = mem0_store.search("alice@example.test", "durable memory", limit=5)
    assert [item["id"] for item in found["results"]] == ["vector-1", "graph-2"]
