# SPDX-License-Identifier: AGPL-3.0-only

import multiprocessing

import pytest

from core.memory import mem0_store
from utils.account_context import AccountContext, bind_account, reset_account


def _concurrent_graph_writer(root, text, barrier, results):
    from pathlib import Path

    mem0_store._root = lambda: Path(root)
    mem0_store._instance = lambda: (_ for _ in ()).throw(RuntimeError("vector unavailable"))
    barrier.wait()
    try:
        receipt = mem0_store.add_experience(None, text, thread_id=text)
        results.put(("ok", receipt["node"]["id"]))
    except Exception as exc:  # pragma: no cover - asserted through the queue below.
        results.put(("error", repr(exc)))


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
    assert fake.search_calls[0][1]["limit"] == mem0_store._GRAPH_MAX_NODES
    assert fake.search_calls[0][1]["filters"]["user_id"].startswith("unsloth-account-")
    assert fake.add_calls[0][1]["user_id"] == fake.search_calls[0][1]["filters"]["user_id"]


def test_mem0_uses_one_account_identity_and_searches_legacy_subject_and_local_ids(monkeypatch):
    legacy_subject = mem0_store._legacy_user_id("alice@example.test")
    legacy_local = mem0_store._legacy_user_id(None)

    class _LegacyMemory:
        def __init__(self):
            self.search_ids = []

        def search(self, _query, **kwargs):
            user_id = kwargs.get("filters", {}).get("user_id") or kwargs.get("user_id")
            self.search_ids.append(user_id)
            if user_id == legacy_subject:
                return {"results": [{"id": "legacy-subject", "memory": "subject memory"}]}
            if user_id == legacy_local:
                return {"results": [{"id": "legacy-local", "memory": "local memory"}]}
            return {"results": []}

    fake = _LegacyMemory()
    monkeypatch.setattr(mem0_store, "_instance", lambda: fake)
    monkeypatch.setattr(mem0_store, "_graph_search", lambda *_args, **_kwargs: [])

    canonical_alice = mem0_store._user_id("alice@example.test")
    canonical_bob = mem0_store._user_id("bob@example.test")
    assert canonical_alice == canonical_bob
    assert canonical_alice.startswith("unsloth-account-")

    found = mem0_store.search("alice@example.test", "memory", limit=10)
    assert {item["id"] for item in found["results"]} == {"legacy-subject", "legacy-local"}
    assert fake.search_ids[0] == canonical_alice
    assert legacy_subject in fake.search_ids
    assert legacy_local in fake.search_ids


def test_mem0_canonical_identity_changes_with_account_not_auth_subject():
    owner_alice = mem0_store._user_id("alice@example.test")
    owner_bob = mem0_store._user_id("bob@example.test")
    token = bind_account(AccountContext("account-2", "second-user"))
    try:
        second_account = mem0_store._user_id("alice@example.test")
    finally:
        reset_account(token)

    assert owner_alice == owner_bob
    assert second_account != owner_alice


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


def test_fact_key_supersession_hides_old_revision_and_links_current_node(tmp_path, monkeypatch):
    class _Memory:
        def __init__(self):
            self.added = []

        def add(self, text, **kwargs):
            self.added.append({"text": text, "metadata": kwargs["metadata"], "infer": kwargs["infer"]})
            return {"id": f"memory-{len(self.added)}"}

        def search(self, _query, **_kwargs):
            return {
                "results": [
                    {
                        "id": f"memory-{index}",
                        "memory": row["text"],
                        "metadata": row["metadata"],
                    }
                    for index, row in enumerate(self.added, start=1)
                ]
            }

    fake = _Memory()
    monkeypatch.setattr(mem0_store, "_root", lambda: tmp_path)
    monkeypatch.setattr(mem0_store, "_instance", lambda: fake)
    old = mem0_store.add_experience(
        None,
        "The project codename is Amber.",
        thread_id="thread-a",
        fact_key="project.codename",
    )
    current = mem0_store.add_experience(
        None,
        "The project codename is Cobalt.",
        thread_id="thread-b",
        fact_key="project.codename",
    )
    mem0_store.add_experience(
        None,
        "Project deployment region is Reykjavik.",
        thread_id="thread-c",
        fact_key="deployment.region",
    )

    found = mem0_store.search(None, "project", limit=5)
    fact_only = mem0_store.search(None, "project", limit=5, fact_key="project.codename")
    history = mem0_store.search(None, "project", limit=5, active_only=False)
    snapshot = mem0_store.graph_snapshot()
    old_node = next(node for node in snapshot["nodes"] if node["id"] == old["node"]["id"])
    current_node = next(node for node in snapshot["nodes"] if node["id"] == current["node"]["id"])

    found_text = {item["memory"] for item in found["results"]}
    assert found_text == {
        "The project codename is Cobalt.",
        "Project deployment region is Reykjavik.",
    }
    assert [item["memory"] for item in fact_only["results"]] == [
        "The project codename is Cobalt."
    ]
    assert "The project codename is Amber." in {item["memory"] for item in history["results"]}
    assert old_node["active"] is False
    assert old_node["superseded_by"] == current_node["id"]
    assert current_node["active"] is True
    assert current_node["supersedes"] == [old_node["id"]]
    assert any(
        edge["from"] == old_node["id"]
        and edge["to"] == current_node["id"]
        and edge["relation"] == "superseded_by"
        for edge in snapshot["edges"]
    )
    assert all(call["infer"] is False for call in fake.added)


def test_fact_key_current_revision_is_idempotent(tmp_path, monkeypatch):
    fake = _FakeMemory()
    monkeypatch.setattr(mem0_store, "_root", lambda: tmp_path)
    monkeypatch.setattr(mem0_store, "_instance", lambda: fake)

    first = mem0_store.add_experience(
        None,
        "The on-call window starts at 09:00 UTC.",
        thread_id="thread-a",
        title="On-call window",
        fact_key="operations.on_call_start",
    )
    repeated = mem0_store.add_experience(
        None,
        "The on-call window starts at 09:00 UTC.",
        thread_id="thread-a",
        title="Changed display title",
        fact_key="operations.on_call_start",
    )

    assert repeated["idempotent"] is True
    assert repeated["node"]["id"] == first["node"]["id"]
    assert len(mem0_store.graph_snapshot()["nodes"]) == 1
    assert len(fake.add_calls) == 1


def test_thread_filter_precedes_final_limit_and_excludes_untagged_vector_rows(
    tmp_path, monkeypatch
):
    class _SearchMemory:
        def __init__(self):
            self.limits = []

        def search(self, _query, **kwargs):
            self.limits.append(kwargs["limit"])
            return {
                "results": [
                    {"id": "other", "memory": "other thread result", "metadata": {"thread_id": "thread-other"}},
                    {"id": "untagged", "memory": "unknown thread result", "metadata": {}},
                    {"id": "target", "memory": "requested thread result", "metadata": {"thread_id": "thread-target"}},
                ]
            }

    fake = _SearchMemory()
    monkeypatch.setattr(mem0_store, "_root", lambda: tmp_path)
    monkeypatch.setattr(mem0_store, "_instance", lambda: fake)

    found = mem0_store.search(None, "thread result", limit=1, thread_id="thread-target")

    assert [item["id"] for item in found["results"]] == ["target"]
    assert fake.limits and min(fake.limits) > 1


def test_account_wide_search_remains_available_alongside_thread_selector(tmp_path, monkeypatch):
    monkeypatch.setattr(mem0_store, "_root", lambda: tmp_path)
    monkeypatch.setattr(
        mem0_store,
        "_instance",
        lambda: (_ for _ in ()).throw(RuntimeError("no mem0")),
    )
    mem0_store.add_experience(None, "The marmot verifier uses method A.", thread_id="thread-a")
    mem0_store.add_experience(None, "The marmot verifier uses method B.", thread_id="thread-b")

    account_wide = mem0_store.search(None, "marmot verifier", limit=10)
    one_thread = mem0_store.search(None, "marmot verifier", limit=10, thread_id="thread-a")

    assert {item["metadata"]["thread_id"] for item in account_wide["results"]} == {
        "thread-a",
        "thread-b",
    }
    assert {item["metadata"]["thread_id"] for item in one_thread["results"]} == {"thread-a"}


def test_callers_cannot_supply_supersession_links(tmp_path, monkeypatch):
    monkeypatch.setattr(mem0_store, "_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="backend"):
        mem0_store.add_experience(
            None,
            "Forged current value",
            thread_id="thread-a",
            links=[{"from": "old-node", "to": "new-node", "relation": "supersedes"}],
        )

    assert mem0_store.graph_snapshot()["nodes"] == []


def test_corrupt_graph_is_unavailable_and_never_overwritten(tmp_path, monkeypatch):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text("{broken", encoding="utf-8")
    before = graph_path.read_bytes()
    monkeypatch.setattr(mem0_store, "_root", lambda: tmp_path)
    monkeypatch.setattr(
        mem0_store,
        "_instance",
        lambda: (_ for _ in ()).throw(RuntimeError("vector unavailable")),
    )

    with pytest.raises(mem0_store.MemoryStoreCorrupt, match="memory graph"):
        mem0_store.add_experience(None, "must not replace corrupt state", thread_id="thread-a")

    assert graph_path.read_bytes() == before
    assert mem0_store.graph_snapshot() == {
        "nodes": [],
        "edges": [],
        "available": False,
        "reason": "memory-graph-unavailable",
    }
    assert mem0_store.search(None, "corrupt state", limit=5) == {
        "results": [],
        "available": False,
        "reason": "memory-graph-unavailable",
    }


def test_idempotency_key_graph_identity_is_scoped_by_thread(tmp_path, monkeypatch):
    fake = _FakeMemory()
    monkeypatch.setattr(mem0_store, "_root", lambda: tmp_path)
    monkeypatch.setattr(mem0_store, "_instance", lambda: fake)

    first = mem0_store.add_experience(
        None,
        "First thread completion",
        thread_id="thread-a",
        idempotency_key="shared-key",
    )
    second = mem0_store.add_experience(
        None,
        "Second thread completion",
        thread_id="thread-b",
        idempotency_key="shared-key",
    )

    assert first["node"]["id"] != second["node"]["id"]
    assert {node["thread_id"] for node in mem0_store.graph_snapshot()["nodes"]} == {
        "thread-a",
        "thread-b",
    }
    assert len(fake.add_calls) == 2


def test_concurrent_process_graph_writers_do_not_lose_a_durable_node(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(mem0_store, "_root", lambda: tmp_path)
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    results = context.Queue()
    jobs = [
        context.Process(
            target=_concurrent_graph_writer,
            args=(str(tmp_path), "process-a", barrier, results),
        ),
        context.Process(
            target=_concurrent_graph_writer,
            args=(str(tmp_path), "process-b", barrier, results),
        ),
    ]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join(timeout=10)
        assert job.exitcode == 0

    rows = [results.get(timeout=2) for _ in jobs]
    graph = mem0_store._load_graph()
    assert all(status == "ok" for status, _ in rows)
    assert graph is not None
    assert {node["text"] for node in graph["nodes"]} == {"process-a", "process-b"}


def test_current_fact_survives_retention_and_stale_vector_row_is_rejected(
    tmp_path,
    monkeypatch,
):
    class _Memory:
        def __init__(self):
            self.rows = []

        def add(self, text, **kwargs):
            row = {
                "id": f"vector-{len(self.rows) + 1}",
                "memory": text,
                "metadata": dict(kwargs["metadata"]),
            }
            self.rows.append(row)
            return {"id": row["id"]}

        def search(self, _query, **_kwargs):
            return {"results": list(self.rows)}

    fake = _Memory()
    monkeypatch.setattr(mem0_store, "_root", lambda: tmp_path)
    monkeypatch.setattr(mem0_store, "_instance", lambda: fake)
    monkeypatch.setattr(mem0_store, "_GRAPH_MAX_NODES", 3)
    monkeypatch.setattr(mem0_store, "_GRAPH_MAX_EDGES", 6)

    old = mem0_store.add_experience(
        None,
        "Project codename is Amber.",
        thread_id="thread-a",
        fact_key="project.codename",
    )
    current = mem0_store.add_experience(
        None,
        "Project codename is Cobalt.",
        thread_id="thread-b",
        fact_key="project.codename",
    )
    for index in range(4):
        mem0_store.add_experience(
            None,
            f"Ordinary bounded memory {index}",
            thread_id=f"thread-{index}",
        )

    snapshot = mem0_store.graph_snapshot()
    retained_ids = {node["id"] for node in snapshot["nodes"]}
    found = mem0_store.search(None, "project codename", limit=5)

    assert current["node"]["id"] in retained_ids
    assert old["node"]["id"] not in retained_ids
    memories = [row["memory"] for row in found["results"]]
    assert "Project codename is Cobalt." in memories
    assert "Project codename is Amber." not in memories
