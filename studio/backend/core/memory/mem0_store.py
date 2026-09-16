# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Privacy-first Mem0 adapter for Unsloth Studio.

Mem0 is optional at import time and never receives a conversation by accident.
The adapter uses an account-scoped local Qdrant directory and a local
HuggingFace embedder. The Mem0 LLM is pointed at Studio's local OpenAI-compatible
endpoint by default, and telemetry is disabled before importing Mem0. If the
optional package or local embedding runtime is absent, the existing bounded JSON
learning ledger remains the source of truth.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from utils.paths import account_path, ensure_dir

_LOCK = threading.RLock()
_INSTANCES: dict[str, Any] = {}
_MAX_EXPERIENCE_CHARS = 8_000
_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_GRAPH_MAX_NODES = 400
_GRAPH_MAX_EDGES = 800


def _root() -> Path:
    path = account_path("learning/mem0")
    ensure_dir(path)
    return path


def _user_id(subject: str | None) -> str:
    # Do not put an email or account subject into the vector store.
    digest = hashlib.sha256(str(subject or "local").encode("utf-8")).hexdigest()[:32]
    return f"unsloth-{digest}"


def _import_mem0() -> Any:
    # Mem0's default telemetry is on. Set this before any mem0 import, including
    # its package-level initialization.
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from mem0 import Memory

    return Memory


def _config() -> dict[str, Any]:
    root = _root()
    embedding_model = os.environ.get("UNSLOTH_MEM0_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL).strip()
    embedding_dims = int(os.environ.get("UNSLOTH_MEM0_EMBEDDING_DIMS", "384"))
    llm_model = os.environ.get("UNSLOTH_MEM0_LLM_MODEL", "local").strip() or "local"
    llm_base_url = os.environ.get("UNSLOTH_MEM0_LLM_BASE_URL", "http://127.0.0.1:8888/v1").strip()
    llm_provider = os.environ.get("UNSLOTH_MEM0_LLM_PROVIDER", "openai").strip().lower() or "openai"
    if llm_provider == "ollama":
        llm_config: dict[str, Any] = {
            "model": llm_model,
            "ollama_base_url": os.environ.get("UNSLOTH_MEM0_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        }
    else:
        llm_config = {
            "model": llm_model,
            # The placeholder key is never sent to a remote endpoint by this
            # adapter unless the user explicitly overrides the base URL.
            "api_key": os.environ.get("UNSLOTH_MEM0_LLM_API_KEY", "local"),
            "openai_base_url": llm_base_url,
        }
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "unsloth_learning",
                "embedding_model_dims": embedding_dims,
                "path": str(root / "qdrant"),
            },
        },
        "llm": {"provider": llm_provider, "config": llm_config},
        "embedder": {
            "provider": "huggingface",
            "config": {"model": embedding_model, "embedding_dims": embedding_dims},
        },
        "history_db_path": str(root / "history.db"),
    }


def _instance() -> Any:
    key = str(_root()) + "\0" + os.environ.get("UNSLOTH_MEM0_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)
    with _LOCK:
        if key not in _INSTANCES:
            Memory = _import_mem0()
            _INSTANCES[key] = Memory.from_config(_config())
        return _INSTANCES[key]


def status() -> dict[str, Any]:
    try:
        import importlib.util

        installed = importlib.util.find_spec("mem0") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        installed = False
    configured = bool(os.environ.get("UNSLOTH_MEM0_LLM_BASE_URL", "http://127.0.0.1:8888/v1"))
    return {
        "installed": installed,
        "enabled": installed,
        "available": installed,
        "configured": configured,
        "backend": "mem0 + local qdrant + local HuggingFace embeddings" if installed else "bounded local learning ledger",
        "privacy": "Account-scoped local storage; Mem0 telemetry disabled; no remote endpoint unless explicitly configured.",
        "root": str(_root()),
        "error": None,
    }


def _graph_path() -> Path:
    return _root() / "graph.json"


def _empty_graph() -> dict[str, Any]:
    return {"nodes": [], "edges": []}


def _load_graph() -> dict[str, Any]:
    path = _graph_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_graph()
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _empty_graph()
    if not isinstance(raw, dict):
        return _empty_graph()
    nodes = [item for item in raw.get("nodes", []) if isinstance(item, dict)]
    edges = [item for item in raw.get("edges", []) if isinstance(item, dict)]
    return {"nodes": nodes[-_GRAPH_MAX_NODES:], "edges": edges[-_GRAPH_MAX_EDGES:]}


def _save_graph(graph: dict[str, Any]) -> None:
    path = _graph_path()
    ensure_dir(path.parent)
    payload = {
        "nodes": list(graph.get("nodes", []))[-_GRAPH_MAX_NODES:],
        "edges": list(graph.get("edges", []))[-_GRAPH_MAX_EDGES:],
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=".mem0-graph-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _node_id(title: str, text: str, thread_id: str) -> str:
    digest = hashlib.sha256(f"{title}\0{text}\0{thread_id}".encode("utf-8")).hexdigest()[:16]
    return f"n-{digest}"


def _append_graph_node(
    *,
    title: str,
    text: str,
    kind: str,
    thread_id: str,
    entities: list[str] | None,
    links: list[dict[str, str]] | None,
) -> dict[str, Any]:
    with _LOCK:
        graph = _load_graph()
        node_id = _node_id(title, text, thread_id)
        node = {
            "id": node_id,
            "title": title[:240],
            "text": text[:_MAX_EXPERIENCE_CHARS],
            "kind": kind,
            "thread_id": thread_id[:200],
            "entities": [str(item)[:80] for item in (entities or []) if str(item).strip()][:12],
            "created_at": int(time.time() * 1_000),
        }
        graph["nodes"] = [item for item in graph["nodes"] if item.get("id") != node_id]
        graph["nodes"].append(node)
        for link in links or []:
            source = str(link.get("from") or "").strip()[:80]
            target = str(link.get("to") or "").strip()[:80]
            relation = str(link.get("relation") or "related").strip()[:40] or "related"
            if source and target:
                graph["edges"].append(
                    {"from": source, "to": target, "relation": relation, "node_id": node_id}
                )
        _save_graph(graph)
        return node


def graph_snapshot(subject: str | None = None) -> dict[str, Any]:
    _ = subject
    graph = _load_graph()
    return {
        "nodes": graph.get("nodes", []),
        "edges": graph.get("edges", []),
        "available": True,
    }


def thread_has_memory(thread_id: str | None) -> bool:
    if not thread_id:
        return False
    needle = str(thread_id)[:200]
    return any(node.get("thread_id") == needle for node in _load_graph().get("nodes", []))


def add_experience(
    subject: str | None,
    text: str,
    *,
    thread_id: str | None = None,
    kind: str = "experience",
    title: str | None = None,
    entities: list[str] | None = None,
    links: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    text = str(text or "").strip()[:_MAX_EXPERIENCE_CHARS]
    if not text:
        return {"stored": False, "reason": "empty"}
    heading = (title or text.split("\n", 1)[0]).strip()[:240] or kind
    node = _append_graph_node(
        title=heading,
        text=text,
        kind=str(kind or "experience")[:40],
        thread_id=str(thread_id or ""),
        entities=entities,
        links=links,
    )
    try:
        memory = _instance()
        result = memory.add(
            text,
            user_id=_user_id(subject),
            metadata={
                "source": "unsloth-studio",
                "kind": kind,
                "thread_id": str(thread_id or "")[:200],
                "title": heading,
                "node_id": node["id"],
            },
            infer=False,
        )
        return {"stored": True, "result": result, "node": node}
    except Exception as error:  # noqa: BLE001 -- graph is enough; Mem0 package is optional
        return {"stored": True, "node": node, "reason": str(error)[:1_000]}


def _graph_search(query: str, limit: int) -> list[dict[str, Any]]:
    tokens = {token for token in query.lower().split() if len(token) >= 3}
    scored: list[tuple[int, dict[str, Any]]] = []
    graph = _load_graph()
    for node in graph.get("nodes", []):
        hay = " ".join(
            [
                str(node.get("title") or ""),
                str(node.get("text") or ""),
                " ".join(node.get("entities") or []),
            ]
        ).lower()
        score = sum(1 for token in tokens if token in hay)
        if score:
            scored.append((score, node))
    scored.sort(key=lambda item: -item[0])
    hits = []
    neighbor_names = {str(node.get("title") or "") for _, node in scored[:limit]}
    for score, node in scored[:limit]:
        related = [
            edge
            for edge in graph.get("edges", [])
            if edge.get("node_id") == node.get("id")
            or edge.get("from") in neighbor_names
            or edge.get("to") in neighbor_names
        ]
        hits.append(
            {
                "id": node.get("id"),
                "memory": node.get("text"),
                "title": node.get("title"),
                "score": score,
                "source": "mem0-graph",
                "metadata": {
                    "kind": node.get("kind"),
                    "thread_id": node.get("thread_id"),
                    "entities": node.get("entities") or [],
                    "links": related[:8],
                },
            }
        )
    return hits


def search(subject: str | None, query: str, limit: int = 5) -> dict[str, Any]:
    query = str(query or "").strip()[:2_000]
    if not query:
        return {"results": [], "available": False, "reason": "empty"}
    cap = max(1, min(int(limit), 10))
    results: list[Any] = []
    vector_ok = False
    try:
        memory = _instance()
        user_id = _user_id(subject)
        try:
            # Mem0 2.x moved entity selectors under ``filters``.  Keep a
            # narrow fallback for older installations rather than treating an
            # API-shape mismatch as vector-memory unavailability.
            found = memory.search(query, filters={"user_id": user_id}, limit=cap)
        except (TypeError, ValueError) as error:
            text = str(error)
            if "filters" not in text and "Top-level entity parameters" not in text:
                raise
            found = memory.search(query, user_id=user_id, limit=cap)
        if isinstance(found, dict):
            vector_rows = found.get("results")
            if isinstance(vector_rows, list):
                results.extend(item for item in vector_rows if isinstance(item, dict))
                vector_ok = True
        elif isinstance(found, list):
            results.extend(item for item in found if isinstance(item, dict))
            vector_ok = True
    except Exception:
        vector_ok = False
    graph_hits = _graph_search(query, cap)

    def _keys(item: dict[str, Any]) -> set[str]:
        keys: set[str] = set()
        item_id = str(item.get("id") or "").strip()
        if item_id:
            keys.add(f"id:{item_id}")
        text = str(item.get("memory") or item.get("text") or "").strip()
        if text:
            # Vector Mem0 and the bounded graph intentionally store the same
            # experience for fail-open redundancy. Do not spend two recall slots
            # on byte-equivalent copies merely because the stores use different ids.
            keys.add("text:" + " ".join(text.casefold().split()))
        return keys

    seen: set[str] = set()
    for item in results:
        if isinstance(item, dict):
            seen.update(_keys(item))
    for hit in graph_hits:
        keys = _keys(hit)
        if keys and keys.intersection(seen):
            continue
        results.append(hit)
        seen.update(keys)
    available = vector_ok or bool(graph_hits)
    return {"results": results[:cap], "available": available}
