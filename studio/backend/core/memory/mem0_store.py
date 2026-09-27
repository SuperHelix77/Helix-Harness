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

import base64
import binascii
import contextlib
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Unix/macOS is the qualified runtime.
    fcntl = None

from utils.account_context import current_account
from utils.paths import account_path, ensure_dir

_LOCK = threading.RLock()
_INSTANCES: dict[str, Any] = {}
_MAX_EXPERIENCE_CHARS = 8_000
_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_TYPED_SCHEMA = "helix.memory.typed.v1"
_TYPED_MAX_NODES = 400
_GRAPH_MAX_NODES = 800
_GRAPH_MAX_EDGES = 1_600
_OBSERVATION_SOURCE_TOOLS = frozenset(
    {"web_search", "search_knowledge_base", "read_observation"}
)


class MemoryStoreCorrupt(RuntimeError):
    """Durable memory exists but cannot be safely interpreted."""


@contextlib.contextmanager
def _memory_state_transaction():
    """Serialize graph and typed-sidecar read/modify/replace transactions."""

    with _LOCK:
        root = _root()
        ensure_dir(root)
        with (root / ".memory-state.lock").open("a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _root() -> Path:
    path = account_path("learning/mem0")
    ensure_dir(path)
    return path


def _legacy_user_id(subject: str | None) -> str:
    """Return the pre-account-scope Mem0 identity for compatibility reads."""
    digest = hashlib.sha256(str(subject or "local").encode("utf-8")).hexdigest()[:32]
    return f"unsloth-{digest}"


def _user_id(subject: str | None = None) -> str:
    """Canonical vector identity for this account-scoped Mem0 store.

    The Qdrant directory itself is already isolated by immutable account id. New
    writes therefore use one stable identity per account instead of fragmenting a
    single store by whichever auth subject happened to reach the caller.
    """
    _ = subject
    account_id = str(current_account().account_id or "owner")
    digest = hashlib.sha256(f"account:{account_id}".encode("utf-8")).hexdigest()[:32]
    return f"unsloth-account-{digest}"


def _search_user_ids(subject: str | None) -> list[str]:
    """Canonical identity first, then legacy subject/local identities."""
    account = current_account()
    legacy_subjects = [subject, account.username, "local"]
    identities = [_user_id(subject)]
    for legacy_subject in legacy_subjects:
        legacy = _legacy_user_id(legacy_subject)
        if legacy not in identities:
            identities.append(legacy)
    return identities


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


def _typed_path() -> Path:
    return _root() / "typed.json"


def _empty_graph() -> dict[str, Any]:
    return {"nodes": [], "edges": []}


def _empty_typed() -> dict[str, Any]:
    return {"schema": _TYPED_SCHEMA, "nodes": []}


def _load_graph() -> dict[str, Any] | None:
    path = _graph_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_graph()
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    raw_nodes = raw.get("nodes")
    raw_edges = raw.get("edges")
    if (
        not isinstance(raw_nodes, list)
        or not isinstance(raw_edges, list)
        or len(raw_nodes) > _GRAPH_MAX_NODES
    ):
        return None

    nodes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    active_facts: dict[str, str] = {}
    for item in raw_nodes:
        if not isinstance(item, dict):
            return None
        node_id = str(item.get("id") or "").strip()
        if not node_id or node_id in seen_ids:
            return None
        seen_ids.add(node_id)
        if "active" in item and type(item["active"]) is not bool:
            return None
        fact_key = str(item.get("fact_key") or "").strip()
        if fact_key and item.get("active", True) is not False:
            prior = active_facts.get(fact_key)
            if prior is not None and prior != node_id:
                return None
            active_facts[fact_key] = node_id
        nodes.append(item)
    if any(not isinstance(item, dict) for item in raw_edges):
        return None
    edges = list(raw_edges)
    return {"nodes": nodes[-_GRAPH_MAX_NODES:], "edges": edges[-_GRAPH_MAX_EDGES:]}


def _load_typed() -> dict[str, Any] | None:
    """Load the account-scoped typed sidecar, rejecting malformed state."""

    path = _typed_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_typed()
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or raw.get("schema") != _TYPED_SCHEMA:
        return None
    nodes = raw.get("nodes")
    if not isinstance(nodes, list) or len(nodes) > _TYPED_MAX_NODES:
        return None
    account_id = str(current_account().account_id or "owner")
    cleaned: list[dict[str, Any]] = []
    for item in nodes:
        if not isinstance(item, dict):
            return None
        if str(item.get("account_id") or "") != account_id:
            return None
        if not isinstance(item.get("node_id"), str) or not item["node_id"]:
            return None
        if not isinstance(item.get("record_id"), str) or not item["record_id"]:
            return None
        if not isinstance(item.get("thread_id"), str) or not item["thread_id"]:
            return None
        if type(item.get("active")) is not bool:
            return None
        if not isinstance(item.get("record"), dict):
            return None
        if not isinstance(item.get("references"), list):
            return None
        if not isinstance(item.get("evidence"), dict):
            return None
        cleaned.append(item)
    return {"schema": _TYPED_SCHEMA, "nodes": cleaned}


def _save_typed(typed: dict[str, Any]) -> None:
    path = _typed_path()
    ensure_dir(path.parent)
    nodes = list(typed.get("nodes", []))
    if len(nodes) > _TYPED_MAX_NODES:
        raise MemoryStoreCorrupt("typed sidecar capacity exceeded")
    payload = {
        "schema": _TYPED_SCHEMA,
        "nodes": nodes,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=".mem0-typed-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:
        pass


def _retained_graph_nodes(
    nodes: list[dict[str, Any]],
    *,
    pinned_node_ids: set[str],
) -> list[dict[str, Any]]:
    """Retain typed evidence and current facts before ordinary recent nodes."""

    if len(pinned_node_ids) > _GRAPH_MAX_NODES:
        raise MemoryStoreCorrupt("typed evidence exceeds the graph retention bound")
    selected_indexes: set[int] = set()
    selected_ids = set(pinned_node_ids)
    for index, node in enumerate(nodes):
        if str(node.get("id") or "") in pinned_node_ids:
            selected_indexes.add(index)
    present_ids = {str(node.get("id") or "") for node in nodes}
    if pinned_node_ids - present_ids:
        raise MemoryStoreCorrupt("typed evidence is missing from the memory graph")

    remaining = _GRAPH_MAX_NODES - len(selected_indexes)
    current_facts = [
        index
        for index, node in enumerate(nodes)
        if index not in selected_indexes
        and str(node.get("id") or "") not in selected_ids
        and node.get("active", True) is not False
        and str(node.get("fact_key") or "").strip()
    ]
    if len(current_facts) > remaining:
        raise MemoryStoreCorrupt("current facts exceed the graph retention bound")
    for index in current_facts:
        selected_indexes.add(index)
        selected_ids.add(str(nodes[index].get("id") or ""))

    remaining = _GRAPH_MAX_NODES - len(selected_indexes)
    ordinary = [
        index
        for index, node in enumerate(nodes)
        if index not in selected_indexes
        and str(node.get("id") or "") not in selected_ids
    ]
    if remaining > 0:
        for index in ordinary[-remaining:]:
            selected_indexes.add(index)
            selected_ids.add(str(nodes[index].get("id") or ""))
    return [node for index, node in enumerate(nodes) if index in selected_indexes]


def _save_graph(
    graph: dict[str, Any],
    *,
    pinned_node_ids: set[str],
) -> set[str]:
    path = _graph_path()
    ensure_dir(path.parent)
    nodes = _retained_graph_nodes(
        [item for item in graph.get("nodes", []) if isinstance(item, dict)],
        pinned_node_ids=pinned_node_ids,
    )
    retained_ids = {str(item.get("id") or "") for item in nodes}
    retained_ids.discard("")
    edges = [
        item
        for item in graph.get("edges", [])
        if isinstance(item, dict)
        and (
            str(item.get("node_id") or "") in retained_ids
            or str(item.get("from") or "") in retained_ids
            or str(item.get("to") or "") in retained_ids
        )
    ]
    payload = {
        "nodes": nodes,
        "edges": edges[-_GRAPH_MAX_EDGES:],
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
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:
        pass
    return retained_ids


def _node_id(
    title: str,
    text: str,
    thread_id: str,
    idempotency_key: str | None = None,
    *,
    fact_key: str | None = None,
    predecessor_ids: list[str] | None = None,
) -> str:
    if str(idempotency_key or "").strip():
        material = f"idempotency\0{thread_id}\0{idempotency_key}"
    elif str(fact_key or "").strip():
        # Include the prior active revision so restoring an old value later is a
        # new node, while retrying the current revision remains idempotent.
        material = f"fact\0{fact_key}\0{text}\0{','.join(sorted(predecessor_ids or []))}"
    else:
        material = f"{title}\0{text}\0{thread_id}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"n-{digest}"


def _append_graph_node(
    *,
    title: str,
    text: str,
    kind: str,
    thread_id: str,
    entities: list[str] | None,
    links: list[dict[str, str]] | None,
    idempotency_key: str | None = None,
    fact_key: str | None = None,
    binding: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    with _memory_state_transaction():
        typed = _load_typed()
        if typed is None:
            raise MemoryStoreCorrupt("typed sidecar is unavailable")
        graph = _load_graph()
        if graph is None:
            raise MemoryStoreCorrupt("memory graph is unavailable")
        fact_key = str(fact_key or "").strip()[:240] or None
        active_predecessors = [
            item
            for item in graph["nodes"]
            if fact_key
            and str(item.get("fact_key") or "") == fact_key
            and item.get("active", True) is not False
        ]
        current_match = next(
            (
                item
                for item in reversed(active_predecessors)
                if str(item.get("text") or "") == text
            ),
            None,
        )
        if current_match is not None:
            return current_match, False

        predecessor_ids = [str(item.get("id") or "") for item in active_predecessors]
        node_id = _node_id(
            title,
            text,
            thread_id,
            idempotency_key,
            fact_key=fact_key,
            predecessor_ids=predecessor_ids,
        )
        normalized_key = str(idempotency_key or "").strip()[:240]
        keyed_existing = next(
            (
                item
                for item in graph["nodes"]
                if normalized_key
                and str(item.get("idempotency_key") or "") == normalized_key
                and str(item.get("thread_id") or "") == str(thread_id)[:200]
            ),
            None,
        )
        if keyed_existing is not None and str(keyed_existing.get("id") or "") != node_id:
            raise MemoryStoreCorrupt(
                "idempotency key is already bound to different graph identity"
            )
        existing = next(
            (item for item in graph["nodes"] if str(item.get("id") or "") == node_id),
            None,
        )
        if existing is not None:
            if str(idempotency_key or "").strip() and (
                str(existing.get("thread_id") or "") != str(thread_id or "")[:200]
                or str(existing.get("text") or "") != text[:_MAX_EXPERIENCE_CHARS]
            ):
                raise MemoryStoreCorrupt(
                    "idempotency key is already bound to different memory content"
                )
            if binding is not None and existing.get("binding") != binding:
                raise MemoryStoreCorrupt(
                    "idempotency key is already bound to different receipt identity"
                )
            # Non-fact experiences keep their original deterministic identity.
            # A matching fact node can only be current here after the predecessor
            # check above; this branch also protects legacy graph collisions.
            return existing, False

        created_at = int(time.time() * 1_000)
        node = {
            "id": node_id,
            "title": title[:240],
            "text": text[:_MAX_EXPERIENCE_CHARS],
            "kind": kind,
            "thread_id": thread_id[:200],
            "entities": [str(item)[:80] for item in (entities or []) if str(item).strip()][:12],
            "created_at": created_at,
            "active": True,
        }
        if fact_key:
            node["fact_key"] = fact_key
        if predecessor_ids:
            node["supersedes"] = predecessor_ids
            for predecessor in active_predecessors:
                predecessor["active"] = False
                predecessor["superseded_by"] = node_id
                graph["edges"].append(
                    {
                        "from": str(predecessor.get("id") or ""),
                        "to": node_id,
                        "relation": "superseded_by",
                        "node_id": node_id,
                    }
                )
        if str(idempotency_key or "").strip():
            node["idempotency_key"] = str(idempotency_key).strip()[:240]
        if binding is not None:
            node["binding"] = dict(binding)
        graph["nodes"].append(node)
        for link in links or []:
            source = str(link.get("from") or "").strip()[:80]
            target = str(link.get("to") or "").strip()[:80]
            relation = str(link.get("relation") or "related").strip()[:40] or "related"
            if source and target:
                graph["edges"].append(
                    {"from": source, "to": target, "relation": relation, "node_id": node_id}
                )
        pinned_node_ids = {
            str(item.get("node_id") or "")
            for item in typed.get("nodes", [])
            if str(item.get("node_id") or "")
        }
        retained_ids = _save_graph(graph, pinned_node_ids=pinned_node_ids)
        if node_id not in retained_ids:
            raise MemoryStoreCorrupt("memory graph retention capacity is exhausted")
        return node, True


def graph_snapshot(subject: str | None = None) -> dict[str, Any]:
    _ = subject
    graph = _load_graph()
    if graph is None:
        return {
            "nodes": [],
            "edges": [],
            "available": False,
            "reason": "memory-graph-unavailable",
        }
    return {
        "nodes": graph.get("nodes", []),
        "edges": graph.get("edges", []),
        "available": True,
    }


def _encode_evidence(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _decode_evidence(value: object) -> bytes | None:
    if not isinstance(value, str):
        return None
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError, binascii.Error):
        return None


def _graph_node_for_typed(node_id: str) -> dict[str, Any] | None:
    node_id = str(node_id or "").strip()
    if not node_id:
        return None
    graph = _load_graph()
    if graph is None:
        return None
    for node in graph.get("nodes", []):
        if str(node.get("id") or "") == node_id:
            return node
    return None


def _graph_supersedes(node: dict[str, Any]) -> list[str]:
    values = node.get("supersedes")
    if values is None:
        return []
    if not isinstance(values, list):
        raise MemoryStoreCorrupt("memory graph supersession is malformed")
    return [str(value).strip()[:128] for value in values if str(value).strip()]


def _graph_node_for_idempotency_key(
    idempotency_key: str,
    *,
    thread_id: str | None = None,
) -> dict[str, Any] | None:
    """Return the sole local graph witness for a backend idempotency key."""

    key = str(idempotency_key or "").strip()[:240]
    if not key:
        return None
    graph = _load_graph()
    if graph is None:
        raise MemoryStoreCorrupt("memory graph is unavailable")
    matches = [
        item
        for item in graph.get("nodes", [])
        if isinstance(item, dict) and str(item.get("idempotency_key") or "") == key
        and (
            thread_id is None
            or str(item.get("thread_id") or "") == str(thread_id)[:200]
        )
    ]
    if len(matches) > 1:
        raise MemoryStoreCorrupt("memory graph idempotency key is ambiguous")
    return matches[0] if matches else None


def _trusted_fact_supersession(
    successor: dict[str, Any],
    predecessor: dict[str, Any],
) -> bool:
    """Whether graph state proves a backend-owned fact revision transition."""

    if (
        successor.get("active", True) is not True
        or predecessor.get("active", True) is not False
        or str(successor.get("fact_key") or "")
        != str(predecessor.get("fact_key") or "")
    ):
        return False
    successor_id = str(successor.get("id") or "")
    predecessor_id = str(predecessor.get("id") or "")
    if not successor_id or not predecessor_id:
        return False
    try:
        supersedes = _graph_supersedes(successor)
    except MemoryStoreCorrupt:
        return False
    return (
        supersedes == [predecessor_id]
        and str(predecessor.get("superseded_by") or "") == successor_id
    )


def _trusted_predecessor_activity_change(predecessor: dict[str, Any]) -> bool:
    """Whether graph state proves this node was superseded by a current fact."""

    graph = _load_graph()
    predecessor_id = str(predecessor.get("id") or "")
    successor_id = str(predecessor.get("superseded_by") or "")
    if graph is None or not predecessor_id or not successor_id:
        return False
    successor = next(
        (
            item
            for item in graph.get("nodes", [])
            if str(item.get("id") or "") == successor_id
        ),
        None,
    )
    return (
        successor is not None
        and _trusted_fact_supersession(successor, predecessor)
    )


def _register_typed_memory_locked(
    node_id: str,
    record_id: str,
    *,
    epistemic_class: Any,
    evidence: bytes | Mapping[str, bytes],
    record_text: str | None = None,
    record_type: str = "observation",
    evidence_id: str | None = None,
    media_type: str = "text/plain",
    thread_id: str | None = None,
    fact_key: str | None = None,
) -> dict[str, Any]:
    """Bind a backend-created graph node to immutable typed evidence.

    This is a backend-only registration API.  Account, thread, node activity,
    fact key, and supersession are read from the durable graph; caller-supplied
    values may only be checked for exact agreement.  The public memory routes
    never expose this function or its authority-bearing arguments.
    """

    from core.helix_engine.interlingua import EpistemicClass, Reference

    node = _graph_node_for_typed(node_id)
    if node is None:
        return {"stored": False, "reason": "node-not-found"}
    node_id = str(node.get("id") or "")
    graph_thread = str(node.get("thread_id") or "")
    if not graph_thread:
        return {"stored": False, "reason": "thread-provenance-missing"}
    if thread_id is not None and str(thread_id) != graph_thread:
        return {"stored": False, "reason": "thread-binding-mismatch"}
    graph_fact_key = str(node.get("fact_key") or "") or None
    if fact_key is not None and str(fact_key) != (graph_fact_key or ""):
        return {"stored": False, "reason": "fact-key-binding-mismatch"}
    try:
        graph_supersedes = _graph_supersedes(node)
    except MemoryStoreCorrupt:
        return {"stored": False, "reason": "graph-supersession-malformed"}
    expected_active = node.get("active", True) is not False
    account_id = str(current_account().account_id or "owner")

    record_id = str(record_id or "").strip()
    if not record_id:
        return {"stored": False, "reason": "record-id-missing"}
    graph_text = str(node.get("text") or "").strip()
    if record_text is not None and str(record_text).strip() != graph_text:
        return {"stored": False, "reason": "record-text-binding-mismatch"}
    record_text = graph_text
    if not record_text:
        return {"stored": False, "reason": "record-text-missing"}
    try:
        epistemic_class = EpistemicClass(epistemic_class)
    except (TypeError, ValueError):
        return {"stored": False, "reason": "invalid-epistemic-class"}
    if record_type not in {"observation", "claim"}:
        return {"stored": False, "reason": "invalid-record-type"}
    graph_kind = str(node.get("kind") or "")
    if graph_kind == "completed-turn-model-claim" and (
        epistemic_class is not EpistemicClass.MODEL_CLAIM
        or record_type != "claim"
    ):
        return {
            "stored": False,
            "reason": "completed-turn-memory-must-remain-model-claim",
        }
    if graph_kind == "durable-tool-observation" and (
        epistemic_class is not EpistemicClass.OBSERVED
        or record_type != "observation"
    ):
        return {
            "stored": False,
            "reason": "durable-observation-must-remain-observed",
        }

    if isinstance(evidence, bytes):
        if evidence_id is None:
            return {"stored": False, "reason": "evidence-id-missing"}
        evidence_items = {str(evidence_id): evidence}
    elif isinstance(evidence, Mapping):
        evidence_items = {
            str(key): value for key, value in evidence.items() if isinstance(value, bytes)
        }
        if len(evidence_items) != len(evidence):
            return {"stored": False, "reason": "evidence-must-be-bytes"}
    else:
        return {"stored": False, "reason": "invalid-evidence"}

    references: list[dict[str, str | int]] = []
    encoded_evidence: dict[str, str] = {}
    try:
        for reference_id, payload in sorted(evidence_items.items()):
            reference = Reference.from_bytes(reference_id, payload, media_type)
            references.append(
                {
                    "id": reference.id,
                    "sha256": reference.sha256,
                    "byte_length": reference.byte_length,
                    "media_type": reference.media_type,
                }
            )
            encoded_evidence[reference.id] = _encode_evidence(payload)
    except (TypeError, ValueError) as error:
        return {"stored": False, "reason": str(error)[:240]}

    record = {
        "type": record_type,
        "id": record_id,
        "text": record_text,
        "epistemic_class": epistemic_class.value,
        "evidence_refs": [item["id"] for item in references],
    }
    with _LOCK:
        typed = _load_typed()
        if typed is None:
            return {"stored": False, "reason": "typed-sidecar-unavailable"}
        existing_by_node = {
            str(item.get("node_id") or ""): item for item in typed["nodes"]
        }
        existing = existing_by_node.get(node_id)
        idempotent = existing is not None
        if existing is not None:
            immutable_mismatch = any(
                existing.get(field) != expected
                for field, expected in (
                    ("node_id", node_id),
                    ("record_id", record_id),
                    ("account_id", account_id),
                    ("thread_id", graph_thread),
                    ("fact_key", graph_fact_key),
                    ("supersedes", graph_supersedes),
                )
            )
            active_mismatch = existing.get("active") is not expected_active
            if immutable_mismatch or (
                active_mismatch
                and not (
                    graph_fact_key
                    and _trusted_predecessor_activity_change(node)
                )
            ) or any(
                existing.get(field) != expected
                for field, expected in (
                ("record_id", record_id),
                ("record", record),
                ("references", references),
                ("evidence", encoded_evidence),
                )
            ):
                return {"stored": False, "reason": "typed-binding-immutable"}
        else:
            if len(typed["nodes"]) >= _TYPED_MAX_NODES:
                return {"stored": False, "reason": "typed-sidecar-capacity"}
            if any(
                str(item.get("record_id") or "") == record_id
                for item in typed["nodes"]
            ):
                return {"stored": False, "reason": "record-id-already-bound"}
            existing = {
                "node_id": node_id,
                "record_id": record_id,
                "account_id": account_id,
                "thread_id": graph_thread,
                "active": expected_active,
                "supersedes": graph_supersedes,
                "fact_key": graph_fact_key,
                "record": record,
                "references": references,
                "evidence": encoded_evidence,
            }
            typed["nodes"].append(existing)
        # Activity and explicit graph supersession are backend state, not
        # caller/model assertions.  Update them on every trusted registration.
        if graph_fact_key:
            for sibling in typed["nodes"]:
                sibling_id = str(sibling.get("node_id") or "")
                sibling_node = _graph_node_for_typed(sibling_id)
                if sibling_node is None:
                    continue
                if str(sibling_node.get("fact_key") or "") != graph_fact_key:
                    continue
                sibling_expected_active = (
                    sibling_node.get("active", True) is not False
                )
                if (
                    sibling.get("active") is not sibling_expected_active
                    and not _trusted_predecessor_activity_change(sibling_node)
                ):
                    return {
                        "stored": False,
                        "reason": "typed-binding-immutable",
                    }
                sibling["active"] = sibling_expected_active
        existing["active"] = expected_active
        existing["thread_id"] = graph_thread
        existing["supersedes"] = graph_supersedes
        if graph_fact_key:
            existing["fact_key"] = graph_fact_key
        elif "fact_key" in existing:
            existing.pop("fact_key")
        _save_typed(typed)
    return {
        "stored": True,
        "node_id": node_id,
        "record_id": record_id,
        "idempotent": idempotent,
    }


def register_typed_memory(
    node_id: str,
    record_id: str,
    *,
    epistemic_class: Any,
    evidence: bytes | Mapping[str, bytes],
    record_text: str | None = None,
    record_type: str = "observation",
    evidence_id: str | None = None,
    media_type: str = "text/plain",
    thread_id: str | None = None,
    fact_key: str | None = None,
) -> dict[str, Any]:
    """Bind graph-backed typed evidence inside one serialized state transaction."""

    with _memory_state_transaction():
        return _register_typed_memory_locked(
            node_id,
            record_id,
            epistemic_class=epistemic_class,
            evidence=evidence,
            record_text=record_text,
            record_type=record_type,
            evidence_id=evidence_id,
            media_type=media_type,
            thread_id=thread_id,
            fact_key=fact_key,
        )


def _completed_turn_memory_identity(
    *,
    thread_id: str,
    idempotency_key: str,
) -> tuple[str, str]:
    account_id = str(current_account().account_id or "owner")
    digest = hashlib.sha256(
        (
            "helix.completed-turn-memory.v1\0"
            f"{account_id}\0{thread_id}\0{idempotency_key}"
        ).encode("utf-8")
    ).hexdigest()
    return f"mem-{digest[:32]}", f"evidence-{digest[:32]}"


def _completed_turn_memory_node(
    *,
    thread_id: str,
    idempotency_key: str,
) -> dict[str, Any] | None:
    graph = _load_graph()
    if graph is None:
        return None
    return next(
        (
            item
            for item in reversed(graph.get("nodes", []))
            if isinstance(item, dict)
            and str(item.get("idempotency_key") or "") == idempotency_key
            and str(item.get("thread_id") or "") == thread_id
        ),
        None,
    )


def completed_turn_memory_receipt_for_idempotency(
    idempotency_key: str,
    *,
    thread_id: str,
) -> dict[str, Any] | None:
    """Return a replay witness only after graph and typed trust both committed."""

    from core.helix_engine.interlingua import Claim, EpistemicClass

    key = str(idempotency_key or "").strip()[:240]
    thread = str(thread_id or "").strip()[:200]
    if not key or not thread:
        return None
    node = _completed_turn_memory_node(thread_id=thread, idempotency_key=key)
    if node is None:
        return None
    record_id, evidence_id = _completed_turn_memory_identity(
        thread_id=thread,
        idempotency_key=key,
    )
    loaded = _typed_state_and_bindings()
    if loaded is None:
        return None
    state, bindings, payloads = loaded
    node_id = str(node.get("id") or "")
    binding = bindings.get(node_id)
    account_id = str(current_account().account_id or "owner")
    text = str(node.get("text") or "")
    if (
        node_id != _node_id("", "", thread, key)
        or str(node.get("kind") or "") != "completed-turn-model-claim"
        or str(node.get("idempotency_key") or "") != key
        or str(node.get("thread_id") or "") != thread
        or node.get("active", True) is not True
        or binding is None
        or binding.record_id != record_id
        or binding.account_id != account_id
        or binding.thread_id != thread
        or binding.active is not True
    ):
        return None
    record = next((item for item in state.claims if item.id == record_id), None)
    if (
        not isinstance(record, Claim)
        or record.epistemic_class is not EpistemicClass.MODEL_CLAIM
        or record.text != text
        or record.evidence_refs != (evidence_id,)
        or payloads.get(evidence_id) != text.encode("utf-8")
    ):
        return None
    return {
        "stored": True,
        "node": node,
        "node_id": node_id,
        "record_id": record_id,
        "epistemic_class": "MODEL_CLAIM",
        "model_facing": False,
        "reason": "recovered_from_idempotent_typed_graph",
        "idempotent": True,
    }


def admit_completed_turn_memory(
    subject: str | None,
    text: str,
    *,
    thread_id: str,
    idempotency_key: str,
    title: str | None = None,
) -> dict[str, Any]:
    """Commit one finalized assistant claim to graph + typed memory.

    This backend-only path never accepts a caller-selected epistemic class.
    Assistant output is always a ``MODEL_CLAIM`` bound to the exact bytes that
    produced it. The graph and typed sidecar are canonical local witnesses;
    vector indexing starts only after both witnesses exist and remains optional
    redundancy.
    """

    normalized_text = str(text or "").strip()[:_MAX_EXPERIENCE_CHARS]
    normalized_thread = str(thread_id or "").strip()[:200]
    normalized_key = str(idempotency_key or "").strip()[:240]
    if not normalized_text:
        return {"stored": False, "reason": "empty"}
    if not normalized_thread:
        return {"stored": False, "reason": "thread-provenance-missing"}
    if not normalized_key:
        return {"stored": False, "reason": "finalization-idempotency-key-missing"}

    prior = completed_turn_memory_receipt_for_idempotency(
        normalized_key,
        thread_id=normalized_thread,
    )
    if prior is not None:
        return prior

    heading = (title or "Completed assistant turn").strip()[:240]
    graph_witness = _graph_node_for_idempotency_key(
        normalized_key,
    )
    if graph_witness is not None:
        prior = completed_turn_memory_receipt_for_idempotency(
            normalized_key,
            thread_id=normalized_thread,
        )
        if prior is None:
            raise MemoryStoreCorrupt(
                "completed-turn memory replay witness failed validation"
            )
        return prior
    node, created = _append_graph_node(
        title=heading,
        text=normalized_text,
        kind="completed-turn-model-claim",
        thread_id=normalized_thread,
        entities=None,
        links=None,
        idempotency_key=normalized_key,
    )
    if not created:
        recovered = completed_turn_memory_receipt_for_idempotency(
            normalized_key,
            thread_id=normalized_thread,
        )
        if recovered is not None:
            return recovered
        raise MemoryStoreCorrupt(
            "completed-turn memory replay witness failed validation"
        )
    node_id = str(node.get("id") or "")
    if not node_id:
        raise MemoryStoreCorrupt("completed-turn memory graph node has no identity")
    record_id, evidence_id = _completed_turn_memory_identity(
        thread_id=normalized_thread,
        idempotency_key=normalized_key,
    )
    from core.helix_engine.interlingua import EpistemicClass

    typed = register_typed_memory(
        node_id,
        record_id,
        epistemic_class=EpistemicClass.MODEL_CLAIM,
        evidence={evidence_id: normalized_text.encode("utf-8")},
        record_text=normalized_text,
        record_type="claim",
        thread_id=normalized_thread,
    )
    if typed.get("stored") is not True:
        raise MemoryStoreCorrupt(
            f"completed-turn typed admission failed: {typed.get('reason') or 'unknown'}"
        )

    result: dict[str, Any] = {
        "stored": True,
        "node": node,
        "node_id": node_id,
        "record_id": record_id,
        "epistemic_class": "MODEL_CLAIM",
        "model_facing": False,
        "idempotent": bool(typed.get("idempotent")),
    }
    try:
        memory = _instance()
        result["result"] = memory.add(
            normalized_text,
            user_id=_user_id(subject),
            metadata={
                "source": "helix-completed-turn",
                "kind": "completed-turn-model-claim",
                "thread_id": normalized_thread,
                "title": heading,
                "node_id": node_id,
                "record_id": record_id,
                "epistemic_class": "MODEL_CLAIM",
                "model_facing": False,
                "active": True,
            },
            infer=False,
        )
        result["reason"] = "typed_model_claim_committed"
    except Exception as error:  # noqa: BLE001 -- typed graph is canonical
        result["reason"] = str(error)[:1_000] or "vector-index-unavailable"
    return result


def _observation_receipt_value(
    receipt: dict[str, Any],
    *,
    camel: str,
    snake: str,
    default: Any = None,
) -> Any:
    value = receipt.get(camel)
    if value is None:
        value = receipt.get(snake)
    return default if value is None else value


def _observation_identity_fields(receipt: dict[str, Any]) -> dict[str, Any]:
    terminal_seq = _observation_receipt_value(
        receipt,
        camel="terminalSeq",
        snake="terminal_seq",
        default=0,
    )
    return {
        "run_id": str(
            _observation_receipt_value(receipt, camel="runId", snake="run_id", default="")
        ).strip(),
        "execution_id": str(
            _observation_receipt_value(
                receipt,
                camel="executionId",
                snake="execution_id",
                default="",
            )
        ).strip(),
        "thread_id": str(
            _observation_receipt_value(
                receipt,
                camel="threadId",
                snake="thread_id",
                default="",
            )
        ).strip(),
        "tool_name": str(
            _observation_receipt_value(
                receipt,
                camel="toolName",
                snake="tool_name",
                default="",
            )
        ).strip(),
        "tool_call_id": str(
            _observation_receipt_value(
                receipt,
                camel="toolCallId",
                snake="tool_call_id",
                default="",
            )
        ).strip(),
        "authority_kind": str(
            _observation_receipt_value(
                receipt,
                camel="authorityKind",
                snake="authority_kind",
                default="",
            )
        ).strip(),
        "receipt_ref": str(
            _observation_receipt_value(
                receipt,
                camel="receiptRef",
                snake="receipt_ref",
                default="",
            )
        ).strip(),
        "receipt_digest": str(
            _observation_receipt_value(
                receipt,
                camel="receiptDigest",
                snake="receipt_digest",
                default="",
            )
        ).strip(),
        "terminal_seq": (
            terminal_seq
            if isinstance(terminal_seq, int) and not isinstance(terminal_seq, bool)
            else 0
        ),
    }


def _observation_result_value(receipt: dict[str, Any]) -> Any:
    return receipt.get("result")


_OBSERVATION_MISSING = object()


def _observation_marker(
    receipt: dict[str, Any],
    *,
    camel: str,
    snake: str,
) -> Any:
    if camel in receipt:
        return receipt[camel]
    if snake in receipt:
        return receipt[snake]
    return _OBSERVATION_MISSING


def _observation_bounded_result(receipt: dict[str, Any]) -> tuple[str, str] | None:
    """Return the exact stored projection and its digest for a terminal row."""

    result = _observation_result_value(receipt)
    if not isinstance(result, str):
        return None
    bounded = result.strip()
    if not bounded:
        return None
    bounded = bounded[:_MAX_EXPERIENCE_CHARS]
    if not bounded:
        return None
    try:
        digest = hashlib.sha256(bounded.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        return None
    return bounded, digest


def _observation_receipt_state(
    receipt: dict[str, Any],
) -> tuple[str, str | None]:
    """Extract only explicitly successful terminal receipt markers."""

    state = _observation_receipt_value(
        receipt,
        camel="executionState",
        snake="execution_state",
        default="",
    )
    controller_error = _observation_receipt_value(
        receipt,
        camel="controllerIsError",
        snake="controller_is_error",
        default=None,
    )
    error = _observation_receipt_value(
        receipt,
        camel="error",
        snake="error_message",
        default=None,
    )
    completion = receipt.get("completion")
    if str(state) != "finished":
        return "receipt-not-finished", None
    if error not in (None, ""):
        return "receipt-error-present", None
    if controller_error is not None and controller_error is not False and controller_error != 0:
        return "receipt-controller-error", None
    if not isinstance(completion, dict):
        return "authoritative-receipt-completion-missing", None
    fields = _observation_identity_fields(receipt)
    if (
        completion.get("terminal_state") != "finished"
        or completion.get("selected_result")
        != _observation_result_value(receipt)
        or completion.get("error") is not None
        or (
            completion.get("controller_is_error") is not None
            and completion.get("controller_is_error") is not False
        )
        or completion.get("receipt_ref") != fields["receipt_ref"]
        or str(completion.get("execution_id") or "") != fields["execution_id"]
        or str(completion.get("authority_kind") or "") != fields["authority_kind"]
    ):
        return "authoritative-receipt-completion-mismatch", None
    return "", "finished"


def _observation_authority_receipt(
    subject: str | None,
    receipt: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve a submitted receipt against the account-private terminal row."""

    if not isinstance(receipt, dict):
        return None, "receipt-not-object"
    fields = _observation_identity_fields(receipt)
    if not all(
        (
            fields["run_id"],
            fields["execution_id"],
            fields["thread_id"],
            fields["tool_name"],
            fields["tool_call_id"],
        )
    ):
        return None, "receipt-identity-incomplete"
    owner = str(receipt.get("ownerSubject") or subject or "").strip()
    if not owner:
        return None, "receipt-owner-missing"
    account_id = str(current_account().account_id or "owner")
    from storage import chat_generation_runs_db as runs_db

    authoritative = runs_db.get_finished_tool_receipt(
        fields["run_id"],
        fields["execution_id"],
        owner_subject=owner,
        account_id=account_id,
    )
    if authoritative is None:
        return None, "authoritative-receipt-unavailable"
    authoritative_fields = _observation_identity_fields(authoritative)
    if (
        authoritative_fields["authority_kind"] not in {"approved", "ungated"}
        or authoritative_fields["receipt_ref"]
        != f"tool-receipt:{authoritative_fields['execution_id']}"
        or not re.fullmatch(r"[0-9a-f]{64}", authoritative_fields["receipt_digest"])
        or authoritative_fields["terminal_seq"] <= 0
    ):
        return None, "authoritative-receipt-identity-invalid"
    for key in (
        "run_id",
        "execution_id",
        "thread_id",
        "tool_name",
        "tool_call_id",
        "authority_kind",
        "receipt_ref",
        "receipt_digest",
        "terminal_seq",
    ):
        if authoritative_fields[key] != fields[key]:
            return None, f"authoritative-receipt-{key}-mismatch"
    if authoritative_fields["tool_name"] not in _OBSERVATION_SOURCE_TOOLS:
        return None, "tool-not-observation-source"
    if str(authoritative.get("ownerSubject") or "") != owner:
        return None, "authoritative-receipt-owner-mismatch"
    if str(
        _observation_receipt_value(
            authoritative,
            camel="backendAccountId",
            snake="backend_account_id",
            default="",
        )
        or ""
    ) != account_id:
        return None, "authoritative-receipt-account-mismatch"
    submitted_account = _observation_marker(
        receipt,
        camel="backendAccountId",
        snake="backend_account_id",
    )
    if (
        submitted_account is not _OBSERVATION_MISSING
        and str(submitted_account or "") != account_id
    ):
        return None, "authoritative-receipt-account-mismatch"
    if _observation_result_value(authoritative) != _observation_result_value(receipt):
        return None, "authoritative-receipt-result-mismatch"
    for camel, snake in (
        ("executionState", "execution_state"),
        ("error", "error_message"),
        ("controllerIsError", "controller_is_error"),
    ):
        submitted_marker = _observation_marker(
            receipt,
            camel=camel,
            snake=snake,
        )
        if (
            submitted_marker is not _OBSERVATION_MISSING
            and submitted_marker
            != _observation_marker(authoritative, camel=camel, snake=snake)
        ):
            return None, f"authoritative-receipt-{snake}-mismatch"
    if (
        "completion" in receipt
        and receipt.get("completion") != authoritative.get("completion")
    ):
        return None, "authoritative-receipt-completion-mismatch"
    for camel, snake, field in (
        ("cardCallId", "card_call_id", "card_call_id"),
        ("sessionId", "session_id", "session_id"),
        ("argumentsFingerprint", "arguments_fingerprint", "arguments_fingerprint"),
        ("approvalId", "approval_id", "approval_id"),
    ):
        submitted = _observation_marker(
            receipt,
            camel=camel,
            snake=snake,
        )
        if submitted is _OBSERVATION_MISSING:
            continue
        if submitted != _observation_marker(
            authoritative,
            camel=camel,
            snake=snake,
        ):
            return None, f"authoritative-receipt-{field}-mismatch"
    reason, terminal_state = _observation_receipt_state(authoritative)
    if reason:
        return None, reason
    bounded = _observation_bounded_result(authoritative)
    if bounded is None:
        return None, "receipt-result-empty"
    return authoritative, None


def admit_durable_observation(
    subject: str | None,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Admit one successful backend tool receipt as a typed observation.

    This path accepts only trusted, already-issued receipt metadata. It never
    accepts a tool call, prose, or caller-selected epistemic class. The durable
    receipt is the evidence source; the graph/sidecar copy is an indexed,
    immutable convenience projection.
    """

    authoritative, reason = _observation_authority_receipt(subject, receipt)
    if authoritative is None:
        return {"stored": False, "reason": reason or "authoritative-receipt-unavailable"}
    receipt = authoritative
    from core.memory.experience_idempotency import run_idempotent_memory_experience

    identity = _observation_identity_fields(receipt)
    key = f"durable-observation:{identity['run_id']}:{identity['execution_id']}"
    payload = {
        "runId": identity["run_id"],
        "executionId": identity["execution_id"],
        "threadId": identity["thread_id"],
        "toolName": identity["tool_name"],
        "toolCallId": identity["tool_call_id"],
        "authorityKind": identity["authority_kind"],
        "receiptRef": identity["receipt_ref"],
        "receiptDigest": identity["receipt_digest"],
        "terminalSeq": identity["terminal_seq"],
        "result": _observation_result_value(receipt),
    }
    if _graph_node_for_idempotency_key(key) is not None:
        replay = durable_observation_receipt_for_idempotency(
            key,
            thread_id=identity["thread_id"],
            receipt=receipt,
        )
        if replay is None:
            raise MemoryStoreCorrupt(
                "durable observation replay witness failed validation"
            )
        return replay
    return run_idempotent_memory_experience(
        idempotency_key=key,
        thread_id=identity["thread_id"],
        payload=payload,
        operation=lambda: _admit_durable_observation_once(subject, receipt),
        recovery_probe=lambda: durable_observation_receipt_for_idempotency(
            key,
            thread_id=identity["thread_id"],
            receipt=receipt,
        ),
    )


def _admit_durable_observation_once(
    subject: str | None,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Commit one already-authorized durable tool receipt."""

    from core.helix_engine.interlingua import EpistemicClass
    run_id = str(receipt.get("run_id") or receipt.get("runId") or "").strip()[:240]
    execution_id = str(
        receipt.get("execution_id") or receipt.get("executionId") or ""
    ).strip()[:240]
    thread_id = str(receipt.get("thread_id") or receipt.get("threadId") or "").strip()[:200]
    tool_name = str(receipt.get("tool_name") or receipt.get("toolName") or "").strip()[:240]
    tool_call_id = str(
        receipt.get("tool_call_id") or receipt.get("toolCallId") or ""
    ).strip()[:500]
    authority_kind = str(
        receipt.get("authority_kind") or receipt.get("authorityKind") or ""
    ).strip()[:40]
    receipt_ref = str(receipt.get("receipt_ref") or receipt.get("receiptRef") or "").strip()[:300]
    receipt_digest = str(
        receipt.get("receipt_digest") or receipt.get("receiptDigest") or ""
    ).strip()[:128]
    terminal_seq = receipt.get("terminal_seq", receipt.get("terminalSeq"))
    result = receipt.get("result")
    if not all((run_id, execution_id, thread_id, tool_name, tool_call_id)):
        return {"stored": False, "reason": "receipt-identity-incomplete"}
    if authority_kind not in {"approved", "ungated"}:
        return {"stored": False, "reason": "receipt-authority-invalid"}
    if receipt_ref != f"tool-receipt:{execution_id}":
        return {"stored": False, "reason": "receipt-reference-mismatch"}
    if not re.fullmatch(r"[0-9a-f]{64}", receipt_digest):
        return {"stored": False, "reason": "receipt-digest-invalid"}
    if isinstance(terminal_seq, bool) or not isinstance(terminal_seq, int) or terminal_seq <= 0:
        return {"stored": False, "reason": "receipt-terminal-sequence-invalid"}
    if not isinstance(result, str) or not result.strip():
        return {"stored": False, "reason": "receipt-result-empty"}
    bounded = _observation_bounded_result(receipt)
    if bounded is None:
        return {"stored": False, "reason": "receipt-result-empty"}
    result, result_sha256 = bounded
    # A selected result is an observation only when the durable row explicitly
    # says it was successful. This does not claim the result proves a task.
    state_reason, _terminal_state = _observation_receipt_state(receipt)
    if state_reason:
        return {"stored": False, "reason": state_reason}

    account_id = str(current_account().account_id or "owner")
    digest_material = (
        f"helix.durable-observation.v1\0{account_id}\0{run_id}\0{thread_id}\0"
        f"{execution_id}\0{receipt_digest}"
    ).encode("utf-8")
    identity = hashlib.sha256(digest_material).hexdigest()
    record_id = f"obs-{identity[:32]}"
    evidence_id = f"receipt-{identity[:32]}"
    evidence = {
        "schema_version": "helix.durable-tool-receipt.v1",
        "run_id": run_id,
        "execution_id": execution_id,
        "thread_id": thread_id,
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "authority_kind": authority_kind,
        "receipt_ref": receipt_ref,
        "receipt_digest": receipt_digest,
        "terminal_seq": terminal_seq,
        "selected_result": result,
        "result_sha256": result_sha256,
    }
    evidence_bytes = json.dumps(
        evidence,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    key = f"durable-observation:{run_id}:{execution_id}"
    binding = {
        "schema_version": "helix.durable-observation.v1",
        "account_id": account_id,
        "run_id": run_id,
        "execution_id": execution_id,
        "thread_id": thread_id,
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "authority_kind": authority_kind,
        "receipt_ref": receipt_ref,
        "receipt_digest": receipt_digest,
        "terminal_seq": terminal_seq,
        "result_sha256": result_sha256,
    }
    title = f"{tool_name} observation {execution_id[:12]}"
    node, created = _append_graph_node(
        title=title,
        text=result,
        kind="durable-tool-observation",
        thread_id=thread_id,
        entities=None,
        links=None,
        idempotency_key=key,
        binding=binding,
    )
    if not created:
        # A graph node is not sufficient evidence. Recover only when the
        # authoritative receipt, graph binding, typed OBSERVED record, and raw
        # evidence all agree. Otherwise the immutable registration below must
        # fail closed instead of reasserting a caller-visible OBSERVED class.
        recovered = durable_observation_receipt_for_idempotency(
            key,
            thread_id=thread_id,
            receipt=receipt,
        )
        if recovered is not None:
            return recovered
        raise MemoryStoreCorrupt(
            "durable observation replay witness failed validation"
        )
    record_id_actual = record_id
    typed = register_typed_memory(
        str(node["id"]),
        record_id_actual,
        epistemic_class=EpistemicClass.OBSERVED,
        evidence={evidence_id: evidence_bytes},
        record_text=result,
        record_type="observation",
        thread_id=thread_id,
    )
    if typed.get("stored") is not True:
        raise MemoryStoreCorrupt(
            f"durable observation typed admission failed: {typed.get('reason') or 'unknown'}"
        )
    result_value = {
        "stored": True,
        "node": node,
        "node_id": node["id"],
        "record_id": record_id_actual,
        "run_id": run_id,
        "execution_id": execution_id,
        "thread_id": thread_id,
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "authority_kind": authority_kind,
        "epistemic_class": "OBSERVED",
        "model_facing": True,
        "idempotent": True,
        "reason": "durable_tool_observation_committed",
        "receipt_ref": receipt_ref,
        "receipt_digest": receipt_digest,
        "terminal_seq": terminal_seq,
    }
    try:
        _instance().add(
            result,
            user_id=_user_id(subject),
            metadata={
                "source": "helix-durable-tool-observation",
                "kind": "durable-tool-observation",
                "thread_id": thread_id,
                "title": title,
                "node_id": node["id"],
                "record_id": record_id_actual,
                "epistemic_class": "OBSERVED",
                "model_facing": True,
                "active": True,
                **binding,
            },
            infer=False,
        )
    except Exception:
        # The graph binding, typed evidence, and authoritative tool receipt
        # are canonical. Mem0/Qdrant is an optional index and must not change
        # the durable receipt shape or make replay ambiguous.
        pass
    return result_value


def durable_observation_receipt_for_idempotency(
    idempotency_key: str,
    *,
    thread_id: str,
    receipt: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a replay witness only after graph and typed state both commit."""

    from core.helix_engine.interlingua import EpistemicClass, Observation

    key = str(idempotency_key or "").strip()[:240]
    thread = str(thread_id or "").strip()[:200]
    if (
        not key
        or not thread
        or not key.startswith("durable-observation:")
        or not isinstance(receipt, dict)
    ):
        return None
    authoritative, _reason = _observation_authority_receipt(
        str(receipt.get("ownerSubject") or "") if isinstance(receipt, dict) else "",
        receipt,
    )
    if authoritative is None:
        return None
    fields = _observation_identity_fields(authoritative)
    if (
        fields["thread_id"] != thread
        or key != f"durable-observation:{fields['run_id']}:{fields['execution_id']}"
    ):
        return None
    if fields["tool_name"] not in _OBSERVATION_SOURCE_TOOLS:
        return None
    bounded = _observation_bounded_result(authoritative)
    if bounded is None:
        return None
    result, result_sha256 = bounded

    account_id = str(current_account().account_id or "owner")
    digest_material = (
        f"helix.durable-observation.v1\0{account_id}\0{fields['run_id']}\0"
        f"{fields['thread_id']}\0{fields['execution_id']}\0"
        f"{fields['receipt_digest']}"
    ).encode("utf-8")
    identity = hashlib.sha256(digest_material).hexdigest()
    record_id = f"obs-{identity[:32]}"
    evidence_id = f"receipt-{identity[:32]}"
    evidence = {
        "schema_version": "helix.durable-tool-receipt.v1",
        "run_id": fields["run_id"],
        "execution_id": fields["execution_id"],
        "thread_id": fields["thread_id"],
        "tool_name": fields["tool_name"],
        "tool_call_id": fields["tool_call_id"],
        "authority_kind": fields["authority_kind"],
        "receipt_ref": fields["receipt_ref"],
        "receipt_digest": fields["receipt_digest"],
        "terminal_seq": fields["terminal_seq"],
        "selected_result": result,
        "result_sha256": result_sha256,
    }
    expected_binding = {
        "schema_version": "helix.durable-observation.v1",
        "account_id": account_id,
        "run_id": fields["run_id"],
        "execution_id": fields["execution_id"],
        "thread_id": fields["thread_id"],
        "tool_name": fields["tool_name"],
        "tool_call_id": fields["tool_call_id"],
        "authority_kind": fields["authority_kind"],
        "receipt_ref": fields["receipt_ref"],
        "receipt_digest": fields["receipt_digest"],
        "terminal_seq": fields["terminal_seq"],
        "result_sha256": result_sha256,
    }
    graph = _load_graph()
    if graph is None:
        return None
    node = next(
        (
            item
            for item in reversed(graph.get("nodes", []))
            if isinstance(item, dict)
            and str(item.get("idempotency_key") or "") == key
            and str(item.get("thread_id") or "") == thread
        ),
        None,
    )
    if node is None:
        return None
    node_id = str(node.get("id") or "")
    title = f"{fields['tool_name']} observation {fields['execution_id'][:12]}"
    if (
        node_id != _node_id(title, result, thread, key)
        or str(node.get("idempotency_key") or "") != key
        or str(node.get("kind") or "") != "durable-tool-observation"
        or str(node.get("title") or "") != title
        or str(node.get("text") or "") != result
        or str(node.get("thread_id") or "") != thread
        or node.get("active", True) is not True
        or node.get("binding") != expected_binding
    ):
        return None
    try:
        if _graph_supersedes(node) != []:
            return None
    except MemoryStoreCorrupt:
        return None
    loaded = _typed_state_and_bindings()
    if loaded is None:
        return None
    state, bindings, payloads = loaded
    binding = bindings.get(node_id)
    if (
        binding is None
        or binding.record_id != record_id
        or binding.account_id != account_id
        or binding.thread_id != thread
        or binding.active is not True
    ):
        return None
    record = next(
        (item for item in state.observations if item.id == record_id),
        None,
    )
    if (
        not isinstance(record, Observation)
        or record.epistemic_class is not EpistemicClass.OBSERVED
        or record.text != result
        or record.evidence_refs != (evidence_id,)
    ):
        return None
    raw_evidence = payloads.get(evidence_id)
    if raw_evidence is None:
        return None
    try:
        decoded_evidence = json.loads(raw_evidence.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if decoded_evidence != evidence:
        return None
    return {
        "stored": True,
        "node": node,
        "node_id": node_id,
        "record_id": binding.record_id,
        "run_id": fields["run_id"],
        "execution_id": fields["execution_id"],
        "thread_id": fields["thread_id"],
        "tool_name": fields["tool_name"],
        "tool_call_id": fields["tool_call_id"],
        "authority_kind": fields["authority_kind"],
        "epistemic_class": "OBSERVED",
        "model_facing": True,
        "idempotent": True,
        "reason": "durable_tool_observation_committed",
        "receipt_ref": fields["receipt_ref"],
        "receipt_digest": fields["receipt_digest"],
        "terminal_seq": fields["terminal_seq"],
    }


def _typed_state_and_bindings() -> tuple[Any, dict[str, Any], dict[str, bytes]] | None:
    """Build a validated Interlingua state from the account sidecar."""

    from core.helix_engine.interlingua import (
        Claim,
        EpistemicClass,
        InterlinguaState,
        Observation,
        Reference,
        Relationship,
        RelationshipKind,
    )
    from core.memory.typed_retrieval import BackendNodeBinding

    typed = _load_typed()
    if typed is None:
        return None
    claims: list[Claim] = []
    observations: list[Observation] = []
    references: list[Reference] = []
    references_by_id: dict[str, Reference] = {}
    bindings: dict[str, Any] = {}
    payloads: dict[str, bytes] = {}
    graph = _load_graph()
    if graph is None:
        return None
    graph_nodes = {
        str(node.get("id") or ""): node for node in graph.get("nodes", [])
    }
    for item in typed.get("nodes", []):
        try:
            node_id = str(item["node_id"])
            graph_node = graph_nodes.get(node_id)
            if graph_node is None:
                # Retention from an older build or a damaged graph can orphan one
                # sidecar row. Skip only that binding; other verified records remain
                # usable and the orphan cannot be projected without its graph receipt.
                continue
            if str(item["account_id"]) != str(current_account().account_id or "owner"):
                return None
            if str(item["thread_id"]) != str(graph_node.get("thread_id") or ""):
                return None
            if item["active"] is not (graph_node.get("active", True) is not False):
                return None
            try:
                graph_supersedes = _graph_supersedes(graph_node)
            except MemoryStoreCorrupt:
                return None
            if item.get("supersedes", []) != graph_supersedes:
                return None
            graph_fact_key = str(graph_node.get("fact_key") or "") or None
            if str(item.get("fact_key") or "") != (graph_fact_key or ""):
                return None
            record = item["record"]
            if str(record.get("id") or "") != str(item["record_id"]):
                return None
            record_type = record["type"]
            record_id = record["id"]
            record_text = record["text"]
            epistemic_class = EpistemicClass(record["epistemic_class"])
            evidence_refs = tuple(record["evidence_refs"])
            if record_type == "claim":
                claims.append(Claim(record_id, record_text, epistemic_class, evidence_refs))
            elif record_type == "observation":
                observations.append(
                    Observation(record_id, record_text, epistemic_class, evidence_refs)
                )
            else:
                return None
            for raw_reference in item["references"]:
                reference = Reference(
                    id=raw_reference["id"],
                    sha256=raw_reference["sha256"],
                    byte_length=raw_reference["byte_length"],
                    media_type=raw_reference["media_type"],
                )
                prior_reference = references_by_id.get(reference.id)
                if prior_reference is not None and prior_reference != reference:
                    return None
                if prior_reference is None:
                    references_by_id[reference.id] = reference
                    references.append(reference)
                payload = _decode_evidence(item["evidence"].get(reference.id))
                if payload is None or not reference.verifies(payload):
                    return None
                payloads[reference.id] = payload
            bindings[str(item["node_id"])] = BackendNodeBinding(
                node_id=str(item["node_id"]),
                record_id=str(item["record_id"]),
                account_id=str(item["account_id"]),
                thread_id=str(item["thread_id"]),
                active=bool(item["active"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    record_by_node = {
        node_id: binding.record_id for node_id, binding in bindings.items()
    }
    relationships: list[Relationship] = []
    relationship_keys: set[tuple[str, str, str]] = set()
    for node_id, binding in bindings.items():
        node = graph_nodes.get(node_id, {})
        targets = list(node.get("supersedes") or [])
        # Also honor the reverse pointer written by the backend when a legacy
        # graph contains only ``superseded_by``.
        for candidate_id, candidate in graph_nodes.items():
            if str(candidate.get("superseded_by") or "") == node_id:
                targets.append(candidate_id)
        for target_node_id in targets:
            target_record = record_by_node.get(str(target_node_id))
            if target_record and target_record != binding.record_id:
                try:
                    relationship_key = (
                        binding.record_id,
                        target_record,
                        RelationshipKind.SUPERSEDES.value,
                    )
                    if relationship_key in relationship_keys:
                        continue
                    relationships.append(
                        Relationship(
                            binding.record_id,
                            target_record,
                            RelationshipKind.SUPERSEDES,
                        )
                    )
                    relationship_keys.add(relationship_key)
                except ValueError:
                    return None
    try:
        state = InterlinguaState(
            claims=tuple(claims),
            observations=tuple(observations),
            references=tuple(references),
            relationships=tuple(relationships),
        )
    except (TypeError, ValueError):
        return None
    return state, bindings, payloads


def typed_bindings() -> dict[str, Any]:
    """Expose only the validated backend node bindings for the active account."""

    loaded = _typed_state_and_bindings()
    return {} if loaded is None else loaded[1]


def search_typed(
    subject: str | None,
    query: str,
    limit: int = 5,
    *,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Search and project typed memory, failing closed on missing trust data."""

    from core.memory.typed_retrieval import (
        bind_backend_hits,
        project_typed_context,
        projection_to_dict,
    )

    try:
        raw = search(
            subject,
            query,
            limit,
            thread_id=thread_id,
            active_only=True,
            candidate_limit=128,
            result_limit=128,
            graph_first=True,
        )
    except Exception:
        return {
            "available": False,
            "status": "unavailable",
            "items": [],
            "results": [],
            "exclusions": [],
        }
    try:
        loaded = _typed_state_and_bindings()
        if loaded is None:
            return {
                "available": False,
                "status": "unavailable",
                "items": [],
                "results": [],
                "exclusions": [],
            }
        state, bindings, payloads = loaded
        candidates = bind_backend_hits(
            query=str(query or ""),
            account_id=str(current_account().account_id or "owner"),
            thread_id=thread_id,
            hits=raw.get("results", []),
            bindings=bindings,
        )
        projection = project_typed_context(
            query=query,
            account_id=str(current_account().account_id or "owner"),
            thread_id=thread_id,
            candidates=candidates,
            state=state,
            evidence_payloads=payloads,
            limit=max(1, min(int(limit), 10)),
        )
        result = projection_to_dict(projection)
    except (TypeError, ValueError, KeyError, OSError):
        return {
            "available": False,
            "status": "unavailable",
            "items": [],
            "results": [],
            "exclusions": [],
        }
    result["available"] = bool(result["items"])
    result["results"] = result["items"]
    return result


def has_any_memory() -> bool:
    """Whether this account-scoped memory graph contains at least one experience."""
    graph = _load_graph()
    return bool(graph is not None and graph.get("nodes"))


def thread_has_memory(thread_id: str | None) -> bool:
    if not thread_id:
        return False
    needle = str(thread_id)[:200]
    graph = _load_graph()
    return bool(
        graph is not None
        and any(node.get("thread_id") == needle for node in graph.get("nodes", []))
    )


def experience_receipt_for_idempotency(
    idempotency_key: str,
    *,
    thread_id: str | None = None,
) -> dict[str, Any] | None:
    """Return a replay-safe local witness for a keyed experience, if one committed.

    The bounded graph write is the first side effect in add_experience. Therefore:
    a present node proves the logical experience committed locally and a missing
    node proves the Mem0 vector add could not yet have started.
    """
    key = str(idempotency_key or "").strip()[:240]
    if not key:
        return None
    thread = str(thread_id or "")[:200]
    with _LOCK:
        graph = _load_graph()
        if graph is None:
            return None
        node = next(
            (
                item
                for item in reversed(graph.get("nodes", []))
                if isinstance(item, dict)
                and str(item.get("idempotency_key") or "") == key
                and str(item.get("thread_id") or "") == thread
            ),
            None,
        )
    if node is None:
        return None
    return {
        "stored": True,
        "node": node,
        "reason": "recovered_from_idempotent_local_graph",
        "idempotent": True,
    }


def add_experience(
    subject: str | None,
    text: str,
    *,
    thread_id: str | None = None,
    kind: str = "experience",
    title: str | None = None,
    entities: list[str] | None = None,
    links: list[dict[str, str]] | None = None,
    idempotency_key: str | None = None,
    fact_key: str | None = None,
) -> dict[str, Any]:
    """Persist an experience, optionally replacing the backend-owned fact key.

    ``fact_key`` is an internal selector supplied by trusted backend code. It is
    deliberately not inferred from text and is not accepted from tool metadata.
    """
    text = str(text or "").strip()[:_MAX_EXPERIENCE_CHARS]
    if not text:
        return {"stored": False, "reason": "empty"}
    heading = (title or text.split("\n", 1)[0]).strip()[:240] or kind
    if any(
        isinstance(link, dict)
        and (
            any(str(key).casefold().replace("_", " ").startswith("supersed") for key in link)
            or "supersed" in str(link.get("relation") or "").casefold()
        )
        for link in links or []
    ):
        raise ValueError("supersession is assigned by the memory backend")
    if idempotency_key:
        prior = experience_receipt_for_idempotency(
            idempotency_key,
            thread_id=thread_id,
        )
        if prior is not None:
            return prior
    node, created = _append_graph_node(
        title=heading,
        text=text,
        kind=str(kind or "experience")[:40],
        thread_id=str(thread_id or ""),
        entities=entities,
        links=links,
        idempotency_key=idempotency_key,
        fact_key=fact_key,
    )
    if not created:
        return {
            "stored": True,
            "node": node,
            "reason": "current_memory_already_present",
            "idempotent": True,
        }
    try:
        memory = _instance()
        metadata = {
            "source": "unsloth-studio",
            "kind": kind,
            "thread_id": str(thread_id or "")[:200],
            "title": heading,
            "node_id": node["id"],
            "active": True,
        }
        if node.get("fact_key"):
            metadata["fact_key"] = node["fact_key"]
        result = memory.add(
            text,
            user_id=_user_id(subject),
            metadata=metadata,
            infer=False,
        )
        return {"stored": True, "result": result, "node": node}
    except Exception as error:  # noqa: BLE001 -- graph is enough; Mem0 package is optional
        return {"stored": True, "node": node, "reason": str(error)[:1_000]}


def _graph_search(
    query: str,
    limit: int,
    *,
    fact_key: str | None = None,
    thread_id: str | None = None,
    active_only: bool = True,
) -> list[dict[str, Any]]:
    tokens = {token for token in query.lower().split() if len(token) >= 3}
    scored: list[tuple[int, dict[str, Any]]] = []
    graph = _load_graph()
    if graph is None:
        return []
    for node in graph.get("nodes", []):
        if active_only and node.get("active", True) is False:
            continue
        if fact_key and str(node.get("fact_key") or "") != fact_key:
            continue
        if thread_id is not None and str(node.get("thread_id") or "") != thread_id:
            continue
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
    scored.sort(
        key=lambda item: (
            -item[0],
            -int(item[1].get("created_at") or 0),
            str(item[1].get("id") or ""),
        )
    )
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
                    "fact_key": node.get("fact_key"),
                    "active": node.get("active", True),
                    "node_id": node.get("id"),
                    "entities": node.get("entities") or [],
                    "links": related[:8],
                },
            }
        )
    return hits


def search(
    subject: str | None,
    query: str,
    limit: int = 5,
    *,
    fact_key: str | None = None,
    thread_id: str | None = None,
    active_only: bool = True,
    candidate_limit: int | None = None,
    result_limit: int | None = None,
    graph_first: bool = False,
) -> dict[str, Any]:
    query = str(query or "").strip()[:2_000]
    if not query:
        return {"results": [], "available": False, "reason": "empty"}
    cap = max(1, min(int(limit), 10))
    result_cap = (
        cap
        if result_limit is None
        else min(max(int(result_limit), cap), _GRAPH_MAX_NODES)
    )
    fact_key = str(fact_key or "").strip()[:240] or None
    thread_filter = None if thread_id is None else str(thread_id)[:200]
    # Pull a candidate window at least as large as the graph so stale or
    # cross-thread vector rows do not consume the final cap.
    candidate_hint = min(max(int(candidate_limit or 0), 0), _GRAPH_MAX_NODES)
    candidate_cap = max(cap, _GRAPH_MAX_NODES, candidate_hint)
    graph = _load_graph()
    if graph is None:
        return {
            "results": [],
            "available": False,
            "reason": "memory-graph-unavailable",
        }
    graph_nodes = {
        str(node.get("id") or ""): node
        for node in graph.get("nodes", [])
        if str(node.get("id") or "")
    }
    current_by_fact = {
        str(node.get("fact_key") or ""): node
        for node in graph.get("nodes", [])
        if node.get("fact_key") and node.get("active", True) is not False
    }
    vector_results: list[Any] = []
    vector_ok = False
    try:
        memory = _instance()
        for user_id in _search_user_ids(subject):
            try:
                try:
                    # Mem0 2.x moved entity selectors under ``filters``. Keep a
                    # narrow fallback for older installations rather than treating an
                    # API-shape mismatch as vector-memory unavailability.
                    found = memory.search(
                        query,
                        filters={"user_id": user_id},
                        limit=candidate_cap,
                    )
                except (TypeError, ValueError) as error:
                    text = str(error)
                    if "filters" not in text and "Top-level entity parameters" not in text:
                        raise
                    found = memory.search(query, user_id=user_id, limit=candidate_cap)
                if isinstance(found, dict):
                    vector_rows = found.get("results")
                    if isinstance(vector_rows, list):
                        vector_results.extend(
                            item for item in vector_rows if isinstance(item, dict)
                        )
                        vector_ok = True
                elif isinstance(found, list):
                    vector_results.extend(
                        item for item in found if isinstance(item, dict)
                    )
                    vector_ok = True
            except Exception:
                # One stale/unsupported legacy selector must not hide canonical
                # results or the graph fallback.
                continue
    except Exception:
        vector_ok = False

    def _eligible(item: dict[str, Any]) -> bool:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        bound_node_id = str(metadata.get("node_id") or item.get("node_id") or "")
        node_id = bound_node_id or str(item.get("id") or "")
        node = graph_nodes.get(node_id)
        if bound_node_id and node is None:
            return False
        item_thread = str(metadata.get("thread_id") or "")
        item_fact_key = str(metadata.get("fact_key") or "")

        if thread_filter is not None:
            # Missing thread provenance is not evidence that a memory belongs to
            # this conversation. Filter before applying the caller's final cap.
            if node is not None:
                if str(node.get("thread_id") or "") != thread_filter:
                    return False
            elif item_thread != thread_filter:
                return False
        if fact_key is not None:
            if node is not None:
                if str(node.get("fact_key") or "") != fact_key:
                    return False
            elif item_fact_key != fact_key:
                return False
        if active_only:
            if item.get("active") is False or metadata.get("active") is False:
                return False
            if node is not None and node.get("active", True) is False:
                return False
            if item_fact_key and item_fact_key in current_by_fact:
                current = current_by_fact[item_fact_key]
                current_id = str(current.get("id") or "")
                if node_id != current_id:
                    return False
        return True

    vector_results = [item for item in vector_results if _eligible(item)]
    graph_hits = _graph_search(
        query,
        candidate_cap,
        fact_key=fact_key,
        thread_id=thread_filter,
        active_only=active_only,
    )

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
    deduplicated: list[dict[str, Any]] = []
    ordered_results = (
        [*graph_hits, *vector_results]
        if graph_first
        else [*vector_results, *graph_hits]
    )
    for item in ordered_results:
        if not isinstance(item, dict):
            continue
        keys = _keys(item)
        if keys and keys.intersection(seen):
            continue
        deduplicated.append(item)
        seen.update(keys)
    available = vector_ok or bool(graph_hits)
    return {"results": deduplicated[:result_cap], "available": available}
