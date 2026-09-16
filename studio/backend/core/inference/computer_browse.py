# SPDX-License-Identifier: AGPL-3.0-only
"""Chromium-acceptable browse actions plus a session-scoped live feed."""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

_LOCK = threading.Lock()
_FEED: dict[str, list[dict[str, Any]]] = {}
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def feed_session_id(session_id: str | None = None, thread_id: str | None = None) -> str:
    """Same id execute_tool receives as session_id, then thread_id, then default."""
    for value in (session_id, thread_id):
        text = str(value or "").strip()
        if text:
            return text
    return "default"


def record_live_feed(session_id: str, event: dict[str, Any]) -> dict[str, Any]:
    payload = {"session_id": session_id, **event}
    with _LOCK:
        _FEED.setdefault(session_id, []).append(payload)
        _FEED[session_id] = _FEED[session_id][-200:]
    return payload


def list_live_feed(session_id: str) -> list[dict[str, Any]]:
    with _LOCK:
        return list(_FEED.get(session_id, []))


def _fetch_page(url: str, timeout: float = 8.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "HelixHarness/1.0 (+https://github.com/SuperHelix77/Helix-Harness)"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(120_000)
            charset = response.headers.get_content_charset() or "utf-8"
            html = raw.decode(charset, errors="replace")
            final_url = str(response.geturl())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
        return {"ok": False, "error": str(error)[:500], "url": url, "engine": "urllib"}
    title_match = _TITLE_RE.search(html)
    title = _TAG_RE.sub("", title_match.group(1)).strip() if title_match else ""
    snippet = " ".join(_TAG_RE.sub(" ", html).split())[:800]
    return {
        "ok": True,
        "url": final_url,
        "title": title[:200],
        "snippet": snippet,
        "engine": "urllib",
    }


def browse_action(action: str, *, url: str | None = None, session_id: str = "default") -> dict[str, Any]:
    name = str(action or "").strip().lower()
    if name not in {"navigate", "browse", "snapshot", "back"}:
        return {"ok": False, "error": f"unsupported browse action '{action}'"}
    parsed = urlparse(str(url or ""))
    if name in {"navigate", "browse"}:
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return {"ok": False, "error": "browse requires an http(s) URL"}
        page = _fetch_page(str(url))
        event = {
            "action": "navigate",
            "url": page.get("url") or url,
            "kind": "browse",
            "ok": bool(page.get("ok")),
            "title": page.get("title") or "",
            "snippet": page.get("snippet") or page.get("error") or "",
            "engine": page.get("engine") or "urllib",
        }
        return record_live_feed(session_id, event)
    event = {
        "action": name,
        "url": url,
        "kind": "browse",
        "ok": True,
    }
    return record_live_feed(session_id, event)


def map_computer_action(arguments: dict[str, Any]) -> dict[str, Any]:
    action = str(arguments.get("action") or "").strip().lower()
    if action in {"browse", "navigate"}:
        return {
            "kind": "browse",
            "action": "browse",
            "url": arguments.get("url"),
        }
    return {
        "kind": "computer",
        "action": action or "screenshot",
        "x": arguments.get("x"),
        "y": arguments.get("y"),
    }


def feed_json(session_id: str) -> str:
    return json.dumps({"session_id": session_id, "events": list_live_feed(session_id)})
