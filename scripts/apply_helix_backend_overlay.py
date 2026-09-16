#!/usr/bin/env python3
"""Atomically apply a partial Helix backend overlay onto an installed Studio backend."""
from __future__ import annotations
import importlib.metadata, importlib.util, json, os, shutil, subprocess, sys, uuid
from pathlib import Path

EXPECTED = "helix.adaptive.backend.v1"
MEMORY_REQUIREMENT = "mem0ai>=2.0.20,<3.0"


def _memory_runtime_ready() -> bool:
    try:
        from packaging.version import Version

        version = Version(importlib.metadata.version("mem0ai"))
        if not (Version("2.0.20") <= version < Version("3.0")):
            return False
        import qdrant_client  # noqa: F401
    except Exception:
        return False
    return True


def _ensure_optional_memory_runtime() -> bool:
    """Best-effort install of the local Mem0/Qdrant runtime.

    Helix memory has a bounded JSON/graph fallback, so an offline package install
    must never make Studio repair/update fail. The overlay itself remains the
    compatibility contract; this optional dependency only upgrades retrieval.
    """
    if _memory_runtime_ready():
        return True
    if os.environ.get("HELIX_SKIP_MEMORY_INSTALL", "").strip() == "1":
        print("[helix] Mem0 runtime install skipped by HELIX_SKIP_MEMORY_INSTALL=1", file=sys.stderr)
        return False
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", MEMORY_REQUIREMENT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=180,
        )
    except Exception as error:
        print(f"[helix] optional Mem0 runtime install failed open: {error}", file=sys.stderr)
        return False
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "pip failed").strip().splitlines()[-1:]
        print(
            "[helix] optional Mem0 runtime unavailable; bounded local graph remains active: "
            + (detail[0] if detail else "pip failed"),
            file=sys.stderr,
        )
        return False
    importlib.invalidate_caches()
    ready = _memory_runtime_ready()
    if not ready:
        print(
            "[helix] Mem0 package install completed but runtime validation failed; "
            "bounded local graph remains active",
            file=sys.stderr,
        )
    return ready


def main() -> None:
    if len(sys.argv) != 2: raise SystemExit("usage: apply_helix_backend_overlay.py <overlay-dir>")
    overlay = Path(sys.argv[1]).resolve()
    meta_path = overlay / "HELIX_BACKEND_MANIFEST.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("contract") != EXPECTED: raise SystemExit(f"unexpected Helix backend contract: {meta.get('contract')!r}")
    spec = importlib.util.find_spec("studio")
    locations = list(spec.submodule_search_locations or ()) if spec else []
    if not locations: raise SystemExit("installed studio package not found")
    studio_root = Path(locations[0]).resolve()
    current = studio_root / "backend"
    if not current.is_dir(): raise SystemExit(f"installed backend not found: {current}")
    stage = studio_root / f".backend-helix-{uuid.uuid4().hex}"
    backup = studio_root / f".backend-pre-helix-{uuid.uuid4().hex}"
    shutil.copytree(current, stage, symlinks=False)
    try:
        for rel in meta.get("files", []):
            src, dst = overlay / rel, stage / rel
            if not src.is_file(): raise RuntimeError(f"overlay file missing: {rel}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        os.replace(current, backup)
        try: os.replace(stage, current)
        except BaseException:
            os.replace(backup, current)
            raise
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    else:
        shutil.rmtree(backup, ignore_errors=True)
    sys.path.insert(0, str(current))
    from state.tool_policy import HELIX_HARNESS_BACKEND_CONTRACT, require_tool_access
    from core.helix_engine.controller import run_closed_loop
    assert HELIX_HARNESS_BACKEND_CONTRACT == EXPECTED
    assert callable(require_tool_access) and callable(run_closed_loop)
    memory_ready = _ensure_optional_memory_runtime()
    print(
        f"applied {EXPECTED} ({meta.get('file_count')} overrides); "
        f"mem0={'ready' if memory_ready else 'fallback'}"
    )
if __name__ == "__main__": main()
