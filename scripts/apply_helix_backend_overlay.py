#!/usr/bin/env python3
"""Atomically apply a partial Helix backend overlay onto an installed Studio backend."""
from __future__ import annotations
import importlib.util, json, os, shutil, sys, uuid
from pathlib import Path

EXPECTED = "helix.adaptive.backend.v1"

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
    print(f"applied {EXPECTED} ({meta.get('file_count')} overrides)")
if __name__ == "__main__": main()
