#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Helix Harness v3 builds require macOS." >&2
  exit 1
fi

# The flavor identity and app-only output are fixed by this entry point. Allow
# ordinary build options, but do not let passthrough arguments select another
# config or bundle type.
for arg in "$@"; do
  case "$arg" in
    --config|-c|--config=*|-c?*|--bundles|-b|--bundles=*|-b?*|--no-bundle|--)
      echo "This build wrapper owns --config and --bundles; conflicting option: $arg" >&2
      exit 2
      ;;
  esac
done

# Tauri packages the staged backend manifest into the app. Refuse to compile
# when it is malformed, stale against its staged files, or out of sync with
# the source runtime; this check only reads files and never rewrites staging.
python3 - "$ROOT" <<'PY'
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath


ROOT = Path(sys.argv[1])
EXPECTED_VERSION = "3.0.1"
EXPECTED_PRODUCT = "Helix Harness v3"
EXPECTED_BUNDLE_ID = "ai.helix.harness.v3"
EXPECTED_SCHEME = "helixharness-v3"
EXPECTED_BACKEND_SCHEMA = "helix.backend-overlay.v1"
EXCLUDED_DIRS = {"tests", "requirements", "__pycache__", ".pytest_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"v3 build preflight failed: {message}")


tauri_root = ROOT / "studio/src-tauri"
config_path = tauri_root / "tauri.helix-v3.conf.json"
try:
    config = json.loads(config_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"v3 build preflight failed: cannot read v3 Tauri config: {error}")

require(isinstance(config, dict), "v3 Tauri config must be a JSON object")
require(config.get("productName") == EXPECTED_PRODUCT, "v3 product name drifted")
require(config.get("identifier") == EXPECTED_BUNDLE_ID, "v3 bundle identifier drifted")
windows = config.get("app", {}).get("windows", [])
require(
    isinstance(windows, list)
    and bool(windows)
    and isinstance(windows[0], dict)
    and windows[0].get("title") == EXPECTED_PRODUCT,
    "v3 window title drifted",
)
schemes = config.get("plugins", {}).get("deep-link", {}).get("desktop", {}).get("schemes")
require(schemes == [EXPECTED_SCHEME], "v3 deep-link scheme drifted")

cargo_path = tauri_root / "Cargo.toml"
cargo_text = cargo_path.read_text(encoding="utf-8")
package = re.search(r"(?ms)^\[package\]\s*(.*?)(?=^\[|\Z)", cargo_text)
version = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', package.group(1)) if package else None
require(version is not None and version.group(1) == EXPECTED_VERSION, "Cargo package version must be 3.0.0")

source = ROOT / "studio/backend"
overlay = tauri_root / "artifacts/helix-backend"
manifest_path = overlay / "HELIX_BACKEND_MANIFEST.json"
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"v3 build preflight failed: cannot read staged backend manifest: {error}")

require(isinstance(manifest, dict), "staged backend manifest must be a JSON object")
contract_text = (source / "state/tool_policy.py").read_text(encoding="utf-8")
contract_match = re.search(
    r'(?m)^HELIX_HARNESS_BACKEND_CONTRACT\s*=\s*"([^"]+)"\s*$', contract_text
)
require(contract_match is not None, "source backend contract is unavailable")
require(manifest.get("schema_version") == EXPECTED_BACKEND_SCHEMA, "staged backend schema drifted")
require(manifest.get("contract") == contract_match.group(1), "staged backend contract differs from source")

files = manifest.get("files")
require(
    isinstance(files, list)
    and bool(files)
    and all(isinstance(entry, str) for entry in files)
    and isinstance(manifest.get("file_count"), int)
    and not isinstance(manifest.get("file_count"), bool)
    and manifest.get("file_count") == len(files)
    and files == sorted(set(files)),
    "staged backend file manifest is malformed",
)
require(
    isinstance(manifest.get("tree_sha256"), str)
    and re.fullmatch(r"[0-9a-f]{64}", manifest["tree_sha256"]) is not None,
    "staged backend tree digest is malformed",
)


def runtime_files(directory: Path, *, ignored_names: set[str] | None = None) -> list[str]:
    ignored_names = ignored_names or set()
    paths: list[str] = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise SystemExit(f"v3 build preflight failed: symlink in backend staging: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(directory)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES or path.name == ".DS_Store" or path.name in ignored_names:
            continue
        paths.append(relative.as_posix())
    return sorted(paths)


require(runtime_files(source) == files, "staged backend file list differs from source runtime")
require(
    runtime_files(overlay, ignored_names={manifest_path.name}) == files,
    "staged backend contains files outside its manifest",
)


def tree_digest(directory: Path, entries: list[str]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        relative = PurePosixPath(entry)
        require(
            not relative.is_absolute()
            and "\\" not in entry
            and relative.as_posix() == entry
            and all(part not in ("", ".", "..") for part in relative.parts),
            f"unsafe path in staged backend manifest: {entry!r}",
        )
        path = directory.joinpath(*relative.parts)
        require(path.is_file() and not path.is_symlink(), f"backend file is missing or unsafe: {entry}")
        digest.update(entry.encode() + b"\0" + hashlib.sha256(path.read_bytes()).hexdigest().encode() + b"\n")
    return digest.hexdigest()


staged_digest = tree_digest(overlay, files)
require(staged_digest == manifest["tree_sha256"], "staged backend files do not match their manifest")
require(
    tree_digest(source, files) == staged_digest,
    "staged backend is stale against studio/backend source",
)
PY

cd "$ROOT/studio"

# Keep release binaries free of checkout and Cargo-registry paths while
# retaining useful source locations in diagnostics.
build_cargo_home="${CARGO_HOME:-}"
if [[ -z "$build_cargo_home" && -n "${HOME:-}" ]]; then
  build_cargo_home="${HOME}/.cargo"
fi
release_remap="--remap-path-prefix=$ROOT=/helix-source"
if [[ -n "$build_cargo_home" && -d "$build_cargo_home" ]]; then
  release_remap+=" --remap-path-prefix=$build_cargo_home=/cargo"
fi
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }$release_remap"

exec npx --prefix . tauri build \
  --bundles app \
  --config src-tauri/tauri.helix-v3.conf.json \
  --config '{"version":"3.0.1"}' \
  "$@"
