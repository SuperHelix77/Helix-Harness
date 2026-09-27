#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Fail-closed static qualification for one explicit Helix v3 app candidate.

The gate is read-only. It hashes the complete app bundle, binds that receipt to
the current repository source identity, verifies the embedded Helix backend
manifest against both source and package bytes, and inspects plist and macOS
signature metadata. It never launches the app, installs or stages files,
loads a model, reads user data, or uses the network.

Static identity and distribution trust are deliberately separate verdicts. A
correctly sealed ad-hoc bundle may be ``qualified_static_identity`` while its
distribution trust remains unqualified. Only a Developer ID signature that
passes Gatekeeper and has a stapled notarization ticket is distribution-ready.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import plistlib
import re
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


GATE_VERSION = "helix-v3-package-static-qualification-gate.v1"
CONFIG_SCHEMA = "helix-v3-package-static-qualification-config.v1"
RECEIPT_SCHEMA = "helix-v3-package-static-qualification-receipt.v1"
APP_FILE_RECEIPT_SCHEMA = "helix-v3-package-static-app-file-receipt.v1"
SOURCE_STATUS_SCHEMA = "helix-v3-package-static-source-status.v1"
BACKEND_MANIFEST_SCHEMA = "helix.backend-overlay.v1"
BACKEND_CONTRACT = "helix.adaptive.backend.v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("helix_v3_package_static_gate.json")
SOURCE_IDENTITY_PATH = REPO_ROOT / "scripts" / "helix-v3-source-identity.py"

BACKEND_MANIFEST_RELATIVE = Path(
    "Contents/Resources/helix-backend/HELIX_BACKEND_MANIFEST.json"
)
EXCLUDED_BACKEND_DIRS = {
    "tests",
    "requirements",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
}
EXCLUDED_BACKEND_SUFFIXES = {".pyc", ".pyo"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_OID_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
COMMAND_TIMEOUT_S = 20.0
READ_CHUNK_BYTES = 1024 * 1024

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class GateError(RuntimeError):
    """A required package or source identity invariant is unavailable."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is not allowed: {key}")
        result[key] = value
    return result


def _parse_json_bytes(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise GateError(f"{label} is malformed JSON") from exc


def load_config(path: Path = CONFIG_PATH) -> tuple[dict[str, Any], str]:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise GateError("candidate static-gate config is unavailable") from exc
    value = _parse_json_bytes(raw, label="candidate static-gate config")
    if not isinstance(value, dict):
        raise GateError("candidate static-gate config is not a JSON object")
    return value, _sha256_bytes(raw)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("unexpected=" + ",".join(extra))
        raise GateError(f"{label} has an unsupported field set ({'; '.join(details)})")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise GateError(f"{label} is not a lowercase SHA-256")
    return value


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        config,
        {"schema_version", "gate_id", "app", "backend", "trust_policy"},
        "candidate static-gate config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GateError("candidate static-gate config schema is unsupported")
    if config.get("gate_id") != "HELIX-V3-PACKAGE-STATIC-QUALIFICATION-V1":
        raise GateError("candidate static-gate id is unsupported")

    app = config.get("app")
    if not isinstance(app, Mapping):
        raise GateError("candidate app identity is missing")
    _require_exact_keys(
        app,
        {
            "bundle_name",
            "bundle_identifier",
            "display_name",
            "version",
            "executable_relative_path",
            "executable_sha256",
            "info_plist_sha256",
            "file_count",
            "total_bytes",
            "tree_sha256",
        },
        "candidate app identity",
    )
    for key in ("bundle_name", "bundle_identifier", "display_name", "version"):
        if not isinstance(app.get(key), str) or not app[key]:
            raise GateError(f"candidate app {key} is malformed")
    _require_safe_relative_path(app.get("executable_relative_path"), "app executable")
    _require_sha256(app.get("executable_sha256"), "app executable SHA-256")
    _require_sha256(app.get("info_plist_sha256"), "Info.plist SHA-256")
    _require_sha256(app.get("tree_sha256"), "app tree SHA-256")
    for key in ("file_count", "total_bytes"):
        if (
            not isinstance(app.get(key), int)
            or isinstance(app.get(key), bool)
            or app[key] < 1
        ):
            raise GateError(f"candidate app {key} is malformed")

    backend = config.get("backend")
    if not isinstance(backend, Mapping):
        raise GateError("candidate backend identity is missing")
    _require_exact_keys(
        backend,
        {
            "manifest_relative_path",
            "manifest_sha256",
            "schema_version",
            "contract",
            "file_count",
            "tree_sha256",
        },
        "candidate backend identity",
    )
    _require_safe_relative_path(
        backend.get("manifest_relative_path"), "backend manifest"
    )
    if backend.get("manifest_relative_path") != BACKEND_MANIFEST_RELATIVE.as_posix():
        raise GateError("candidate backend manifest location is unsupported")
    _require_sha256(backend.get("manifest_sha256"), "backend manifest SHA-256")
    if backend.get("schema_version") != BACKEND_MANIFEST_SCHEMA:
        raise GateError("candidate backend manifest schema is unsupported")
    if backend.get("contract") != BACKEND_CONTRACT:
        raise GateError("candidate backend contract is unsupported")
    _require_sha256(backend.get("tree_sha256"), "backend tree SHA-256")
    if (
        not isinstance(backend.get("file_count"), int)
        or isinstance(backend.get("file_count"), bool)
        or backend["file_count"] < 1
    ):
        raise GateError("candidate backend file_count is malformed")

    trust = config.get("trust_policy")
    if not isinstance(trust, Mapping):
        raise GateError("candidate trust policy is missing")
    _require_exact_keys(
        trust,
        {"qualification_target", "required_team_identifier"},
        "candidate trust policy",
    )
    if trust.get("qualification_target") not in {"static_identity", "distribution"}:
        raise GateError("candidate trust qualification target is unsupported")
    team = trust.get("required_team_identifier")
    if team is not None and (not isinstance(team, str) or not team.strip()):
        raise GateError("candidate required team identifier is malformed")

    return {
        "app": dict(app),
        "backend": dict(backend),
        "trust": dict(trust),
    }


def _require_safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise GateError(f"{label} is not a safe relative POSIX path")
    if any(ord(character) < 32 for character in value):
        raise GateError(f"{label} contains a control character")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise GateError(f"{label} is not a safe relative POSIX path")
    return value


def _absolute_existing_directory(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(value).expanduser()
    try:
        absolute = path.absolute()
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise GateError(f"{label} does not resolve") from exc
    if absolute.is_symlink() or resolved.is_symlink() or not resolved.is_dir():
        raise GateError(f"{label} is not a regular directory")
    return resolved


def _regular_file(root: Path, relative: str, label: str) -> Path:
    relative_path = Path(_require_safe_relative_path(relative, label))
    path = root
    for index, part in enumerate(relative_path.parts):
        path = path / part
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise GateError(f"{label} is unavailable: {relative}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise GateError(f"{label} contains a symlink component: {relative}")
        final = index == len(relative_path.parts) - 1
        if final and not stat.S_ISREG(metadata.st_mode):
            raise GateError(f"{label} is not a regular file: {relative}")
        if not final and not stat.S_ISDIR(metadata.st_mode):
            raise GateError(f"{label} has a non-directory component: {relative}")
    return path


def _iter_regular_files(root: Path, label: str) -> list[tuple[str, Path]]:
    """Enumerate a tree without following symlinks and reject empty trees."""

    files: list[tuple[str, Path]] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise GateError(f"{label} tree is unavailable") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise GateError(f"{label} contains an unreadable entry") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise GateError(f"{label} contains a symlink: {path.relative_to(root)}")
            if stat.S_ISDIR(metadata.st_mode):
                visit(path)
            elif stat.S_ISREG(metadata.st_mode):
                relative = path.relative_to(root).as_posix()
                files.append((_require_safe_relative_path(relative, label), path))
            else:
                raise GateError(
                    f"{label} contains a non-regular entry: {path.relative_to(root)}"
                )

    visit(root)
    if not files:
        raise GateError(f"{label} tree is empty")
    return sorted(files, key=lambda item: item[0])


def _backend_source_files(root: Path) -> list[str]:
    result: list[str] = []
    for relative, _path in _iter_regular_files(root, "backend source"):
        parts = PurePosixPath(relative).parts
        if any(part in EXCLUDED_BACKEND_DIRS for part in parts):
            continue
        name = parts[-1]
        if (
            PurePosixPath(relative).suffix in EXCLUDED_BACKEND_SUFFIXES
            or name == ".DS_Store"
        ):
            continue
        result.append(relative)
    return result


def _app_file_receipt(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    total_bytes = 0
    for relative, path in _iter_regular_files(root, "app bundle"):
        try:
            before = path.lstat()
            size = before.st_size
            sha256 = _sha256_file(path)
            after = path.lstat()
        except OSError as exc:
            raise GateError(f"app file changed or is unavailable: {relative}") from exc
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or size != after.st_size:
            raise GateError(f"app file changed while hashing: {relative}")
        digest.update(relative.encode("utf-8") + b"\0" + sha256.encode("ascii") + b"\n")
        total_bytes += size
        files.append({"path": relative, "size_bytes": size, "sha256": sha256})
    return {
        "schema_version": APP_FILE_RECEIPT_SCHEMA,
        "path_algorithm": "utf8-path-nul-sha256-newline-v1",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
        "files": files,
    }


def _tree_digest(root: Path, entries: Sequence[str], label: str) -> str:
    digest = hashlib.sha256()
    for relative in entries:
        safe = _require_safe_relative_path(relative, label)
        path = _regular_file(root, safe, label)
        digest.update(
            safe.encode("utf-8")
            + b"\0"
            + _sha256_file(path).encode("ascii")
            + b"\n"
        )
    return digest.hexdigest()


def _load_source_identity_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "helix_v3_package_gate_source_identity", SOURCE_IDENTITY_PATH
    )
    if spec is None or spec.loader is None:
        raise GateError("current source identity helper is unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise GateError("current source identity helper could not be loaded") from exc
    return module


def _git(
    repo_root: Path, command_runner: CommandRunner, *args: str
) -> subprocess.CompletedProcess[str]:
    try:
        result = command_runner(
            ["git", *args],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="surrogateescape",
            check=False,
            timeout=COMMAND_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GateError(f"Git identity command failed: git {' '.join(args)}") from exc
    if result.returncode != 0:
        raise GateError(f"Git identity command failed: git {' '.join(args)}")
    return result


def _validate_source_identity(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise GateError("current source identity is not an object")
    required = {
        "schema_version",
        "repo",
        "git_head",
        "branch",
        "dirty",
        "status_count",
        "exclusions",
        "excluded_symlink_count",
        "excluded_symlinks",
        "file_count",
        "total_bytes",
        "tree_sha256",
    }
    missing = sorted(required - set(identity))
    if missing:
        raise GateError("current source identity is missing required fields")
    if identity.get("schema_version") != "path-nul-sha256-newline-v2-artifacts-excluded":
        raise GateError("current source identity schema is unsupported")
    if not isinstance(identity.get("repo"), str) or not identity["repo"]:
        raise GateError("current source identity repository is malformed")
    if (
        not isinstance(identity.get("git_head"), str)
        or GIT_OID_RE.fullmatch(identity["git_head"]) is None
    ):
        raise GateError("current source identity HEAD is malformed")
    if not isinstance(identity.get("branch"), (str, type(None))):
        raise GateError("current source identity branch is malformed")
    if not isinstance(identity.get("dirty"), bool):
        raise GateError("current source identity dirty flag is malformed")
    for key in ("status_count", "excluded_symlink_count", "file_count", "total_bytes"):
        if not isinstance(identity.get(key), int) or isinstance(identity.get(key), bool):
            raise GateError(f"current source identity {key} is malformed")
    if identity["file_count"] < 1 or identity["total_bytes"] < 0:
        raise GateError("current source identity counts are malformed")
    _require_sha256(identity.get("tree_sha256"), "current source tree SHA-256")
    return dict(identity)


def capture_source_identity(
    repo_root: Path,
    *,
    command_runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    module = _load_source_identity_module()
    try:
        identity = _validate_source_identity(module.compute_identity(repo_root))
    except module.SourceIdentityError as exc:
        raise GateError("current source tree identity could not be computed") from exc

    top_level = _git(
        repo_root, command_runner, "rev-parse", "--show-toplevel"
    ).stdout.rstrip("\r\n")
    try:
        resolved_top_level = Path(top_level).resolve(strict=True)
    except OSError as exc:
        raise GateError("Git top-level path is unavailable") from exc
    if resolved_top_level != repo_root:
        raise GateError("source identity repository does not match Git top-level")

    head = _git(repo_root, command_runner, "rev-parse", "--verify", "HEAD").stdout.strip()
    head_tree = _git(
        repo_root, command_runner, "rev-parse", "--verify", "HEAD^{tree}"
    ).stdout.strip()
    if GIT_OID_RE.fullmatch(head) is None or GIT_OID_RE.fullmatch(head_tree) is None:
        raise GateError("Git HEAD identity is malformed")
    if head != identity["git_head"]:
        raise GateError("source identity HEAD changed during qualification")
    status = _git(
        repo_root,
        command_runner,
        "status",
        "--porcelain=v2",
        "--untracked-files=all",
        "-z",
    ).stdout.encode("utf-8", "surrogateescape")
    status_records = [record for record in status.split(b"\0") if record]
    return {
        "identity": identity,
        "status_manifest": {
            "schema_version": SOURCE_STATUS_SCHEMA,
            "git_head": head,
            "git_head_tree_oid": head_tree,
            "branch": identity.get("branch"),
            "dirty": identity["dirty"],
            "status_count": identity["status_count"],
            "status_record_count": len(status_records),
            "status_porcelain_v2_sha256": _sha256_bytes(status),
        },
    }


def _read_backend_manifest(
    app_root: Path, expected: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = expected["manifest_relative_path"]
    path = _regular_file(app_root, relative, "backend manifest")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise GateError("backend manifest is unavailable") from exc
    manifest_sha256 = _sha256_bytes(raw)
    if manifest_sha256 != expected["manifest_sha256"]:
        raise GateError("backend manifest bytes do not match the candidate config")
    value = _parse_json_bytes(raw, label="backend manifest")
    if not isinstance(value, dict):
        raise GateError("backend manifest is not a JSON object")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "contract",
            "file_count",
            "tree_sha256",
            "files",
            "excluded_dirs",
        },
        "backend manifest",
    )
    if value.get("schema_version") != expected["schema_version"]:
        raise GateError("backend manifest schema differs from the candidate config")
    if value.get("contract") != expected["contract"]:
        raise GateError("backend manifest contract differs from the candidate config")
    excluded_dirs = value.get("excluded_dirs")
    if (
        not isinstance(excluded_dirs, list)
        or any(not isinstance(item, str) for item in excluded_dirs)
        or len(excluded_dirs) != len(set(excluded_dirs))
        or set(excluded_dirs) != EXCLUDED_BACKEND_DIRS
    ):
        raise GateError("backend manifest excluded_dirs policy is unsupported")
    if (
        not isinstance(value.get("file_count"), int)
        or isinstance(value.get("file_count"), bool)
        or value["file_count"] != expected["file_count"]
        or value["file_count"] < 1
    ):
        raise GateError("backend manifest file_count is malformed or differs")
    if (
        not isinstance(value.get("tree_sha256"), str)
        or SHA256_RE.fullmatch(value["tree_sha256"]) is None
        or value["tree_sha256"] != expected["tree_sha256"]
    ):
        raise GateError("backend manifest tree SHA-256 is malformed or differs")
    files = value.get("files")
    if not isinstance(files, list) or not files or len(files) != value["file_count"]:
        raise GateError("backend manifest file list is malformed")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_path in files:
        safe = _require_safe_relative_path(raw_path, "backend manifest file")
        if safe in seen:
            raise GateError(f"backend manifest contains a duplicate path: {safe}")
        if safe == relative:
            raise GateError("backend manifest cannot list itself as an owned file")
        seen.add(safe)
        normalized.append(safe)
    if normalized != sorted(normalized):
        raise GateError("backend manifest file list is not canonically sorted")
    return value, {
        "path": relative,
        "excluded_dirs": list(excluded_dirs),
        "files": list(normalized),
        "sha256": manifest_sha256,
        "schema_version": value["schema_version"],
        "contract": value["contract"],
        "file_count": value["file_count"],
        "tree_sha256": value["tree_sha256"],
    }


def _compare_backend(
    app_root: Path,
    source_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_relative = manifest["path"]
    backend_root = app_root / Path(manifest_relative).parent
    manifest_name = Path(manifest_relative).name
    if backend_root.is_symlink() or not backend_root.is_dir():
        raise GateError("packaged backend root is not a regular directory")

    files = list(manifest["files"])
    manifest_set = set(files)
    package_set = {
        relative
        for relative, _path in _iter_regular_files(backend_root, "packaged backend")
        if relative != manifest_name
    }
    source_set = set(_backend_source_files(source_root))

    comparable_files = sorted(manifest_set & source_set & package_set)
    source_entries_tree = _tree_digest(
        source_root, comparable_files, "backend source entry"
    )
    package_entries_tree = _tree_digest(
        backend_root, comparable_files, "packaged backend entry"
    )
    source_actual_tree = _tree_digest(source_root, sorted(source_set), "backend source tree")
    package_actual_tree = _tree_digest(
        backend_root, sorted(package_set), "packaged backend tree"
    )

    stale_files: list[dict[str, str]] = []
    for relative in files:
        source_path = source_root.joinpath(*PurePosixPath(relative).parts)
        package_path = backend_root.joinpath(*PurePosixPath(relative).parts)
        source_sha256 = (
            _sha256_file(source_path) if relative in source_set else None
        )
        package_sha256 = (
            _sha256_file(package_path) if relative in package_set else None
        )
        if source_sha256 != package_sha256:
            stale_files.append(
                {
                    "path": relative,
                    "source_sha256": source_sha256,
                    "package_sha256": package_sha256,
                }
            )

    file_sets_match = source_set == manifest_set == package_set
    trees_match = (
        file_sets_match
        and source_entries_tree == package_entries_tree
        and source_actual_tree == package_actual_tree
        and source_actual_tree == manifest["tree_sha256"]
    )
    return {
        "manifest": dict(manifest),
        "file_sets_match": file_sets_match,
        "trees_match": trees_match,
        "source_file_count": len(source_set),
        "package_file_count": len(package_set),
        "source_actual_tree_sha256": source_actual_tree,
        "package_actual_tree_sha256": package_actual_tree,
        "source_manifest_entries_tree_sha256": source_entries_tree,
        "package_manifest_entries_tree_sha256": package_entries_tree,
        "source_only_files": sorted(source_set - manifest_set),
        "package_only_files": sorted(package_set - manifest_set),
        "missing_source_files": sorted(manifest_set - source_set),
        "missing_package_files": sorted(manifest_set - package_set),
        "stale_files": stale_files,
        "bytes_match": file_sets_match and not stale_files,
    }


def _inspect_plist(
    app_root: Path, expected: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _regular_file(app_root, "Contents/Info.plist", "Info.plist")
    try:
        raw = path.read_bytes()
        value = plistlib.loads(raw)
    except (OSError, plistlib.InvalidFileException, ValueError) as exc:
        raise GateError("Info.plist is malformed") from exc
    if not isinstance(value, dict):
        raise GateError("Info.plist is not a dictionary")
    actual = {
        "CFBundleIdentifier": value.get("CFBundleIdentifier"),
        "CFBundleDisplayName": value.get("CFBundleDisplayName"),
        "CFBundleShortVersionString": value.get("CFBundleShortVersionString"),
        "CFBundleVersion": value.get("CFBundleVersion"),
        "CFBundleExecutable": value.get("CFBundleExecutable"),
    }
    expected_actual = {
        "CFBundleIdentifier": expected["bundle_identifier"],
        "CFBundleDisplayName": expected["display_name"],
        "CFBundleShortVersionString": expected["version"],
        "CFBundleVersion": expected["version"],
        "CFBundleExecutable": Path(expected["executable_relative_path"]).name,
    }
    if actual != expected_actual:
        raise GateError("Info.plist identity differs from the candidate config")
    if _sha256_bytes(raw) != expected["info_plist_sha256"]:
        raise GateError("Info.plist bytes differ from the candidate config")
    return actual, {"path": "Contents/Info.plist", "sha256": _sha256_bytes(raw)}


def _run_tool(
    command: Sequence[str], command_runner: CommandRunner
) -> dict[str, Any]:
    try:
        result = command_runner(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=COMMAND_TIMEOUT_S,
        )
    except FileNotFoundError:
        return {
            "command": list(command),
            "status": "unavailable",
            "returncode": None,
            "output": "required local metadata tool is unavailable",
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "command": list(command),
            "status": "error",
            "returncode": None,
            "output": f"{type(exc).__name__}: {exc}",
        }
    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    return {
        "command": list(command),
        "status": "pass" if result.returncode == 0 else "fail",
        "returncode": result.returncode,
        "output": output,
    }


def _parse_codesign_details(output: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "identifier": None,
        "team_identifier": None,
        "signature": None,
        "cdhash": None,
        "runtime_version": None,
    }
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip().lower().replace(" ", "_").replace("-", "_")
        if key == "teamidentifier":
            key = "team_identifier"
        value = value.strip()
        if key in {"identifier", "team_identifier", "signature", "cdhash", "runtime_version"}:
            if key == "team_identifier" and value == "not set":
                value = None
            parsed[key] = value
    return parsed


def _distribution_trust(
    signature_kind: str,
    team_identifier: str | None,
    required_team_identifier: str | None,
    notarization_passed: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if signature_kind != "developer_id":
        reasons.append("signature_is_not_developer_id")
    if not team_identifier:
        reasons.append("team_identifier_is_missing")
    elif required_team_identifier and team_identifier != required_team_identifier:
        reasons.append("team_identifier_differs")
    if not notarization_passed:
        reasons.append("notarization_ticket_is_not_stapled")
    return not reasons, reasons


def inspect_macos_trust(
    app_root: Path,
    trust_policy: Mapping[str, Any],
    *,
    command_runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    verify = _run_tool(
        [
            "/usr/bin/codesign",
            "--verify",
            "--deep",
            "--strict",
            "--verbose=4",
            str(app_root),
        ],
        command_runner,
    )
    details = _run_tool(
        ["/usr/bin/codesign", "-dv", "--verbose=4", str(app_root)],
        command_runner,
    )
    notarization = _run_tool(
        ["/usr/bin/xcrun", "stapler", "validate", str(app_root)],
        command_runner,
    )
    parsed = (
        _parse_codesign_details(details["output"])
        if details["status"] == "pass"
        else {
            "identifier": None,
            "team_identifier": None,
            "signature": None,
            "cdhash": None,
            "runtime_version": None,
        }
    )
    signature_value = parsed.get("signature")
    if signature_value == "adhoc":
        signature_kind = "ad_hoc"
    elif signature_value == "Developer ID Application":
        signature_kind = "developer_id"
    else:
        signature_kind = "unknown"
    codesign_valid = verify["status"] == "pass" and signature_kind != "unknown"
    distribution_qualified, reasons = _distribution_trust(
        signature_kind,
        parsed.get("team_identifier"),
        trust_policy.get("required_team_identifier"),
        notarization["status"] == "pass",
    )
    if signature_kind == "ad_hoc" and codesign_valid:
        static_trust_status = "ad_hoc_valid_not_distribution_qualified"
    elif codesign_valid:
        static_trust_status = "signed"
    else:
        static_trust_status = "signature_invalid_or_unavailable"
    return {
        "codesign_verify": verify,
        "codesign_details": details,
        "gatekeeper_assessment": {
            "status": "not_run",
            "reason": "gate is network-free; local signature and stapled ticket decide trust",
        },
        "notarization": notarization,
        "identity": parsed,
        "signature_kind": signature_kind,
        "codesign_valid": codesign_valid,
        "static_trust_status": static_trust_status,
        "distribution_trust": {
            "qualified": distribution_qualified,
            "status": "qualified" if distribution_qualified else "not_qualified",
            "reasons": reasons,
        },
    }


def _base_receipt(
    *,
    app_path: str,
    repo_root: str,
    config_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "gate_version": GATE_VERSION,
        "gate_id": "HELIX-V3-PACKAGE-STATIC-QUALIFICATION-V1",
        "status": "not_qualified",
        "checked_at": _utc_now(),
        "inputs": {
            "app_path": app_path,
            "repo_root": repo_root,
            "config_sha256": config_sha256,
        },
        "mode": {
            "metadata_only": True,
            "launches_app": False,
            "uses_overlay_or_staging": False,
            "loads_model": False,
            "uses_network": False,
            "reads_user_data": False,
        },
        "errors": [],
        "caveats": [
            "Static qualification does not execute or behaviorally validate the app.",
            "A valid ad-hoc signature is static identity evidence, not distribution trust.",
        ],
    }


def qualify_package(
    app_path: str | os.PathLike[str],
    repo_root: str | os.PathLike[str],
    config: Mapping[str, Any],
    *,
    config_sha256: str = "inline",
    command_runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    """Return a fail-closed receipt; malformed inputs never raise to the caller."""

    app_argument = str(Path(app_path).expanduser().absolute())
    repo_argument = str(Path(repo_root).expanduser().absolute())
    receipt = _base_receipt(
        app_path=app_argument,
        repo_root=repo_argument,
        config_sha256=config_sha256,
    )
    try:
        pinned = _validate_config(config)
        app_root = _absolute_existing_directory(app_path, "explicit app bundle")
        source_root = _absolute_existing_directory(repo_root, "source repository")
        if app_root.name != pinned["app"]["bundle_name"]:
            raise GateError("explicit app bundle name differs from the candidate config")

        source_before = capture_source_identity(
            source_root, command_runner=command_runner
        )
        receipt["source"] = {
            **source_before,
            "stable_during_qualification": False,
        }

        file_receipt = _app_file_receipt(app_root)
        app_tree_matches = (
            file_receipt["file_count"] == pinned["app"]["file_count"]
            and file_receipt["total_bytes"] == pinned["app"]["total_bytes"]
            and file_receipt["tree_sha256"] == pinned["app"]["tree_sha256"]
        )
        plist_identity, plist_receipt = _inspect_plist(app_root, pinned["app"])
        executable_relative = pinned["app"]["executable_relative_path"]
        executable_path = _regular_file(app_root, executable_relative, "app executable")
        executable_sha256 = _sha256_file(executable_path)
        executable_matches = executable_sha256 == pinned["app"]["executable_sha256"]
        receipt["app"] = {
            "path": str(app_root),
            "name": app_root.name,
            "file_receipt": file_receipt,
            "expected_file_count": pinned["app"]["file_count"],
            "expected_total_bytes": pinned["app"]["total_bytes"],
            "expected_tree_sha256": pinned["app"]["tree_sha256"],
            "tree_matches_expected": app_tree_matches,
            "info_plist": {**plist_receipt, "identity": plist_identity},
            "executable": {
                "path": executable_relative,
                "sha256": executable_sha256,
                "expected_sha256": pinned["app"]["executable_sha256"],
                "matches_expected": executable_matches,
            },
        }

        manifest, manifest_receipt = _read_backend_manifest(app_root, pinned["backend"])
        backend = _compare_backend(
            app_root,
            source_root / "studio" / "backend",
            manifest_receipt,
        )
        receipt["backend"] = backend

        signature = inspect_macos_trust(
            app_root, pinned["trust"], command_runner=command_runner
        )
        receipt["signature"] = signature
        final_file_receipt = _app_file_receipt(app_root)
        app_bytes_stable = final_file_receipt == file_receipt
        if not app_bytes_stable:
            raise GateError("app bundle bytes changed during qualification")
        receipt["app"]["file_receipt_stable_during_qualification"] = app_bytes_stable

        source_after = capture_source_identity(
            source_root, command_runner=command_runner
        )
        if source_after != source_before:
            raise GateError("current source identity changed during qualification")
        receipt["source"] = {
            **source_after,
            "stable_during_qualification": True,
        }

        checks = {
            "source_identity_bound": True,
            "source_identity_stable_during_qualification": True,
            "complete_app_file_receipt_matches": app_tree_matches,
            "app_bytes_stable_during_qualification": app_bytes_stable,
            "plist_identity_matches": True,
            "executable_sha256_matches": executable_matches,
            "backend_manifest_matches_config": True,
            "backend_file_sets_match": backend["file_sets_match"],
            "backend_trees_match": backend["trees_match"],
            "backend_source_package_bytes_match": backend["bytes_match"],
            "codesign_valid": signature["codesign_valid"],
        }
        distribution_qualified = signature["distribution_trust"]["qualified"]
        checks["distribution_trust"] = distribution_qualified
        receipt["checks"] = checks

        if not all(value for key, value in checks.items() if key != "distribution_trust"):
            raise GateError("one or more static package identity checks failed")
        target = pinned["trust"]["qualification_target"]
        if target == "distribution":
            if not distribution_qualified:
                receipt["errors"].append(
                    "distribution trust is required but the package does not qualify"
                )
                return receipt
            receipt["status"] = "qualified_distribution"
        else:
            receipt["status"] = "qualified_static_identity"
        return receipt
    except Exception as exc:
        receipt["errors"].append(f"{type(exc).__name__}: {exc}")
        return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-path", required=True, help="explicit .app bundle to inspect")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="current source repository identity root",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config, config_sha256 = load_config(args.config)
    except GateError as exc:
        receipt = _base_receipt(
            app_path=str(args.app_path),
            repo_root=str(args.repo_root),
            config_sha256="unavailable",
        )
        receipt["errors"].append(f"GateError: {exc}")
    else:
        receipt = qualify_package(
            args.app_path,
            args.repo_root,
            config,
            config_sha256=config_sha256,
        )
    print(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if receipt["status"] in {"qualified_static_identity", "qualified_distribution"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
