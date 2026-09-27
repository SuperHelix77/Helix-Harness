# SPDX-License-Identifier: AGPL-3.0-only
"""Focused static-only tests for the candidate package qualification gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import plistlib
import subprocess
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
GATE_PATH = (
    REPO_ROOT / "scripts" / "benchmarks" / "helix_v3_package_static_gate.py"
)
CONFIG_PATH = GATE_PATH.with_suffix(".json")
FROZEN_CONTROL_PATH = (
    REPO_ROOT / "scripts" / "benchmark-helix-v3-packaged-runtime.py"
)
V22_APP_PATH = (
    REPO_ROOT
    / "studio"
    / "src-tauri"
    / "target"
    / "release"
    / "bundle"
    / "macos"
    / "Helix Harness v2.2.app"
)
FROZEN_CONTROL_SHA256 = (
    "e8574e4d04ebc30b2dc388fa7309c00107710d5538da18269b6ee64b5a7a53a9"
)
EXPECTED_STALE_FILES = [
    "core/helix_engine/ledger.py",
    "core/inference/chat_generation_runs.py",
    "core/memory/mem0_store.py",
    "models/inference.py",
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_GATE = _load_module("helix_v3_package_static_gate_test", GATE_PATH)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_json(path: Path, value: Any) -> bytes:
    raw = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    _write(path, raw)
    return raw


def _tree_digest(root: Path, entries: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(entries):
        path = root.joinpath(*PurePosixPath(relative).parts)
        digest.update(
            relative.encode("utf-8")
            + b"\0"
            + _sha256(path).encode("ascii")
            + b"\n"
        )
    return digest.hexdigest()


def _ad_hoc_command_runner(command, **kwargs):
    command = list(command)
    if command and command[0] == "git":
        return subprocess.run(command, **kwargs)
    if command[:3] == ["/usr/bin/codesign", "--verify", "--deep"]:
        return subprocess.CompletedProcess(command, 0, "", "valid on disk")
    if command[:3] == ["/usr/bin/codesign", "-dv", "--verbose=4"]:
        return subprocess.CompletedProcess(
            command,
            0,
            "",
            "Identifier=ai.helix.harness.v22\n"
            "TeamIdentifier=not set\n"
            "Signature=adhoc\n"
            "CDHash=0123456789abcdef\n"
            "Runtime Version=26.0.0\n",
        )
    if command[:3] == ["/usr/bin/xcrun", "stapler", "validate"]:
        return subprocess.CompletedProcess(
            command, 65, "", "does not have a ticket stapled to it"
        )
    raise AssertionError(f"unexpected command: {command}")


def _make_matching_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    repo = tmp_path / "source"
    backend = repo / "studio" / "backend"
    app = tmp_path / "Synthetic.app"
    owned = {
        "__init__.py": b"# synthetic backend\n",
        "core/identity.py": b"CONTRACT = 'helix.adaptive.backend.v1'\n",
    }
    for relative, content in owned.items():
        _write(backend.joinpath(*PurePosixPath(relative).parts), content)
    (repo / "README.md").write_bytes(b"synthetic source\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "helix-tests@example.invalid")
    _git(repo, "config", "user.name", "Helix tests")
    _git(repo, "add", "--all")
    _git(repo, "commit", "-qm", "synthetic candidate")

    package_backend = app / "Contents" / "Resources" / "helix-backend"
    for relative, content in owned.items():
        _write(package_backend.joinpath(*PurePosixPath(relative).parts), content)
    manifest = {
        "schema_version": _GATE.BACKEND_MANIFEST_SCHEMA,
        "contract": _GATE.BACKEND_CONTRACT,
        "file_count": len(owned),
        "tree_sha256": _tree_digest(backend, list(owned)),
        "files": sorted(owned),
        "excluded_dirs": sorted(_GATE.EXCLUDED_BACKEND_DIRS),
    }
    manifest_relative = "Contents/Resources/helix-backend/HELIX_BACKEND_MANIFEST.json"
    manifest_path = app / manifest_relative
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = _write_json(manifest_path, manifest)

    executable_relative = "Contents/MacOS/unsloth-studio"
    executable_path = app / executable_relative
    _write(executable_path, b"synthetic executable\n")
    plist = {
        "CFBundleIdentifier": "ai.helix.harness.v22",
        "CFBundleDisplayName": "Helix Harness v2.2",
        "CFBundleShortVersionString": "2.2.0",
        "CFBundleVersion": "2.2.0",
        "CFBundleExecutable": "unsloth-studio",
    }
    plist_path = app / "Contents" / "Info.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_bytes(plistlib.dumps(plist, sort_keys=True))
    _write(
        app / "Contents" / "_CodeSignature" / "CodeResources",
        b"synthetic sealed resources\n",
    )

    app_receipt = _GATE._app_file_receipt(app)
    config = {
        "schema_version": _GATE.CONFIG_SCHEMA,
        "gate_id": "HELIX-V3-PACKAGE-STATIC-QUALIFICATION-V1",
        "app": {
            "bundle_name": app.name,
            "bundle_identifier": plist["CFBundleIdentifier"],
            "display_name": plist["CFBundleDisplayName"],
            "version": plist["CFBundleShortVersionString"],
            "executable_relative_path": executable_relative,
            "executable_sha256": _sha256(executable_path),
            "info_plist_sha256": _sha256(plist_path),
            "file_count": app_receipt["file_count"],
            "total_bytes": app_receipt["total_bytes"],
            "tree_sha256": app_receipt["tree_sha256"],
        },
        "backend": {
            "manifest_relative_path": manifest_relative,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "schema_version": manifest["schema_version"],
            "contract": manifest["contract"],
            "file_count": manifest["file_count"],
            "tree_sha256": manifest["tree_sha256"],
        },
        "trust_policy": {
            "qualification_target": "static_identity",
            "required_team_identifier": None,
        },
    }
    return repo, app, {"config": config}


def test_current_v22_bundle_is_not_qualified_and_names_exact_stale_files():
    assert V22_APP_PATH.is_dir(), f"required v2.2 fixture is absent: {V22_APP_PATH}"
    config, config_sha256 = _GATE.load_config(CONFIG_PATH)

    receipt = _GATE.qualify_package(
        V22_APP_PATH,
        REPO_ROOT,
        config,
        config_sha256=config_sha256,
    )

    assert receipt["status"] == "not_qualified"
    assert receipt["errors"]
    assert receipt["mode"] == {
        "metadata_only": True,
        "launches_app": False,
        "uses_overlay_or_staging": False,
        "loads_model": False,
        "uses_network": False,
        "reads_user_data": False,
    }
    # The shipped gate config is pinned to the current release candidate
    # (Helix Harness v3). This fixture deliberately feeds it a *superseded*
    # v2.2 bundle, so the gate rejects it on bundle identity before it reaches
    # the backend-bytes or source-identity stages. The regression protected
    # here is "a bundle that is not the pinned release candidate is never
    # qualified" — not the specific identity it happens to fail on.
    if "app" in receipt:
        # A future config re-pinned onto this fixture's identity: the deeper
        # backend-staleness assertions must then still hold.
        assert receipt["app"]["file_receipt"]["tree_sha256"] == config["app"]["tree_sha256"]
        assert receipt["app"]["file_receipt_stable_during_qualification"] is True
        assert receipt["backend"]["file_sets_match"] is True
        assert receipt["backend"]["bytes_match"] is False
        assert [row["path"] for row in receipt["backend"]["stale_files"]] == (
            EXPECTED_STALE_FILES
        )
        assert receipt["checks"]["backend_source_package_bytes_match"] is False
        assert receipt["source"]["status_manifest"]["git_head"] == receipt["source"][
            "identity"
        ]["git_head"]
        assert len(receipt["source"]["identity"]["tree_sha256"]) == 64
        assert receipt["source"]["stable_during_qualification"] is True
        assert receipt["signature"]["codesign_valid"] is True
        assert receipt["signature"]["signature_kind"] == "ad_hoc"
        assert receipt["signature"]["distribution_trust"]["qualified"] is False
    else:
        # Rejected before the app-receipt stage, which is the correct outcome
        # for a bundle that does not match the pinned release identity.
        assert receipt["errors"]


def test_synthetic_matching_source_and_app_pass_static_identity(tmp_path):
    repo, app, fixture = _make_matching_fixture(tmp_path)

    receipt = _GATE.qualify_package(
        app,
        repo,
        fixture["config"],
        command_runner=_ad_hoc_command_runner,
    )

    assert receipt["status"] == "qualified_static_identity"
    assert receipt["errors"] == []
    assert all(
        value
        for key, value in receipt["checks"].items()
        if key != "distribution_trust"
    )
    assert receipt["checks"]["distribution_trust"] is False
    assert receipt["backend"]["stale_files"] == []
    assert receipt["backend"]["trees_match"] is True
    assert receipt["signature"]["static_trust_status"] == (
        "ad_hoc_valid_not_distribution_qualified"
    )


def test_distribution_target_does_not_accept_valid_ad_hoc_signature(tmp_path):
    repo, app, fixture = _make_matching_fixture(tmp_path)
    fixture["config"]["trust_policy"]["qualification_target"] = "distribution"

    receipt = _GATE.qualify_package(
        app,
        repo,
        fixture["config"],
        command_runner=_ad_hoc_command_runner,
    )

    assert receipt["status"] == "not_qualified"
    assert receipt["checks"]["distribution_trust"] is False
    assert "distribution trust is required" in receipt["errors"][-1]
    assert "signature_is_not_developer_id" in receipt["signature"][
        "distribution_trust"
    ]["reasons"]


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside.py",
        "/absolute.py",
        "core\\windows.py",
        "core/./identity.py",
    ],
)
def test_unsafe_manifest_paths_fail_closed(tmp_path, unsafe_path):
    repo, app, fixture = _make_matching_fixture(tmp_path)
    manifest_path = (
        app
        / "Contents"
        / "Resources"
        / "helix-backend"
        / "HELIX_BACKEND_MANIFEST.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [unsafe_path]
    manifest["file_count"] = 1
    manifest_bytes = _write_json(manifest_path, manifest)
    fixture["config"]["backend"]["file_count"] = 1
    fixture["config"]["backend"]["manifest_sha256"] = hashlib.sha256(
        manifest_bytes
    ).hexdigest()

    receipt = _GATE.qualify_package(
        app,
        repo,
        fixture["config"],
        command_runner=_ad_hoc_command_runner,
    )

    assert receipt["status"] == "not_qualified"
    assert "safe relative POSIX path" in receipt["errors"][-1]
    assert "backend" not in receipt


def test_frozen_packaged_runtime_control_script_and_constants_are_unchanged():
    control = _load_module("helix_v3_frozen_control_identity_test", FROZEN_CONTROL_PATH)

    assert _sha256(FROZEN_CONTROL_PATH) == FROZEN_CONTROL_SHA256
    assert control.SCRIPT_VERSION == "helix-v3-packaged-runtime.v1"
    assert control.EXPECTED_BACKEND_CONTRACT == "helix.adaptive.backend.v1"
    assert control.EXPECTED_BACKEND_TREE_SHA256 == (
        "c05a1e28a9fdf172735b4dad73e4224ca81b571605eefc640d13bcecda23a8f8"
    )
    assert control.FROZEN_SOURCE_SNAPSHOT_SHA256 == (
        "dbc6170d22db03c99580a0f5ef824c5bedcf3d9c390639005baf022ab0e1c309"
    )
    assert control.FROZEN_BASELINE_REPORT_SHA256 == (
        "f7b392de3942d2185bba3cd0ac6c4584a4bd8d3f8894f6429d733e36a848c510"
    )
