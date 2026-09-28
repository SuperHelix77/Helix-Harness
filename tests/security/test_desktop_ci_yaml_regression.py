# SPDX-License-Identifier: AGPL-3.0-only
"""Keep the desktop release workflow and its embedded contract probe executable."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/desktop-app-clean-machine-ci.yml"


def _workflow():
    # BaseLoader preserves GitHub's `on` key instead of YAML 1.1's boolean coercion.
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _installer_script():
    steps = _workflow()["jobs"]["macos"]["steps"]
    return next(step["run"] for step in steps if step.get("name") ==
                "Run the bundled installer, the path first launch takes")


def _probe():
    match = re.search(r'^"\$PY" - <<\'PYHELIX\'\n(.*?)\nPYHELIX$',
                      _installer_script(), re.MULTILINE | re.DOTALL)
    assert match is not None, "backend-contract probe must remain in the shell step"
    source = match.group(1) + "\n"
    compile(source, "<desktop-backend-contract-probe>", "exec")
    return source


def test_workflow_parses_and_retains_all_platform_jobs():
    workflow = _workflow()
    assert {"macos", "linux", "windows", "appimage-pr-build", "appimage-portability",
            "appimage-model-download"} <= workflow["jobs"].keys()
    assert {"pull_request", "workflow_dispatch", "schedule"} <= workflow["on"].keys()


def test_installer_script_and_embedded_python_are_syntactically_valid():
    result = subprocess.run(["bash", "-n"], input=_installer_script(), text=True,
                            capture_output=True, timeout=5, check=False)
    assert result.returncode == 0, result.stderr
    assert "here-document" not in result.stderr, result.stderr
    _probe()


@pytest.mark.parametrize("contract,include_tool,expected_success", [
    ("helix.adaptive.backend.v1", True, True),
    ("wrong-backend-contract", True, False),
    ("helix.adaptive.backend.v1", False, False),
])
def test_contract_probe_accepts_only_the_expected_backend(
    tmp_path, contract, include_tool, expected_success,
):
    studio = tmp_path / "studio"
    policy = studio / "backend/state/tool_policy.py"
    controller = studio / "backend/core/helix_engine/controller.py"
    policy.parent.mkdir(parents=True)
    controller.parent.mkdir(parents=True)
    (studio / "__init__.py").write_text("", encoding="utf-8")
    policy.write_text(
        f"HELIX_HARNESS_BACKEND_CONTRACT = {contract!r}\n" +
        ("def require_tool_access(): pass\n" if include_tool else ""), encoding="utf-8",
    )
    controller.write_text("def run_closed_loop(): pass\n", encoding="utf-8")
    result = subprocess.run([sys.executable, "-"], input=_probe(), text=True,
                            capture_output=True, timeout=5, cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH": str(tmp_path)}, check=False)
    assert (result.returncode == 0) is expected_success, result.stderr
    if expected_success:
        assert "Helix backend contract loaded" in result.stdout
