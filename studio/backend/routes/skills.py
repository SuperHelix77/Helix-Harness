# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local Codex/Claude Code skill management.

Skill files are plain text and are never executed by this route. The manager only
reads the two user-owned skill roots and accepts GitHub HTTPS repositories (or an
explicit local directory) as installation sources.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from auth.authentication import get_current_subject
from core.inference.skills import (
    SkillError,
    SkillNotFoundError,
    list_skills,
    set_skill_enabled,
)

router = APIRouter()

SkillTarget = Literal["codex", "claude", "both"]
_SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_GITHUB_SOURCE = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?/?$")


class CreateSkillRequest(BaseModel):
    name: str = Field(min_length = 1, max_length = 64)
    description: str = Field(min_length = 1, max_length = 500)
    instructions: str = Field(min_length = 1, max_length = 100_000)
    target: SkillTarget = "codex"


class InstallSkillRequest(BaseModel):
    source: str = Field(min_length = 1, max_length = 2_000)
    target: SkillTarget = "codex"


def _roots() -> dict[str, Path]:
    home = Path.home()
    return {"codex": home / ".codex" / "skills", "claude": home / ".claude" / "skills"}


def _valid_name(name: str) -> str:
    normalized = name.strip().lower()
    if not _SKILL_NAME.fullmatch(normalized):
        raise HTTPException(
            status_code = 400,
            detail = "Skill names may use lowercase letters, numbers, dots, underscores, and hyphens.",
        )
    return normalized


def _description(path: Path) -> str:
    try:
        text = (path / "SKILL.md").read_text(encoding = "utf-8")[:4_000]
    except (OSError, UnicodeError):
        return "Local skill"
    match = re.search(r"(?im)^description:\s*(.+?)\s*$", text)
    return (match.group(1).strip().strip('"\'') if match else "Local skill")[:500]


def _skill_marker(path: Path) -> Path:
    marker = path / "SKILL.md"
    if not path.is_dir() or path.is_symlink() or not marker.is_file() or marker.is_symlink():
        raise HTTPException(status_code = 404, detail = "Local skill not found.")
    root = path.parent.resolve()
    resolved = marker.resolve()
    if root not in resolved.parents:
        raise HTTPException(status_code = 400, detail = "Invalid local skill path.")
    return marker


def _skill_dirs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    if (root / "SKILL.md").is_file():
        candidates.append(root)
    for parent in (root, root / "skills"):
        if not parent.is_dir():
            continue
        for child in sorted(parent.iterdir()):
            if child.is_dir() and not child.is_symlink() and (child / "SKILL.md").is_file():
                candidates.append(child)
    return list(dict.fromkeys(candidates))


def _list_skills() -> list[dict[str, object]]:
    roots = _roots()
    by_name: dict[str, dict[str, object]] = {}
    for ecosystem, root in roots.items():
        for path in _skill_dirs(root):
            name = path.name
            row = by_name.setdefault(
                name,
                {"name": name, "description": _description(path), "path": str(path), "ecosystems": []},
            )
            ecosystems = row["ecosystems"]
            if ecosystem not in ecosystems:
                ecosystems.append(ecosystem)
    return sorted(by_name.values(), key = lambda row: str(row["name"]))


def _targets(target: SkillTarget) -> list[tuple[str, Path]]:
    roots = _roots()
    return [(name, roots[name]) for name in ("codex", "claude") if target == "both" or target == name]


def _copy_skill(source: Path, target_root: Path, name: str) -> None:
    marker = source / "SKILL.md"
    if not marker.is_file() or marker.is_symlink() or source.is_symlink():
        raise HTTPException(status_code = 400, detail = f"{source} does not contain a SKILL.md file.")
    # Do not follow a repository symlink into an arbitrary local path while
    # copying an otherwise harmless-looking skill package.
    if any(path.is_symlink() for path in source.rglob("*")):
        raise HTTPException(status_code = 400, detail = f"Skill package contains symlinks: {source}")
    raw_destination = target_root / name
    if raw_destination.is_symlink():
        raise HTTPException(status_code = 409, detail = f"Skill destination is a symlink: {name}")
    destination = raw_destination.resolve()
    root = target_root.resolve()
    if root not in destination.parents:
        raise HTTPException(status_code = 400, detail = "Invalid skill destination.")
    target_root.mkdir(parents = True, exist_ok = True)
    if destination.exists() and not destination.is_dir():
        raise HTTPException(status_code = 409, detail = f"Skill destination already exists: {name}")
    if destination.exists() and any(path.is_symlink() for path in destination.rglob("*")):
        raise HTTPException(status_code = 409, detail = f"Skill destination contains symlinks: {name}")
    shutil.copytree(source, destination, dirs_exist_ok = True, symlinks = False)


def _source_directory(source: str) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    parsed = urlparse(source)
    if parsed.scheme or parsed.netloc:
        if not _GITHUB_SOURCE.fullmatch(source.strip()):
            raise HTTPException(status_code = 400, detail = "Only a direct GitHub HTTPS repository URL is allowed.")
        checkout = tempfile.TemporaryDirectory(prefix = "unsloth-skill-")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", source.strip(), checkout.name],
                check = True,
                capture_output = True,
                text = True,
                timeout = 180,
            )
        except FileNotFoundError as error:
            checkout.cleanup()
            raise HTTPException(status_code = 503, detail = "git is not installed on this Mac.") from error
        except subprocess.TimeoutExpired as error:
            checkout.cleanup()
            raise HTTPException(status_code = 504, detail = "The skill repository clone timed out.") from error
        except subprocess.CalledProcessError as error:
            checkout.cleanup()
            detail = (error.stderr or "GitHub clone failed.").strip()[-1_000:]
            raise HTTPException(status_code = 400, detail = detail) from error
        return Path(checkout.name), checkout
    local = Path(source).expanduser().resolve()
    if not local.is_dir():
        raise HTTPException(status_code = 400, detail = "Skill source must be a GitHub repository or local directory.")
    return local, None


def _install(source: str, target: SkillTarget) -> list[dict[str, object]]:
    root, checkout = _source_directory(source)
    try:
        sources = _skill_dirs(root)
        if not sources:
            raise HTTPException(status_code = 400, detail = "The source contains no SKILL.md skill package.")
        planned: list[tuple[Path, Path, str]] = []
        for skill in sources:
            name = _valid_name(skill.name)
            for _, destination_root in _targets(target):
                raw_destination = destination_root / name
                if raw_destination.is_symlink() or raw_destination.exists():
                    raise HTTPException(status_code = 409, detail = f"Skill destination already exists: {name}")
                planned.append((skill, destination_root, name))
        for skill, destination_root, name in planned:
            _copy_skill(skill, destination_root, name)
        return _list_skills()
    finally:
        if checkout is not None:
            checkout.cleanup()


class SkillRecord(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    name: str
    description: str
    source: Literal["agents", "claude", "bundled"]
    enabled: bool
    valid: bool
    shadowed: bool
    shadowed_by: Optional[Literal["agents", "claude", "bundled"]] = None
    error: Optional[str] = None
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Optional[dict[str, str]] = None
    allowed_tools: Optional[str] = None


class SkillEnabledRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    enabled: StrictBool


@router.get("", response_model = list[SkillRecord])
def get_skills(current_subject: str = Depends(get_current_subject)) -> list[dict[str, Any]]:
    """Preserve the upstream Agent Skills catalog contract."""
    try:
        records = list_skills()
    except SkillError as exc:
        raise HTTPException(status_code = 500, detail = "Could not read Agent Skills.") from exc
    try:
        from routes.inference import _invalidate_agent_skills_cache
        _invalidate_agent_skills_cache()
    except ImportError:
        pass
    return records


@router.get("/local")
def get_local_skills(current_subject: str = Depends(get_current_subject)):
    """Return the Codex/Claude local manager's simpler skill shape."""
    return {"skills": _list_skills()}


@router.put("/{name}/enabled", response_model = SkillRecord)
def update_skill_enabled(
    name: str,
    payload: SkillEnabledRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    try:
        updated = set_skill_enabled(name, payload.enabled)
        try:
            from routes.inference import _invalidate_agent_skills_cache
            _invalidate_agent_skills_cache()
        except ImportError:
            pass
        return updated
    except SkillNotFoundError as exc:
        raise HTTPException(status_code = 404, detail = str(exc)) from exc
    except SkillError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc


@router.get("/{name}")
def read_skill(name: str, current_subject: str = Depends(get_current_subject)):
    normalized = _valid_name(name)
    for _, root in _roots().items():
        candidate = root / normalized
        if not candidate.exists() or candidate.is_symlink():
            continue
        try:
            marker = _skill_marker(candidate)
            instructions = marker.read_text(encoding = "utf-8")
        except (OSError, UnicodeError) as error:
            raise HTTPException(status_code = 400, detail = "Could not read local skill.") from error
        if len(instructions) > 100_000:
            raise HTTPException(status_code = 413, detail = "Local skill instructions are too large to invoke.")
        return {
            "name": normalized,
            "description": _description(candidate),
            "instructions": instructions,
        }
    raise HTTPException(status_code = 404, detail = "Local skill not found.")


@router.post("/create")
def create_skill(payload: CreateSkillRequest, current_subject: str = Depends(get_current_subject)):
    name = _valid_name(payload.name)
    description = payload.description.strip()
    instructions = payload.instructions.strip()
    if not description or not instructions:
        raise HTTPException(status_code = 400, detail = "Skill description and instructions are required.")
    body = f"---\nname: {name}\ndescription: {description}\n---\n\n{instructions}\n"
    targets = _targets(payload.target)
    destinations = [root / name for _, root in targets]
    if any(destination.is_symlink() or destination.exists() for destination in destinations):
        raise HTTPException(status_code = 409, detail = f"Skill already exists: {name}")
    for _, root in targets:
        destination = root / name
        root.mkdir(parents = True, exist_ok = True)
        destination.mkdir()
        (destination / "SKILL.md").write_text(body, encoding = "utf-8")
    return {"skills": _list_skills()}


@router.post("/install")
def install_skill(payload: InstallSkillRequest, current_subject: str = Depends(get_current_subject)):
    return {"skills": _install(payload.source, payload.target)}
