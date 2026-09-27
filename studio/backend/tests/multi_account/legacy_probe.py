# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fresh-process upgrade simulation; main is deliberately imported after seeding."""

import importlib
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from .seed import (
    MESSAGE_ID,
    PASSWORD,
    SENTINEL,
    THREAD_ID,
    old_auth_row,
    seed_legacy_install,
)

_DATABASE_FILES = {"auth/auth.db", "studio.db", "rag/rag.db"}


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _legacy_schema(conn: sqlite3.Connection) -> dict[str, tuple[str, ...]]:
    schema: dict[str, tuple[str, ...]] = {}
    table_names = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    for (table_name,) in table_names:
        columns = tuple(
            row[1]
            for row in conn.execute(
                f"PRAGMA table_info({_quote_identifier(table_name)})"
            ).fetchall()
        )
        if columns:
            schema[table_name] = columns
    return schema


def _legacy_table_rows(
    path: Path,
    schema: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, tuple[tuple[object, ...], ...]]:
    """Capture legacy table data without depending on physical SQLite bytes.

    Opening the current backend legitimately checkpoints WAL pages and adds
    migration tables/columns.  The upgrade invariant is that every row from
    the frozen legacy schema remains present, not that the container file is
    byte-identical.
    """

    result: dict[str, tuple[tuple[object, ...], ...]] = {}
    with closing(sqlite3.connect(path)) as conn:
        table_schema = schema or _legacy_schema(conn)
        for table_name, columns in table_schema.items():
            quoted_columns = ", ".join(_quote_identifier(column) for column in columns)
            order_columns = ", ".join(_quote_identifier(column) for column in columns)
            rows = conn.execute(
                f"SELECT {quoted_columns} FROM {_quote_identifier(table_name)} "
                f"ORDER BY {order_columns}"
            ).fetchall()
            result[table_name] = tuple(tuple(row) for row in rows)
    return result


def _legacy_file_hashes(home: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(home)): path.read_bytes()
        for path in home.rglob("*")
        if path.is_file() and str(path.relative_to(home)) not in _DATABASE_FILES
    }


def main() -> None:
    home = Path(os.environ["UNSLOTH_STUDIO_HOME"])
    original = seed_legacy_install(home)
    with closing(sqlite3.connect(home / "studio.db")) as conn:
        legacy_schema_before = _legacy_schema(conn)
    legacy_rows_before = _legacy_table_rows(home / "studio.db", legacy_schema_before)
    non_database_files_before = _legacy_file_hashes(home)
    auth_row = old_auth_row(home / "auth" / "auth.db")
    application = importlib.import_module("main")
    for name, payload in non_database_files_before.items():
        assert (home / name).read_bytes() == payload, f"Import changed {name}"
    assert not (home / "accounts").exists()

    from fastapi.testclient import TestClient
    from auth import policy, storage
    from storage import credential_secrets, rag_db, studio_db
    from core.rag import store
    from utils.account_context import OWNER
    from utils.paths import workspace_root
    from core.inference.tools import sandbox_root

    assert storage.get_account("unsloth") == OWNER
    assert old_auth_row(home / "auth" / "auth.db") == auth_row
    assert workspace_root() == home
    assert Path(sandbox_root()) == home / "sandbox"
    assert studio_db.get_chat_thread(THREAD_ID)["title"] == SENTINEL
    assert studio_db.get_chat_message(THREAD_ID, MESSAGE_ID)["content"][0]["text"] == SENTINEL
    assert studio_db.list_chat_settings()["theme"] == "dark"
    assert studio_db.get_app_setting("legacy-owner-setting") == {"preserve": True}
    assert credential_secrets.get_hf_token() == "hf_legacy_private"
    with closing(rag_db.get_metadata_connection()) as conn:
        assert store.get_kb(conn, "legacy-kb")["name"] == SENTINEL
    assert policy.login_mode() == "single"

    client = TestClient(application.app)
    try:
        response = client.get("/api/auth/status")
        assert response.status_code == 200, response.text
        assert response.json() == {
            "initialized": True,
            "default_username": "unsloth",
            "requires_password_change": False,
            "bootstrap_deadline_seconds": None,
            "login_mode": "single",
            "full_access": True,
        }
        login = client.post("/api/auth/login", json = {"username": "unsloth", "password": PASSWORD})
        assert login.status_code == 200, login.text
        assert login.json()["access_token"]
    finally:
        client.close()
    legacy_rows_after = _legacy_table_rows(home / "studio.db", legacy_schema_before)
    assert legacy_rows_after == legacy_rows_before, "Owner read changed legacy studio rows"
    for name, payload in non_database_files_before.items():
        assert (home / name).read_bytes() == payload, f"Owner read changed {name}"
    assert old_auth_row(home / "auth" / "auth.db") == auth_row
    assert not (home / "accounts").exists()
    print(json.dumps({"preserved_files": len(original), "owner_login": True}))


if __name__ == "__main__":
    main()
