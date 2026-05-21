from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ApprovalRepository:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with closing(sqlite3.connect(self._db_path)) as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    tool_args TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    approval_id TEXT,
                    provider TEXT NOT NULL,
                    request_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def create_pending(
        self,
        *,
        approval_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        summary: str,
        user_message: str,
        provider: str,
        expires_at: datetime,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with closing(sqlite3.connect(self._db_path)) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO approvals (
                    approval_id, status, tool_name, tool_args, summary, user_message,
                    provider, expires_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    "pending",
                    tool_name,
                    json.dumps(tool_args),
                    summary,
                    user_message,
                    provider,
                    expires_at.isoformat(),
                    now,
                    now,
                ),
            )
            connection.commit()

    def get(self, approval_id: str) -> dict[str, Any] | None:
        with closing(sqlite3.connect(self._db_path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["tool_args"] = json.loads(data["tool_args"])
        data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return data

    def update_status(self, approval_id: str, status: str) -> None:
        with closing(sqlite3.connect(self._db_path)) as connection:
            connection.execute(
                "UPDATE approvals SET status = ?, updated_at = ? WHERE approval_id = ?",
                (status, datetime.now(UTC).isoformat(), approval_id),
            )
            connection.commit()

    def log_run(
        self,
        *,
        approval_id: str | None,
        provider: str,
        request_text: str,
        status: str,
        response_json: dict[str, Any],
    ) -> None:
        with closing(sqlite3.connect(self._db_path)) as connection:
            connection.execute(
                """
                INSERT INTO agent_runs (approval_id, provider, request_text, status, response_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    provider,
                    request_text,
                    status,
                    json.dumps(response_json),
                    datetime.now(UTC).isoformat(),
                ),
            )
            connection.commit()
