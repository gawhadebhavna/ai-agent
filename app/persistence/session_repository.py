from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

# Patterns used to detect and redact credential-like values before persistence
_CRED_PATTERNS = [
    re.compile(r"(?i)(password|passwd|pwd|secret|api[_-]?key|token|client_secret)"),
]


def _redact_message(text: str) -> str:
    """Return text with credentials replaced by [REDACTED]."""
    for pattern in _CRED_PATTERNS:
        if pattern.search(text):
            # Redact the entire value after known keywords
            text = re.sub(
                r"(?i)(\b(?:password|passwd|pwd|secret|api[_-]?key|token|client_secret)\b\s*[=:\"' ]+)([^\s\"',;}{]+)",
                r"\1[REDACTED]",
                text,
            )
    return text


class SessionRepository:
    """Manages migration sessions and conversation history in SQLite."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id      TEXT PRIMARY KEY,
                    migration_phase TEXT NOT NULL DEFAULT 'INIT',
                    cloud_provider  TEXT,
                    source_system   TEXT,
                    phase_data      TEXT NOT NULL DEFAULT '{}',
                    completed_phases TEXT NOT NULL DEFAULT '[]',
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_history (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
                    role            TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    redacted_content TEXT NOT NULL,
                    timestamp       TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pending_approvals (
                    approval_id     TEXT PRIMARY KEY,
                    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
                    status          TEXT NOT NULL DEFAULT 'pending',
                    tool_name       TEXT NOT NULL,
                    tool_args       TEXT NOT NULL DEFAULT '{}',
                    summary         TEXT NOT NULL,
                    cloud_resource_type TEXT,
                    expires_at      TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    resolved_at     TEXT
                );
                """
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, session_id: str) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, migration_phase, phase_data, completed_phases, created_at, updated_at)
                VALUES (?, 'INIT', '{}', '[]', ?, ?)
                """,
                (session_id, now, now),
            )
            conn.commit()
        return self.get_session(session_id)  # type: ignore[return-value]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["phase_data"] = json.loads(d["phase_data"])
        d["completed_phases"] = json.loads(d["completed_phases"])
        return d

    def update_phase(
        self,
        session_id: str,
        phase: str,
        *,
        phase_data_patch: dict[str, Any] | None = None,
        cloud_provider: str | None = None,
        source_system: str | None = None,
    ) -> None:
        session = self.get_session(session_id)
        if session is None:
            return
        now = datetime.now(UTC).isoformat()
        merged_data: dict[str, Any] = session["phase_data"]
        if phase_data_patch:
            merged_data.update(phase_data_patch)

        # Track completed phases
        completed: list[str] = session["completed_phases"]
        prev = session["migration_phase"]
        if prev != phase and prev not in completed:
            completed.append(prev)

        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.execute(
                """
                UPDATE sessions
                SET migration_phase = ?,
                    phase_data = ?,
                    completed_phases = ?,
                    cloud_provider = COALESCE(?, cloud_provider),
                    source_system = COALESCE(?, source_system),
                    updated_at = ?
                WHERE session_id = ?
                """,
                (
                    phase,
                    json.dumps(merged_data),
                    json.dumps(completed),
                    cloud_provider,
                    source_system,
                    now,
                    session_id,
                ),
            )
            conn.commit()

    def patch_phase_data(self, session_id: str, patch: dict[str, Any]) -> None:
        """Merge patch into phase_data without changing the phase."""
        session = self.get_session(session_id)
        if session is None:
            return
        merged: dict[str, Any] = session["phase_data"]
        merged.update(patch)
        now = datetime.now(UTC).isoformat()
        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.execute(
                "UPDATE sessions SET phase_data = ?, updated_at = ? WHERE session_id = ?",
                (json.dumps(merged), now, session_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def append_message(self, session_id: str, role: str, content: str) -> None:
        redacted = _redact_message(content)
        now = datetime.now(UTC).isoformat()
        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.execute(
                """
                INSERT INTO conversation_history (session_id, role, content, redacted_content, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, redacted, now),
            )
            conn.commit()

    def get_history(self, session_id: str, limit: int = 30) -> list[dict[str, Any]]:
        """Return redacted conversation history (newest last)."""
        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT role, redacted_content AS content, timestamp
                FROM conversation_history
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return list(reversed([dict(r) for r in rows]))

    def get_raw_history(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Return raw (un-redacted) history for LLM context injection (credentials stay in mem)."""
        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT role, content, timestamp
                FROM conversation_history
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return list(reversed([dict(r) for r in rows]))

    # ------------------------------------------------------------------
    # Pending approvals
    # ------------------------------------------------------------------

    def create_approval(
        self,
        *,
        approval_id: str,
        session_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        summary: str,
        expires_at: datetime,
        cloud_resource_type: str | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pending_approvals
                    (approval_id, session_id, status, tool_name, tool_args, summary, cloud_resource_type, expires_at, created_at)
                VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    session_id,
                    tool_name,
                    json.dumps(tool_args),
                    summary,
                    cloud_resource_type,
                    expires_at.isoformat(),
                    now,
                ),
            )
            conn.commit()

    def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM pending_approvals WHERE approval_id = ?", (approval_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["tool_args"] = json.loads(d["tool_args"])
        return d

    def resolve_approval(self, approval_id: str, status: str) -> None:
        now = datetime.now(UTC).isoformat()
        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.execute(
                "UPDATE pending_approvals SET status = ?, resolved_at = ? WHERE approval_id = ?",
                (status, now, approval_id),
            )
            conn.commit()

    def list_pending_approvals(self, session_id: str) -> list[dict[str, Any]]:
        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM pending_approvals
                WHERE session_id = ? AND status = 'pending'
                ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["tool_args"] = json.loads(d["tool_args"])
            result.append(d)
        return result
