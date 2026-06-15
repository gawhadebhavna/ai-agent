from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.agents.migration_orchestrator import MigrationOrchestratorService
from app.dependencies import get_migration_orchestrator, get_session_repository
from app.persistence.session_repository import SessionRepository
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    SessionCreateResponse,
    SessionStatusResponse,
    ApprovalItem,
    MigrationPhase,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@router.post("/sessions", response_model=SessionCreateResponse, status_code=201)
def create_session(
    orchestrator: MigrationOrchestratorService = Depends(get_migration_orchestrator),
) -> SessionCreateResponse:
    """Create a new migration session. Returns session_id."""
    return orchestrator.create_session()


# ---------------------------------------------------------------------------
# Main chat endpoint
# ---------------------------------------------------------------------------

@router.post("/{session_id}/message", response_model=ChatResponse)
def send_message(
    session_id: str,
    body: ChatRequest,
    orchestrator: MigrationOrchestratorService = Depends(get_migration_orchestrator),
) -> ChatResponse:
    """
    Send a message to the migration assistant.

    Handles:
    - Normal conversational messages (routed through review_incoming → phase node → review_outgoing)
    - Approval resolutions (body.approval_id + body.approve)
    """
    return orchestrator.handle_message(
        session_id=session_id,
        message=body.message,
        approval_id=body.approval_id,
        approve=body.approve,
    )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@router.get("/{session_id}/history")
def get_history(
    session_id: str,
    limit: int = 30,
    session_repo: SessionRepository = Depends(get_session_repository),
) -> dict:
    """Return paginated redacted conversation history for a session."""
    session = session_repo.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    history = session_repo.get_history(session_id, limit=limit)
    return {
        "session_id": session_id,
        "history": history,
        "count": len(history),
    }


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get("/{session_id}/status", response_model=SessionStatusResponse)
def get_status(
    session_id: str,
    session_repo: SessionRepository = Depends(get_session_repository),
) -> SessionStatusResponse:
    """Return current phase, completed phases, and pending approvals."""
    session = session_repo.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    pending_rows = session_repo.list_pending_approvals(session_id)
    pending = [
        ApprovalItem(
            approval_id=r["approval_id"],
            summary=r["summary"],
            tool_name=r["tool_name"],
            tool_args=r["tool_args"],
            expires_at=r["expires_at"],
            cloud_resource_type=r.get("cloud_resource_type"),
        )
        for r in pending_rows
    ]

    phase = session.get("migration_phase", MigrationPhase.INIT)
    return SessionStatusResponse(
        session_id=session_id,
        migration_phase=MigrationPhase(phase) if phase in MigrationPhase.__members__ else MigrationPhase.INIT,
        completed_phases=session.get("completed_phases", []),
        pending_approvals=pending,
        phase_data=session.get("phase_data", {}),
    )


# ---------------------------------------------------------------------------
# Server-Sent Events stream
# ---------------------------------------------------------------------------

@router.get("/{session_id}/events")
async def event_stream(
    session_id: str,
    session_repo: SessionRepository = Depends(get_session_repository),
) -> StreamingResponse:
    """
    SSE stream for a session.

    Emits phase-change events and approval events.
    Clients connect and receive real-time updates without polling.
    """

    async def generate() -> AsyncGenerator[str, None]:
        last_phase = None
        last_approval_count = -1

        for _ in range(120):  # Max 120 iterations × 2s = 4 min window
            await asyncio.sleep(2)
            session = session_repo.get_session(session_id)
            if session is None:
                yield "event: error\ndata: {\"message\": \"Session not found\"}\n\n"
                break

            phase = session.get("migration_phase", MigrationPhase.INIT)
            pending = session_repo.list_pending_approvals(session_id)

            if phase != last_phase:
                last_phase = phase
                payload = json.dumps({"migration_phase": phase, "completed_phases": session.get("completed_phases", [])})
                yield f"event: phase_change\ndata: {payload}\n\n"

            if len(pending) != last_approval_count:
                last_approval_count = len(pending)
                approvals = [
                    {
                        "approval_id": r["approval_id"],
                        "summary": r["summary"],
                        "tool_name": r["tool_name"],
                        "expires_at": r["expires_at"],
                        "cloud_resource_type": r.get("cloud_resource_type"),
                    }
                    for r in pending
                ]
                payload = json.dumps({"pending_approvals": approvals})
                yield f"event: approvals_update\ndata: {payload}\n\n"

        yield "event: stream_end\ndata: {}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
