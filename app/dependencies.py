from __future__ import annotations

from fastapi import Request

from app.agents.migration_orchestrator import MigrationOrchestratorService
from app.agents.source_registry import SourceSystemRegistry
from app.persistence.session_repository import SessionRepository


def get_session_repository(request: Request) -> SessionRepository:
    return request.app.state.session_repository


def get_migration_orchestrator(request: Request) -> MigrationOrchestratorService:
    return request.app.state.migration_orchestrator


def get_source_registry(request: Request) -> SourceSystemRegistry:
    return request.app.state.source_registry
