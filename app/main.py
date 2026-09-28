from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agents.migration_orchestrator import MigrationOrchestratorService
from app.agents.source_registry import SourceSystemRegistry
from app.api.router import api_router
from app.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import get_logger, setup_logging
from app.persistence.approval_repository import ApprovalRepository
from app.persistence.session_repository import SessionRepository

setup_logging()
logger = get_logger("migration")


async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "=== Migration AI Tool v2 starting ===",
            extra={"phase": "STARTUP", "session_id": "-"},
        )

        # Persistence
        session_repo = SessionRepository(settings.sqlite_path_obj)
        # Keep ApprovalRepository for backward-compat with existing SQLite schema
        approval_repo = ApprovalRepository(settings.sqlite_path_obj)

        app.state.session_repository = session_repo
        app.state.approval_repository = approval_repo

        # Source registry
        source_registry = SourceSystemRegistry()
        app.state.source_registry = source_registry
        logger.info(
            "Source registry: %d plugins registered",
            len(source_registry.list_sources()),
            extra={"phase": "STARTUP", "session_id": "-"},
        )

        # Migration orchestrator
        try:
            orchestrator = MigrationOrchestratorService(
                settings=settings,
                session_repo=session_repo,
                source_registry=source_registry,
            )
            app.state.migration_orchestrator = orchestrator
            logger.info(
                "Migration orchestrator ready | llm_provider=%s",
                settings.llm_provider,
                extra={"phase": "STARTUP", "session_id": "-"},
            )
        except Exception as exc:
            logger.error(
                "Failed to initialize migration orchestrator: %s",
                exc,
                extra={"phase": "STARTUP", "session_id": "-"},
            )
            app.state.migration_orchestrator = None

        logger.info(
            "=== Migration AI Tool v2 ready ===",
            extra={"phase": "STARTUP", "session_id": "-"},
        )
        yield

        logger.info("Migration AI Tool shutting down.", extra={"phase": "STARTUP", "session_id": "-"})

    application = FastAPI(
        title="Migration AI Tool",
        version="2.0.0",
        lifespan=lifespan,
    )

    application.add_exception_handler(AppError, app_error_handler)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(api_router)

    @application.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "Migration AI Tool v2"}

    return application


app = create_app()
