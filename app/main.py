from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agents.azure_blob_agent_service import AzureBlobAgentService
from app.agents.blob_providers import AzureBlobRequestPlanner
from app.agents.graph import AgentGraphBuilder
from app.agents.providers import LLMRequestPlanner
from app.agents.sap_graph import SAPGraphBuilder
from app.agents.sap_service import SAPAgentService
from app.agents.blob_tools import build_blob_tools
from app.agents.sap_tools import build_sap_tools
from app.agents.service import AgentService
from app.agents.tools import build_s3_tools
from app.agents.unified_service import UnifiedLLMService
from app.api.router import api_router
from app.config import get_settings
from app.core.exceptions import AppError
from app.persistence.approval_repository import ApprovalRepository
from app.services.aws import AWSClientFactory
from app.services.azure import AzureBlobClientFactory
from app.services.blob import AzureBlobService
from app.services.s3 import S3Service


async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        approval_repository = ApprovalRepository(settings.sqlite_path_obj)
        aws_client_factory = AWSClientFactory(settings)
        s3_service = S3Service(aws_client_factory, settings)
        planner = LLMRequestPlanner(settings)
        tools = build_s3_tools(s3_service)
        graph = AgentGraphBuilder(planner=planner, tools=tools, settings=settings).compile()
        agent_service = AgentService(
            graph=graph,
            settings=settings,
            approval_repository=approval_repository,
            provider_name=f"aws_agent:{planner.provider_name}",
        )

        app.state.settings = settings
        app.state.approval_repository = approval_repository
        app.state.s3_service = s3_service
        app.state.agent_service = agent_service
        app.state.blob_service = None
        app.state.azure_blob_agent_service = None
        app.state.sap_agent_service = None

        if settings.has_azure_blob_credentials:
            azure_blob_factory = AzureBlobClientFactory(settings)
            azure_blob_service = AzureBlobService(azure_blob_factory, settings)
            app.state.blob_service = azure_blob_service
            blob_planner = AzureBlobRequestPlanner(settings)
            app.state.azure_blob_agent_service = AzureBlobAgentService(
                planner=blob_planner,
                blob_service=azure_blob_service,
                settings=settings,
                approval_repository=approval_repository,
                provider_name=f"azure_blob_agent:{settings.llm_provider.lower().strip()}",
            )

            if settings.has_sap_api_credentials:
                sap_tools = {**build_sap_tools(settings), **build_blob_tools(azure_blob_service)}
                sap_graph = SAPGraphBuilder(
                    tools=sap_tools, settings=settings
                ).compile()
                app.state.sap_agent_service = SAPAgentService(
                    graph=sap_graph,
                    settings=settings,
                    approval_repository=approval_repository,
                )

        app.state.unified_llm_service = UnifiedLLMService(
            aws_agent_service=agent_service,
            approval_repository=approval_repository,
            azure_blob_agent_service=app.state.azure_blob_agent_service,
            sap_agent_service=app.state.sap_agent_service,
        )

        yield

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.include_router(api_router)
    app.add_exception_handler(AppError, app_error_handler)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
