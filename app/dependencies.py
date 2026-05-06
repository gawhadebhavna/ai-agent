from __future__ import annotations

from fastapi import Request

from app.agents.sap_service import SAPAgentService
from app.agents.service import AgentService
from app.core.exceptions import ProviderConfigurationError
from app.services.blob import AzureBlobService
from app.services.s3 import S3Service


def get_s3_service(request: Request) -> S3Service:
    return request.app.state.s3_service


def get_blob_service(request: Request) -> AzureBlobService:
    service = request.app.state.blob_service
    if service is None:
        raise ProviderConfigurationError(
            "Azure Blob Storage is not available. Configure AZURE_STORAGE_CONNECTION_STRING."
        )
    return service


def get_agent_service(request: Request) -> AgentService:
    return request.app.state.agent_service


def get_sap_agent_service(request: Request) -> SAPAgentService:
    service = request.app.state.sap_agent_service
    if service is None:
        raise ProviderConfigurationError(
            "SAP agent is not available. Configure AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_ENDPOINT, and AZURE_STORAGE_CONNECTION_STRING."
        )
    return service
