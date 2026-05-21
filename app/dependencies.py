from __future__ import annotations

from fastapi import Request

from app.agents.unified_service import UnifiedLLMService
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


def get_unified_llm_service(request: Request) -> UnifiedLLMService:
    return request.app.state.unified_llm_service
