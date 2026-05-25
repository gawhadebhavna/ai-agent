from __future__ import annotations

from typing import Any

from app.agents.azure_blob_agent_service import AzureBlobAgentService
from app.agents.azure_mcp_service import AzureMCPAgentService
from app.agents.sap_service import SAPAgentService
from app.agents.service import AgentService
from app.core.exceptions import ProviderConfigurationError
from app.persistence.approval_repository import ApprovalRepository
from app.schemas.agent import AgentS3Request
from app.schemas.azure_agent import AzureAgentRequest
from app.schemas.azure_mcp_agent import AzureMCPAgentRequest
from app.schemas.llm import LLMToolDomain, UnifiedLLMRequest, UnifiedLLMResponse
from app.schemas.sap_agent import SAPAgentRequest


class UnifiedLLMService:
    """Centralized entry point for LLM-driven AWS, Azure, and SAP tool calls."""

    def __init__(
        self,
        *,
        aws_agent_service: AgentService,
        approval_repository: ApprovalRepository,
        azure_blob_agent_service: AzureBlobAgentService | None = None,
        azure_mcp_agent_service: AzureMCPAgentService | None = None,
        sap_agent_service: SAPAgentService | None = None,
    ) -> None:
        self._aws_agent_service = aws_agent_service
        self._azure_blob_agent_service = azure_blob_agent_service
        self._azure_mcp_agent_service = azure_mcp_agent_service
        self._sap_agent_service = sap_agent_service
        self._approval_repository = approval_repository

    def handle(self, request: UnifiedLLMRequest) -> UnifiedLLMResponse:
        domain = self._resolve_domain(request)

        if domain == LLMToolDomain.aws:
            response = self._aws_agent_service.handle(
                AgentS3Request(
                    message=request.message,
                    approval_id=request.approval_id,
                    approve=request.approve,
                )
            )
            return self._wrap_response(domain=domain, payload=self._as_payload(response))

        if domain == LLMToolDomain.azure:
            if self._azure_blob_agent_service is None:
                raise ProviderConfigurationError(
                    "Azure tool calls are not available. Configure AZURE_STORAGE_CONNECTION_STRING."
                )
            response = self._azure_blob_agent_service.handle(
                AzureAgentRequest(
                    message=request.message,
                    approval_id=request.approval_id,
                    approve=request.approve,
                )
            )
            return self._wrap_response(domain=domain, payload=self._as_payload(response))

        if domain == LLMToolDomain.azure_mcp:
            if self._azure_mcp_agent_service is None:
                raise ProviderConfigurationError(
                    "Azure MCP is not available. Ensure Node.js is installed and Azure CLI is authenticated (run 'az login')."
                )
            response = self._azure_mcp_agent_service.handle(
                AzureMCPAgentRequest(
                    message=request.message,
                    approval_id=request.approval_id,
                    approve=request.approve,
                )
            )
            return self._wrap_response(domain=domain, payload=self._as_payload(response))

        if self._sap_agent_service is None:
            raise ProviderConfigurationError(
                "SAP tool calls are not available. Configure SAP API and Azure Blob credentials."
            )
        response = self._sap_agent_service.handle(
            SAPAgentRequest(
                message=request.message,
                approval_id=request.approval_id,
                approve=request.approve,
            )
        )
        return self._wrap_response(domain=LLMToolDomain.sap, payload=self._as_payload(response))

    def _resolve_domain(self, request: UnifiedLLMRequest) -> LLMToolDomain:
        if request.domain != LLMToolDomain.auto:
            return request.domain

        if request.approval_id and request.approve is not None:
            record = self._approval_repository.get(request.approval_id)
            if record:
                provider = str(record.get("provider", "")).lower()
                if provider.startswith("sap_agent"):
                    return LLMToolDomain.sap
                if provider.startswith("azure_mcp_agent"):
                    return LLMToolDomain.azure_mcp
                if provider.startswith("azure_blob_agent"):
                    return LLMToolDomain.azure
                return LLMToolDomain.aws

        message = request.message.lower()
        if any(token in message for token in ["sap", "kna1", "mara", "vbfa", "vbkd", "vbpa"]):
            return LLMToolDomain.sap
        
        # Azure MCP keywords (Key Vault, Storage, Databricks, SQL, Cosmos, AKS, etc.)
        azure_mcp_keywords = [
            # Key Vault
            "key vault", "keyvault", "secret",
            # Storage (when using MCP operations)
            "blob container", "blob storage", "upload blob", "download blob",
            # Databricks
            "databricks", "dbx", "cluster", "notebook", "spark",
            # Other Azure Services
            "cosmos", "sql database", "aks", "kubernetes", "resource group",
            "subscription"
        ]
        if any(token in message for token in azure_mcp_keywords):
            return LLMToolDomain.azure_mcp
        
        # Azure Blob (legacy direct blob service)
        if any(token in message for token in ["azure", "blob", "container", "parquet"]):
            return LLMToolDomain.azure
        
        return LLMToolDomain.aws

    @staticmethod
    def _as_payload(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if isinstance(response, dict):
            return response
        raise ProviderConfigurationError("Unified service received an invalid agent response payload.")

    @staticmethod
    def _wrap_response(*, domain: LLMToolDomain, payload: dict[str, Any]) -> UnifiedLLMResponse:
        return UnifiedLLMResponse(
            domain=domain,
            status=str(payload.get("status", "error")),
            message=str(payload.get("message", "")),
            result=payload.get("result"),
            approval=payload.get("approval"),
        )
