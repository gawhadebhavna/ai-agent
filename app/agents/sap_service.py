from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.config import get_settings
from app.core.exceptions import ApprovalError
from app.persistence.approval_repository import ApprovalRepository
from app.schemas.sap_agent import (
    SAPAgentRequest,
    SAPAgentResponse,
    SAPAgentStatus,
    SAPApprovalProposal,
)


class SAPAgentService:
    """Orchestrates SAP agent requests with approval gating.

    Mirrors AgentService for the S3 flow.
    """

    def __init__(
        self,
        *,
        graph,
        settings: Settings,
        approval_repository: ApprovalRepository,
    ) -> None:
        self._graph = graph
        self._settings = settings
        self._approval_repository = approval_repository

    def handle(self, request: SAPAgentRequest) -> SAPAgentResponse:
        if request.approval_id and request.approve is not None:
            return self._resume(request)
        return self._invoke_new(request)

    def _invoke_new(self, request: SAPAgentRequest) -> SAPAgentResponse:
        approval_id = str(uuid4())
        config = {"configurable": {"thread_id": approval_id}}
        result = self._graph.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )

        if "__interrupt__" in result:
            interrupt_value = result["__interrupt__"][0].value
            expires_at = datetime.fromisoformat(interrupt_value["expires_at"])
            self._approval_repository.create_pending(
                approval_id=approval_id,
                tool_name=interrupt_value["tool_name"],
                tool_args=interrupt_value["tool_args"],
                summary=interrupt_value["summary"],
                user_message=request.message,
                provider="sap_agent",
                expires_at=expires_at,
            )
            return SAPAgentResponse(
                status=SAPAgentStatus.approval_required,
                message="Approval is required before writing to Azure Blob Storage.",
                approval=SAPApprovalProposal(
                    approval_id=approval_id,
                    tool_name=interrupt_value["tool_name"],
                    tool_args=interrupt_value["tool_args"],
                    summary=interrupt_value["summary"],
                    expires_at=expires_at,
                ),
            )

        return SAPAgentResponse.model_validate(result["final_response"])

    def _resume(self, request: SAPAgentRequest) -> SAPAgentResponse:
        record = self._approval_repository.get(request.approval_id or "")
        if record is None:
            raise ApprovalError("Approval request was not found.")
        if record["status"] != "pending":
            raise ApprovalError(f"Approval request is already {record['status']}.")
        if record["expires_at"] < datetime.now(UTC):
            self._approval_repository.update_status(record["approval_id"], "expired")
            raise ApprovalError("Approval request has expired.")

        if request.approve is False:
            self._approval_repository.update_status(record["approval_id"], "rejected")
            return SAPAgentResponse(
                status=SAPAgentStatus.rejected,
                message="The write to Azure Blob was rejected.",
            )

        self._approval_repository.update_status(record["approval_id"], "approved")
        config = {"configurable": {"thread_id": record["approval_id"]}}
        result = self._graph.invoke(Command(resume=True), config=config)
        self._approval_repository.update_status(record["approval_id"], "executed")
        return SAPAgentResponse.model_validate(result["final_response"])
