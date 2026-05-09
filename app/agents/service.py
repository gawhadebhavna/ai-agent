from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from langgraph.types import Command

from app.config import get_settings
from app.core.exceptions import ApprovalError
from app.persistence.approval_repository import ApprovalRepository
from app.schemas.agent import AgentS3Request, AgentS3Response, AgentStatus, ApprovalProposal


class AgentService:
    def __init__(
        self,
        *,
        graph,
        settings: Settings,
        approval_repository: ApprovalRepository,
        provider_name: str,
    ) -> None:
        self._graph = graph
        self._settings = settings
        self._approval_repository = approval_repository
        self._provider_name = provider_name

    def handle(self, request: AgentS3Request) -> AgentS3Response:
        if request.approval_id and request.approve is not None:
            return self._resume(request)
        return self._invoke_new(request)

    def _invoke_new(self, request: AgentS3Request) -> AgentS3Response:
        approval_id = str(uuid4())
        config = {"configurable": {"thread_id": approval_id}}
        result = self._graph.invoke({"message": request.message}, config=config)
        if "__interrupt__" in result:
            interrupt_value = result["__interrupt__"][0].value
            expires_at = datetime.fromisoformat(interrupt_value["expires_at"])
            self._approval_repository.create_pending(
                approval_id=approval_id,
                tool_name=interrupt_value["tool_name"],
                tool_args=interrupt_value["tool_args"],
                summary=interrupt_value["summary"],
                user_message=request.message,
                provider=self._provider_name,
                expires_at=expires_at,
            )
            response = AgentS3Response(
                status=AgentStatus.approval_required,
                message="Approval is required before this write action can be executed.",
                approval=ApprovalProposal(
                    approval_id=approval_id,
                    tool_name=interrupt_value["tool_name"],
                    tool_args=interrupt_value["tool_args"],
                    summary=interrupt_value["summary"],
                    expires_at=expires_at,
                ),
            )
            self._log_run(approval_id=approval_id, request_text=request.message, response=response)
            return response

        response = AgentS3Response.model_validate(result["final_response"])
        self._log_run(approval_id=None, request_text=request.message, response=response)
        return response

    def _resume(self, request: AgentS3Request) -> AgentS3Response:
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
            response = AgentS3Response(
                status=AgentStatus.rejected,
                message="The requested write action was rejected.",
            )
            self._log_run(
                approval_id=record["approval_id"],
                request_text=request.message,
                response=response,
            )
            return response

        self._approval_repository.update_status(record["approval_id"], "approved")
        config = {"configurable": {"thread_id": record["approval_id"]}}
        result = self._graph.invoke(Command(resume=True), config=config)
        response = AgentS3Response.model_validate(result["final_response"])
        self._approval_repository.update_status(record["approval_id"], "executed")
        self._log_run(
            approval_id=record["approval_id"],
            request_text=request.message,
            response=response,
        )
        return response

    def _log_run(self, *, approval_id: str | None, request_text: str, response: AgentS3Response) -> None:
        self._approval_repository.log_run(
            approval_id=approval_id,
            provider=self._provider_name,
            request_text=request_text,
            status=response.status.value,
            response_json=response.model_dump(mode="json"),
        )
