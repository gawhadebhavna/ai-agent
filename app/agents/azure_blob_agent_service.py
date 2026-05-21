from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.config import Settings
from app.core.exceptions import ApprovalError
from app.persistence.approval_repository import ApprovalRepository
from app.schemas.azure_agent import (
    AzureAgentRequest,
    AzureAgentResponse,
    AzureAgentStatus,
    AzureApprovalProposal,
    AzureBlobActionPlan,
    AzureToolName,
)
from app.services.blob import AzureBlobService


class AzureBlobAgentService:
    """Executes LLM-planned Azure Blob actions with approval gating for writes."""

    def __init__(
        self,
        *,
        planner,
        blob_service: AzureBlobService,
        settings: Settings,
        approval_repository: ApprovalRepository,
        provider_name: str,
    ) -> None:
        self._planner = planner
        self._blob_service = blob_service
        self._settings = settings
        self._approval_repository = approval_repository
        self._provider_name = provider_name

    def handle(self, request: AzureAgentRequest) -> AzureAgentResponse:
        if request.approval_id and request.approve is not None:
            return self._resume(request)
        return self._invoke_new(request)

    def _invoke_new(self, request: AzureAgentRequest) -> AzureAgentResponse:
        try:
            plan: AzureBlobActionPlan = self._planner.plan(message=request.message)
        except Exception as exc:
            return AzureAgentResponse(
                status=AzureAgentStatus.error,
                message=(
                    "I could not plan that Azure Blob request due to a planner/schema validation issue. "
                    "Please retry with an explicit request, for example: "
                    "Write JSON {\"status\":\"ok\"} to blob_path reports/output.json. "
                    f"Details: {exc}"
                ),
            )

        if plan.tool_name == AzureToolName.unsupported:
            return AzureAgentResponse(
                status=AzureAgentStatus.error,
                message=plan.final_response or plan.summary,
            )

        if plan.requires_approval:
            approval_id = str(uuid4())
            expires_at = datetime.now(UTC) + timedelta(minutes=self._settings.approval_ttl_minutes)
            self._approval_repository.create_pending(
                approval_id=approval_id,
                tool_name=plan.tool_name.value,
                tool_args=plan.tool_args(),
                summary=plan.summary,
                user_message=request.message,
                provider=self._provider_name,
                expires_at=expires_at,
            )
            return AzureAgentResponse(
                status=AzureAgentStatus.approval_required,
                message="Approval is required before writing to Azure Blob Storage.",
                approval=AzureApprovalProposal(
                    approval_id=approval_id,
                    tool_name=plan.tool_name.value,
                    tool_args=plan.tool_args(),
                    summary=plan.summary,
                    expires_at=expires_at,
                ),
            )

        result = self._execute_tool(tool_name=plan.tool_name.value, tool_args=plan.tool_args())
        return AzureAgentResponse(
            status=AzureAgentStatus.completed,
            message=plan.summary,
            result=result,
        )

    def _resume(self, request: AzureAgentRequest) -> AzureAgentResponse:
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
            return AzureAgentResponse(
                status=AzureAgentStatus.rejected,
                message="The requested Azure Blob write action was rejected.",
            )

        self._approval_repository.update_status(record["approval_id"], "approved")
        result = self._execute_tool(tool_name=record["tool_name"], tool_args=record["tool_args"])
        self._approval_repository.update_status(record["approval_id"], "executed")
        return AzureAgentResponse(
            status=AzureAgentStatus.completed,
            message="Azure Blob write action completed successfully.",
            result=result,
        )

    def _execute_tool(self, *, tool_name: str, tool_args: dict) -> dict:
        if tool_name == AzureToolName.list_blobs.value:
            return self._blob_service.list_blobs(
                prefix=tool_args.get("prefix"),
                max_results=int(tool_args.get("max_results", 100)),
            )
        if tool_name == AzureToolName.read_text.value:
            return self._blob_service.read_text(blob_path=tool_args["blob_path"])
        if tool_name == AzureToolName.read_json.value:
            return self._blob_service.read_json(blob_path=tool_args["blob_path"])
        if tool_name == AzureToolName.write_text.value:
            return self._blob_service.write_text(
                content=tool_args["content"],
                blob_path=tool_args["blob_path"],
                content_type=tool_args.get("content_type", "text/plain"),
            )
        if tool_name == AzureToolName.write_json.value:
            return self._blob_service.write_json(
                data=tool_args["data"],
                blob_path=tool_args["blob_path"],
            )
        if tool_name == AzureToolName.write_parquet.value:
            return self._blob_service.write_parquet(
                data=tool_args["data"],
                blob_path=tool_args["blob_path"],
            )
        raise ApprovalError(f"Unsupported Azure Blob tool '{tool_name}'.")
