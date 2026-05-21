from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, TypeAlias

from pydantic import BaseModel, Field, model_validator


class AzureAgentStatus(str, Enum):
    completed = "completed"
    approval_required = "approval_required"
    rejected = "rejected"
    error = "error"


class AzureToolName(str, Enum):
    list_blobs = "blob_list_blobs"
    read_text = "blob_read_text"
    read_json = "blob_read_json"
    write_text = "blob_write_text"
    write_json = "blob_write_json"
    write_parquet = "blob_write_parquet"
    unsupported = "unsupported"


AzureRecord: TypeAlias = dict[str, Any]
AzureJSONPayload: TypeAlias = AzureRecord | list[AzureRecord]


class AzureBlobActionPlan(BaseModel):
    tool_name: AzureToolName
    summary: str
    rationale: str
    blob_path: str | None = None
    prefix: str | None = None
    max_results: int = 100
    content: str | None = None
    data: AzureJSONPayload | None = None
    content_type: str = "text/plain"
    final_response: str | None = None
    requires_approval: bool = False

    @model_validator(mode="after")
    def validate_required_fields(self) -> "AzureBlobActionPlan":
        if self.tool_name == AzureToolName.list_blobs:
            if self.max_results < 1:
                raise ValueError("max_results must be at least 1 for blob_list_blobs")
            self.requires_approval = False
            return self

        if self.tool_name == AzureToolName.read_text:
            if not self.blob_path:
                raise ValueError("blob_path is required for blob_read_text")
            self.requires_approval = False
            return self

        if self.tool_name == AzureToolName.read_json:
            if not self.blob_path:
                raise ValueError("blob_path is required for blob_read_json")
            self.requires_approval = False
            return self

        if self.tool_name == AzureToolName.write_text:
            if not self.blob_path:
                raise ValueError("blob_path is required for blob_write_text")
            if self.content is None:
                raise ValueError("content is required for blob_write_text")
            self.requires_approval = True
            return self

        if self.tool_name == AzureToolName.write_json:
            if not self.blob_path:
                raise ValueError("blob_path is required for blob_write_json")
            if not self._is_json_payload(self.data):
                raise ValueError("data must be a JSON object or array of objects for blob_write_json")
            self.requires_approval = True
            return self

        if self.tool_name == AzureToolName.write_parquet:
            if not self.blob_path:
                raise ValueError("blob_path is required for blob_write_parquet")
            if not self._is_record_list(self.data):
                raise ValueError("data must be a list of records for blob_write_parquet")
            self.requires_approval = True
            return self

        return self

    @staticmethod
    def _is_record_list(value: Any) -> bool:
        return isinstance(value, list) and all(isinstance(item, dict) for item in value)

    @classmethod
    def _is_json_payload(cls, value: Any) -> bool:
        if isinstance(value, dict):
            return True
        return cls._is_record_list(value)

    def tool_args(self) -> dict[str, Any]:
        if self.tool_name == AzureToolName.list_blobs:
            return {
                "prefix": self.prefix,
                "max_results": self.max_results,
            }
        if self.tool_name == AzureToolName.read_text:
            return {"blob_path": self.blob_path}
        if self.tool_name == AzureToolName.read_json:
            return {"blob_path": self.blob_path}
        if self.tool_name == AzureToolName.write_text:
            return {
                "blob_path": self.blob_path,
                "content": self.content,
                "content_type": self.content_type or "text/plain",
            }
        if self.tool_name == AzureToolName.write_json:
            return {
                "blob_path": self.blob_path,
                "data": self.data,
            }
        if self.tool_name == AzureToolName.write_parquet:
            return {
                "blob_path": self.blob_path,
                "data": self.data,
            }
        return {}


class RawAzureBlobActionPlan(BaseModel):
    tool_name: AzureToolName
    summary: str
    rationale: str
    blob_path: str | None = None
    prefix: str | None = None
    max_results: int | None = None
    content: str | None = None
    data: AzureJSONPayload | None = None
    content_type: str | None = None
    final_response: str | None = None


class AzureAgentRequest(BaseModel):
    message: str = Field(min_length=1)
    approval_id: str | None = None
    approve: bool | None = None


class AzureApprovalProposal(BaseModel):
    approval_id: str
    tool_name: str
    tool_args: dict[str, Any]
    summary: str
    expires_at: datetime


class AzureAgentResponse(BaseModel):
    status: AzureAgentStatus
    message: str
    result: dict[str, Any] | None = None
    approval: AzureApprovalProposal | None = None
