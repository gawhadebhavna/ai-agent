from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class AgentStatus(str, Enum):
    completed = "completed"
    approval_required = "approval_required"
    rejected = "rejected"
    error = "error"


class ToolName(str, Enum):
    list_buckets = "s3_list_buckets"
    list_objects = "s3_list_objects"
    read_object = "s3_read_object"
    write_object = "s3_write_object"
    unsupported = "unsupported"


class ActionPlan(BaseModel):
    tool_name: ToolName
    summary: str
    rationale: str
    bucket: str | None = None
    key: str | None = None
    prefix: str | None = None
    content: str | None = None
    content_type: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    requires_approval: bool = False
    final_response: str | None = None

    @model_validator(mode="after")
    def validate_required_fields(self) -> "ActionPlan":
        if self.tool_name in {ToolName.list_objects, ToolName.read_object, ToolName.write_object} and not self.bucket:
            raise ValueError("bucket is required for the selected tool")
        if self.tool_name in {ToolName.read_object, ToolName.write_object} and not self.key:
            raise ValueError("key is required for the selected tool")
        if self.tool_name == ToolName.write_object and self.content is None:
            raise ValueError("content is required for write requests")
        if self.tool_name == ToolName.write_object:
            self.requires_approval = True
        return self

    def tool_args(self) -> dict[str, Any]:
        if self.tool_name == ToolName.list_buckets:
            return {}
        if self.tool_name == ToolName.list_objects:
            return {"bucket": self.bucket, "prefix": self.prefix}
        if self.tool_name == ToolName.read_object:
            return {"bucket": self.bucket, "key": self.key}
        if self.tool_name == ToolName.write_object:
            return {
                "bucket": self.bucket,
                "key": self.key,
                "content": self.content,
                "content_type": self.content_type or "application/json",
                "metadata": self.metadata,
            }
        return {}


class RawActionPlan(BaseModel):
    tool_name: ToolName
    summary: str
    rationale: str
    bucket: str | None = None
    key: str | None = None
    prefix: str | None = None
    filename: str | None = None
    content: str | None = None
    content_type: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    final_response: str | None = None


class AgentS3Request(BaseModel):
    message: str = Field(min_length=1)
    approval_id: str | None = None
    approve: bool | None = None


class ApprovalProposal(BaseModel):
    approval_id: str
    tool_name: str
    tool_args: dict[str, Any]
    summary: str
    expires_at: datetime


class AgentS3Response(BaseModel):
    status: AgentStatus
    message: str
    result: dict[str, Any] | None = None
    approval: ApprovalProposal | None = None
