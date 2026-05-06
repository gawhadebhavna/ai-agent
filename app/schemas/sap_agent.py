from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SAPAgentStatus(str, Enum):
    completed = "completed"
    approval_required = "approval_required"
    rejected = "rejected"
    error = "error"
    no_data = "no_data"


class SAPToolName(str, Enum):
    get_business_partners = "get_business_partners"
    get_products = "get_products"
    get_sales_orders = "get_sales_orders"
    trigger_ingestion = "trigger_ingestion"
    unsupported = "unsupported"


class SAPActionPlan(BaseModel):
    fetch_tool: SAPToolName
    fetch_args: dict[str, Any] = Field(default_factory=dict)
    write_after_fetch: bool = False
    table_name: str | None = None
    summary: str
    rationale: str
    requires_approval: bool = False
    final_response: str | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.write_after_fetch:
            self.requires_approval = True


class SAPAgentRequest(BaseModel):
    message: str = Field(min_length=1)
    approval_id: str | None = None
    approve: bool | None = None


class SAPApprovalProposal(BaseModel):
    approval_id: str
    tool_name: str
    tool_args: dict[str, Any]
    summary: str
    expires_at: datetime


class SAPAgentResponse(BaseModel):
    status: SAPAgentStatus
    message: str
    result: Any | None = None
    approval: SAPApprovalProposal | None = None
