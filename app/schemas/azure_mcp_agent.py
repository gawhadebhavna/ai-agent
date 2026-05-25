from __future__ import annotations

from pydantic import BaseModel, Field


class AzureMCPAgentRequest(BaseModel):
    """Request schema for Azure MCP agent operations."""

    message: str = Field(
        min_length=1,
        description="Natural language request for Azure operations via MCP",
    )
    approval_id: str | None = Field(
        default=None,
        description="Optional approval ID for continuing approved operations",
    )
    approve: bool | None = Field(
        default=None,
        description="Whether to approve the pending operation (true) or reject (false)",
    )
