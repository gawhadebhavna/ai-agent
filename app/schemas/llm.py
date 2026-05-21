from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LLMToolDomain(str, Enum):
    auto = "auto"
    aws = "aws"
    azure = "azure"
    sap = "sap"


class UnifiedLLMRequest(BaseModel):
    message: str = Field(min_length=1)
    domain: LLMToolDomain = LLMToolDomain.auto
    approval_id: str | None = None
    approve: bool | None = None


class UnifiedLLMResponse(BaseModel):
    domain: LLMToolDomain
    status: str
    message: str
    result: Any | None = None
    approval: dict[str, Any] | None = None
