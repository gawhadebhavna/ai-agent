from __future__ import annotations

from fastapi import APIRouter, Depends

from app.agents.unified_service import UnifiedLLMService
from app.dependencies import get_unified_llm_service
from app.schemas.llm import UnifiedLLMRequest, UnifiedLLMResponse

router = APIRouter()


@router.post("/run", response_model=UnifiedLLMResponse)
def run_llm_tool_call(
    payload: UnifiedLLMRequest,
    llm_service: UnifiedLLMService = Depends(get_unified_llm_service),
) -> UnifiedLLMResponse:
    return llm_service.handle(payload)
