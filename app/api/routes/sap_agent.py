from __future__ import annotations

from fastapi import APIRouter, Depends

from app.agents.sap_service import SAPAgentService
from app.dependencies import get_sap_agent_service
from app.schemas.sap_agent import SAPAgentRequest, SAPAgentResponse

router = APIRouter()


@router.post("/run", response_model=SAPAgentResponse)
def run_sap_agent(
    payload: SAPAgentRequest,
    sap_service: SAPAgentService = Depends(get_sap_agent_service),
) -> SAPAgentResponse:
    return sap_service.handle(payload)
