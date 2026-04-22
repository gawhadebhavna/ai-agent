from __future__ import annotations

from fastapi import APIRouter, Depends

from app.agents.service import AgentService
from app.dependencies import get_agent_service
from app.schemas.agent import AgentS3Request, AgentS3Response

router = APIRouter()


@router.post("/s3", response_model=AgentS3Response)
def handle_s3_agent_request(
    payload: AgentS3Request,
    agent_service: AgentService = Depends(get_agent_service),
) -> AgentS3Response:
    return agent_service.handle(payload)
