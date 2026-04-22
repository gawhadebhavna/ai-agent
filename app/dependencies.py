from __future__ import annotations

from fastapi import Request

from app.agents.service import AgentService
from app.services.s3 import S3Service


def get_s3_service(request: Request) -> S3Service:
    return request.app.state.s3_service


def get_agent_service(request: Request) -> AgentService:
    return request.app.state.agent_service
