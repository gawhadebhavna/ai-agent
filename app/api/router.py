from __future__ import annotations

from fastapi import APIRouter

from app.api.routes.agent import router as agent_router
from app.api.routes.s3 import router as s3_router

api_router = APIRouter()
api_router.include_router(s3_router, prefix="/v1/s3", tags=["s3"])
api_router.include_router(agent_router, prefix="/v1/agent", tags=["agent"])
