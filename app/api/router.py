from __future__ import annotations

from fastapi import APIRouter

from app.api.routes.agent import router as agent_router
from app.api.routes.blob import router as blob_router
from app.api.routes.s3 import router as s3_router
from app.api.routes.sap_agent import router as sap_agent_router

api_router = APIRouter()
api_router.include_router(s3_router, prefix="/v1/s3", tags=["s3"])
api_router.include_router(blob_router, prefix="/v1/blob", tags=["blob"])
api_router.include_router(agent_router, prefix="/v1/agent", tags=["agent"])
api_router.include_router(sap_agent_router, prefix="/v1/sap", tags=["sap"])
