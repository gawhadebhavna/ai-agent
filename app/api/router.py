from __future__ import annotations

from fastapi import APIRouter

from app.api.routes.blob import router as blob_router
from app.api.routes.llm import router as llm_router
from app.api.routes.s3 import router as s3_router

api_router = APIRouter()
api_router.include_router(s3_router, prefix="/v1/s3", tags=["s3"])
api_router.include_router(blob_router, prefix="/v1/blob", tags=["blob"])
api_router.include_router(llm_router, prefix="/v1/llm", tags=["llm"])
