from __future__ import annotations

from fastapi import APIRouter

from app.api.routes.chat import router as chat_router
from app.api.routes.meta import router as meta_router

api_router = APIRouter()
api_router.include_router(chat_router, prefix="/v1/chat", tags=["chat"])
api_router.include_router(meta_router, prefix="/v1", tags=["meta"])
