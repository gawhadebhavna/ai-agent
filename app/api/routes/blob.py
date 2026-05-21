from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_blob_service
from app.schemas.blob import (
    BlobWriteJsonRequest,
    BlobWriteParquetRequest,
    BlobWriteResponse,
    BlobWriteTextRequest,
)
from app.services.blob import AzureBlobService

router = APIRouter()


@router.put("/text", response_model=BlobWriteResponse)
def write_text(
    payload: BlobWriteTextRequest,
    blob_service: AzureBlobService = Depends(get_blob_service),
) -> BlobWriteResponse:
    result = blob_service.write_text(
        content=payload.content,
        blob_path=payload.blob_path,
        content_type=payload.content_type,
    )
    return BlobWriteResponse(**result)


@router.put("/json", response_model=BlobWriteResponse)
def write_json(
    payload: BlobWriteJsonRequest,
    blob_service: AzureBlobService = Depends(get_blob_service),
) -> BlobWriteResponse:
    result = blob_service.write_json(data=payload.data, blob_path=payload.blob_path)
    return BlobWriteResponse(**result)


@router.put("/parquet", response_model=BlobWriteResponse)
def write_parquet(
    payload: BlobWriteParquetRequest,
    blob_service: AzureBlobService = Depends(get_blob_service),
) -> BlobWriteResponse:
    result = blob_service.write_parquet(data=payload.data, blob_path=payload.blob_path)
    return BlobWriteResponse(**result)
