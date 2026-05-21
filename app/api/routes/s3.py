from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_s3_service
from app.schemas.s3 import (
    BucketListResponse,
    ObjectListResponse,
    S3ObjectReadResponse,
    S3ObjectWriteRequest,
    S3ObjectWriteResponse,
)
from app.services.s3 import S3Service

router = APIRouter()


@router.get("/buckets", response_model=BucketListResponse)
def list_buckets(s3_service: S3Service = Depends(get_s3_service)) -> BucketListResponse:
    return BucketListResponse(buckets=s3_service.list_buckets())


@router.get("/objects", response_model=ObjectListResponse)
def list_objects(
    bucket: str,
    prefix: str | None = None,
    s3_service: S3Service = Depends(get_s3_service),
) -> ObjectListResponse:
    return s3_service.list_objects(bucket=bucket, prefix=prefix)


@router.get("/object", response_model=S3ObjectReadResponse)
def read_object(
    bucket: str,
    key: str,
    s3_service: S3Service = Depends(get_s3_service),
) -> S3ObjectReadResponse:
    return s3_service.read_object(bucket=bucket, key=key)


@router.put("/object", response_model=S3ObjectWriteResponse)
def write_object(
    payload: S3ObjectWriteRequest,
    s3_service: S3Service = Depends(get_s3_service),
) -> S3ObjectWriteResponse:
    return s3_service.write_object(payload)
