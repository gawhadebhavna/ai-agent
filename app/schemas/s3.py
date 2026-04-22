from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class BucketListResponse(BaseModel):
    buckets: list[str]


class S3ObjectSummary(BaseModel):
    key: str
    size: int
    etag: str | None = None
    last_modified: datetime | None = None


class ObjectListResponse(BaseModel):
    bucket: str
    prefix: str | None = None
    objects: list[S3ObjectSummary]


class S3ObjectReadRequest(BaseModel):
    bucket: str = Field(min_length=1)
    key: str = Field(min_length=1)


class S3ObjectReadResponse(BaseModel):
    bucket: str
    key: str
    content: str
    content_type: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    etag: str | None = None
    version_id: str | None = None
    last_modified: datetime | None = None


class S3ObjectWriteRequest(BaseModel):
    bucket: str = Field(min_length=1)
    key: str = Field(min_length=1)
    content_type: str = "application/json"
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)


class S3ObjectWriteResponse(BaseModel):
    bucket: str
    key: str
    etag: str | None = None
    version_id: str | None = None
    message: str
