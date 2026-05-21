from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BlobWriteTextRequest(BaseModel):
    blob_path: str = Field(min_length=1, description="Destination path inside the container, e.g. reports/output.txt")
    content: str
    content_type: str = "text/plain"


class BlobWriteJsonRequest(BaseModel):
    blob_path: str = Field(min_length=1, description="Destination path inside the container, e.g. data/result.json")
    data: Any


class BlobWriteParquetRequest(BaseModel):
    blob_path: str = Field(min_length=1, description="Destination path inside the container, e.g. tables/records.parquet")
    data: list[dict]


class BlobWriteResponse(BaseModel):
    status: str
    blob_path: str
    bytes_written: int | None = None
    rows_written: int | None = None
    message: str | None = None
