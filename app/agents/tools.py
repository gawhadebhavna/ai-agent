from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.schemas.s3 import S3ObjectWriteRequest
from app.services.s3 import S3Service


class ListObjectsArgs(BaseModel):
    bucket: str = Field(min_length=1)
    prefix: str | None = None


class ReadObjectArgs(BaseModel):
    bucket: str = Field(min_length=1)
    key: str = Field(min_length=1)


class WriteObjectArgs(BaseModel):
    bucket: str = Field(min_length=1)
    key: str = Field(min_length=1)
    content: str
    content_type: str = "application/json"
    metadata: dict[str, str] = Field(default_factory=dict)


def build_s3_tools(s3_service: S3Service) -> dict[str, StructuredTool]:
    def list_buckets() -> dict:
        return {"buckets": s3_service.list_buckets()}

    def list_objects(bucket: str, prefix: str | None = None) -> dict:
        return s3_service.list_objects(bucket=bucket, prefix=prefix).model_dump(mode="json")

    def read_object(bucket: str, key: str) -> dict:
        return s3_service.read_object(bucket=bucket, key=key).model_dump(mode="json")

    def write_object(
        bucket: str,
        key: str,
        content: str,
        content_type: str = "application/json",
        metadata: dict[str, str] | None = None,
    ) -> dict:
        return s3_service.write_object(
            S3ObjectWriteRequest(
                bucket=bucket,
                key=key,
                content=content,
                content_type=content_type,
                metadata=metadata or {},
            )
        ).model_dump(mode="json")

    return {
        "s3_list_buckets": StructuredTool.from_function(
            func=list_buckets,
            name="s3_list_buckets",
            description="List the S3 buckets the application can access.",
        ),
        "s3_list_objects": StructuredTool.from_function(
            func=list_objects,
            name="s3_list_objects",
            description="List objects in a bucket and optional prefix.",
            args_schema=ListObjectsArgs,
        ),
        "s3_read_object": StructuredTool.from_function(
            func=read_object,
            name="s3_read_object",
            description="Read the content and metadata of an S3 object.",
            args_schema=ReadObjectArgs,
        ),
        "s3_write_object": StructuredTool.from_function(
            func=write_object,
            name="s3_write_object",
            description="Write a text or JSON object into S3. Requires approval before execution.",
            args_schema=WriteObjectArgs,
        ),
    }
