from __future__ import annotations

from botocore.exceptions import ClientError

from app.config import get_settings
from app.core.exceptions import AccessDeniedError, NotFoundError
from app.schemas.s3 import (
    ObjectListResponse,
    S3ObjectReadResponse,
    S3ObjectSummary,
    S3ObjectWriteRequest,
    S3ObjectWriteResponse,
)
from app.services.aws import AWSClientFactory


class S3Service:
    def __init__(self, client_factory: AWSClientFactory, settings: Settings) -> None:
        self._client_factory = client_factory
        self._settings = settings

    def list_buckets(self) -> list[str]:
        client = self._client_factory.s3()
        response = client.list_buckets()
        buckets = [item["Name"] for item in response.get("Buckets", [])]
        if self._settings.allowed_bucket_list:
            allowed = set(self._settings.allowed_bucket_list)
            buckets = [bucket for bucket in buckets if bucket in allowed]
        return buckets

    def list_objects(self, bucket: str, prefix: str | None = None) -> ObjectListResponse:
        self._ensure_bucket_allowed(bucket)
        self._ensure_prefix_allowed(bucket, prefix)
        client = self._client_factory.s3()
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix or "")
        objects = [
            S3ObjectSummary(
                key=item["Key"],
                size=item["Size"],
                etag=item.get("ETag"),
                last_modified=item.get("LastModified"),
            )
            for item in response.get("Contents", [])
        ]
        return ObjectListResponse(bucket=bucket, prefix=prefix, objects=objects)

    def read_object(self, bucket: str, key: str) -> S3ObjectReadResponse:
        self._ensure_bucket_allowed(bucket)
        self._ensure_prefix_allowed(bucket, key)
        client = self._client_factory.s3()
        try:
            response = client.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code in {"NoSuchKey", "404"}:
                raise NotFoundError(f"S3 object '{key}' was not found in bucket '{bucket}'.") from exc
            raise

        body = response["Body"].read().decode("utf-8", errors="replace")
        return S3ObjectReadResponse(
            bucket=bucket,
            key=key,
            content=body,
            content_type=response.get("ContentType"),
            metadata=response.get("Metadata", {}),
            etag=response.get("ETag"),
            version_id=response.get("VersionId"),
            last_modified=response.get("LastModified"),
        )

    def write_object(self, request: S3ObjectWriteRequest) -> S3ObjectWriteResponse:
        self._ensure_bucket_allowed(request.bucket)
        self._ensure_prefix_allowed(request.bucket, request.key)
        client = self._client_factory.s3()
        response = client.put_object(
            Bucket=request.bucket,
            Key=request.key,
            Body=request.content.encode("utf-8"),
            ContentType=request.content_type,
            Metadata=request.metadata,
        )
        return S3ObjectWriteResponse(
            bucket=request.bucket,
            key=request.key,
            etag=response.get("ETag"),
            version_id=response.get("VersionId"),
            message=f"Object '{request.key}' was written to bucket '{request.bucket}'.",
        )

    def _ensure_bucket_allowed(self, bucket: str) -> None:
        allowed_buckets = self._settings.allowed_bucket_list
        if allowed_buckets and bucket not in allowed_buckets:
            raise AccessDeniedError(f"Bucket '{bucket}' is not in the allowlist.")

    def _ensure_prefix_allowed(self, bucket: str, key_or_prefix: str | None) -> None:
        if not key_or_prefix:
            return
        allowed_prefixes = self._settings.allowed_prefixes.get(bucket, [])
        if allowed_prefixes and not any(key_or_prefix.startswith(prefix) for prefix in allowed_prefixes):
            raise AccessDeniedError(
                f"Key or prefix '{key_or_prefix}' is not allowed for bucket '{bucket}'."
            )
