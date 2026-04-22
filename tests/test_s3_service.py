from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pytest

from app.config import Settings
from app.core.exceptions import AccessDeniedError
from app.schemas.s3 import S3ObjectWriteRequest
from app.services.s3 import S3Service


class FakeS3Client:
    def list_buckets(self):
        return {"Buckets": [{"Name": "allowed-bucket"}, {"Name": "blocked-bucket"}]}

    def list_objects_v2(self, *, Bucket: str, Prefix: str):
        return {
            "Contents": [
                {
                    "Key": f"{Prefix}file.json",
                    "Size": 12,
                    "ETag": "etag-1",
                    "LastModified": datetime(2026, 4, 15, 0, 0, 0),
                }
            ]
        }

    def get_object(self, *, Bucket: str, Key: str):
        return {
            "Body": BytesIO(b'{"hello":"world"}'),
            "ContentType": "application/json",
            "Metadata": {"source": "test"},
            "ETag": "etag-2",
            "VersionId": "v1",
        }

    def put_object(self, **kwargs):
        self.last_put = kwargs
        return {"ETag": "etag-3", "VersionId": "v2"}


class FakeFactory:
    def __init__(self):
        self.client = FakeS3Client()

    def s3(self):
        return self.client


def build_settings() -> Settings:
    return Settings(
        allowed_buckets="allowed-bucket",
        allowed_prefixes_json='{"allowed-bucket": ["landing/"]}',
        sqlite_path="./var/test-app.db",
    )


def test_list_buckets_filters_allowlist():
    service = S3Service(FakeFactory(), build_settings())
    assert service.list_buckets() == ["allowed-bucket"]


def test_read_object_returns_content():
    service = S3Service(FakeFactory(), build_settings())
    response = service.read_object("allowed-bucket", "landing/input.json")
    assert response.content == '{"hello":"world"}'
    assert response.metadata == {"source": "test"}


def test_write_object_blocks_disallowed_prefix():
    service = S3Service(FakeFactory(), build_settings())
    with pytest.raises(AccessDeniedError):
        service.write_object(
            S3ObjectWriteRequest(
                bucket="allowed-bucket",
                key="raw/output.json",
                content="{}",
            )
        )
