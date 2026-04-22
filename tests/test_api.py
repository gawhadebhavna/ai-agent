from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.router import api_router
from app.core.exceptions import AccessDeniedError
from app.dependencies import get_agent_service, get_s3_service
from app.main import app_error_handler


class StubS3Service:
    def list_buckets(self):
        return ["bucket-a"]

    def list_objects(self, bucket: str, prefix: str | None = None):
        return {
            "bucket": bucket,
            "prefix": prefix,
            "objects": [{"key": "landing/file.json", "size": 1, "etag": None, "last_modified": None}],
        }

    def read_object(self, bucket: str, key: str):
        return {
            "bucket": bucket,
            "key": key,
            "content": "{}",
            "content_type": "application/json",
            "metadata": {},
            "etag": None,
            "version_id": None,
            "last_modified": None,
        }

    def write_object(self, payload):
        return {
            "bucket": payload.bucket,
            "key": payload.key,
            "etag": "etag",
            "version_id": None,
            "message": "written",
        }


class DeniedS3Service(StubS3Service):
    def read_object(self, bucket: str, key: str):
        raise AccessDeniedError("blocked")


class StubAgentService:
    def handle(self, payload):
        return {
            "status": "approval_required",
            "message": "Approval is required.",
            "result": None,
            "approval": {
                "approval_id": "approval-1",
                "tool_name": "s3_write_object",
                "tool_args": {"bucket": "bucket-a", "key": "landing/out.json"},
                "summary": "Write file",
                "expires_at": "2026-04-15T00:00:00+00:00",
            },
        }


def build_test_app(*, s3_service, agent_service) -> FastAPI:
    app = FastAPI()
    app.include_router(api_router)
    app.add_exception_handler(AccessDeniedError, app_error_handler)
    app.dependency_overrides[get_s3_service] = lambda: s3_service
    app.dependency_overrides[get_agent_service] = lambda: agent_service
    return app


def test_list_buckets_endpoint():
    client = TestClient(build_test_app(s3_service=StubS3Service(), agent_service=StubAgentService()))
    response = client.get("/v1/s3/buckets")
    assert response.status_code == 200
    assert response.json() == {"buckets": ["bucket-a"]}


def test_agent_endpoint_returns_approval_payload():
    client = TestClient(build_test_app(s3_service=StubS3Service(), agent_service=StubAgentService()))
    response = client.post("/v1/agent/s3", json={"message": "write a file"})
    assert response.status_code == 200
    assert response.json()["status"] == "approval_required"
    assert response.json()["approval"]["tool_name"] == "s3_write_object"


def test_app_error_handler_maps_service_errors():
    client = TestClient(build_test_app(s3_service=DeniedS3Service(), agent_service=StubAgentService()))
    response = client.get("/v1/s3/object", params={"bucket": "bucket-a", "key": "landing/in.json"})
    assert response.status_code == 403
    assert response.json() == {"error": {"code": "access_denied", "message": "blocked"}}
