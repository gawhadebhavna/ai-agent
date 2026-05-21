from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd

from app.config import Settings
from app.services.azure import AzureBlobClientFactory


class AzureBlobService:
    """General-purpose Azure Blob Storage service.

    Supports reading/listing plus writing text, JSON, raw bytes, and Parquet (from records).
    Mirrors the S3Service pattern and is not tied to any specific data source.
    """

    def __init__(self, client_factory: AzureBlobClientFactory, settings: Settings) -> None:
        self._client_factory = client_factory
        self._settings = settings

    def write_bytes(
        self,
        data: bytes,
        blob_path: str,
        content_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        """Upload raw bytes to the configured container at blob_path."""
        from azure.storage.blob import ContentSettings
        client = self._client_factory.blob_service()
        blob_client = client.get_blob_client(
            container=self._settings.azure_container_name,
            blob=blob_path,
        )
        blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
        return {"status": "success", "blob_path": blob_path, "bytes_written": len(data)}

    def write_text(
        self,
        content: str,
        blob_path: str,
        content_type: str = "text/plain",
    ) -> dict[str, Any]:
        """Upload a UTF-8 string to the configured container at blob_path."""
        return self.write_bytes(content.encode("utf-8"), blob_path, content_type)

    def write_json(
        self,
        data: Any,
        blob_path: str,
    ) -> dict[str, Any]:
        """Serialize data as JSON and upload to blob_path."""
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        return self.write_bytes(payload, blob_path, "application/json")

    def write_parquet(
        self,
        data: list[dict],
        blob_path: str,
    ) -> dict[str, Any]:
        """Serialize a list of records as Parquet and upload to blob_path."""
        if not data:
            return {"status": "error", "message": "No data to write", "blob_path": blob_path}

        df = pd.DataFrame(data)
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        raw = buffer.getvalue()

        result = self.write_bytes(raw, blob_path, "application/octet-stream")
        result["rows_written"] = len(df)
        return result

    def read_bytes(self, blob_path: str) -> dict[str, Any]:
        """Download raw blob bytes from the configured container."""
        client = self._client_factory.blob_service()
        blob_client = client.get_blob_client(
            container=self._settings.azure_container_name,
            blob=blob_path,
        )
        raw = blob_client.download_blob().readall()
        properties = blob_client.get_blob_properties()
        content_settings = getattr(properties, "content_settings", None)
        content_type = getattr(content_settings, "content_type", None) or "application/octet-stream"
        return {
            "status": "success",
            "blob_path": blob_path,
            "bytes_read": len(raw),
            "content_type": content_type,
            "data": raw,
        }

    def read_text(self, blob_path: str) -> dict[str, Any]:
        """Download a blob as UTF-8 text."""
        payload = self.read_bytes(blob_path)
        raw = payload.pop("data", b"")
        if not isinstance(raw, (bytes, bytearray)):
            return {
                "status": "error",
                "blob_path": blob_path,
                "message": "Unexpected blob payload type while reading text content.",
            }
        payload["content"] = bytes(raw).decode("utf-8", errors="replace")
        return payload

    def read_json(self, blob_path: str) -> dict[str, Any]:
        """Download a blob and parse it as JSON."""
        payload = self.read_text(blob_path)
        content = payload.get("content")
        if not isinstance(content, str):
            return {
                "status": "error",
                "blob_path": blob_path,
                "message": "No text content was available for JSON parsing.",
            }
        try:
            payload["data"] = json.loads(content)
            return payload
        except json.JSONDecodeError:
            return {
                "status": "error",
                "blob_path": blob_path,
                "content_type": payload.get("content_type"),
                "bytes_read": payload.get("bytes_read"),
                "message": "Blob content is not valid JSON.",
            }

    def list_blobs(self, prefix: str | None = None, max_results: int = 100) -> dict[str, Any]:
        """List blobs in the configured container, optionally filtered by prefix."""
        client = self._client_factory.blob_service()
        container_client = client.get_container_client(self._settings.azure_container_name)
        blob_iter = container_client.list_blobs(name_starts_with=prefix or None)

        items: list[dict[str, Any]] = []
        for blob in blob_iter:
            if len(items) >= max_results:
                break
            content_settings = getattr(blob, "content_settings", None)
            content_type = getattr(content_settings, "content_type", None)
            last_modified = getattr(blob, "last_modified", None)
            items.append(
                {
                    "name": str(getattr(blob, "name", "")),
                    "size": int(getattr(blob, "size", 0) or 0),
                    "content_type": content_type,
                    "last_modified": last_modified.isoformat() if last_modified else None,
                }
            )

        return {
            "status": "success",
            "container": self._settings.azure_container_name,
            "prefix": prefix,
            "count": len(items),
            "blobs": items,
        }

