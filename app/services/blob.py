from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd

from app.config import Settings
from app.services.azure import AzureBlobClientFactory


class AzureBlobService:
    """General-purpose Azure Blob Storage service.

    Supports writing text, JSON, raw bytes, and Parquet (from records).
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

