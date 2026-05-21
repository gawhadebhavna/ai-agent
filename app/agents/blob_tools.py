from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.services.blob import AzureBlobService


class WriteBlobTextArgs(BaseModel):
    content: str = Field(description="Text content to upload.")
    blob_path: str = Field(description="Destination path in the container, e.g. reports/output.txt")
    content_type: str = Field(default="text/plain", description="MIME type.")


class WriteBlobJsonArgs(BaseModel):
    data: Any = Field(description="JSON-serializable data to upload.")
    blob_path: str = Field(description="Destination path in the container, e.g. data/result.json")


class WriteBlobParquetArgs(BaseModel):
    data: list[dict] = Field(description="List of records to serialize as Parquet.")
    blob_path: str = Field(description="Destination path in the container, e.g. tables/customers.parquet")


def build_blob_tools(blob_service: AzureBlobService) -> dict[str, StructuredTool]:
    """Build general-purpose Azure Blob write tools.

    These are completely data-source agnostic — usable by any agent or graph
    that needs to persist content to Azure Blob Storage.
    All writes should require approval when used inside an approval-gated graph.
    """

    def write_text(content: str, blob_path: str, content_type: str = "text/plain") -> dict:
        return blob_service.write_text(content=content, blob_path=blob_path, content_type=content_type)

    def write_json(data: Any, blob_path: str) -> dict:
        return blob_service.write_json(data=data, blob_path=blob_path)

    def write_parquet(data: list[dict], blob_path: str) -> dict:
        return blob_service.write_parquet(data=data, blob_path=blob_path)

    return {
        "blob_write_text": StructuredTool.from_function(
            func=write_text,
            name="blob_write_text",
            description=(
                "Write a text or CSV string to Azure Blob Storage at the given path. "
                "Requires approval before execution."
            ),
            args_schema=WriteBlobTextArgs,
        ),
        "blob_write_json": StructuredTool.from_function(
            func=write_json,
            name="blob_write_json",
            description=(
                "Serialize data as JSON and write to Azure Blob Storage at the given path. "
                "Requires approval before execution."
            ),
            args_schema=WriteBlobJsonArgs,
        ),
        "blob_write_parquet": StructuredTool.from_function(
            func=write_parquet,
            name="blob_write_parquet",
            description=(
                "Serialize a list of records as Parquet and write to Azure Blob Storage. "
                "Requires approval before execution."
            ),
            args_schema=WriteBlobParquetArgs,
        ),
    }
