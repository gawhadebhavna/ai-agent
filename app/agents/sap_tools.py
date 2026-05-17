from __future__ import annotations

from typing import Any

import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from requests.auth import HTTPBasicAuth

from app.config import Settings
from app.core.exceptions import ProviderConfigurationError

SUPPORTED_SAP_TABLES = ("KNA1", "MARA", "VBFA", "VBKD", "VBPA")

_TABLE_ALIASES: dict[str, str] = {
    "business_partner": "KNA1",
    "business_partners": "KNA1",
    "customer": "KNA1",
    "customers": "KNA1",
    "customer_master": "KNA1",
    "material": "MARA",
    "materials": "MARA",
    "product": "MARA",
    "products": "MARA",
    "product_master": "MARA",
    "sales_document_flow": "VBFA",
    "document_flow": "VBFA",
    "sales_flow": "VBFA",
    "order_flow": "VBFA",
    "orders": "VBFA",
    "sales_orders": "VBFA",
    "sales_document_business_data": "VBKD",
    "business_data": "VBKD",
    "billing_business_data": "VBKD",
    "sales_document_partners": "VBPA",
    "document_partners": "VBPA",
    "partners": "VBPA",
}


class GetTableDataArgs(BaseModel):
    table_name: str = Field(
        description="SAP table name or alias. Supported tables: KNA1, MARA, VBFA, VBKD, VBPA."
    )
    filter_query: str | None = Field(
        default=None,
        description="Optional OData-style equality filter for $filter, e.g. ORT01 eq 'DXB'.",
    )
    select_query: str | None = Field(
        default=None,
        description="Optional comma-separated SAP fields for $select, e.g. KUNNR,NAME1,ORT01.",
    )
    top: int | None = Field(default=None, ge=0, description="Optional row limit mapped to $top.")
    skip: int = Field(default=0, ge=0, description="Optional number of rows to skip mapped to $skip.")
    search: str | None = Field(default=None, description="Optional free-text search across row values.")


class GetTableMetadataArgs(BaseModel):
    table_name: str = Field(
        description="SAP table name or alias. Supported tables: KNA1, MARA, VBFA, VBKD, VBPA."
    )


def normalize_sap_table_name(table_name: str) -> str:
    normalized = "_".join(str(table_name).strip().lower().replace("-", "_").split())
    return _TABLE_ALIASES.get(normalized, str(table_name).strip().upper())


def _sap_get(endpoint: str, params: dict | None = None, *, settings: Settings) -> Any:
    """Make an authenticated GET request to the mock SAP table API."""
    if not settings.has_sap_api_credentials:
        raise ProviderConfigurationError(
            "SAP_NGROK_BASE_URL, SAP_API_USERNAME, and SAP_API_PASSWORD are required."
        )

    url = f"{settings.sap_ngrok_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    response = requests.get(
        url,
        params=params,
        auth=HTTPBasicAuth(settings.sap_api_username, settings.sap_api_password),
        headers={"ngrok-skip-browser-warning": "true"},
        timeout=30,
        verify=settings.ssl_verify,
    )
    response.raise_for_status()
    return response.json()


def _extract_results(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return []

    for key in ("results", "data", "rows", "records", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return value

    legacy_results = payload.get("d", {}).get("results")
    if isinstance(legacy_results, list):
        return legacy_results

    return []


def call_sap_api(endpoint: str, params: dict | None = None, *, settings: Settings) -> list[dict]:
    """Fetch SAP records and return a stable record list.

    The dummy SAP API now returns table payloads as {"table", "count", "results"}
    instead of the older SAP-style {"d": {"results": [...]}} wrapper. This helper
    accepts both shapes so legacy callers still get a list of records.
    """
    return _extract_results(_sap_get(endpoint, params=params, settings=settings))


def list_sap_tables(*, settings: Settings) -> dict[str, Any]:
    payload = _sap_get("/tables", settings=settings)
    if isinstance(payload, dict):
        return payload
    return {"count": len(payload) if isinstance(payload, list) else 0, "tables": payload}


def get_sap_table_metadata(table_name: str, *, settings: Settings) -> dict[str, Any]:
    resolved_table = normalize_sap_table_name(table_name)
    payload = _sap_get(f"/tables/{resolved_table}/metadata", settings=settings)
    if isinstance(payload, dict):
        return payload
    return {"table": resolved_table, "field_count": 0, "fields": []}


def get_sap_table_data(
    table_name: str,
    *,
    filter_query: str | None = None,
    select_query: str | None = None,
    top: int | None = None,
    skip: int = 0,
    search: str | None = None,
    settings: Settings,
) -> dict[str, Any]:
    resolved_table = normalize_sap_table_name(table_name)
    params: dict[str, Any] = {}
    if filter_query:
        params["$filter"] = filter_query
    if select_query:
        params["$select"] = select_query
    if top is not None:
        params["$top"] = top
    if skip:
        params["$skip"] = skip
    if search:
        params["search"] = search

    payload = _sap_get(f"/tables/{resolved_table}/data", params=params or None, settings=settings)
    results = _extract_results(payload)
    if isinstance(payload, dict):
        count = payload.get("count")
        return {
            "table": str(payload.get("table") or resolved_table),
            "count": count if isinstance(count, int) else len(results),
            "results": results,
        }
    return {"table": resolved_table, "count": len(results), "results": results}


def trigger_databricks_ingestion(api_url: str, *, settings: Settings) -> dict[str, Any]:
    """Trigger a Databricks job to ingest SAP data."""
    if not settings.has_databricks_credentials:
        raise ProviderConfigurationError(
            "DATABRICKS_URL and DATABRICKS_TOKEN are required."
        )
    url = f"{settings.databricks_url}/api/2.1/jobs/run-now"
    headers = {"Authorization": f"Bearer {settings.databricks_token}"}
    payload = {
        "job_id": settings.databricks_job_id,
        "notebook_params": {"api_url": api_url},
    }
    response = requests.post(url, headers=headers, json=payload, timeout=30, verify=settings.ssl_verify)
    return response.json()


def build_sap_tools(settings: Settings) -> dict[str, StructuredTool]:
    """Build StructuredTool wrappers for SAP table fetch operations."""

    def _list_sap_tables() -> dict:
        return list_sap_tables(settings=settings)

    def _get_sap_table_metadata(table_name: str) -> dict:
        return get_sap_table_metadata(table_name, settings=settings)

    def _get_sap_table_data(
        table_name: str,
        filter_query: str | None = None,
        select_query: str | None = None,
        top: int | None = None,
        skip: int = 0,
        search: str | None = None,
    ) -> dict:
        return get_sap_table_data(
            table_name,
            filter_query=filter_query,
            select_query=select_query,
            top=top,
            skip=skip,
            search=search,
            settings=settings,
        )

    return {
        "list_sap_tables": StructuredTool.from_function(
            func=_list_sap_tables,
            name="list_sap_tables",
            description="List SAP tables available from the dummy SAP table API.",
        ),
        "get_sap_table_metadata": StructuredTool.from_function(
            func=_get_sap_table_metadata,
            name="get_sap_table_metadata",
            description=(
                "Fetch SAP table field metadata. Supported tables: "
                f"{', '.join(SUPPORTED_SAP_TABLES)}."
            ),
            args_schema=GetTableMetadataArgs,
        ),
        "get_sap_table_data": StructuredTool.from_function(
            func=_get_sap_table_data,
            name="get_sap_table_data",
            description=(
                "Fetch SAP table data through /tables/{tablename}/data. "
                "Supports $filter, $select, $top, $skip, and search. "
                f"Supported tables: {', '.join(SUPPORTED_SAP_TABLES)}."
            ),
            args_schema=GetTableDataArgs,
        ),
    }
