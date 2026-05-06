from __future__ import annotations

from typing import Any

import requests
from langchain_core.tools import StructuredTool
from requests.auth import HTTPBasicAuth

from app.config import Settings
from app.core.exceptions import ProviderConfigurationError

_TOOL_TABLE_MAPPING: dict[str, str] = {
    "get_business_partners": "customers",
    "get_products": "products",
    "get_sales_orders": "sales_orders",
}


def call_sap_api(endpoint: str, params: dict | None = None, *, settings: Settings) -> list[dict]:
    """Make an authenticated GET request to the mock SAP endpoint."""
    if not settings.has_sap_api_credentials:
        raise ProviderConfigurationError(
            "SAP_NGROK_BASE_URL, SAP_API_USERNAME, and SAP_API_PASSWORD are required."
        )
    url = f"{settings.sap_ngrok_base_url}{endpoint}"
    response = requests.get(
        url,
        params=params,
        auth=HTTPBasicAuth(settings.sap_api_username, settings.sap_api_password),
        headers={"ngrok-skip-browser-warning": "true"},
        timeout=30,
        verify=settings.ssl_verify,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("d", {}).get("results", [])


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


# ---------------------------------------------------------------------------
# StructuredTool definitions (used by SAPGraphBuilder)
# ---------------------------------------------------------------------------


def build_sap_tools(settings: Settings) -> dict[str, StructuredTool]:
    """Build StructuredTool wrappers for SAP fetch operations.

    These tools only fetch data from the SAP API — no storage logic.
    Write operations are handled separately by build_blob_tools().
    """

    def get_business_partners(city: str | None = None, top: int | None = None) -> dict:
        params: dict = {}
        if city:
            params["$filter"] = f"City eq '{city}'"
        if top:
            params["$top"] = top
        return {
            "results": call_sap_api(
                "/sap/opu/odata/sap/API_BUSINESS_PARTNER/A_BusinessPartner",
                params, settings=settings,
            )
        }

    def get_products(
        product_type: str | None = None,
        top: int | None = None,
        search: str | None = None,
        exclude_type: str | None = None,
    ) -> dict:
        params: dict = {}
        if top:
            params["$top"] = top
        if search:
            params["search"] = search
        if product_type:
            params["product_type"] = product_type
        if exclude_type:
            params["exclude_type"] = exclude_type
        return {
            "results": call_sap_api(
                "/sap/opu/odata/sap/API_PRODUCT_SRV/A_Product",
                params, settings=settings,
            )
        }

    def get_sales_orders(
        customer: str | None = None,
        top: int | None = None,
        search: str | None = None,
    ) -> dict:
        params: dict = {}
        if customer:
            params["customer"] = customer
        if top:
            params["$top"] = top
        if search:
            params["search"] = search
        return {
            "results": call_sap_api(
                "/sap/opu/odata/sap/API_SALES_ORDER_SRV/A_SalesOrder",
                params, settings=settings,
            )
        }

    return {
        "get_business_partners": StructuredTool.from_function(
            func=get_business_partners,
            name="get_business_partners",
            description="Fetch SAP business partners (customers). Supports city filter and top-N.",
        ),
        "get_products": StructuredTool.from_function(
            func=get_products,
            name="get_products",
            description="Fetch SAP products. Supports product_type, search, exclude_type and top-N.",
        ),
        "get_sales_orders": StructuredTool.from_function(
            func=get_sales_orders,
            name="get_sales_orders",
            description="Fetch SAP sales orders. Supports customer filter, search and top-N.",
        ),
    }
