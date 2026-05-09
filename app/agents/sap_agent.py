from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx
from openai import AzureOpenAI

from app.agents.sap_tools import call_sap_api, trigger_databricks_ingestion
from app.config import get_settings
from app.core.exceptions import ProviderConfigurationError


def _load_prompt(filename: str) -> str:
    path = Path(__file__).parent.parent / "prompts" / filename
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _build_system_prompt() -> str:
    system_prompt = _load_prompt("sap_system_prompt.txt")
    examples = _load_prompt("sap_examples.txt")
    return f"{system_prompt}\n\nExamples:\n{examples}"


_SAP_TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_business_partners",
            "description": (
                "Get business partner data. If city is not provided, return all. "
                "Use 'top' for limiting records."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "top": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_products",
            "description": "Get product data. Supports product_type, search and top.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_type": {"type": "string"},
                    "top": {"type": "integer"},
                    "search": {"type": "string"},
                    "exclude_type": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sales_orders",
            "description": "Get sales order data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer": {"type": "string"},
                    "top": {"type": "integer"},
                    "search": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_ingestion",
            "description": "Trigger Databricks ingestion job for SAP data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "api_url": {"type": "string"},
                },
            },
        },
    },
]


class SAPDataFetcher:
    """Uses Azure OpenAI to map a natural-language request to a SAP API call.

    Responsibility: fetch SAP data only. No write or storage logic lives here.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.has_azure_openai_credentials:
            raise ProviderConfigurationError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are required for the SAP agent."
            )
        self._settings = settings
        self._client = self._build_client()
        self._system_prompt = _build_system_prompt()

    def _build_client(self) -> AzureOpenAI:
        http_client = httpx.Client(verify=self._settings.ssl_verify)
        return AzureOpenAI(
            api_key=self._settings.azure_openai_api_key,
            azure_endpoint=self._settings.azure_openai_endpoint,
            api_version=self._settings.azure_openai_api_version,
            http_client=http_client,
        )

    def plan_and_fetch(self, user_input: str) -> tuple[str, dict, list[dict] | dict]:
        """Ask the LLM to select a tool, then execute it.

        Returns (tool_name, arguments, result).
        """
        response = self._client.chat.completions.create(
            model=self._settings.azure_openai_deployment,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_input},
            ],
            tools=_SAP_TOOLS_SCHEMA,
            tool_choice="auto",
            max_tokens=200,
            temperature=0,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            return ("none", {}, {"message": message.content})

        import json
        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except Exception:
            arguments = {}

        arguments = self._enhance_args(user_input, arguments)
        result = self._execute_fetch(tool_name, arguments)
        return (tool_name, arguments, result)

    def _enhance_args(self, user_input: str, args: dict) -> dict:
        if not isinstance(args, dict):
            args = {}
        text = user_input.lower()
        if args.get("top") is None:
            match = re.search(r"\btop\s+(\d+)", text)
            if match:
                args["top"] = int(match.group(1))
        structured_keywords = [
            "top", "all", "list", "show", "get",
            "customer", "customers", "product", "products",
            "order", "orders", "sales",
        ]
        if not any(word in text for word in structured_keywords) and "search" not in args:
            args["search"] = user_input
        return args

    def _execute_fetch(self, name: str, args: dict) -> Any:
        settings = self._settings

        if name == "get_business_partners":
            params: dict = {}
            if args.get("city"):
                params["$filter"] = f"City eq '{args['city']}'"
            if args.get("top"):
                params["$top"] = args["top"]
            return call_sap_api(
                "/sap/opu/odata/sap/API_BUSINESS_PARTNER/A_BusinessPartner",
                params, settings=settings,
            )

        if name == "get_products":
            params = {}
            if args.get("top"):
                params["$top"] = args["top"]
            if args.get("search"):
                params["search"] = args["search"]
            if args.get("product_type"):
                params["product_type"] = args["product_type"]
            if args.get("exclude_type"):
                params["exclude_type"] = args["exclude_type"]
            return call_sap_api(
                "/sap/opu/odata/sap/API_PRODUCT_SRV/A_Product",
                params, settings=settings,
            )

        if name == "get_sales_orders":
            params = {}
            if args.get("customer"):
                params["customer"] = args["customer"]
            if args.get("top"):
                params["$top"] = args["top"]
            if args.get("search"):
                params["search"] = args["search"]
            return call_sap_api(
                "/sap/opu/odata/sap/API_SALES_ORDER_SRV/A_SalesOrder",
                params, settings=settings,
            )

        if name == "trigger_ingestion":
            api_url = (
                f"{settings.sap_ngrok_base_url}"
                "/sap/opu/odata/sap/API_BUSINESS_PARTNER/A_BusinessPartner"
            )
            return trigger_databricks_ingestion(api_url, settings=settings)

        return {"error": f"Unknown tool: {name}"}


_SAP_TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_business_partners",
            "description": (
                "Get business partner data. If city is not provided, return all. "
                "Use 'top' for limiting records."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "top": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_products",
            "description": "Get product data. Supports product_type, search and top.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_type": {"type": "string"},
                    "top": {"type": "integer"},
                    "search": {"type": "string"},
                    "exclude_type": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sales_orders",
            "description": "Get sales order data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer": {"type": "string"},
                    "top": {"type": "integer"},
                    "search": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_ingestion",
            "description": "Trigger Databricks ingestion job for SAP data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "api_url": {"type": "string"},
                },
            },
        },
    },
]


def _load_prompt(filename: str) -> str:
    path = Path(__file__).parent.parent / "prompts" / filename
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _build_system_prompt() -> str:
    system_prompt = _load_prompt("sap_system_prompt.txt")
    examples = _load_prompt("sap_examples.txt")
    return f"{system_prompt}\n\nExamples:\n{examples}"


class SAPAgent:
    """Azure OpenAI GPT-4o agent for querying SAP entities and writing to Azure Blob.

    Accepts an AzureBlobService dependency (mirrors AgentService receiving graph/tools).
    """

    def __init__(self, settings: Settings, blob_service: AzureBlobService) -> None:
        if not settings.has_azure_openai_credentials:
            raise ProviderConfigurationError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are required for the SAP agent."
            )
        self._settings = settings
        self._blob_service = blob_service
        self._client = self._build_client()
        self._system_prompt = _build_system_prompt()

    def _build_client(self) -> AzureOpenAI:
        http_client = httpx.Client(verify=self._settings.ssl_verify)
        return AzureOpenAI(
            api_key=self._settings.azure_openai_api_key,
            azure_endpoint=self._settings.azure_openai_endpoint,
            api_version=self._settings.azure_openai_api_version,
            http_client=http_client,
        )

    def _is_write_intent(self, text: str) -> bool:
        return any(k in text.lower() for k in _WRITE_KEYWORDS)

    def _enhance_args(self, user_input: str, args: dict) -> dict:
        """Extract top-N and fallback search from free-text when the model misses them."""
        if not isinstance(args, dict):
            args = {}
        text = user_input.lower()

        if args.get("top") is None:
            match = re.search(r"\btop\s+(\d+)", text)
            if match:
                args["top"] = int(match.group(1))

        structured_keywords = [
            "top", "all", "list", "show", "get",
            "customer", "customers", "product", "products",
            "order", "orders", "sales",
        ]
        if not any(word in text for word in structured_keywords) and "search" not in args:
            args["search"] = user_input

        return args

    def _execute_tool(self, name: str, args: dict) -> Any:
        settings = self._settings

        if name == "get_business_partners":
            params: dict = {}
            if args.get("city"):
                params["$filter"] = f"City eq '{args['city']}'"
            if args.get("top"):
                params["$top"] = args["top"]
            return call_sap_api(
                "/sap/opu/odata/sap/API_BUSINESS_PARTNER/A_BusinessPartner",
                params,
                settings=settings,
            )

        if name == "get_products":
            params = {}
            if args.get("top"):
                params["$top"] = args["top"]
            if args.get("search"):
                params["search"] = args["search"]
            if args.get("product_type"):
                params["product_type"] = args["product_type"]
            if args.get("exclude_type"):
                params["exclude_type"] = args["exclude_type"]
            return call_sap_api(
                "/sap/opu/odata/sap/API_PRODUCT_SRV/A_Product",
                params,
                settings=settings,
            )

        if name == "get_sales_orders":
            params = {}
            if args.get("customer"):
                params["customer"] = args["customer"]
            if args.get("top"):
                params["$top"] = args["top"]
            if args.get("search"):
                params["search"] = args["search"]
            return call_sap_api(
                "/sap/opu/odata/sap/API_SALES_ORDER_SRV/A_SalesOrder",
                params,
                settings=settings,
            )

        if name == "trigger_ingestion":
            api_url = (
                f"{settings.sap_ngrok_base_url}"
                "/sap/opu/odata/sap/API_BUSINESS_PARTNER/A_BusinessPartner"
            )
            return trigger_databricks_ingestion(api_url, settings=settings)

        return {"error": f"Unknown tool: {name}"}

    def run(self, user_input: str) -> Any:
        response = self._client.chat.completions.create(
            model=self._settings.azure_openai_deployment,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_input},
            ],
            tools=_SAP_TOOLS_SCHEMA,
            tool_choice="auto",
            max_tokens=200,
            temperature=0,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            return {"message": message.content}

        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name

        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except Exception:
            arguments = {}

        arguments = self._enhance_args(user_input, arguments)
        result = self._execute_tool(tool_name, arguments)

        if isinstance(result, list) and len(result) == 0:
            return {"status": "no_data", "message": "No records found"}

        if self._is_write_intent(user_input) and isinstance(result, list) and result:
            table_name = _TOOL_TABLE_MAPPING.get(tool_name, "default_table")
            write_result = self._blob_service.write_parquet(
                data=result,
                table_name=table_name,
            )
            return {"fetch_result_count": len(result), "azure_blob_write": write_result}

        return result
