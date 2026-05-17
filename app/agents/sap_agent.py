from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx
from openai import AzureOpenAI

from app.agents.sap_tools import (
    get_sap_table_data,
    get_sap_table_metadata,
    list_sap_tables,
    normalize_sap_table_name,
    trigger_databricks_ingestion,
)
from app.config import Settings
from app.core.exceptions import ProviderConfigurationError
from app.services.blob import AzureBlobService

_WRITE_KEYWORDS = frozenset(
    [
        "write",
        "store",
        "save",
        "persist",
        "upload",
        "load into delta",
        "write into delta",
        "push to blob",
    ]
)

_SAP_TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_sap_tables",
            "description": "List SAP tables available from the dummy SAP table API.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sap_table_data",
            "description": (
                "Fetch SAP table rows from /tables/{tablename}/data. "
                "Supported tables are KNA1, MARA, VBFA, VBKD, and VBPA."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "SAP table name or alias, e.g. KNA1, customers, MARA, products.",
                    },
                    "filter_query": {
                        "type": "string",
                        "description": "OData-style equality filter, e.g. ORT01 eq 'DXB'.",
                    },
                    "select_query": {
                        "type": "string",
                        "description": "Comma-separated SAP field names for $select.",
                    },
                    "top": {"type": "integer"},
                    "skip": {"type": "integer"},
                    "search": {"type": "string"},
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sap_table_metadata",
            "description": (
                "Fetch SAP table field metadata from /tables/{tablename}/metadata. "
                "Supported tables are KNA1, MARA, VBFA, VBKD, and VBPA."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string"},
                },
                "required": ["table_name"],
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


def _infer_table_name(user_input: str) -> str | None:
    text = user_input.lower()
    explicit = re.search(r"\b(kna1|mara|vbfa|vbkd|vbpa)\b", text)
    if explicit:
        return explicit.group(1).upper()
    if "customer" in text or "business partner" in text:
        return "KNA1"
    if "product" in text or "material" in text:
        return "MARA"
    if "partner" in text and ("sales" in text or "document" in text or "function" in text):
        return "VBPA"
    if "payment" in text or "billing business" in text or "business data" in text:
        return "VBKD"
    if "document flow" in text or "order flow" in text or "sales order" in text:
        return "VBFA"
    return None


def _records_from_result(result: Any) -> list[dict]:
    if isinstance(result, list):
        return result
    if isinstance(result, dict) and isinstance(result.get("results"), list):
        return result["results"]
    return []


def _enhance_tool_args(user_input: str, tool_name: str, args: dict) -> dict:
    if not isinstance(args, dict):
        args = {}

    text = user_input.lower()
    if args.get("top") is None:
        match = re.search(r"\btop\s+(\d+)", text)
        if match:
            args["top"] = int(match.group(1))

    if tool_name in {"get_sap_table_data", "get_sap_table_metadata"}:
        table_name = args.get("table_name") or _infer_table_name(user_input)
        if table_name:
            args["table_name"] = normalize_sap_table_name(table_name)

    structured_keywords = [
        "top",
        "all",
        "list",
        "show",
        "get",
        "customer",
        "customers",
        "product",
        "products",
        "material",
        "materials",
        "table",
        "metadata",
        "schema",
        "document",
        "order",
        "sales",
    ]
    if tool_name == "get_sap_table_data" and not any(word in text for word in structured_keywords):
        args.setdefault("search", user_input)

    return args


def _execute_fetch_with_settings(settings: Settings, name: str, args: dict) -> Any:
    if name == "list_sap_tables":
        return list_sap_tables(settings=settings)

    if name == "get_sap_table_metadata":
        table_name = args.get("table_name")
        if not table_name:
            return {"error": "table_name is required."}
        return get_sap_table_metadata(table_name, settings=settings)

    if name == "get_sap_table_data":
        table_name = args.get("table_name")
        if not table_name:
            return {"error": "table_name is required."}
        return get_sap_table_data(
            table_name,
            filter_query=args.get("filter_query"),
            select_query=args.get("select_query"),
            top=args.get("top"),
            skip=args.get("skip") or 0,
            search=args.get("search"),
            settings=settings,
        )

    if name == "trigger_ingestion":
        base_url = (settings.sap_ngrok_base_url or "").rstrip("/")
        api_url = args.get("api_url") or (
            f"{base_url}/tables/KNA1/data"
        )
        return trigger_databricks_ingestion(api_url, settings=settings)

    return {"error": f"Unknown tool: {name}"}


class SAPDataFetcher:
    """Uses Azure OpenAI to map a natural-language request to a SAP table API call.

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

        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except Exception:
            arguments = {}

        arguments = self._enhance_args(user_input, tool_name, arguments)
        result = self._execute_fetch(tool_name, arguments)
        return (tool_name, arguments, result)

    def _enhance_args(self, user_input: str, tool_name: str, args: dict) -> dict:
        return _enhance_tool_args(user_input, tool_name, args)

    def _execute_fetch(self, name: str, args: dict) -> Any:
        return _execute_fetch_with_settings(self._settings, name, args)


class SAPAgent:
    """Legacy Azure OpenAI agent for querying SAP tables and writing to Azure Blob."""

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
        return any(keyword in text.lower() for keyword in _WRITE_KEYWORDS)

    def _enhance_args(self, user_input: str, tool_name: str, args: dict) -> dict:
        return _enhance_tool_args(user_input, tool_name, args)

    def _execute_tool(self, name: str, args: dict) -> Any:
        return _execute_fetch_with_settings(self._settings, name, args)

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

        arguments = self._enhance_args(user_input, tool_name, arguments)
        result = self._execute_tool(tool_name, arguments)
        records = _records_from_result(result)

        if tool_name == "get_sap_table_data" and not records:
            return {"status": "no_data", "message": "No records found"}

        if self._is_write_intent(user_input) and records:
            table_name = normalize_sap_table_name(arguments.get("table_name", "sap_table"))
            write_result = self._blob_service.write_parquet(
                data=records,
                blob_path=f"sap_tables/{table_name}/{table_name}.parquet",
            )
            return {"fetch_result_count": len(records), "azure_blob_write": write_result}

        return result
