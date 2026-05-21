from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from app.agents.model_factory import build_runtime_chat_model
from app.schemas.azure_agent import (
    AzureBlobActionPlan,
    AzureToolName,
    RawAzureBlobActionPlan,
)

PLANNER_PROMPT = """You are an Azure Blob Storage planning assistant.
Your job is to convert a user's request into exactly one safe Azure Blob action.

Allowed tools:
- blob_list_blobs: list blob names, optionally filtered by prefix
- blob_read_text: read blob content as UTF-8 text
- blob_read_json: read and parse blob content as JSON
- blob_write_text: write plain text or CSV content
- blob_write_json: write JSON data
- blob_write_parquet: write tabular records as parquet
- unsupported: use when the request is missing required details

Safety rules:
- Only return one action plan.
- Read/list actions do not require approval.
- All write operations require approval.
- For blob_list_blobs, include prefix if provided and max_results when requested.
- For blob_read_text and blob_read_json, include blob_path.
- For write actions, include blob_path and the required payload fields.
- For blob_write_text include content and content_type if known.
- For blob_write_json include data as JSON-serializable object/array.
- For blob_write_parquet include data as a list of record objects.
- If required fields are missing, return unsupported and clearly explain missing fields.

User request: {message}
"""


class AzureBlobRequestPlanner:
    def __init__(self, settings) -> None:
        self._model = build_runtime_chat_model(settings)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PLANNER_PROMPT),
                ("human", "Return the best single Azure Blob action plan."),
            ]
        )

    def plan(self, *, message: str) -> AzureBlobActionPlan:
        chain = self._prompt | self._with_structured_output()
        raw_plan = chain.invoke({"message": message})
        return self._normalize_plan(raw_plan=raw_plan, message=message)

    def _with_structured_output(self):
        try:
            return self._model.with_structured_output(
                RawAzureBlobActionPlan,
                method="function_calling",
            )
        except TypeError:
            # Backward-compatible fallback for older model wrappers/tests
            # that do not accept the `method` kwarg.
            return self._model.with_structured_output(RawAzureBlobActionPlan)

    def _normalize_plan(self, *, raw_plan: RawAzureBlobActionPlan, message: str) -> AzureBlobActionPlan:
        blob_path = raw_plan.blob_path or self._extract_blob_path_from_message(message)
        prefix = raw_plan.prefix or self._extract_prefix_from_message(message)
        max_results = raw_plan.max_results or self._extract_max_results_from_message(message) or 100
        content = raw_plan.content or self._extract_content_from_message(message)
        data = raw_plan.data if raw_plan.data is not None else self._extract_json_from_message(message)

        if raw_plan.tool_name == AzureToolName.list_blobs:
            return AzureBlobActionPlan(
                tool_name=raw_plan.tool_name,
                summary=raw_plan.summary,
                rationale=raw_plan.rationale,
                prefix=prefix,
                max_results=max(1, min(int(max_results), 1000)),
                final_response=raw_plan.final_response,
            )

        if raw_plan.tool_name in {AzureToolName.read_text, AzureToolName.read_json}:
            if not blob_path:
                return self._build_unsupported_plan(missing_fields=["blob_path"], rationale=raw_plan.rationale)
            return AzureBlobActionPlan(
                tool_name=raw_plan.tool_name,
                summary=raw_plan.summary,
                rationale=raw_plan.rationale,
                blob_path=blob_path,
                final_response=raw_plan.final_response,
            )

        if raw_plan.tool_name == AzureToolName.write_parquet and isinstance(data, dict):
            data = [data]

        if raw_plan.tool_name == AzureToolName.write_json and data is None and content:
            data = self._parse_json_string(content)

        if raw_plan.tool_name == AzureToolName.write_text:
            missing_fields: list[str] = []
            if not blob_path:
                missing_fields.append("blob_path")
            if content is None:
                missing_fields.append("content")
            if missing_fields:
                return self._build_unsupported_plan(missing_fields=missing_fields, rationale=raw_plan.rationale)
            return AzureBlobActionPlan(
                tool_name=raw_plan.tool_name,
                summary=raw_plan.summary,
                rationale=raw_plan.rationale,
                blob_path=blob_path,
                content=content,
                content_type=raw_plan.content_type or self._infer_content_type(blob_path=blob_path, content=content),
                final_response=raw_plan.final_response,
            )

        if raw_plan.tool_name == AzureToolName.write_json:
            missing_fields = []
            if not blob_path:
                missing_fields.append("blob_path")
            if data is None:
                missing_fields.append("data")
            elif not self._is_json_payload(data):
                missing_fields.append("data as JSON object or array of objects")
            if missing_fields:
                return self._build_unsupported_plan(missing_fields=missing_fields, rationale=raw_plan.rationale)
            return AzureBlobActionPlan(
                tool_name=raw_plan.tool_name,
                summary=raw_plan.summary,
                rationale=raw_plan.rationale,
                blob_path=blob_path,
                data=data,
                final_response=raw_plan.final_response,
            )

        if raw_plan.tool_name == AzureToolName.write_parquet:
            missing_fields = []
            if not blob_path:
                missing_fields.append("blob_path")
            if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
                missing_fields.append("data as list[record]")
            if missing_fields:
                return self._build_unsupported_plan(missing_fields=missing_fields, rationale=raw_plan.rationale)
            return AzureBlobActionPlan(
                tool_name=raw_plan.tool_name,
                summary=raw_plan.summary,
                rationale=raw_plan.rationale,
                blob_path=blob_path,
                data=data,
                final_response=raw_plan.final_response,
            )

        fallback_plan = self._infer_read_or_list_plan(
            raw_plan=raw_plan,
            message=message,
            blob_path=blob_path,
            prefix=prefix,
            max_results=max_results,
        )
        if fallback_plan is not None:
            return fallback_plan

        return AzureBlobActionPlan(
            tool_name=AzureToolName.unsupported,
            summary=raw_plan.summary,
            rationale=raw_plan.rationale,
            final_response=raw_plan.final_response
            or "I could not safely plan that Azure Blob request. Please include blob_path and payload.",
        )

    @staticmethod
    def _extract_blob_path_from_message(message: str) -> str | None:
        patterns = [
            r"\bblob[_ ]?path\s+(?:is\s+)?([A-Za-z0-9!_\-./]+)",
            r"\bpath\s+(?:is\s+)?([A-Za-z0-9!_\-./]+)",
            r"\b(?:read|from|get|open|download)\s+([A-Za-z0-9!_\-./]+\.(?:txt|csv|json|parquet))",
            r"\bto\s+([A-Za-z0-9!_\-./]+\.(?:txt|csv|json|parquet))",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().strip('"').strip("'")
        return None

    @staticmethod
    def _extract_content_from_message(message: str) -> str | None:
        patterns = [
            r"\bcontent(?: is)?\s*:\s*(.+)$",
            r"\bcontent(?: is)?\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _extract_prefix_from_message(message: str) -> str | None:
        patterns = [
            r"\bprefix\s+(?:is\s+)?([A-Za-z0-9!_\-./]+)",
            r"\bunder\s+([A-Za-z0-9!_\-./]+/)",
            r"\bin\s+folder\s+([A-Za-z0-9!_\-./]+/)",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().strip('"').strip("'")
        return None

    @staticmethod
    def _extract_max_results_from_message(message: str) -> int | None:
        patterns = [
            r"\btop\s+(\d{1,4})\b",
            r"\bfirst\s+(\d{1,4})\b",
            r"\bmax(?:imum)?\s+(\d{1,4})\b",
            r"\blimit\s+(\d{1,4})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if not match:
                continue
            value = int(match.group(1))
            if value > 0:
                return value
        return None

    @classmethod
    def _extract_json_from_message(cls, message: str) -> Any | None:
        json_candidate = cls._extract_content_from_message(message)
        if json_candidate:
            parsed = cls._parse_json_string(json_candidate)
            if parsed is not None:
                return parsed

        for pattern in [r"\{.*\}", r"\[.*\]"]:
            match = re.search(pattern, message, flags=re.DOTALL)
            if not match:
                continue
            parsed = cls._parse_json_string(match.group(0).strip())
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _parse_json_string(raw: str) -> Any | None:
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _is_json_payload(data: Any) -> bool:
        if isinstance(data, dict):
            return True
        return isinstance(data, list) and all(isinstance(item, dict) for item in data)

    def _infer_read_or_list_plan(
        self,
        *,
        raw_plan: RawAzureBlobActionPlan,
        message: str,
        blob_path: str | None,
        prefix: str | None,
        max_results: int,
    ) -> AzureBlobActionPlan | None:
        text = message.lower()
        list_tokens = {"list", "enumerate", "show blobs", "what blobs", "which blobs"}
        read_tokens = {"read", "download", "fetch", "get content", "open"}

        if any(token in text for token in list_tokens):
            return AzureBlobActionPlan(
                tool_name=AzureToolName.list_blobs,
                summary="List blobs in the configured Azure container.",
                rationale=raw_plan.rationale or "Inferred list intent from user request keywords.",
                prefix=prefix,
                max_results=max(1, min(int(max_results), 1000)),
                final_response=raw_plan.final_response,
            )

        if any(token in text for token in read_tokens):
            if not blob_path:
                return self._build_unsupported_plan(
                    missing_fields=["blob_path"],
                    rationale=raw_plan.rationale or "Inferred read intent from user request keywords.",
                )
            tool_name = AzureToolName.read_json if blob_path.lower().endswith(".json") else AzureToolName.read_text
            return AzureBlobActionPlan(
                tool_name=tool_name,
                summary=f"Read blob content from {blob_path}.",
                rationale=raw_plan.rationale or "Inferred read intent from user request keywords.",
                blob_path=blob_path,
                final_response=raw_plan.final_response,
            )

        return None

    @staticmethod
    def _infer_content_type(*, blob_path: str | None, content: str | None) -> str:
        lower_path = (blob_path or "").lower()
        if lower_path.endswith(".csv"):
            return "text/csv"
        if lower_path.endswith(".json"):
            return "application/json"
        if content and (content.strip().startswith("{") or content.strip().startswith("[")):
            return "application/json"
        return "text/plain"

    @staticmethod
    def _build_unsupported_plan(*, missing_fields: list[str], rationale: str) -> AzureBlobActionPlan:
        missing_text = ", ".join(missing_fields)
        return AzureBlobActionPlan(
            tool_name=AzureToolName.unsupported,
            summary="I need a more explicit Azure Blob request before I can continue.",
            rationale=f"{rationale} Missing required field(s): {missing_text}.",
            final_response=(
                "I could not safely plan that Azure Blob action. "
                f"Please specify the following field(s): {missing_text}."
            ),
        )
