from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate

from app.agents.model_factory import build_runtime_chat_model
from app.config import Settings
from app.schemas.agent import ActionPlan, RawActionPlan, ToolName

PLANNER_PROMPT = """You are an AWS S3 planning assistant.
Your job is to convert a user's natural-language request into exactly one safe S3 action.

Allowed tools:
- s3_list_buckets: use when the user wants buckets
- s3_list_objects: use when the user wants object names under a bucket or prefix
- s3_read_object: use when the user wants file content from S3
- s3_write_object: use when the user explicitly wants to create, upload, save, or overwrite an S3 object
- unsupported: use when the request is missing information or is outside S3 scope

Safety rules:
- Only choose s3_write_object for an explicit write or upload request.
- Writes require approval.
- If the request is ambiguous or missing bucket, key, or content, you may still choose s3_write_object, but you must return any fields you can identify explicitly.
- Never hide bucket, key, prefix, filename, or content only inside the summary. Put them in dedicated fields.
- Keep the summary concise and business-friendly.

Allowed buckets: {allowed_buckets}
Allowed prefixes by bucket: {allowed_prefixes}
Intent hint: {intent_hint}
User request: {message}

Examples:
1. User request: Write a file to bucket agentic-ai-migration-bkt key landing/agent-test.json with content {{"status":"ok"}}
Return: bucket=agentic-ai-migration-bkt, key=landing/agent-test.json, content={{"status":"ok"}}

2. User request: Write file agent-test.json under landing/ in bucket agentic-ai-migration-bkt with content hello
Return: bucket=agentic-ai-migration-bkt, prefix=landing/, filename=agent-test.json, content=hello
"""


class LLMRequestPlanner:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = self._build_model()
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PLANNER_PROMPT),
                ("human", "Return the best single action plan for this request."),
            ]
        )

    @property
    def provider_name(self) -> str:
        return self._settings.llm_provider

    def plan(self, *, message: str, intent_hint: str) -> ActionPlan:
        chain = self._prompt | self._model.with_structured_output(RawActionPlan)
        raw_plan = chain.invoke(
            {
                "message": message,
                "intent_hint": intent_hint,
                "allowed_buckets": self._settings.allowed_bucket_list or ["<not configured>"],
                "allowed_prefixes": self._settings.allowed_prefixes or {"<not configured>": []},
            }
        )
        return self._normalize_plan(raw_plan=raw_plan, message=message)

    def _normalize_plan(self, *, raw_plan: RawActionPlan, message: str) -> ActionPlan:
        bucket = self._recover_bucket(raw_plan=raw_plan, message=message)
        prefix = self._normalize_prefix(raw_plan.prefix or self._extract_prefix_from_message(message))
        filename = raw_plan.filename or self._extract_filename_from_message(message)
        key = self._recover_key(raw_plan=raw_plan, message=message, prefix=prefix, filename=filename)
        content = raw_plan.content or self._extract_content_from_message(message)
        content_type = raw_plan.content_type or self._infer_content_type(content)

        if raw_plan.tool_name == ToolName.write_object:
            missing_fields: list[str] = []
            if not bucket:
                missing_fields.append("bucket")
            if not key:
                missing_fields.append("object key or filename")
            if content is None:
                missing_fields.append("content")
            if missing_fields:
                return self._build_unsupported_plan(
                    missing_fields=missing_fields,
                    rationale=raw_plan.rationale,
                )

        if raw_plan.tool_name == ToolName.list_objects and not bucket:
            return self._build_unsupported_plan(
                missing_fields=["bucket"],
                rationale=raw_plan.rationale,
            )

        if raw_plan.tool_name == ToolName.read_object:
            read_bucket = bucket
            read_key = key or raw_plan.key or self._extract_key_from_message(message)
            missing_fields: list[str] = []
            if not read_bucket:
                missing_fields.append("bucket")
            if not read_key:
                missing_fields.append("object key")
            if missing_fields:
                return self._build_unsupported_plan(
                    missing_fields=missing_fields,
                    rationale=raw_plan.rationale,
                )
            return ActionPlan(
                tool_name=raw_plan.tool_name,
                summary=raw_plan.summary,
                rationale=raw_plan.rationale,
                bucket=read_bucket,
                key=read_key,
                prefix=prefix,
                final_response=raw_plan.final_response,
            )

        if raw_plan.tool_name == ToolName.list_objects:
            return ActionPlan(
                tool_name=raw_plan.tool_name,
                summary=raw_plan.summary,
                rationale=raw_plan.rationale,
                bucket=bucket,
                prefix=prefix,
                final_response=raw_plan.final_response,
            )

        if raw_plan.tool_name == ToolName.list_buckets:
            return ActionPlan(
                tool_name=raw_plan.tool_name,
                summary=raw_plan.summary,
                rationale=raw_plan.rationale,
                final_response=raw_plan.final_response,
            )

        if raw_plan.tool_name == ToolName.unsupported:
            return ActionPlan(
                tool_name=raw_plan.tool_name,
                summary=raw_plan.summary,
                rationale=raw_plan.rationale,
                final_response=raw_plan.final_response
                or "I could not safely plan that S3 action. Please specify the bucket name, object key, and content clearly.",
            )

        return ActionPlan(
            tool_name=raw_plan.tool_name,
            summary=raw_plan.summary,
            rationale=raw_plan.rationale,
            bucket=bucket,
            key=key,
            prefix=prefix,
            content=content,
            content_type=content_type,
            metadata=raw_plan.metadata,
            final_response=raw_plan.final_response,
        )

    def _recover_bucket(self, *, raw_plan: RawActionPlan, message: str) -> str | None:
        if raw_plan.bucket:
            return raw_plan.bucket.strip()
        prompt_bucket = self._extract_bucket_from_message(message)
        if prompt_bucket:
            return prompt_bucket
        if len(self._settings.allowed_bucket_list) == 1:
            return self._settings.allowed_bucket_list[0]
        return None

    def _recover_key(
        self,
        *,
        raw_plan: RawActionPlan,
        message: str,
        prefix: str | None,
        filename: str | None,
    ) -> str | None:
        if raw_plan.key:
            return self._normalize_key(raw_plan.key)
        prompt_key = self._extract_key_from_message(message)
        if prompt_key:
            return self._normalize_key(prompt_key)
        if prefix and filename:
            return self._normalize_key(f"{prefix}{filename}")
        if len(self._settings.allowed_bucket_list) == 1:
            configured_prefixes = self._settings.allowed_prefixes.get(self._settings.allowed_bucket_list[0], [])
            if len(configured_prefixes) == 1 and filename:
                return self._normalize_key(f"{configured_prefixes[0]}{filename}")
        return None

    @staticmethod
    def _normalize_prefix(prefix: str | None) -> str | None:
        if not prefix:
            return None
        clean = prefix.strip().strip('"').strip("'")
        if not clean:
            return None
        return clean if clean.endswith("/") else f"{clean}/"

    @staticmethod
    def _normalize_key(key: str | None) -> str | None:
        if not key:
            return None
        clean = key.strip().strip('"').strip("'")
        return clean.lstrip("/") or None

    @staticmethod
    def _extract_bucket_from_message(message: str) -> str | None:
        patterns = [
            r"\bbucket(?: name)?\s+(?:is\s+)?([a-z0-9][a-z0-9.\-]{1,61}[a-z0-9])",
            r"\bto\s+bucket\s+([a-z0-9][a-z0-9.\-]{1,61}[a-z0-9])",
            r"\bin\s+bucket\s+([a-z0-9][a-z0-9.\-]{1,61}[a-z0-9])",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _extract_key_from_message(message: str) -> str | None:
        patterns = [
            r"\bkey\s+(?:is\s+)?([A-Za-z0-9!_\-./]+)",
            r"\bobject key\s+(?:is\s+)?([A-Za-z0-9!_\-./]+)",
            r"\bto\s+([A-Za-z0-9!_\-./]+/[A-Za-z0-9!_\-.]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _extract_prefix_from_message(message: str) -> str | None:
        patterns = [
            r"\bunder\s+([A-Za-z0-9!_\-./]+/)",
            r"\bprefix\s+(?:is\s+)?([A-Za-z0-9!_\-./]+/)",
            r"\binto\s+([A-Za-z0-9!_\-./]+/)",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _extract_filename_from_message(message: str) -> str | None:
        patterns = [
            r"\bfile(?:name)?\s+(?:is\s+)?([A-Za-z0-9!_\-.]+\.[A-Za-z0-9]+)",
            r"\bnamed\s+([A-Za-z0-9!_\-.]+\.[A-Za-z0-9]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _extract_content_from_message(message: str) -> str | None:
        patterns = [
            r"\bcontent(?: is)?\s*:\s*(.+)$",
            r"\bcontent(?: is)?\s+(.+)$",
            r"\bfile content(?: is)?\s*:\s*(.+)$",
            r"\bfile content(?: is)?\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _infer_content_type(content: str | None) -> str:
        if not content:
            return "application/json"
        stripped = content.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            return "application/json"
        return "text/plain"

    @staticmethod
    def _build_unsupported_plan(*, missing_fields: list[str], rationale: str) -> ActionPlan:
        missing_text = ", ".join(missing_fields)
        return ActionPlan(
            tool_name=ToolName.unsupported,
            summary="I need a more explicit S3 request before I can continue.",
            rationale=f"{rationale} Missing required field(s): {missing_text}.",
            final_response=(
                "I could not safely plan that S3 action. "
                f"Please specify the following field(s): {missing_text}."
            ),
        )

    def _build_model(self):
        return build_runtime_chat_model(self._settings)
