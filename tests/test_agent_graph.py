from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from langgraph.types import Command

from app.agents.graph import AgentGraphBuilder
from app.agents import providers as provider_module
from app.config import Settings
from app.schemas.agent import ActionPlan, RawActionPlan, ToolName
from app.agents.providers import LLMRequestPlanner


class StubPlanner:
    def __init__(self, plans: dict[str, ActionPlan]) -> None:
        self._plans = plans

    def plan(self, *, message: str, intent_hint: str) -> ActionPlan:
        return self._plans[message]


class FakeTool:
    def __init__(self, result: dict):
        self.result = result
        self.calls: list[dict] = []

    def invoke(self, payload: dict):
        self.calls.append(payload)
        return self.result


class FailingPlanner:
    def plan(self, *, message: str, intent_hint: str):
        raise ValueError("bucket is required for the selected tool")


class RawStubModel:
    def __init__(self, raw_plan: RawActionPlan) -> None:
        self._raw_plan = raw_plan

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _payload):
        return self._raw_plan


class RawStubPrompt:
    def __init__(self, raw_plan: RawActionPlan) -> None:
        self._raw_plan = raw_plan

    def __or__(self, model):
        return model


def build_settings(tmp_path: Path) -> Settings:
    return Settings(sqlite_path=str(tmp_path / "graph.db"), approval_ttl_minutes=15)


def test_read_request_executes_without_approval(tmp_path: Path):
    planner = StubPlanner(
        {
            "read file": ActionPlan(
                tool_name=ToolName.read_object,
                summary="Read the requested object.",
                rationale="The user asked to read a file.",
                bucket="landing-bucket",
                key="landing/file.json",
            )
        }
    )
    read_tool = FakeTool({"bucket": "landing-bucket", "key": "landing/file.json", "content": "{}"})
    graph = AgentGraphBuilder(
        planner=planner,
        tools={
            "s3_read_object": read_tool,
            "s3_list_buckets": FakeTool({}),
            "s3_list_objects": FakeTool({}),
            "s3_write_object": FakeTool({}),
        },
        settings=build_settings(tmp_path),
    ).compile()

    result = graph.invoke({"message": "read file"}, config={"configurable": {"thread_id": "read-1"}})

    assert result["final_response"]["status"] == "completed"
    assert result["final_response"]["result"]["content"] == "{}"
    assert read_tool.calls == [{"bucket": "landing-bucket", "key": "landing/file.json"}]


def test_write_request_interrupts_for_approval(tmp_path: Path):
    planner = StubPlanner(
        {
            "write file": ActionPlan(
                tool_name=ToolName.write_object,
                summary="Write the provided content to S3.",
                rationale="The user explicitly requested a write.",
                bucket="landing-bucket",
                key="landing/output.json",
                content='{"ok":true}',
                content_type="application/json",
            )
        }
    )
    graph = AgentGraphBuilder(
        planner=planner,
        tools={
            "s3_read_object": FakeTool({}),
            "s3_list_buckets": FakeTool({}),
            "s3_list_objects": FakeTool({}),
            "s3_write_object": FakeTool({"message": "written"}),
        },
        settings=build_settings(tmp_path),
    ).compile()

    result = graph.invoke({"message": "write file"}, config={"configurable": {"thread_id": "write-1"}})

    interrupt_value = result["__interrupt__"][0].value
    assert interrupt_value["tool_name"] == "s3_write_object"
    assert interrupt_value["tool_args"]["key"] == "landing/output.json"


def test_write_request_resumes_after_approval(tmp_path: Path):
    planner = StubPlanner(
        {
            "write file": ActionPlan(
                tool_name=ToolName.write_object,
                summary="Write the provided content to S3.",
                rationale="The user explicitly requested a write.",
                bucket="landing-bucket",
                key="landing/output.json",
                content='{"ok":true}',
                content_type="application/json",
            )
        }
    )
    write_tool = FakeTool({"message": "written"})
    graph = AgentGraphBuilder(
        planner=planner,
        tools={
            "s3_read_object": FakeTool({}),
            "s3_list_buckets": FakeTool({}),
            "s3_list_objects": FakeTool({}),
            "s3_write_object": write_tool,
        },
        settings=build_settings(tmp_path),
    ).compile()

    thread_id = "write-2"
    graph.invoke({"message": "write file"}, config={"configurable": {"thread_id": thread_id}})
    resumed = graph.invoke(Command(resume=True), config={"configurable": {"thread_id": thread_id}})

    assert resumed["final_response"]["status"] == "completed"
    assert write_tool.calls == [
        {
            "bucket": "landing-bucket",
            "key": "landing/output.json",
            "content": '{"ok":true}',
            "content_type": "application/json",
            "metadata": {},
        }
    ]


def test_planner_failure_returns_error_response(tmp_path: Path):
    graph = AgentGraphBuilder(
        planner=FailingPlanner(),
        tools={
            "s3_read_object": FakeTool({}),
            "s3_list_buckets": FakeTool({}),
            "s3_list_objects": FakeTool({}),
            "s3_write_object": FakeTool({}),
        },
        settings=build_settings(tmp_path),
    ).compile()

    result = graph.invoke({"message": "write to s3"}, config={"configurable": {"thread_id": "fail-1"}})

    assert result["final_response"]["status"] == "error"
    assert "bucket name, object key, and content" in result["final_response"]["message"]


def test_normalizer_uses_explicit_bucket_and_key_from_prompt():
    settings = Settings(
        allowed_buckets="agentic-ai-migration-bkt",
        allowed_prefixes_json='{"agentic-ai-migration-bkt":["landing/"]}',
    )
    raw_plan = RawActionPlan(
        tool_name=ToolName.write_object,
        summary="Write the requested file to S3.",
        rationale="The user requested a write operation.",
        content='{"status":"ok"}',
    )

    with patch.object(provider_module.LLMRequestPlanner, "_build_model", return_value=RawStubModel(raw_plan)):
        planner = LLMRequestPlanner(settings)
        planner._prompt = RawStubPrompt(raw_plan)

    plan = planner.plan(
        message=(
            'Write a file to bucket agentic-ai-migration-bkt key landing/agent-test.json '
            'with content {"status":"ok"}'
        ),
        intent_hint="write",
    )

    assert plan.bucket == "agentic-ai-migration-bkt"
    assert plan.key == "landing/agent-test.json"
    assert plan.content == '{"status":"ok"}'


def test_normalizer_composes_prefix_and_filename():
    settings = Settings(
        allowed_buckets="agentic-ai-migration-bkt",
        allowed_prefixes_json='{"agentic-ai-migration-bkt":["landing/"]}',
    )
    raw_plan = RawActionPlan(
        tool_name=ToolName.write_object,
        summary="Write the requested file to S3.",
        rationale="The user requested a write operation.",
        bucket="agentic-ai-migration-bkt",
        prefix="landing/",
        filename="agent-test.json",
        content='{"status":"ok"}',
    )

    with patch.object(provider_module.LLMRequestPlanner, "_build_model", return_value=RawStubModel(raw_plan)):
        planner = LLMRequestPlanner(settings)
        planner._prompt = RawStubPrompt(raw_plan)

    plan = planner.plan(
        message='Write file agent-test.json under landing/ in bucket agentic-ai-migration-bkt with content {"status":"ok"}',
        intent_hint="write",
    )

    assert plan.bucket == "agentic-ai-migration-bkt"
    assert plan.key == "landing/agent-test.json"


def test_normalizer_falls_back_to_single_allowed_bucket():
    settings = Settings(
        allowed_buckets="agentic-ai-migration-bkt",
        allowed_prefixes_json='{"agentic-ai-migration-bkt":["landing/"]}',
    )
    raw_plan = RawActionPlan(
        tool_name=ToolName.write_object,
        summary="Write the requested file to S3.",
        rationale="The user requested a write operation.",
        key="landing/agent-test.json",
        content='{"status":"ok"}',
    )

    with patch.object(provider_module.LLMRequestPlanner, "_build_model", return_value=RawStubModel(raw_plan)):
        planner = LLMRequestPlanner(settings)
        planner._prompt = RawStubPrompt(raw_plan)

    plan = planner.plan(
        message='Write this JSON to landing/agent-test.json with content {"status":"ok"}',
        intent_hint="write",
    )

    assert plan.bucket == "agentic-ai-migration-bkt"
    assert plan.key == "landing/agent-test.json"


def test_normalizer_returns_targeted_error_for_missing_content():
    settings = Settings(
        allowed_buckets="agentic-ai-migration-bkt",
        allowed_prefixes_json='{"agentic-ai-migration-bkt":["landing/"]}',
    )
    raw_plan = RawActionPlan(
        tool_name=ToolName.write_object,
        summary="Write the requested file to S3.",
        rationale="The user requested a write operation.",
        bucket="agentic-ai-migration-bkt",
        key="landing/agent-test.json",
    )

    with patch.object(provider_module.LLMRequestPlanner, "_build_model", return_value=RawStubModel(raw_plan)):
        planner = LLMRequestPlanner(settings)
        planner._prompt = RawStubPrompt(raw_plan)

    plan = planner.plan(
        message="Write a file to bucket agentic-ai-migration-bkt key landing/agent-test.json",
        intent_hint="write",
    )

    assert plan.tool_name == ToolName.unsupported
    assert "content" in (plan.final_response or "")
