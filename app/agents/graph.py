from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.config import Settings
from app.schemas.agent import ActionPlan, AgentS3Response, AgentStatus

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    SqliteSaver = None


class AgentGraphBuilder:
    def __init__(self, *, planner, tools: dict[str, Any], settings: Settings) -> None:
        self._planner = planner
        self._tools = tools
        self._settings = settings

    def compile(self):
        graph = StateGraph(dict)
        graph.add_node("intake_request", self._intake_request)
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("select_tool", self._select_tool)
        graph.add_node("approval_gate", self._approval_gate)
        graph.add_node("execute_tool", self._execute_tool)
        graph.add_node("finalize_response", self._finalize_response)

        graph.add_edge(START, "intake_request")
        graph.add_edge("intake_request", "classify_intent")
        graph.add_edge("classify_intent", "select_tool")
        graph.add_conditional_edges(
            "select_tool",
            self._route_after_selection,
            {
                "approval_gate": "approval_gate",
                "execute_tool": "execute_tool",
                "finalize_response": "finalize_response",
            },
        )
        graph.add_conditional_edges(
            "approval_gate",
            self._route_after_approval,
            {
                "execute_tool": "execute_tool",
                "finalize_response": "finalize_response",
            },
        )
        graph.add_edge("execute_tool", "finalize_response")
        graph.add_edge("finalize_response", END)
        return graph.compile(checkpointer=self._build_checkpointer())

    def _build_checkpointer(self):
        if SqliteSaver is None:
            return InMemorySaver()
        connection = sqlite3.connect(self._settings.sqlite_path_obj, check_same_thread=False)
        return SqliteSaver(connection)

    def _intake_request(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"message": state["message"]}

    def _classify_intent(self, state: dict[str, Any]) -> dict[str, Any]:
        message = state["message"].lower()
        if any(word in message for word in ["write", "upload", "save", "create", "put", "overwrite"]):
            intent = "write"
        elif any(word in message for word in ["read", "show", "get", "list", "fetch", "download"]):
            intent = "read"
        else:
            intent = "unknown"
        return {"message": state["message"], "intent_hint": intent}

    def _select_tool(self, state: dict[str, Any]) -> dict[str, Any]:
        try:
            plan = self._planner.plan(message=state["message"], intent_hint=state["intent_hint"])
        except Exception as exc:
            plan = self._unsupported_plan(
                message=state["message"],
                reason=(
                    "I could not safely extract the required S3 fields from the request. "
                    "For write operations, include the bucket name, object key, and file content explicitly."
                ),
                error=str(exc),
            )
        return {"plan": plan.model_dump(mode="json")}

    def _route_after_selection(self, state: dict[str, Any]) -> Literal["approval_gate", "execute_tool", "finalize_response"]:
        plan = ActionPlan.model_validate(state["plan"])
        if plan.tool_name.value == "unsupported":
            return "finalize_response"
        if plan.requires_approval:
            return "approval_gate"
        return "execute_tool"

    def _approval_gate(self, state: dict[str, Any]) -> dict[str, Any]:
        plan = ActionPlan.model_validate(state["plan"])
        expires_at = datetime.now(UTC) + timedelta(minutes=self._settings.approval_ttl_minutes)
        approved = interrupt(
            {
                "summary": plan.summary,
                "tool_name": plan.tool_name.value,
                "tool_args": plan.tool_args(),
                "expires_at": expires_at.isoformat(),
            }
        )
        if approved:
            return {"approval_decision": "approved", "plan": state["plan"]}
        return {
            "approval_decision": "rejected",
            "plan": state["plan"],
            "final_response": AgentS3Response(
                status=AgentStatus.rejected,
                message="The requested write action was rejected.",
            ).model_dump(mode="json"),
        }

    def _route_after_approval(self, state: dict[str, Any]) -> Literal["execute_tool", "finalize_response"]:
        if state.get("approval_decision") == "approved":
            return "execute_tool"
        return "finalize_response"

    def _execute_tool(self, state: dict[str, Any]) -> dict[str, Any]:
        plan = ActionPlan.model_validate(state["plan"])
        tool = self._tools[plan.tool_name.value]
        return {"plan": state["plan"], "tool_result": tool.invoke(plan.tool_args())}

    def _finalize_response(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("final_response"):
            return state
        plan = ActionPlan.model_validate(state["plan"])
        if plan.tool_name.value == "unsupported":
            response = AgentS3Response(
                status=AgentStatus.error,
                message=plan.final_response or plan.summary,
                result={"rationale": plan.rationale},
            )
        else:
            response = AgentS3Response(
                status=AgentStatus.completed,
                message=plan.summary,
                result=state.get("tool_result"),
            )
        return {"final_response": response.model_dump(mode="json")}

    @staticmethod
    def _unsupported_plan(*, message: str, reason: str, error: str) -> ActionPlan:
        return ActionPlan(
            tool_name="unsupported",
            summary="I need a more explicit S3 request before I can continue.",
            rationale=f"{reason} Planner error: {error}",
            final_response=(
                "I could not safely plan that S3 action. "
                "Please specify the bucket name, object key, and content clearly."
            ),
        )
