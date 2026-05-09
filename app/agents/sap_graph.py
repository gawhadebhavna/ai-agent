from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import httpx
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from typing_extensions import TypedDict

from app.config import get_settings
from app.schemas.sap_agent import SAPAgentStatus

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    SqliteSaver = None

# Tools whose invocation requires human approval before execution.
_WRITE_TOOL_NAMES = frozenset(["blob_write_text", "blob_write_json", "blob_write_parquet"])

_SYSTEM_PROMPT = """\
You are a SAP data assistant with access to tools for fetching SAP data and persisting it.

Available tool groups:
- Fetch tools: get_business_partners, get_products, get_sales_orders
- Write tools (require approval): blob_write_text, blob_write_json, blob_write_parquet

Workflow:
1. Fetch the requested SAP data using the appropriate fetch tool.
2. If the user wants to store the data, call the correct write tool with the fetched records.
   - Use blob_write_parquet for tabular data (default for SAP data).
   - Derive blob_path from the entity: customers -> customers/customers.parquet,
     products -> products/products.parquet, sales_orders -> sales_orders/sales_orders.parquet.
3. After all operations are complete, summarise what was done in plain language.

Never make up data. Only write data that was actually returned by a fetch tool.
"""


class _SAPAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    final_response: dict | None


class SAPGraphBuilder:
    """Builds a LangGraph ReAct agent for SAP data operations.

    The model drives all tool selection. Write tools are intercepted by an
    approval gate (LangGraph interrupt) before execution.

    Graph flow:
        START -> agent -> execute_tool (SAP fetch) -> agent -> ...
                       -> approval_gate (interrupt)  -> execute_tool (blob write) -> agent
                       -> finalize -> END
    """

    def __init__(self, *, tools: dict[str, Any], settings: Settings) -> None:
        self._tools = tools
        self._settings = settings

    def compile(self):
        llm = self._build_llm()
        bound_llm = llm.bind_tools(list(self._tools.values()))

        def agent(state: _SAPAgentState) -> dict:
            msgs = [SystemMessage(content=_SYSTEM_PROMPT)] + list(state["messages"])
            response = bound_llm.invoke(msgs)
            return {"messages": [response]}

        def route_from_agent(state: _SAPAgentState) -> str:
            last = state["messages"][-1]
            if not getattr(last, "tool_calls", None):
                return "finalize"
            tool_name = last.tool_calls[0]["name"]
            if tool_name in _WRITE_TOOL_NAMES:
                return "approval_gate"
            return "execute_tool"

        def approval_gate(state: _SAPAgentState) -> dict:
            last = state["messages"][-1]
            tool_call = last.tool_calls[0]
            expires_at = datetime.now(UTC) + timedelta(minutes=self._settings.approval_ttl_minutes)
            blob_path: str = tool_call["args"].get("blob_path", "")
            interrupt({
                "summary": (
                    f"Write data to Azure Blob Storage using {tool_call['name']}"
                    + (f" at '{blob_path}'" if blob_path else "")
                ),
                "tool_name": tool_call["name"],
                "tool_args": tool_call["args"],
                "expires_at": expires_at.isoformat(),
            })
            # Execution resumes here only when Command(resume=True) is sent.
            return {}

        def execute_tool(state: _SAPAgentState) -> dict:
            last = state["messages"][-1]
            tool_call = last.tool_calls[0]
            tool = self._tools[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            return {
                "messages": [
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                    )
                ]
            }

        def finalize(state: _SAPAgentState) -> dict:
            last = state["messages"][-1]
            content = last.content if isinstance(last.content, str) else str(last.content)
            return {
                "final_response": {
                    "status": SAPAgentStatus.completed,
                    "message": content,
                    "result": None,
                }
            }

        graph = StateGraph(_SAPAgentState)
        graph.add_node("agent", agent)
        graph.add_node("approval_gate", approval_gate)
        graph.add_node("execute_tool", execute_tool)
        graph.add_node("finalize", finalize)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            route_from_agent,
            {
                "execute_tool": "execute_tool",
                "approval_gate": "approval_gate",
                "finalize": "finalize",
            },
        )
        # After approval the graph resumes into execute_tool, then back to agent.
        graph.add_edge("approval_gate", "execute_tool")
        graph.add_edge("execute_tool", "agent")
        graph.add_edge("finalize", END)

        return graph.compile(checkpointer=self._build_checkpointer())

    def _build_llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_deployment=self._settings.azure_openai_deployment,
            azure_endpoint=self._settings.azure_openai_endpoint,
            api_key=self._settings.azure_openai_api_key,
            api_version=self._settings.azure_openai_api_version,
            http_client=httpx.Client(verify=self._settings.ssl_verify),
            temperature=0,
        )

    def _build_checkpointer(self):
        if SqliteSaver is None:
            return InMemorySaver()
        connection = sqlite3.connect(self._settings.sqlite_path_obj, check_same_thread=False)
        return SqliteSaver(connection)
