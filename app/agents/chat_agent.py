from __future__ import annotations

"""
Conversational migration agent.

A single tool-calling LLM loop (agent ⇄ tools) with:
  - full conversation memory per session via a LangGraph checkpointer
  - a fresh, authoritative state summary injected into the system prompt each turn
  - token-bounded history trimming so long chats stay within the context window
  - human-in-the-loop approvals via LangGraph interrupts inside provisioning tools
"""

from typing import Callable

from langchain_core.messages import SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

_MAX_HISTORY_TOKENS = 12_000


def build_chat_agent(
    llm,
    tools: list,
    build_system_prompt: Callable[[str], str],
    checkpointer,
    parallel_tool_calls: bool | None = None,
):
    bind_kwargs = {} if parallel_tool_calls is None else {"parallel_tool_calls": parallel_tool_calls}
    model = llm.bind_tools(tools, **bind_kwargs)

    def call_model(state: MessagesState, config: RunnableConfig) -> dict:
        session_id = config["configurable"]["thread_id"]
        history = trim_messages(
            state["messages"],
            max_tokens=_MAX_HISTORY_TOKENS,
            strategy="last",
            token_counter=count_tokens_approximately,
            start_on="human",
            include_system=False,
            allow_partial=False,
        )
        response = model.invoke([SystemMessage(content=build_system_prompt(session_id)), *history], config)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
