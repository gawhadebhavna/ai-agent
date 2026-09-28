from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import StructuredTool

from app.core.logging import get_logger, log_tool_call

logger = get_logger("migration.tool_resolver")


def resolve_tool(
    name: str,
    mcp_tools: dict[str, StructuredTool],
    sdk_tools: dict[str, StructuredTool],
    *,
    phase: str = "SYSTEM",
    session_id: str = "-",
) -> StructuredTool | None:
    """
    MCP-first tool resolution.

    Checks MCP tools first; falls back to SDK tools.
    Logs which path was taken for every resolution.

    Args:
        name: Tool name to resolve.
        mcp_tools: Available MCP-backed tools.
        sdk_tools: Available SDK-backed tools.
        phase: Current migration phase (for logging).
        session_id: Current session id (for logging).

    Returns:
        Resolved StructuredTool or None if not found in either set.
    """
    if name in mcp_tools:
        logger.info(
            "[TOOL] %s → using MCP",
            name,
            extra={"phase": phase, "session_id": session_id},
        )
        return mcp_tools[name]

    if name in sdk_tools:
        logger.info(
            "[TOOL] %s → MCP not available, using SDK",
            name,
            extra={"phase": phase, "session_id": session_id},
        )
        return sdk_tools[name]

    logger.warning(
        "[TOOL] %s → NOT FOUND in MCP or SDK",
        name,
        extra={"phase": phase, "session_id": session_id},
    )
    return None


def invoke_tool(
    name: str,
    args: dict[str, Any],
    mcp_tools: dict[str, StructuredTool],
    sdk_tools: dict[str, StructuredTool],
    *,
    phase: str = "SYSTEM",
    session_id: str = "-",
) -> dict[str, Any]:
    """
    Resolve and invoke a tool with MCP-first precedence.

    Args:
        name: Tool name.
        args: Arguments to pass to the tool.
        mcp_tools: Available MCP-backed tools.
        sdk_tools: Available SDK-backed tools.
        phase: Current migration phase.
        session_id: Current session id.

    Returns:
        Tool invocation result as a dict.

    Raises:
        ValueError: If the tool is not found in either set.
    """
    tool = resolve_tool(name, mcp_tools, sdk_tools, phase=phase, session_id=session_id)
    if tool is None:
        raise ValueError(f"Tool '{name}' not found in MCP or SDK tool sets.")

    source = "MCP" if name in mcp_tools else "SDK"
    log_tool_call(logger, name, args, source=source, phase=phase, session_id=session_id)
    result = tool.invoke(args)
    if not isinstance(result, dict):
        result = {"result": result}
    return result


def build_merged_tools(
    mcp_tools: dict[str, StructuredTool],
    sdk_tools: dict[str, StructuredTool],
) -> dict[str, StructuredTool]:
    """
    Merge MCP and SDK tools with MCP taking precedence on name collision.

    Returns a unified tool dict safe to pass to a LangGraph agent.
    """
    merged: dict[str, StructuredTool] = {**sdk_tools}
    merged.update(mcp_tools)  # MCP overwrites SDK on conflict
    return merged
