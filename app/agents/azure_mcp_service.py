from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

UTC = timezone.utc

from app.agents.databricks_tools import build_databricks_tools
from app.agents.mcp_client import AzureMCPClient
from app.agents.mcp_planner import AzureMCPRequestPlanner
from app.agents.mcp_tools import build_mcp_tools
from app.config import Settings
from app.persistence.approval_repository import ApprovalRepository
from app.schemas.agent import AgentS3Response, AgentStatus
from app.schemas.azure_mcp_agent import AzureMCPAgentRequest


class AzureMCPAgentService:
    """
    Agent service powered by Azure MCP Server and Databricks SDK.
    
    Provides access to 43+ Azure services through the Model Context Protocol,
    plus Databricks cluster, notebook, and job operations with approval workflow.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        approval_repository: ApprovalRepository,
    ) -> None:
        self._settings = settings
        self._approval_repository = approval_repository

        # Initialize MCP client and planner
        self._mcp_client = AzureMCPClient(settings)
        self._planner = AzureMCPRequestPlanner(settings)
        
        # Build MCP-based tools
        self._tools = build_mcp_tools(self._mcp_client)
        
        # Add Databricks tools if credentials are configured
        dbx_tools = build_databricks_tools(settings)
        self._tools.update(dbx_tools)

    def handle(self, request: AzureMCPAgentRequest) -> AgentS3Response:
        """
        Handle Azure MCP agent request with approval workflow.
        
        Args:
            request: Azure MCP agent request with message and optional approval info
            
        Returns:
            Agent response with results or approval proposal
        """
        # If this is an approval follow-up, execute the stored action
        if request.approval_id and request.approve is not None:
            return self._handle_approval(request.approval_id, request.approve)

        # Otherwise, plan and execute/propose the action
        try:
            plan = self._planner.plan(message=request.message)
        except Exception as e:
            return AgentS3Response(
                status=AgentStatus.error,
                message=f"Failed to plan Azure operation: {str(e)}",
                result=None,
            )

        # Check if tool exists
        if plan.tool_name not in self._tools:
            return AgentS3Response(
                status=AgentStatus.error,
                message=f"Azure MCP tool '{plan.tool_name}' not found. Available tools: {list(self._tools.keys())}",
                result={"requested_tool": plan.tool_name, "rationale": plan.rationale},
            )

        # If write operation, create approval proposal
        if plan.requires_approval:
            return self._create_approval_proposal(plan, request.message)

        # Execute read operation immediately
        return self._execute_tool(plan)

    def _create_approval_proposal(self, plan, user_message: str) -> AgentS3Response:
        """Create an approval proposal for a write operation."""
        approval_id = str(uuid.uuid4())
        expires_at = datetime.now(UTC) + timedelta(minutes=self._settings.approval_ttl_minutes)
        
        arguments = plan.get_arguments()

        self._approval_repository.create_pending(
            approval_id=approval_id,
            provider=f"azure_mcp_agent:{self._settings.llm_provider}",
            tool_name=plan.tool_name,
            tool_args=arguments,
            summary=plan.rationale,
            user_message=user_message,
            expires_at=expires_at,
        )

        return AgentS3Response(
            status=AgentStatus.approval_required,
            message="This Azure write operation requires approval.",
            result=None,
            approval={
                "approval_id": approval_id,
                "summary": plan.rationale,
                "tool_name": plan.tool_name,
                "tool_args": arguments,
                "expires_at": expires_at.isoformat(),
            },
        )

    def _handle_approval(self, approval_id: str, approve: bool) -> AgentS3Response:
        """Handle approval or rejection of a pending operation."""
        record = self._approval_repository.get(approval_id)
        
        if not record:
            return AgentS3Response(
                status=AgentStatus.error,
                message=f"Approval ID {approval_id} not found or expired.",
                result=None,
            )

        if not approve:
            self._approval_repository.delete(approval_id)
            return AgentS3Response(
                status=AgentStatus.rejected,
                message="The Azure operation was rejected by the user.",
                result=None,
            )

        # Execute the approved operation
        tool_name = record["tool_name"]
        tool_args = record["tool_args"]

        if tool_name not in self._tools:
            return AgentS3Response(
                status=AgentStatus.error,
                message=f"Tool '{tool_name}' not found.",
                result=None,
            )

        tool = self._tools[tool_name]
        
        try:
            result = tool.invoke(tool_args)
            self._approval_repository.delete(approval_id)
            
            return AgentS3Response(
                status=AgentStatus.completed,
                message=f"Successfully executed Azure operation: {record.get('summary', tool_name)}",
                result=result,
            )
        except Exception as e:
            return AgentS3Response(
                status=AgentStatus.error,
                message=f"Failed to execute approved operation: {str(e)}",
                result={"error": str(e), "tool_name": tool_name},
            )

    def _execute_tool(self, plan) -> AgentS3Response:
        """Execute a tool immediately (for read operations)."""
        tool = self._tools[plan.tool_name]
        arguments = plan.get_arguments()
        
        try:
            result = tool.invoke(arguments)
            return AgentS3Response(
                status=AgentStatus.completed,
                message=plan.rationale,
                result=result,
            )
        except Exception as e:
            return AgentS3Response(
                status=AgentStatus.error,
                message=f"Failed to execute Azure operation: {str(e)}",
                result={"error": str(e), "tool_name": plan.tool_name, "arguments": arguments},
            )

    async def initialize(self) -> None:
        """Initialize the MCP client connection."""
        await self._mcp_client.initialize()

    def close(self) -> None:
        """Cleanup MCP client resources."""
        self._mcp_client.close()
