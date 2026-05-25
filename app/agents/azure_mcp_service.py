from __future__ import annotations

from app.agents.graph import AgentGraphBuilder
from app.agents.mcp_client import AzureMCPClient
from app.agents.mcp_tools import build_mcp_tools
from app.agents.providers import LLMRequestPlanner
from app.agents.service import AgentService
from app.config import Settings
from app.persistence.approval_repository import ApprovalRepository
from app.schemas.agent import AgentS3Response
from app.schemas.azure_mcp_agent import AzureMCPAgentRequest


class AzureMCPAgentService:
    """
    Agent service powered by Azure MCP Server.
    
    Provides access to 43+ Azure services through the Model Context Protocol,
    integrated with the existing LangGraph agent architecture and approval flow.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        approval_repository: ApprovalRepository,
    ) -> None:
        self._settings = settings
        self._approval_repository = approval_repository

        # Initialize MCP client
        self._mcp_client = AzureMCPClient(settings)

        # Build MCP-based tools
        tools = build_mcp_tools(self._mcp_client)

        # Use existing LLM request planner
        planner = LLMRequestPlanner(settings)

        # Build agent graph with MCP tools
        graph = AgentGraphBuilder(
            planner=planner,
            tools=tools,
            settings=settings,
        ).compile()

        # Wrap in existing AgentService to reuse approval flow
        self._agent_service = AgentService(
            graph=graph,
            settings=settings,
            approval_repository=approval_repository,
            provider_name=f"azure_mcp_agent:{settings.llm_provider.lower().strip()}",
        )

    def handle(self, request: AzureMCPAgentRequest) -> AgentS3Response:
        """
        Handle Azure MCP agent request.
        
        Delegates to the underlying AgentService which manages the LangGraph
        execution and approval flow.
        
        Args:
            request: Azure MCP agent request with message and optional approval info
            
        Returns:
            Agent response with results or approval proposal
        """
        # Convert to internal AgentS3Request format (reusing existing schema)
        from app.schemas.agent import AgentS3Request

        internal_request = AgentS3Request(
            message=request.message,
            approval_id=request.approval_id,
            approve=request.approve,
        )

        return self._agent_service.handle(internal_request)

    async def initialize(self) -> None:
        """Initialize the MCP client connection."""
        await self._mcp_client.initialize()

    def close(self) -> None:
        """Cleanup MCP client resources."""
        self._mcp_client.close()
