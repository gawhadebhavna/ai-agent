from __future__ import annotations

import asyncio
import json
import os
import subprocess
from typing import Any

from app.config import Settings


class AzureMCPClient:
    """
    Wrapper for Azure MCP Server using stdio transport.
    
    This client manages communication with the Azure MCP Server process,
    which provides access to 43+ Azure services through standardized tools.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._process: subprocess.Popen | None = None
        self._tools_cache: dict[str, Any] = {}
        self._initialized = False
        self._subscription_id = settings.azure_subscription_id

    async def initialize(self) -> None:
        """Initialize connection to Azure MCP Server."""
        if self._initialized:
            return

        try:
            # Pass Azure credentials from settings to MCP server environment
            env = os.environ.copy()
            
            if self._settings.azure_token_credentials:
                env["AZURE_TOKEN_CREDENTIALS"] = self._settings.azure_token_credentials
            if self._settings.azure_tenant_id:
                env["AZURE_TENANT_ID"] = self._settings.azure_tenant_id
            if self._settings.azure_client_id:
                env["AZURE_CLIENT_ID"] = self._settings.azure_client_id
            if self._settings.azure_client_secret:
                env["AZURE_CLIENT_SECRET"] = self._settings.azure_client_secret
            if self._settings.azure_subscription_id:
                env["AZURE_SUBSCRIPTION_ID"] = self._settings.azure_subscription_id

            # Start MCP server process with configured authentication
            self._process = subprocess.Popen(
                ["npx.cmd", "-y", "@azure/mcp@latest", "server", "start"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )

            # Send initialization request
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "core_ai_tool", "version": "0.1.0"},
                },
            }

            self._send_request(init_request)
            response = self._read_response()

            if response and "result" in response:
                # List available tools
                await self._list_tools()
                self._initialized = True
            else:
                raise RuntimeError("Failed to initialize Azure MCP Server")

        except FileNotFoundError:
            raise RuntimeError(
                "npx.cmd not found. Please install Node.js (https://nodejs.org/)"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Azure MCP Server: {e}")

    async def _list_tools(self) -> None:
        """Fetch and cache available tools from MCP server."""
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }

        self._send_request(list_tools_request)
        response = self._read_response()

        if response and "result" in response and "tools" in response["result"]:
            for tool in response["result"]["tools"]:
                self._tools_cache[tool["name"]] = tool

    def _send_request(self, request: dict[str, Any]) -> None:
        """Send JSON-RPC request to MCP server."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("MCP server process not initialized")

        request_str = json.dumps(request) + "\n"
        self._process.stdin.write(request_str)
        self._process.stdin.flush()

    def _read_response(self) -> dict[str, Any] | None:
        """Read JSON-RPC response from MCP server."""
        if not self._process or not self._process.stdout:
            return None

        try:
            response_line = self._process.stdout.readline()
            if response_line:
                return json.loads(response_line)
        except json.JSONDecodeError:
            pass

        return None

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute an MCP tool with the given arguments.
        
        Args:
            tool_name: Name of the MCP tool to execute
            arguments: Dictionary of arguments for the tool
            
        Returns:
            Tool execution result
        """
        if not self._initialized:
            await self.initialize()

        if tool_name not in self._tools_cache:
            raise ValueError(
                f"Tool '{tool_name}' not found. Available tools: {list(self._tools_cache.keys())}"
            )

        call_tool_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        self._send_request(call_tool_request)
        response = self._read_response()

        if response and "result" in response:
            return response["result"]
        elif response and "error" in response:
            raise RuntimeError(
                f"MCP tool error: {response['error'].get('message', 'Unknown error')}"
            )
        else:
            raise RuntimeError("No response from MCP server")

    def get_available_tools(self) -> list[str]:
        """Get list of available MCP tool names."""
        return list(self._tools_cache.keys())

    def get_tool_schema(self, tool_name: str) -> dict[str, Any] | None:
        """Get the schema/description for a specific tool."""
        return self._tools_cache.get(tool_name)

    def close(self) -> None:
        """Cleanup and close the MCP server connection."""
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=5)
            self._process = None
        self._initialized = False

    def __del__(self):
        """Ensure process cleanup on deletion."""
        self.close()
