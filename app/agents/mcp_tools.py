from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.agents.mcp_client import AzureMCPClient


class KeyVaultSecretsArgs(BaseModel):
    """Arguments for listing Key Vault secrets."""

    vault_name: str = Field(description="Name of the Azure Key Vault")


class KeyVaultSecretArgs(BaseModel):
    """Arguments for getting a Key Vault secret."""

    vault_name: str = Field(description="Name of the Azure Key Vault")
    secret_name: str = Field(description="Name of the secret to retrieve")


class StorageContainersArgs(BaseModel):
    """Arguments for listing storage containers."""

    storage_account: str = Field(description="Name of the Azure Storage account")


class ResourceGroupsArgs(BaseModel):
    """Arguments for listing resource groups."""

    subscription_id: str | None = Field(
        default=None, description="Optional subscription ID"
    )


class CosmosQueryArgs(BaseModel):
    """Arguments for querying Cosmos DB."""

    database_name: str = Field(description="Name of the Cosmos DB database")
    container_name: str = Field(description="Name of the container")
    query: str = Field(description="SQL query to execute")


class SQLDatabasesArgs(BaseModel):
    """Arguments for listing Azure SQL databases."""

    server_name: str = Field(description="Name of the Azure SQL server")
    resource_group: str = Field(description="Name of the resource group")


def build_mcp_tools(mcp_client: AzureMCPClient) -> dict[str, StructuredTool]:
    """
    Build LangChain StructuredTools from Azure MCP Server.

    Creates high-level tools for common Azure operations that internally
    use the MCP client to interact with Azure services.

    Args:
        mcp_client: Initialized Azure MCP client instance

    Returns:
        Dictionary mapping tool names to StructuredTool instances
    """

    def _run_async(coro):
        """Helper to run async operations in sync context."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    # Azure Key Vault tools
    def list_key_vault_secrets(vault_name: str) -> dict[str, Any]:
        """List all secrets in an Azure Key Vault."""
        return _run_async(
            mcp_client.call_tool(
                "azure_keyvault_list_secrets", {"vaultName": vault_name}
            )
        )

    def get_key_vault_secret(vault_name: str, secret_name: str) -> dict[str, Any]:
        """Get a specific secret from Azure Key Vault."""
        return _run_async(
            mcp_client.call_tool(
                "azure_keyvault_get_secret",
                {"vaultName": vault_name, "secretName": secret_name},
            )
        )

    # Azure Storage tools
    def list_storage_containers(storage_account: str) -> dict[str, Any]:
        """List all containers in an Azure Storage account."""
        return _run_async(
            mcp_client.call_tool(
                "azure_storage_list_containers", {"storageAccount": storage_account}
            )
        )

    def list_storage_accounts(subscription_id: str | None = None) -> dict[str, Any]:
        """List all Azure Storage accounts in the subscription."""
        params = {}
        if subscription_id:
            params["subscriptionId"] = subscription_id
        return _run_async(mcp_client.call_tool("azure_storage_list_accounts", params))

    # Azure Resource Management tools
    def list_resource_groups(subscription_id: str | None = None) -> dict[str, Any]:
        """List all resource groups in the Azure subscription."""
        params = {}
        if subscription_id:
            params["subscriptionId"] = subscription_id
        return _run_async(mcp_client.call_tool("azure_resources_list_groups", params))

    # Azure Cosmos DB tools
    def query_cosmos_db(
        database_name: str, container_name: str, query: str
    ) -> dict[str, Any]:
        """Execute a SQL query against Azure Cosmos DB."""
        return _run_async(
            mcp_client.call_tool(
                "azure_cosmos_query",
                {
                    "databaseName": database_name,
                    "containerName": container_name,
                    "query": query,
                },
            )
        )

    def list_cosmos_databases(account_name: str) -> dict[str, Any]:
        """List all databases in a Cosmos DB account."""
        return _run_async(
            mcp_client.call_tool("azure_cosmos_list_databases", {"accountName": account_name})
        )

    # Azure SQL tools
    def list_sql_databases(server_name: str, resource_group: str) -> dict[str, Any]:
        """List all databases in an Azure SQL server."""
        return _run_async(
            mcp_client.call_tool(
                "azure_sql_list_databases",
                {"serverName": server_name, "resourceGroup": resource_group},
            )
        )

    def list_sql_servers(resource_group: str | None = None) -> dict[str, Any]:
        """List all Azure SQL servers."""
        params = {}
        if resource_group:
            params["resourceGroup"] = resource_group
        return _run_async(mcp_client.call_tool("azure_sql_list_servers", params))

    # Build tool registry
    tools = {
        # Key Vault tools
        "list_key_vault_secrets": StructuredTool.from_function(
            func=list_key_vault_secrets,
            name="list_key_vault_secrets",
            description="List all secrets in an Azure Key Vault by vault name",
            args_schema=KeyVaultSecretsArgs,
        ),
        "get_key_vault_secret": StructuredTool.from_function(
            func=get_key_vault_secret,
            name="get_key_vault_secret",
            description="Get a specific secret value from Azure Key Vault",
            args_schema=KeyVaultSecretArgs,
        ),
        # Storage tools
        "list_storage_containers": StructuredTool.from_function(
            func=list_storage_containers,
            name="list_storage_containers",
            description="List all containers in an Azure Storage account",
            args_schema=StorageContainersArgs,
        ),
        "list_storage_accounts": StructuredTool.from_function(
            func=list_storage_accounts,
            name="list_storage_accounts",
            description="List all Azure Storage accounts in the subscription",
        ),
        # Resource Management tools
        "list_resource_groups": StructuredTool.from_function(
            func=list_resource_groups,
            name="list_resource_groups",
            description="List all resource groups in the Azure subscription",
        ),
        # Cosmos DB tools
        "query_cosmos_db": StructuredTool.from_function(
            func=query_cosmos_db,
            name="query_cosmos_db",
            description="Execute a SQL query against Azure Cosmos DB",
            args_schema=CosmosQueryArgs,
        ),
        "list_cosmos_databases": StructuredTool.from_function(
            func=list_cosmos_databases,
            name="list_cosmos_databases",
            description="List all databases in an Azure Cosmos DB account",
        ),
        # SQL Database tools
        "list_sql_databases": StructuredTool.from_function(
            func=list_sql_databases,
            name="list_sql_databases",
            description="List all databases in an Azure SQL server",
            args_schema=SQLDatabasesArgs,
        ),
        "list_sql_servers": StructuredTool.from_function(
            func=list_sql_servers,
            name="list_sql_servers",
            description="List all Azure SQL servers in the subscription or resource group",
        ),
    }

    return tools
