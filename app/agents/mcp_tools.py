from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import StructuredTool

from app.agents.mcp_client import AzureMCPClient


def build_mcp_tools(mcp_client: AzureMCPClient) -> dict[str, StructuredTool]:
    """
    Build LangChain StructuredTools from Azure MCP Server.

    Creates wrapper tools that invoke the actual MCP server tools directly.

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

    def _add_subscription(kwargs: dict) -> dict:
        """Add subscription ID to kwargs if available and not already present."""
        if mcp_client._subscription_id and "subscription" not in kwargs:
            kwargs = {**kwargs, "subscription": mcp_client._subscription_id}
        return kwargs

    # Azure Resource Groups
    def list_resource_groups(**kwargs) -> dict[str, Any]:
        """List all resource groups in the Azure subscription."""
        kwargs = _add_subscription(kwargs)
        return _run_async(mcp_client.call_tool("group_list", kwargs))

    def list_resources_in_group(**kwargs) -> dict[str, Any]:
        """List all resources in a resource group."""
        kwargs = _add_subscription(kwargs)
        return _run_async(mcp_client.call_tool("group_resource_list", kwargs))

    # Azure Subscriptions
    def list_subscriptions(**kwargs) -> dict[str, Any]:
        """List all Azure subscriptions."""
        return _run_async(mcp_client.call_tool("subscription_list", kwargs))

    # Azure Key Vault
    def keyvault_operation(**kwargs) -> dict[str, Any]:
        """Perform Azure Key Vault operations."""
        kwargs = _add_subscription(kwargs)
        return _run_async(mcp_client.call_tool("keyvault", kwargs))

    # Azure Storage
    def storage_operation(**kwargs) -> dict[str, Any]:
        """Perform Azure Storage operations."""
        kwargs = _add_subscription(kwargs)
        return _run_async(mcp_client.call_tool("storage", kwargs))

    # Azure SQL
    def sql_operation(**kwargs) -> dict[str, Any]:
        """Perform Azure SQL operations."""
        kwargs = _add_subscription(kwargs)
        return _run_async(mcp_client.call_tool("sql", kwargs))

    # Azure Cosmos DB
    def cosmos_operation(**kwargs) -> dict[str, Any]:
        """Perform Azure Cosmos DB operations."""
        kwargs = _add_subscription(kwargs)
        return _run_async(mcp_client.call_tool("cosmos", kwargs))

    # Azure Compute
    def compute_operation(**kwargs) -> dict[str, Any]:
        """Perform Azure Compute operations (VMs, etc.)."""
        kwargs = _add_subscription(kwargs)
        return _run_async(mcp_client.call_tool("compute", kwargs))

    # Azure Container Registry
    def acr_operation(**kwargs) -> dict[str, Any]:
        """Perform Azure Container Registry operations."""
        kwargs = _add_subscription(kwargs)
        return _run_async(mcp_client.call_tool("acr", kwargs))

    # Azure AKS
    def aks_operation(**kwargs) -> dict[str, Any]:
        """Perform Azure Kubernetes Service (AKS) operations."""
        kwargs = _add_subscription(kwargs)
        return _run_async(mcp_client.call_tool("aks", kwargs))

    # Specific Key Vault Operations
    def create_keyvault(
        vault_name: str,
        resource_group: str,
        location: str = "eastus",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Create a new Azure Key Vault.

        Args:
            vault_name: Name for the new Key Vault
            resource_group: Resource group name
            location: Azure region

        Returns:
            Result of vault creation
        """
        params = {
            "operation": "create",
            "vault_name": vault_name,
            "resource_group": resource_group,
            "location": location,
            **kwargs,
        }
        params = _add_subscription(params)
        return _run_async(mcp_client.call_tool("keyvault", params))

    def list_keyvaults(resource_group: str | None = None, **kwargs) -> dict[str, Any]:
        """List Key Vaults in a resource group or subscription."""
        params = {"operation": "list", **kwargs}
        if resource_group:
            params["resource_group"] = resource_group
        params = _add_subscription(params)
        return _run_async(mcp_client.call_tool("keyvault", params))

    def keyvault_set_secret(
        vault_name: str,
        secret_name: str,
        secret_value: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Set a secret in Azure Key Vault."""
        params = {
            "operation": "set_secret",
            "vault_name": vault_name,
            "secret_name": secret_name,
            "secret_value": secret_value,
            **kwargs,
        }
        params = _add_subscription(params)
        return _run_async(mcp_client.call_tool("keyvault", params))

    def keyvault_get_secret(
        vault_name: str,
        secret_name: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Get a secret from Azure Key Vault."""
        params = {
            "operation": "get_secret",
            "vault_name": vault_name,
            "secret_name": secret_name,
            **kwargs,
        }
        params = _add_subscription(params)
        return _run_async(mcp_client.call_tool("keyvault", params))

    # Specific Storage Operations
    def create_storage_account(
        account_name: str,
        resource_group: str,
        location: str = "eastus",
        **kwargs,
    ) -> dict[str, Any]:
        """Create a new Azure Storage Account."""
        params = {
            "operation": "create_account",
            "account_name": account_name,
            "resource_group": resource_group,
            "location": location,
            **kwargs,
        }
        params = _add_subscription(params)
        return _run_async(mcp_client.call_tool("storage", params))

    def create_blob_container(
        account_name: str,
        container_name: str,
        resource_group: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Create a new blob container in a storage account.

        Args:
            account_name: Storage account name
            container_name: Container name
            resource_group: Resource group name (optional)

        Returns:
            Result of container creation
        """
        params = {
            "operation": "create_container",
            "account_name": account_name,
            "container_name": container_name,
            **kwargs,
        }
        if resource_group:
            params["resource_group"] = resource_group
        params = _add_subscription(params)
        return _run_async(mcp_client.call_tool("storage", params))

    def list_blob_containers(
        account_name: str,
        resource_group: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """List blob containers in a storage account."""
        params = {
            "operation": "list_containers",
            "account_name": account_name,
            **kwargs,
        }
        if resource_group:
            params["resource_group"] = resource_group
        params = _add_subscription(params)
        return _run_async(mcp_client.call_tool("storage", params))

    def upload_blob(
        account_name: str,
        container_name: str,
        blob_name: str,
        content: str,
        resource_group: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Upload a blob to Azure Blob Storage.

        Args:
            account_name: Storage account name
            container_name: Container name
            blob_name: Blob name (path in container)
            content: Content to upload
            resource_group: Resource group name (optional)

        Returns:
            Result of blob upload
        """
        params = {
            "operation": "upload_blob",
            "account_name": account_name,
            "container_name": container_name,
            "blob_name": blob_name,
            "content": content,
            **kwargs,
        }
        if resource_group:
            params["resource_group"] = resource_group
        params = _add_subscription(params)
        return _run_async(mcp_client.call_tool("storage", params))

    def download_blob(
        account_name: str,
        container_name: str,
        blob_name: str,
        resource_group: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Download a blob from Azure Blob Storage.

        Args:
            account_name: Storage account name
            container_name: Container name
            blob_name: Blob name (path in container)
            resource_group: Resource group name (optional)

        Returns:
            Blob content
        """
        params = {
            "operation": "download_blob",
            "account_name": account_name,
            "container_name": container_name,
            "blob_name": blob_name,
            **kwargs,
        }
        if resource_group:
            params["resource_group"] = resource_group
        params = _add_subscription(params)
        return _run_async(mcp_client.call_tool("storage", params))

    def list_blobs(
        account_name: str,
        container_name: str,
        prefix: str | None = None,
        resource_group: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """List blobs in a container with optional prefix filter."""
        params = {
            "operation": "list_blobs",
            "account_name": account_name,
            "container_name": container_name,
            **kwargs,
        }
        if prefix:
            params["prefix"] = prefix
        if resource_group:
            params["resource_group"] = resource_group
        params = _add_subscription(params)
        return _run_async(mcp_client.call_tool("storage", params))

    # Build tool registry
    tools = {
        # Resource Management
        "group_list": StructuredTool.from_function(
            func=list_resource_groups,
            name="group_list",
            description="List all resource groups in the Azure subscription",
        ),
        "group_resource_list": StructuredTool.from_function(
            func=list_resources_in_group,
            name="group_resource_list",
            description="List all resources in a specific resource group",
        ),
        "subscription_list": StructuredTool.from_function(
            func=list_subscriptions,
            name="subscription_list",
            description="List all Azure subscriptions available",
        ),
        # Azure Key Vault - Generic
        "keyvault": StructuredTool.from_function(
            func=keyvault_operation,
            name="keyvault",
            description="Perform Azure Key Vault operations (list secrets, get secrets, manage vaults)",
        ),
        # Azure Key Vault - Specific Operations
        "keyvault_create": StructuredTool.from_function(
            func=create_keyvault,
            name="keyvault_create",
            description="Create a new Azure Key Vault in a resource group",
        ),
        "keyvault_list": StructuredTool.from_function(
            func=list_keyvaults,
            name="keyvault_list",
            description="List Azure Key Vaults in a resource group or subscription",
        ),
        "keyvault_set_secret": StructuredTool.from_function(
            func=keyvault_set_secret,
            name="keyvault_set_secret",
            description="Set a secret value in Azure Key Vault",
        ),
        "keyvault_get_secret": StructuredTool.from_function(
            func=keyvault_get_secret,
            name="keyvault_get_secret",
            description="Get a secret value from Azure Key Vault",
        ),
        # Azure Storage - Generic
        "storage": StructuredTool.from_function(
            func=storage_operation,
            name="storage",
            description="Perform Azure Storage operations (list accounts, containers, blobs)",
        ),
        # Azure Storage - Specific Operations
        "storage_create_account": StructuredTool.from_function(
            func=create_storage_account,
            name="storage_create_account",
            description="Create a new Azure Storage Account",
        ),
        "storage_create_container": StructuredTool.from_function(
            func=create_blob_container,
            name="storage_create_container",
            description="Create a new blob container in a storage account",
        ),
        "storage_list_containers": StructuredTool.from_function(
            func=list_blob_containers,
            name="storage_list_containers",
            description="List blob containers in a storage account",
        ),
        "storage_upload_blob": StructuredTool.from_function(
            func=upload_blob,
            name="storage_upload_blob",
            description="Upload a blob to Azure Blob Storage container",
        ),
        "storage_download_blob": StructuredTool.from_function(
            func=download_blob,
            name="storage_download_blob",
            description="Download a blob from Azure Blob Storage container",
        ),
        "storage_list_blobs": StructuredTool.from_function(
            func=list_blobs,
            name="storage_list_blobs",
            description="List blobs in a container with optional prefix filter",
        ),
        # Other Azure Services
        "sql": StructuredTool.from_function(
            func=sql_operation,
            name="sql",
            description="Perform Azure SQL operations (list servers, databases, query)",
        ),
        "cosmos": StructuredTool.from_function(
            func=cosmos_operation,
            name="cosmos",
            description="Perform Azure Cosmos DB operations (list databases, query)",
        ),
        "compute": StructuredTool.from_function(
            func=compute_operation,
            name="compute",
            description="Perform Azure Compute operations (VMs, scale sets, disks)",
        ),
        "acr": StructuredTool.from_function(
            func=acr_operation,
            name="acr",
            description="Perform Azure Container Registry operations",
        ),
        "aks": StructuredTool.from_function(
            func=aks_operation,
            name="aks",
            description="Perform Azure Kubernetes Service (AKS) operations",
        ),
    }

    return tools
