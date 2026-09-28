from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger("migration.adf")


# ---------------------------------------------------------------------------
# Argument models
# ---------------------------------------------------------------------------

class CreateAdfArgs(BaseModel):
    factory_name: str = Field(description="Name for the Azure Data Factory resource.")
    resource_group: str = Field(description="Azure resource group name.")
    location: str = Field(default="eastus", description="Azure region.")


class CreateLinkedServiceArgs(BaseModel):
    factory_name: str = Field(description="ADF factory name.")
    resource_group: str = Field(description="Azure resource group name.")
    service_name: str = Field(description="Name for the linked service.")
    service_type: str = Field(description="Type: 'storage', 'databricks', or 'keyvault'.")
    connection_string: str | None = Field(default=None, description="Connection string (for storage).")
    workspace_url: str | None = Field(default=None, description="Databricks workspace URL.")
    access_token_secret: str | None = Field(default=None, description="Key Vault secret name for Databricks token.")
    vault_url: str | None = Field(default=None, description="Key Vault URL (for keyvault linked service).")


class CreatePipelineArgs(BaseModel):
    factory_name: str = Field(description="ADF factory name.")
    resource_group: str = Field(description="Azure resource group name.")
    pipeline_name: str = Field(description="Pipeline name.")
    pipeline_type: str = Field(description="Layer type: 'landing', 'bronze', or 'silver'.")
    cluster_id: str | None = Field(default=None, description="Databricks cluster ID.")
    notebook_path: str | None = Field(default=None, description="Databricks notebook path.")


class TriggerPipelineArgs(BaseModel):
    factory_name: str = Field(description="ADF factory name.")
    resource_group: str = Field(description="Azure resource group name.")
    pipeline_name: str = Field(description="Pipeline name to trigger.")
    parameters: dict[str, str] = Field(default_factory=dict, description="Optional pipeline parameters.")


class GetPipelineStatusArgs(BaseModel):
    factory_name: str = Field(description="ADF factory name.")
    resource_group: str = Field(description="Azure resource group name.")
    run_id: str = Field(description="Pipeline run ID.")


# ---------------------------------------------------------------------------
# SDK implementation helpers
# ---------------------------------------------------------------------------

def _get_adf_client(subscription_id: str, credential):
    from azure.mgmt.datafactory import DataFactoryManagementClient
    return DataFactoryManagementClient(credential, subscription_id)


def _get_credential(tenant_id: str | None = None, client_id: str | None = None, client_secret: str | None = None):
    if tenant_id and client_id and client_secret:
        from azure.identity import ClientSecretCredential
        return ClientSecretCredential(tenant_id, client_id, client_secret)
    from azure.identity import DefaultAzureCredential
    return DefaultAzureCredential()


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def _create_adf_factory(
    factory_name: str,
    resource_group: str,
    location: str = "eastus",
    *,
    subscription_id: str,
    credential,
) -> dict[str, Any]:
    logger.info("[ADF] create_factory %s in %s/%s", factory_name, resource_group, location)
    from azure.mgmt.datafactory.models import Factory
    client = _get_adf_client(subscription_id, credential)
    factory = client.factories.create_or_update(resource_group, factory_name, Factory(location=location))
    return {
        "status": "created",
        "factory_name": factory.name,
        "location": factory.location,
        "id": factory.id,
    }


def _create_linked_service(
    factory_name: str,
    resource_group: str,
    service_name: str,
    service_type: str,
    *,
    connection_string: str | None = None,
    workspace_url: str | None = None,
    access_token_secret: str | None = None,
    vault_url: str | None = None,
    subscription_id: str,
    credential,
) -> dict[str, Any]:
    logger.info("[ADF] create_linked_service %s (type=%s) in factory %s", service_name, service_type, factory_name)
    from azure.mgmt.datafactory.models import LinkedServiceResource

    if service_type == "storage":
        from azure.mgmt.datafactory.models import AzureBlobStorageLinkedService
        ls = AzureBlobStorageLinkedService(connection_string=connection_string)
    elif service_type == "databricks":
        from azure.mgmt.datafactory.models import AzureDatabricksLinkedService, SecretBase
        from azure.mgmt.datafactory.models import AzureKeyVaultSecretReference
        token_ref = AzureKeyVaultSecretReference(
            store={"type": "LinkedServiceReference", "referenceName": "KeyVaultLinkedService"},
            secret_name=access_token_secret or "databricks-token",
        )
        ls = AzureDatabricksLinkedService(domain=workspace_url, access_token=token_ref)
    elif service_type == "keyvault":
        from azure.mgmt.datafactory.models import AzureKeyVaultLinkedService
        ls = AzureKeyVaultLinkedService(base_url=vault_url)
    else:
        raise ValueError(f"Unknown service_type: {service_type}")

    client = _get_adf_client(subscription_id, credential)
    resource = LinkedServiceResource(properties=ls)
    result = client.linked_services.create_or_update(resource_group, factory_name, service_name, resource)
    return {"status": "created", "service_name": result.name, "type": service_type}


def _create_pipeline(
    factory_name: str,
    resource_group: str,
    pipeline_name: str,
    pipeline_type: str,
    *,
    cluster_id: str | None = None,
    notebook_path: str | None = None,
    subscription_id: str,
    credential,
) -> dict[str, Any]:
    logger.info("[ADF] create_pipeline %s (layer=%s) in factory %s", pipeline_name, pipeline_type, factory_name)
    from azure.mgmt.datafactory.models import (
        DatabricksNotebookActivity,
        LinkedServiceReference,
        PipelineResource,
    )
    client = _get_adf_client(subscription_id, credential)
    activity = DatabricksNotebookActivity(
        name=f"{pipeline_type}_notebook",
        notebook_path=notebook_path or f"/migrations/{pipeline_type}",
        linked_service_name=LinkedServiceReference(reference_name="DatabricksLinkedService"),
        base_parameters={"layer": pipeline_type, "cluster_id": cluster_id or ""},
    )
    pipeline = PipelineResource(activities=[activity])
    result = client.pipelines.create_or_update(resource_group, factory_name, pipeline_name, pipeline)
    return {"status": "created", "pipeline_name": result.name, "layer": pipeline_type}


def _trigger_pipeline_run(
    factory_name: str,
    resource_group: str,
    pipeline_name: str,
    parameters: dict[str, str] | None = None,
    *,
    subscription_id: str,
    credential,
) -> dict[str, Any]:
    logger.info("[ADF] trigger_run %s in factory %s", pipeline_name, factory_name)
    client = _get_adf_client(subscription_id, credential)
    response = client.pipelines.create_run(
        resource_group, factory_name, pipeline_name, parameters=parameters or {}
    )
    return {"status": "triggered", "run_id": response.run_id, "pipeline_name": pipeline_name}


def _get_pipeline_run_status(
    factory_name: str,
    resource_group: str,
    run_id: str,
    *,
    subscription_id: str,
    credential,
) -> dict[str, Any]:
    client = _get_adf_client(subscription_id, credential)
    run = client.pipeline_runs.get(resource_group, factory_name, run_id)
    return {
        "run_id": run_id,
        "status": run.status,
        "pipeline_name": run.pipeline_name,
        "start_time": str(run.run_start) if run.run_start else None,
        "end_time": str(run.run_end) if run.run_end else None,
        "duration_ms": run.duration_in_ms,
    }


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_adf_tools(
    *,
    subscription_id: str,
    tenant_id: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    mcp_tools: dict[str, StructuredTool] | None = None,
) -> dict[str, StructuredTool]:
    """
    Build ADF StructuredTools. MCP tools are checked first via tool_resolver.
    SDK tools are always built as fallback.
    """
    if not subscription_id:
        return {}

    credential = _get_credential(tenant_id, client_id, client_secret)

    # Partially apply credentials into each function
    def make_create_adf(factory_name: str, resource_group: str, location: str = "eastus") -> dict:
        return _create_adf_factory(factory_name, resource_group, location, subscription_id=subscription_id, credential=credential)

    def make_create_linked_service(factory_name: str, resource_group: str, service_name: str, service_type: str, connection_string: str | None = None, workspace_url: str | None = None, access_token_secret: str | None = None, vault_url: str | None = None) -> dict:
        return _create_linked_service(factory_name, resource_group, service_name, service_type, connection_string=connection_string, workspace_url=workspace_url, access_token_secret=access_token_secret, vault_url=vault_url, subscription_id=subscription_id, credential=credential)

    def make_create_pipeline(factory_name: str, resource_group: str, pipeline_name: str, pipeline_type: str, cluster_id: str | None = None, notebook_path: str | None = None) -> dict:
        return _create_pipeline(factory_name, resource_group, pipeline_name, pipeline_type, cluster_id=cluster_id, notebook_path=notebook_path, subscription_id=subscription_id, credential=credential)

    def make_trigger_run(factory_name: str, resource_group: str, pipeline_name: str, parameters: dict[str, str] | None = None) -> dict:
        return _trigger_pipeline_run(factory_name, resource_group, pipeline_name, parameters, subscription_id=subscription_id, credential=credential)

    def make_get_status(factory_name: str, resource_group: str, run_id: str) -> dict:
        return _get_pipeline_run_status(factory_name, resource_group, run_id, subscription_id=subscription_id, credential=credential)

    sdk_tools: dict[str, StructuredTool] = {
        "create_adf_factory": StructuredTool.from_function(make_create_adf, args_schema=CreateAdfArgs, description="Create an Azure Data Factory resource."),
        "create_adf_linked_service": StructuredTool.from_function(make_create_linked_service, args_schema=CreateLinkedServiceArgs, description="Create a linked service in ADF (storage, databricks, or keyvault)."),
        "create_adf_pipeline": StructuredTool.from_function(make_create_pipeline, args_schema=CreatePipelineArgs, description="Create a Databricks notebook pipeline in ADF."),
        "trigger_adf_pipeline_run": StructuredTool.from_function(make_trigger_run, args_schema=TriggerPipelineArgs, description="Trigger an ADF pipeline run."),
        "get_adf_pipeline_run_status": StructuredTool.from_function(make_get_status, args_schema=GetPipelineStatusArgs, description="Get the status of an ADF pipeline run."),
    }

    # MCP-first: if MCP exposes matching tools, they override SDK tools
    if mcp_tools:
        from app.agents.tool_resolver import build_merged_tools
        return build_merged_tools(mcp_tools, sdk_tools)

    return sdk_tools
