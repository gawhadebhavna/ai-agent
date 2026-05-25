from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field

from app.agents.model_factory import build_runtime_chat_model
from app.config import Settings


class AzureMCPActionPlan(BaseModel):
    """Structured action plan for Azure MCP and Databricks operations."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(description="The tool to invoke (Azure MCP or Databricks)")
    
    # Common Azure Parameters
    resource_group: str | None = Field(default=None, description="Azure resource group name")
    location: str | None = Field(default=None, description="Azure region/location (e.g., eastus)")
    subscription_id: str | None = Field(default=None, description="Azure subscription ID")
    
    # Storage Parameters
    storage_account: str | None = Field(default=None, description="Azure storage account name (alias: account_name)")
    account_name: str | None = Field(default=None, description="Azure storage account name")
    container_name: str | None = Field(default=None, description="Blob container name")
    blob_name: str | None = Field(default=None, description="Blob name/path in container")
    prefix: str | None = Field(default=None, description="Blob prefix for filtering")
    content: str | None = Field(default=None, description="Content to upload to blob")
    
    # Key Vault Parameters
    vault_name: str | None = Field(default=None, description="Azure Key Vault name")
    secret_name: str | None = Field(default=None, description="Secret name in Key Vault")
    secret_value: str | None = Field(default=None, description="Secret value to set")
    
    # Databricks Cluster Parameters
    cluster_name: str | None = Field(default=None, description="Databricks cluster name")
    cluster_id: str | None = Field(default=None, description="Databricks cluster ID")
    spark_version: str | None = Field(default=None, description="Spark runtime version (e.g., 13.3.x-scala2.12)")
    node_type_id: str | None = Field(default=None, description="Azure VM instance type (e.g., Standard_DS3_v2)")
    num_workers: int | None = Field(default=None, description="Number of worker nodes")
    
    # Databricks Notebook Parameters
    notebook_path: str | None = Field(default=None, description="Workspace path to notebook (e.g., /Users/user@domain.com/notebook)")
    language: str | None = Field(default=None, description="Notebook language: PYTHON, SQL, SCALA, or R")
    
    # Databricks Job Parameters
    job_id: int | None = Field(default=None, description="Databricks job ID")
    run_id: int | None = Field(default=None, description="Databricks run ID")
    parameters_json: str | None = Field(default=None, description="Job/Notebook parameters as JSON string (e.g., '{\"key\":\"value\"}')")
    
    # Other Azure Services
    server_name: str | None = Field(default=None, description="Server name (SQL, Cosmos, etc.)")
    
    # Metadata
    rationale: str = Field(description="Brief explanation of the action")
    requires_approval: bool = Field(
        default=False, description="Whether this action needs user approval"
    )
    
    def get_arguments(self) -> dict[str, Any]:
        """Extract non-None arguments as a dictionary for tool invocation."""
        import json
        
        excluded = {"tool_name", "rationale", "requires_approval", "parameters_json"}
        args = {
            key: value
            for key, value in self.model_dump().items()
            if value is not None and key not in excluded
        }
        
        # Handle storage_account alias for account_name
        if args.get("storage_account") and not args.get("account_name"):
            args["account_name"] = args.pop("storage_account")
        elif args.get("storage_account"):
            args.pop("storage_account")
        
        # Parse parameters_json if provided
        if self.parameters_json:
            try:
                args["parameters"] = json.loads(self.parameters_json)
            except json.JSONDecodeError:
                pass  # Ignore invalid JSON
        
        return args


AZURE_MCP_PLANNER_PROMPT = """You are an Azure cloud and Databricks operations planning assistant.
Your job is to convert a user's natural-language request into a specific action plan.

Available tools:

**Azure MCP Tools (via Azure MCP Server):**
- group_list: List all resource groups
- group_resource_list: List resources in a resource group
- subscription_list: List all Azure subscriptions

**Key Vault Operations:**
- keyvault: Generic Key Vault operations
- keyvault_create: Create a new Key Vault
- keyvault_list: List Key Vaults
- keyvault_set_secret: Set a secret in Key Vault
- keyvault_get_secret: Get a secret from Key Vault

**Storage Operations:**
- storage: Generic storage operations
- storage_create_account: Create a new Storage Account
- storage_create_container: Create a blob container
- storage_list_containers: List blob containers
- storage_upload_blob: Upload a blob to a container
- storage_download_blob: Download a blob from a container
- storage_list_blobs: List blobs in a container

**Other Azure Services:**
- sql: Azure SQL operations
- cosmos: Cosmos DB operations
- compute: VM and compute operations
- acr: Container Registry operations
- aks: Kubernetes Service operations

**Databricks Tools (via Databricks SDK):**
- dbx_create_cluster: Create a new Databricks cluster
- dbx_list_clusters: List all clusters
- dbx_start_cluster: Start a stopped cluster
- dbx_stop_cluster: Stop a running cluster
- dbx_upload_notebook: Upload a notebook to workspace
- dbx_list_notebooks: List notebooks in workspace
- dbx_run_notebook: Run a notebook as a job
- dbx_run_job: Run an existing job by ID
- dbx_get_run_status: Get status of a job run

Parameter mapping:
- resource_group: Azure resource group name
- location: Azure region (eastus, westus, etc.)
- account_name/storage_account: Storage account name
- container_name: Blob container name
- blob_name: Blob path in container (e.g., "folder/file.txt")
- prefix: Blob prefix filter (e.g., "folder/")
- content: Content to upload
- vault_name: Key Vault name
- secret_name: Secret name
- secret_value: Secret value
- cluster_name: Databricks cluster name
- cluster_id: Databricks cluster ID
- notebook_path: Workspace path (e.g., "/Users/user@domain.com/notebook")
- language: Notebook language (PYTHON, SQL, SCALA, R)
- job_id: Databricks job ID
- run_id: Databricks run ID
- parameters_json: Job/Notebook parameters as JSON string (e.g., '{{"param1":"value1"}}')

Safety rules:
- READ operations (list, get, download) → requires_approval: false
- WRITE operations (create, upload, set, run, start) → requires_approval: true
- DELETE operations (delete, stop) → requires_approval: true

User request: {message}

Examples:

1. "Create a Key Vault named prod-vault in resource group my-rg in eastus"
   → tool_name: "keyvault_create"
   → vault_name: "prod-vault"
   → resource_group: "my-rg"
   → location: "eastus"
   → requires_approval: true

2. "Create a blob container called data-container in storage account mystorageacct"
   → tool_name: "storage_create_container"
   → account_name: "mystorageacct"
   → container_name: "data-container"
   → requires_approval: true

3. "Upload file.txt with content 'Hello World' to container data-container in storage account mystorageacct"
   → tool_name: "storage_upload_blob"
   → account_name: "mystorageacct"
   → container_name: "data-container"
   → blob_name: "file.txt"
   → content: "Hello World"
   → requires_approval: true

4. "List all blobs in container data-container from storage account mystorageacct"
   → tool_name: "storage_list_blobs"
   → account_name: "mystorageacct"
   → container_name: "data-container"
   → requires_approval: false

5. "Create a Databricks cluster named test-cluster with 2 workers"
   → tool_name: "dbx_create_cluster"
   → cluster_name: "test-cluster"
   → num_workers: 2
   → requires_approval: true

6. "Upload a Python notebook to /Users/me@company.com/analysis with content 'print(1)'"
   → tool_name: "dbx_upload_notebook"
   → notebook_path: "/Users/me@company.com/analysis"
   → content: "print(1)"
   → language: "PYTHON"
   → requires_approval: true

7. "Run notebook /Users/me@company.com/analysis on cluster cluster-123"
   → tool_name: "dbx_run_notebook"
   → notebook_path: "/Users/me@company.com/analysis"
   → cluster_id: "cluster-123"
   → requires_approval: true

8. "List all Databricks clusters"
   → tool_name: "dbx_list_clusters"
   → requires_approval: false
"""


class AzureMCPRequestPlanner:
    """Plans Azure MCP operations from natural language requests."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = build_runtime_chat_model(settings)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", AZURE_MCP_PLANNER_PROMPT),
                ("human", "Return the best Azure MCP action plan for this request."),
            ]
        )

    @property
    def provider_name(self) -> str:
        return self._settings.llm_provider

    def plan(self, *, message: str) -> AzureMCPActionPlan:
        """
        Generate an Azure MCP action plan from a natural language request.

        Args:
            message: User's natural language request

        Returns:
            Structured action plan with tool name, arguments, and approval requirement
        """
        chain = self._prompt | self._model.with_structured_output(AzureMCPActionPlan)
        plan = chain.invoke({"message": message})

        # Auto-detect write operations that need approval
        write_keywords = ["create", "delete", "update", "set", "add", "remove", "write"]
        if any(keyword in plan.tool_name.lower() for keyword in write_keywords):
            plan.requires_approval = True

        return plan
