# Azure MCP and Databricks Tools

This document lists all available tools for Azure and Databricks operations through the `/v1/llm/run` endpoint with `domain: azure_mcp`.

## Authentication Configuration

Set in `.env` file:

```env
# Choose authentication method:
AZURE_TOKEN_CREDENTIALS=ClientSecretCredential   # or AzureCliCredential

# Service Principal credentials (when using ClientSecretCredential):
AZURE_TENANT_ID=<your-tenant-id>
AZURE_CLIENT_ID=<your-client-id>
AZURE_CLIENT_SECRET=<your-client-secret>
AZURE_SUBSCRIPTION_ID=<your-subscription-id>

# Databricks credentials (optional):
DATABRICKS_URL=https://adb-<workspace-id>.<region>.azuredatabricks.net
DATABRICKS_TOKEN=<your-personal-access-token>
```

---

## 📦 Resource Management Tools

### `group_list`
List all resource groups in the Azure subscription.

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "List all resource groups"}'
```

### `group_resource_list`
List all resources in a specific resource group.

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "List all resources in resource group my-rg"}'
```

### `subscription_list`
List all Azure subscriptions available.

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "Show me all Azure subscriptions"}'
```

---

## 🔐 Azure Key Vault Tools

### `keyvault_create`
Create a new Azure Key Vault in a resource group.

**Requires Approval:** ✅ Yes

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "Create a Key Vault named prod-vault in resource group my-rg in eastus"
  }'
```

### `keyvault_list`
List Azure Key Vaults in a resource group or subscription.

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "List all Key Vaults in resource group my-rg"}'
```

### `keyvault_set_secret`
Set a secret value in Azure Key Vault.

**Requires Approval:** ✅ Yes

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "Set secret api-key with value abc123 in Key Vault prod-vault"
  }'
```

### `keyvault_get_secret`
Get a secret value from Azure Key Vault.

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "Get secret api-key from Key Vault prod-vault"}'
```

### `keyvault` (generic)
Perform generic Azure Key Vault operations.

---

## 💾 Azure Storage Tools

### `storage_create_account`
Create a new Azure Storage Account.

**Requires Approval:** ✅ Yes

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "Create storage account mystorageacct in resource group my-rg in eastus"
  }'
```

### `storage_create_container`
Create a new blob container in a storage account.

**Requires Approval:** ✅ Yes

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "Create a blob container named data-container in storage account mystorageacct"
  }'
```

### `storage_list_containers`
List blob containers in a storage account.

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "List all containers in storage account mystorageacct"}'
```

### `storage_upload_blob`
Upload a blob to Azure Blob Storage container.

**Requires Approval:** ✅ Yes

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "Upload file data.txt with content Hello World to container data-container in storage account mystorageacct"
  }'
```

**Upload to folder:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "Upload reports/2024/report.csv with content data to container data-container in storage account mystorageacct"
  }'
```

### `storage_download_blob`
Download a blob from Azure Blob Storage container.

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "Download blob data.txt from container data-container in storage account mystorageacct"
  }'
```

### `storage_list_blobs`
List blobs in a container with optional prefix filter.

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "List all blobs in container data-container from storage account mystorageacct"
  }'
```

**List blobs in folder:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "List all blobs in reports/2024/ folder in container data-container from storage account mystorageacct"
  }'
```

### `storage` (generic)
Perform generic Azure Storage operations.

---

## 🗄️ Other Azure Services

### `sql`
Perform Azure SQL operations (list servers, databases, query).

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "List all SQL servers in my subscription"}'
```

### `cosmos`
Perform Azure Cosmos DB operations (list databases, query).

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "List all Cosmos DB accounts"}'
```

### `compute`
Perform Azure Compute operations (VMs, scale sets, disks).

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "List all virtual machines"}'
```

### `acr`
Perform Azure Container Registry operations.

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "List all container registries"}'
```

### `aks`
Perform Azure Kubernetes Service (AKS) operations.

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "List all AKS clusters"}'
```

---

## 🧮 Databricks Cluster Management

### `dbx_create_cluster`
Create a new Databricks cluster with specified configuration.

**Requires Approval:** ✅ Yes

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "Create a Databricks cluster named analytics-cluster with 3 workers"
  }'
```

**With custom configuration:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "Create a Databricks cluster named ml-cluster with 5 workers using Spark version 13.3.x-scala2.12 and node type Standard_DS4_v2"
  }'
```

### `dbx_list_clusters`
List all Databricks clusters in the workspace.

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "List all Databricks clusters"}'
```

### `dbx_start_cluster`
Start a stopped Databricks cluster.

**Requires Approval:** ✅ Yes

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "Start Databricks cluster with ID 0123-456789-abc123"}'
```

### `dbx_stop_cluster`
Stop a running Databricks cluster.

**Requires Approval:** ✅ Yes

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "Stop Databricks cluster with ID 0123-456789-abc123"}'
```

---

## 📓 Databricks Notebook Operations

### `dbx_upload_notebook`
Upload a notebook to Databricks workspace.

**Requires Approval:** ✅ Yes

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "Upload a Python notebook to /Users/me@company.com/analysis with content print(1+1)"
  }'
```

**SQL Notebook:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "Upload a SQL notebook to /Users/me@company.com/query with content SELECT * FROM table"
  }'
```

### `dbx_list_notebooks`
List notebooks in a workspace path.

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "List all notebooks in /Users/me@company.com/"}'
```

### `dbx_run_notebook`
Run a notebook as a job on Databricks.

**Requires Approval:** ✅ Yes

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "Run notebook /Users/me@company.com/analysis on cluster 0123-456789-abc123"
  }'
```

**Run on new cluster:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "Run notebook /Users/me@company.com/etl on a new cluster"
  }'
```

---

## 🔄 Databricks Job Operations

### `dbx_run_job`
Run an existing Databricks job by job ID.

**Requires Approval:** ✅ Yes

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "Run Databricks job with ID 123"}'
```

### `dbx_get_run_status`
Get the status of a Databricks job run.

**Example Prompt:**
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "Get status of Databricks run 456789"}'
```

---

## 🔒 Approval Workflow

All write operations (create, upload, update, delete, run, start, stop) require approval before execution.

### Step 1: Submit Request
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "Create a Key Vault named test-vault in resource group my-rg in eastus"
  }'
```

### Step 2: Receive Approval Proposal
```json
{
  "domain": "azure_mcp",
  "status": "pending_approval",
  "message": "...",
  "approval": {
    "approval_id": "abc123-def456",
    "action_summary": "Create Key Vault 'test-vault' in resource group 'my-rg'",
    "expires_at": "2026-05-26T20:30:00Z"
  }
}
```

### Step 3: Approve
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "",
    "approval_id": "abc123-def456",
    "approve": true
  }'
```

### Step 3 (Alternative): Reject
```bash
curl -X POST http://localhost:8000/v1/llm/run \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "azure_mcp",
    "message": "",
    "approval_id": "abc123-def456",
    "approve": false
  }'
```

---

## 📝 Notes

- **Read Operations**: No approval required (list, get, download, show)
- **Write Operations**: Require approval (create, upload, update, delete, run, start, stop)
- **Approval TTL**: 30 minutes by default (configurable via `APPROVAL_TTL_MINUTES`)
- **Domain Auto-Detection**: Requests containing keywords like "databricks", "cluster", "notebook", "key vault", "blob storage" automatically route to `azure_mcp` domain
- **Natural Language**: All operations support natural language prompts - no need to memorize exact parameter names

---

## 🚀 Quick Examples

**Azure Storage Workflow:**
```bash
# 1. Create container
curl -X POST http://localhost:8000/v1/llm/run -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "Create container mydata in storage account mystorageacct"}'

# 2. Upload file
curl -X POST http://localhost:8000/v1/llm/run -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "Upload data/file.txt with content test data to container mydata in storage mystorageacct"}'

# 3. List files
curl -X POST http://localhost:8000/v1/llm/run -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "List all blobs in data/ folder in container mydata from storage mystorageacct"}'
```

**Databricks Workflow:**
```bash
# 1. Create cluster
curl -X POST http://localhost:8000/v1/llm/run -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "Create a Databricks cluster named etl-cluster with 2 workers"}'

# 2. Upload notebook
curl -X POST http://localhost:8000/v1/llm/run -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "Upload Python notebook to /Users/me@company.com/etl with content print(spark.version)"}'

# 3. Run notebook
curl -X POST http://localhost:8000/v1/llm/run -H "Content-Type: application/json" \
  -d '{"domain": "azure_mcp", "message": "Run notebook /Users/me@company.com/etl on cluster etl-cluster"}'
```

---

## 🔧 Troubleshooting

**Azure MCP Server not available:**
- Ensure Node.js is installed (for `npx` command)
- Check Azure credentials in `.env` file
- Verify `AZURE_TOKEN_CREDENTIALS` is set correctly

**Tenant mismatch error:**
- Ensure `AZURE_TENANT_ID` matches your subscription's tenant
- Run `az account show --subscription <subscription-id>` to verify correct tenant

**Databricks tools not available:**
- Set `DATABRICKS_URL` and `DATABRICKS_TOKEN` in `.env`
- Verify Databricks Personal Access Token is valid

For more information, see [PLAN.md](./PLAN.md)
