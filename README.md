# FastAPI Migration Automation

Local-first backend for migration automation workflows built with:

- `FastAPI` for HTTP APIs
- `boto3` for AWS access
- `LangChain` for model/tool integration
- `LangGraph` for approval-gated agent orchestration
- `Ollama` as the default local model provider
- `OpenAI` as an optional hosted model provider

## Features

- Direct S3 APIs for listing buckets, listing objects, reading objects, and writing text or JSON objects
- Agent endpoint that can interpret natural language requests and execute safe S3 actions
- Unified LLM endpoint that routes to AWS, Azure Blob, or SAP tool calls from one API path
- Human approval gate for write operations using LangGraph interrupts
- SQLite-backed approval metadata and optional LangGraph checkpoint persistence

## Quick Start

1. Create a virtual environment and install dependencies.
2. Create a local `.env` from `.env.example`.
3. Add your AWS credentials and S3 settings to `.env`.
4. Run the API:

```bash
uvicorn app.main:app --reload
```

## Local AWS Setup

For local development, the app can authenticate directly from `.env`. It will try AWS credentials in this order:

1. `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
2. `AWS_PROFILE`
3. Standard boto3 default credential chain

Initial S3 defaults for this project:

```env
AWS_REGION=ap-south-1
ALLOWED_BUCKETS=agentic-ai-migration-bkt
ALLOWED_PREFIXES_JSON={"agentic-ai-migration-bkt":["landing/"]}
```

Example local `.env`:

```env
AWS_REGION=ap-south-1
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_SESSION_TOKEN=
AWS_PROFILE=
ALLOWED_BUCKETS=agentic-ai-migration-bkt
ALLOWED_PREFIXES_JSON={"agentic-ai-migration-bkt":["landing/"]}
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini
AZURE_OPENAI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-15-preview
SQLITE_PATH=./var/app.db
```
Set `LLM_PROVIDER=openai` to use OpenAI API keys (`sk-...`).
Set `LLM_PROVIDER=azure_openai` to use Azure OpenAI credentials.

TLS settings for corporate/proxy environments:

```env
SSL_VERIFY=true
SSL_CA_BUNDLE=
```

If your network uses TLS inspection, set `SSL_CA_BUNDLE` to your corporate CA bundle path.
Use `SSL_VERIFY=false` only as a temporary fallback.

Keep `.env` local only. Do not commit AWS credentials to source control.

## Key Endpoints

- `GET /health`
- `GET /v1/s3/buckets`
- `GET /v1/s3/objects?bucket=...&prefix=...`
- `GET /v1/s3/object?bucket=...&key=...`
- `PUT /v1/s3/object`
- `POST /v1/llm/run`

Unified endpoint payload:

```json
{
  "domain": "auto",
  "message": "Fetch KNA1 data and write parquet to azure blob path sap/KNA1.parquet",
  "approval_id": null,
  "approve": null
}
```

`domain` values:
- `auto` (default): infer `aws`, `azure`, or `sap` from request
- `aws`: force S3 agent flow
- `azure`: force Azure Blob write flow
- `sap`: force SAP agent flow
