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
SQLITE_PATH=./var/app.db
```

Keep `.env` local only. Do not commit AWS credentials to source control.

## Key Endpoints

- `GET /health`
- `GET /v1/s3/buckets`
- `GET /v1/s3/objects?bucket=...&prefix=...`
- `GET /v1/s3/object?bucket=...&key=...`
- `PUT /v1/s3/object`
- `POST /v1/agent/s3`
