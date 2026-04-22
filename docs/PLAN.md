# FastAPI Migration Automation Foundation With LangChain And LangGraph

## Summary
Build a local-first FastAPI backend for S3 list/read/write operations, with an agent endpoint implemented using `LangChain` and `LangGraph`. Use `Ollama` as the default open-source/local model provider and keep `OpenAI` optional behind the same graph-based orchestration. This gives you an agentic architecture in v1 that is already aligned with the larger migration-automation roadmap.

## Key Changes
- Create a layered backend:
  - `api` for FastAPI routes and request/response models
  - `services` for AWS access via `boto3`
  - `agents` for LangChain tools, LangGraph workflow, approval handling, and provider adapters
  - `persistence` for approval state and graph checkpoints using SQLite in v1
- Implement initial direct S3 APIs:
  - `GET /v1/s3/buckets`
  - `GET /v1/s3/objects?bucket=...&prefix=...`
  - `GET /v1/s3/object?bucket=...&key=...`
  - `PUT /v1/s3/object`
- Implement one agent API:
  - `POST /v1/agent/s3`
  - read requests can execute immediately
  - write requests create an approval proposal
  - follow-up request with `approval_id` + `approve=true` executes the stored approved action
- Build the agent flow in `LangGraph`:
  - graph nodes for request intake, intent classification, tool selection, approval decision, tool execution, and final response
  - approval gate modeled as an explicit human-in-the-loop step
  - graph state stores message history, selected tool, tool args, approval metadata, and execution result
- Use `LangChain` for tool and model abstractions:
  - S3 actions exposed as LangChain tools
  - provider-specific chat models plugged into a common orchestration layer
  - prompt templates and structured output parsing handled through LangChain components where useful
- Make `Ollama` the default provider in v1:
  - local model served through Ollama
  - configurable model name via environment settings
  - suitable for local development without per-request API charges
- Keep `OpenAI` optional:
  - supported through LangChain integration
  - selected by config when hosted model quality is preferred
- Enforce v1 safety:
  - bucket/prefix allowlist from config
  - write actions require stored approval before execution
  - no tools beyond the registered S3 actions
  - prompts and graph policy explicitly prevent unapproved mutations
- Keep the design extensible for later migration work:
  - add Glue notebook upload and Glue table CRUD as new tools and graph branches
  - reuse the same approval and execution pattern for destructive or state-changing actions

## AI Libraries
- `langchain`
  - main framework for tool binding, prompts, structured outputs, and provider abstraction
- `langgraph`
  - workflow engine for stateful agent orchestration and approval-gated execution
- `langchain-openai`
  - OpenAI integration package for LangChain when hosted models are enabled
- `langchain-ollama`
  - Ollama integration package for local/open-source models
- `ollama`
  - optional direct client for local validation or lower-level troubleshooting if needed
- `pydantic`
  - schema validation for API payloads, tool arguments, and normalized graph state
- `boto3`
  - AWS SDK for S3 operations
- `langgraph-checkpoint-sqlite` or equivalent SQLite-backed checkpoint support
  - persist graph/approval state locally in v1

Default AI/provider choice for v1:
- default runtime provider: `Ollama`
- optional hosted provider: `OpenAI`
- no Hugging Face integration in v1 unless you specifically want remote open-source hosting instead of local Ollama

## Public Interfaces
- `S3ObjectReadRequest`: `bucket`, `key`
- `S3ObjectWriteRequest`: `bucket`, `key`, `content_type`, `content`, optional `metadata`
- `AgentS3Request`: `message`, optional `approval_id`, optional `approve`
- `AgentS3Response`:
  - direct read/result payload
  - or approval proposal with `approval_id`, `tool_name`, `tool_args`, `summary`, `expires_at`
- Config:
  - AWS region
  - allowed buckets and prefixes
  - SQLite path
  - `LLM_PROVIDER` with values `ollama` or `openai`
  - Ollama base URL and model name
  - OpenAI API key and model name

## Test Plan
- Unit tests for S3 services with mocked AWS clients or `moto`
- Tool tests for each LangChain S3 tool
- LangGraph tests for:
  - read-only request path
  - write request producing approval instead of execution
  - approval follow-up executing only the stored action
  - blocked bucket/prefix requests
  - malformed tool args and provider failures
- API tests for direct S3 endpoints and agent endpoint
- Provider-switch validation:
  - same graph works with `ollama`
  - same graph works with `openai`
  - startup/config errors are surfaced clearly

## Assumptions And Defaults
- `LangChain` and `LangGraph` are first-class dependencies in v1, not deferred.
- `Ollama` is the default because it avoids paid hosted inference for local development.
- `OpenAI` remains optional; the SDK is free, but API usage is paid according to OpenAI’s published pricing.
- v1 supports text/JSON object writes only.
- end-user auth is out of scope for v1; rely on local usage, IAM permissions, and bucket/prefix allowlists.
