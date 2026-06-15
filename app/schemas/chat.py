from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Migration phases — mirrors graph state machine
# ---------------------------------------------------------------------------

class MigrationPhase(str, Enum):
    INIT = "INIT"
    SOURCE_CONFIGURED = "SOURCE_CONFIGURED"
    CLOUD_SELECTED = "CLOUD_SELECTED"
    AZURE_AUTHENTICATED = "AZURE_AUTHENTICATED"  # cloud creds collected from user
    VAULT_SETUP = "VAULT_SETUP"
    STORAGE_PROVISIONED = "STORAGE_PROVISIONED"
    METADATA_EXTRACTED = "METADATA_EXTRACTED"
    DATABRICKS_PROVISIONED = "DATABRICKS_PROVISIONED"
    NOTEBOOKS_UPLOADED = "NOTEBOOKS_UPLOADED"
    METADATA_LOADED = "METADATA_LOADED"
    PIPELINE_CONFIGURED = "PIPELINE_CONFIGURED"
    ADF_PROVISIONED = "ADF_PROVISIONED"
    COMPLETED = "COMPLETED"


class ChatStatus(str, Enum):
    ok = "ok"
    approval_required = "approval_required"
    clarification_required = "clarification_required"
    error = "error"


# ---------------------------------------------------------------------------
# Approval items
# ---------------------------------------------------------------------------

class ApprovalItem(BaseModel):
    approval_id: str
    summary: str
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    expires_at: str
    cloud_resource_type: str | None = None


# ---------------------------------------------------------------------------
# Chat request / response
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(default="")  # empty string allowed when approval_id is set
    approval_id: str | None = None
    approve: bool | None = None


class ChatResponse(BaseModel):
    session_id: str
    message: str
    status: ChatStatus
    migration_phase: MigrationPhase
    completed_phases: list[str] = Field(default_factory=list)
    pending_approvals: list[ApprovalItem] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    phase_data: dict[str, Any] = Field(default_factory=dict)


class SessionCreateResponse(BaseModel):
    session_id: str
    migration_phase: MigrationPhase


class SessionStatusResponse(BaseModel):
    session_id: str
    migration_phase: MigrationPhase
    completed_phases: list[str]
    pending_approvals: list[ApprovalItem]
    phase_data: dict[str, Any]


# ---------------------------------------------------------------------------
# Metadata schemas (for /v1/sources, /v1/clouds, /v1/migration/phases)
# ---------------------------------------------------------------------------

class CredentialField(BaseModel):
    key: str
    label: str
    placeholder: str
    secret: bool = False
    required: bool = True


class SourceSystemMeta(BaseModel):
    id: str
    display_name: str
    icon_key: str
    description: str
    required_cred_fields: list[CredentialField]


class CloudProviderMeta(BaseModel):
    id: str
    display_name: str
    icon_key: str
    description: str
    secrets_service_name: str
    secrets_service_icon: str
    supported_destinations: list[str]


class PhaseInfo(BaseModel):
    id: str
    display_name: str
    description: str
    order: int


# ---------------------------------------------------------------------------
# Internal review structures (not exposed to API)
# ---------------------------------------------------------------------------

class InboundReview(BaseModel):
    intent: str
    contains_credentials: bool = False
    credential_fields: dict[str, str] = Field(default_factory=dict)
    routing: str
    clarification_needed: bool = False
    clarification_prompt: str | None = None


class OutboundReview(BaseModel):
    passes: bool
    failure_reason: str | None = None
    revised_response: str | None = None
