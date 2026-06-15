from __future__ import annotations

from fastapi import APIRouter, Depends

from app.agents.secrets_tools import get_secrets_service_meta
from app.dependencies import get_source_registry
from app.schemas.chat import CloudProviderMeta, PhaseInfo, SourceSystemMeta, MigrationPhase

router = APIRouter()

# ---------------------------------------------------------------------------
# Cloud providers (static registry — only cloud-specific info here)
# ---------------------------------------------------------------------------

_CLOUD_PROVIDERS: list[CloudProviderMeta] = [
    CloudProviderMeta(
        id="azure",
        display_name="Azure",
        icon_key="azure",
        description="Microsoft Azure — Blob Storage, ADLS Gen2, Databricks, and ADF.",
        secrets_service_name="Azure Key Vault",
        secrets_service_icon="keyvault",
        supported_destinations=["azure_blob", "adls_gen2", "databricks"],
    ),
    CloudProviderMeta(
        id="aws",
        display_name="AWS",
        icon_key="aws",
        description="Amazon Web Services — S3, Redshift, and Glue.",
        secrets_service_name="AWS Secrets Manager",
        secrets_service_icon="aws_secrets",
        supported_destinations=["s3", "redshift"],
    ),
    CloudProviderMeta(
        id="gcp",
        display_name="Google Cloud",
        icon_key="gcp",
        description="Google Cloud Platform — Cloud Storage and BigQuery.",
        secrets_service_name="GCP Secret Manager",
        secrets_service_icon="gcp_secrets",
        supported_destinations=["gcs", "bigquery"],
    ),
]

# ---------------------------------------------------------------------------
# Migration phases (ordered)
# ---------------------------------------------------------------------------

_PHASES: list[PhaseInfo] = [
    PhaseInfo(id=MigrationPhase.INIT, display_name="Getting Started", description="Welcome and source system selection.", order=1),
    PhaseInfo(id=MigrationPhase.SOURCE_CONFIGURED, display_name="Source Configured", description="Source system and credentials collected.", order=2),
    PhaseInfo(id=MigrationPhase.CLOUD_SELECTED, display_name="Cloud Selected", description="Target cloud provider and region chosen.", order=3),
    PhaseInfo(id=MigrationPhase.VAULT_SETUP, display_name="Secrets Manager Ready", description="Cloud-native secrets manager configured; credentials stored.", order=4),
    PhaseInfo(id=MigrationPhase.STORAGE_PROVISIONED, display_name="Storage Provisioned", description="Blob containers created for landing, metadata, bronze, silver.", order=5),
    PhaseInfo(id=MigrationPhase.METADATA_EXTRACTED, display_name="Metadata Extracted", description="Source table metadata written to storage.", order=6),
    PhaseInfo(id=MigrationPhase.DATABRICKS_PROVISIONED, display_name="Databricks Ready", description="Databricks workspace and cluster provisioned.", order=7),
    PhaseInfo(id=MigrationPhase.NOTEBOOKS_UPLOADED, display_name="Notebooks Uploaded", description="Migration notebooks uploaded to Databricks workspace.", order=8),
    PhaseInfo(id=MigrationPhase.METADATA_LOADED, display_name="Metadata Loaded", description="Delta schema tables created from metadata.", order=9),
    PhaseInfo(id=MigrationPhase.PIPELINE_CONFIGURED, display_name="Pipeline Configured", description="pipeline_config delta table populated.", order=10),
    PhaseInfo(id=MigrationPhase.ADF_PROVISIONED, display_name="ADF Provisioned", description="Azure Data Factory pipelines created for orchestration.", order=11),
    PhaseInfo(id=MigrationPhase.COMPLETED, display_name="Completed", description="Migration setup complete. Ready to run.", order=12),
]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/sources", response_model=list[SourceSystemMeta])
def list_sources(registry=Depends(get_source_registry)) -> list[SourceSystemMeta]:
    """Return all registered source systems with their required credential fields."""
    return registry.list_sources()


@router.get("/clouds", response_model=list[CloudProviderMeta])
def list_clouds() -> list[CloudProviderMeta]:
    """Return supported cloud providers with their native secrets manager info."""
    return _CLOUD_PROVIDERS


@router.get("/migration/phases", response_model=list[PhaseInfo])
def list_phases() -> list[PhaseInfo]:
    """Return the ordered list of migration phases for the frontend progress tracker."""
    return sorted(_PHASES, key=lambda p: p.order)
