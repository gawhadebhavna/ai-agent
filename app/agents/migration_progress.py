from __future__ import annotations

"""
Derives migration progress from persisted session data.

The session repository is the single source of truth. Phase, the status block
shown to the LLM, the next recommended step and quick-reply suggestions are all
computed from it, so the chat can never drift away from what actually exists.
"""

from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.schemas.chat import MigrationPhase

DEFAULT_RESOURCE_GROUP = "migration-rg"

# ---------------------------------------------------------------------------
# Region catalog
# ---------------------------------------------------------------------------

AZURE_REGIONS: dict[str, str] = {
    "eastus": "East US", "eastus2": "East US 2", "centralus": "Central US",
    "northcentralus": "North Central US", "southcentralus": "South Central US",
    "westcentralus": "West Central US", "westus": "West US", "westus2": "West US 2",
    "westus3": "West US 3", "canadacentral": "Canada Central", "canadaeast": "Canada East",
    "brazilsouth": "Brazil South", "northeurope": "North Europe", "westeurope": "West Europe",
    "uksouth": "UK South", "ukwest": "UK West", "francecentral": "France Central",
    "germanywestcentral": "Germany West Central", "switzerlandnorth": "Switzerland North",
    "norwayeast": "Norway East", "swedencentral": "Sweden Central", "italynorth": "Italy North",
    "polandcentral": "Poland Central", "uaenorth": "UAE North", "qatarcentral": "Qatar Central",
    "southafricanorth": "South Africa North", "centralindia": "Central India",
    "southindia": "South India", "westindia": "West India", "eastasia": "East Asia",
    "southeastasia": "Southeast Asia", "japaneast": "Japan East", "japanwest": "Japan West",
    "koreacentral": "Korea Central", "australiaeast": "Australia East",
    "australiasoutheast": "Australia Southeast",
}

AWS_REGIONS: dict[str, str] = {
    "us-east-1": "US East (N. Virginia)", "us-east-2": "US East (Ohio)",
    "us-west-1": "US West (N. California)", "us-west-2": "US West (Oregon)",
    "ca-central-1": "Canada (Central)", "sa-east-1": "South America (São Paulo)",
    "eu-west-1": "Europe (Ireland)", "eu-west-2": "Europe (London)", "eu-west-3": "Europe (Paris)",
    "eu-central-1": "Europe (Frankfurt)", "eu-north-1": "Europe (Stockholm)",
    "me-central-1": "Middle East (UAE)", "ap-south-1": "Asia Pacific (Mumbai)",
    "ap-southeast-1": "Asia Pacific (Singapore)", "ap-southeast-2": "Asia Pacific (Sydney)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
}

GCP_REGIONS: dict[str, str] = {
    "us-central1": "Iowa", "us-east1": "South Carolina", "us-west1": "Oregon",
    "europe-west1": "Belgium", "europe-west2": "London", "asia-south1": "Mumbai",
    "asia-southeast1": "Singapore",
}

REGIONS_BY_PROVIDER: dict[str, dict[str, str]] = {
    "azure": AZURE_REGIONS,
    "aws": AWS_REGIONS,
    "gcp": GCP_REGIONS,
}

_POPULAR_REGIONS: dict[str, list[str]] = {
    "azure": ["East US", "Central US", "West Europe", "Southeast Asia"],
    "aws": ["us-east-1", "eu-west-1", "ap-south-1", "us-west-2"],
    "gcp": ["us-central1", "europe-west1", "asia-south1"],
}


def normalize_region(provider: str, region: str) -> str | None:
    """Map user/LLM input like 'Central US' or 'centralus' to a canonical region code."""
    catalog = REGIONS_BY_PROVIDER.get(provider, {})
    raw = region.strip().lower()
    if raw in catalog:
        return raw
    compact = raw.replace(" ", "").replace("_", "").replace("(", "").replace(")", "")
    if provider == "azure" and compact in catalog:
        return compact
    for code, display in catalog.items():
        if display.lower() == raw:
            return code
    return None


# ---------------------------------------------------------------------------
# Snapshot of everything the assistant needs to know
# ---------------------------------------------------------------------------

_AZURE_CRED_KEYS = ("tenant_id", "client_id", "client_secret", "subscription_id")


@dataclass
class MigrationSnapshot:
    source_system: str | None
    source_display: str | None
    source_required_keys: list[str]
    cloud_provider: str | None
    cloud_region: str | None
    resource_group: str
    resource_group_set: bool
    azure_creds_from_settings: bool
    azure_creds_collected: list[str]
    azure_authenticated: bool
    vault_name: str | None
    vault_uri: str | None
    stored_secret_names: list[str]
    pending_secret_names: list[str]
    source_creds_from_settings: bool
    storage: dict[str, Any]
    pipeline_tables: list[dict[str, Any]]
    databricks: dict[str, Any]
    databricks_pat_available: bool
    metadata_loaded: bool
    pipeline_configured: bool
    adf: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)

    # --- derived flags -----------------------------------------------------

    @property
    def cloud_target_set(self) -> bool:
        return bool(self.cloud_provider and self.cloud_region)

    @property
    def azure_creds_missing(self) -> list[str]:
        if self.azure_creds_from_settings:
            return []
        return [k for k in _AZURE_CRED_KEYS if k not in self.azure_creds_collected]

    @property
    def cloud_creds_ready(self) -> bool:
        return self.cloud_provider == "azure" and not self.azure_creds_missing

    @property
    def vault_ready(self) -> bool:
        return bool(self.vault_uri)

    @property
    def source_creds_missing(self) -> list[str]:
        if self.source_creds_from_settings:
            return []
        return [k for k in self.source_required_keys if f"source-{k}" not in self.stored_secret_names]

    @property
    def storage_ready(self) -> bool:
        return bool(self.storage.get("provisioned"))

    @property
    def metadata_ready(self) -> bool:
        return bool(self.pipeline_tables)

    @property
    def databricks_ready(self) -> bool:
        return bool(self.databricks.get("workspace_url"))

    @property
    def notebooks_ready(self) -> bool:
        return int(self.databricks.get("notebooks_uploaded_count") or 0) > 0

    @property
    def adf_ready(self) -> bool:
        return bool(self.adf.get("provisioned"))


def build_snapshot(
    session: dict[str, Any],
    settings: Settings,
    source_registry,
    pending_secret_names: list[str] | None = None,
) -> MigrationSnapshot:
    phase_data = session.get("phase_data", {}) or {}
    vault_config = phase_data.get("vault_config", {}) or {}
    azure_creds = vault_config.get("azure_credentials", {}) or {}

    source_system = phase_data.get("source_system") or session.get("source_system")
    plugin = source_registry.get_plugin(source_system) if source_system else None
    stored = list(vault_config.get("stored_secret_names", []) or [])

    return MigrationSnapshot(
        source_system=source_system,
        source_display=plugin.display_name if plugin else None,
        source_required_keys=[f.key for f in plugin.required_cred_fields] if plugin else [],
        cloud_provider=session.get("cloud_provider"),
        cloud_region=phase_data.get("cloud_region"),
        resource_group=vault_config.get("resource_group")
        or settings.azure_resource_group
        or DEFAULT_RESOURCE_GROUP,
        resource_group_set=bool(vault_config.get("resource_group")),
        azure_creds_from_settings=bool(
            settings.has_azure_service_principal and settings.azure_subscription_id
        ) and not azure_creds,
        azure_creds_collected=[k for k in _AZURE_CRED_KEYS if azure_creds.get(k)],
        azure_authenticated=bool(vault_config.get("azure_authenticated")),
        vault_name=vault_config.get("vault_name") if vault_config.get("vault_uri") else None,
        vault_uri=vault_config.get("vault_uri"),
        stored_secret_names=stored,
        pending_secret_names=list(pending_secret_names or []),
        source_creds_from_settings=bool(source_system == "sap_s4" and settings.has_sap_api_credentials),
        storage=phase_data.get("storage_config", {}) or {},
        pipeline_tables=phase_data.get("pipeline_tables", []) or [],
        databricks=phase_data.get("databricks_config", {}) or {},
        databricks_pat_available="databricks-pat" in stored
        or bool(settings.databricks_token and "<" not in (settings.databricks_token or "")),
        metadata_loaded=bool(phase_data.get("metadata_loaded")),
        pipeline_configured=bool(phase_data.get("pipeline_configured")),
        adf=phase_data.get("adf_config", {}) or {},
    )


# ---------------------------------------------------------------------------
# Phase / next step / suggestions
# ---------------------------------------------------------------------------

def derive_phase(s: MigrationSnapshot) -> MigrationPhase:
    if s.adf_ready:
        return MigrationPhase.COMPLETED
    if s.pipeline_configured:
        return MigrationPhase.PIPELINE_CONFIGURED
    if s.metadata_loaded:
        return MigrationPhase.METADATA_LOADED
    if s.notebooks_ready:
        return MigrationPhase.NOTEBOOKS_UPLOADED
    if s.databricks_ready:
        return MigrationPhase.DATABRICKS_PROVISIONED
    if s.metadata_ready:
        return MigrationPhase.METADATA_EXTRACTED
    if s.storage_ready:
        return MigrationPhase.STORAGE_PROVISIONED
    if s.vault_ready:
        return MigrationPhase.VAULT_SETUP
    if s.cloud_creds_ready:
        return MigrationPhase.AZURE_AUTHENTICATED
    if s.cloud_target_set:
        return MigrationPhase.CLOUD_SELECTED
    if s.source_system:
        return MigrationPhase.SOURCE_CONFIGURED
    return MigrationPhase.INIT


def next_step(s: MigrationSnapshot) -> str:
    if not s.source_system:
        return "Ask which source system they want to migrate from, then call select_source_system."
    if not s.cloud_provider:
        return "Ask which target cloud they want (Azure, AWS or GCP)."
    if not s.cloud_region:
        return f"Ask which {s.cloud_provider.upper()} region they want, then call set_cloud_target."
    if s.cloud_provider != "azure":
        return (
            f"Explain that automated provisioning is currently available for Azure only; "
            f"{s.cloud_provider.upper()} support is coming. Offer to switch to Azure."
        )
    if s.azure_creds_missing:
        return (
            "Collect Azure service principal credentials. Still missing: "
            + ", ".join(s.azure_creds_missing)
            + ". Offer the step-by-step guide if they don't have them."
        )
    if not s.vault_ready:
        return "Confirm the resource group (default shown above), then either call create_secrets_vault or, if the user already has a vault, use_existing_vault."
    if s.source_creds_missing:
        return (
            f"Collect {s.source_display or 'source'} connection details. Still missing: "
            + ", ".join(s.source_creds_missing)
            + ". The user should paste them as `key: value` pairs."
        )
    if not s.storage_ready:
        return "Call provision_storage."
    if not s.metadata_ready:
        return "Call extract_source_metadata."
    if not s.databricks_ready:
        return "Call provision_databricks."
    if not s.notebooks_ready:
        if not s.databricks_pat_available:
            return "Ask the user for a Databricks personal access token (starts with `dapi`), with steps to create one."
        return "Call upload_notebooks."
    if not s.metadata_loaded:
        return (
            "Ask the user to run /migrations/generic/01_load_metadata in Databricks; "
            "once they confirm, call confirm_manual_step('metadata_loaded')."
        )
    if not s.pipeline_configured:
        return "Ask whether to include all extracted tables or a subset, then call configure_pipeline_tables."
    if not s.adf_ready:
        return "Call provision_adf."
    return "Setup is complete. Summarise what was created and how to trigger a run."


def suggestions_for(s: MigrationSnapshot, source_names: list[str], approval_pending: bool) -> list[str]:
    if approval_pending:
        return []
    if not s.source_system:
        return source_names
    if not s.cloud_provider:
        return ["Azure", "AWS", "GCP"]
    if not s.cloud_region:
        return _POPULAR_REGIONS.get(s.cloud_provider, [])
    if s.cloud_provider != "azure":
        return ["Switch to Azure"]
    if s.azure_creds_missing:
        return ["I have my credentials", "Service principal guide"]
    if not s.vault_ready:
        if s.resource_group_set:
            return ["Create a new Key Vault", "Use an existing Key Vault"]
        return [f"Use {s.resource_group}", "Use a different resource group"]
    if s.source_creds_missing:
        return ["Which credentials do you need?"]
    if not s.storage_ready:
        return ["Create storage", "What will be created?"]
    if not s.metadata_ready:
        return ["Extract metadata", "Which tables will be scanned?"]
    if not s.databricks_ready:
        return ["Provision Databricks", "What cluster size?"]
    if not s.notebooks_ready:
        return ["Upload notebooks"] if s.databricks_pat_available else ["Show PAT setup steps"]
    if not s.metadata_loaded:
        return ["I've run the metadata notebook", "What does it do?"]
    if not s.pipeline_configured:
        return ["Include all tables", "Let me pick tables"]
    if not s.adf_ready:
        return ["Create Data Factory", "What does ADF do here?"]
    return ["Show summary", "Explain how to trigger a run"]


def render_status(s: MigrationSnapshot) -> str:
    def yes_no(flag: bool, yes: str, no: str = "not yet") -> str:
        return yes if flag else no

    if s.azure_creds_from_settings:
        azure_line = "available from server configuration"
    elif s.azure_creds_missing:
        azure_line = (
            f"collected: {', '.join(s.azure_creds_collected) or 'none'}; "
            f"missing: {', '.join(s.azure_creds_missing)}"
        )
    else:
        azure_line = "all four values collected"
    if s.azure_authenticated:
        azure_line += " (verified)"

    if s.source_creds_from_settings:
        source_creds_line = "available from server configuration"
    elif not s.source_required_keys:
        source_creds_line = "n/a"
    else:
        source_creds_line = (
            f"stored: {', '.join(n for n in s.stored_secret_names if n.startswith('source-')) or 'none'}; "
            f"missing: {', '.join(s.source_creds_missing) or 'none'}"
        )
    if s.pending_secret_names:
        source_creds_line += f"; held until vault exists: {', '.join(s.pending_secret_names)}"

    region_display = ""
    if s.cloud_provider and s.cloud_region:
        region_display = REGIONS_BY_PROVIDER.get(s.cloud_provider, {}).get(s.cloud_region, "")
        region_display = f" ({region_display})" if region_display else ""

    tables = [str(t.get("table_name")) for t in s.pipeline_tables]
    included = [str(t.get("table_name")) for t in s.pipeline_tables if t.get("load_flag", True)]
    source_line = f"{s.source_display} ({s.source_system})" if s.source_system else "not selected"
    storage_line = "{} with containers {}".format(
        s.storage.get("storage_account"), ", ".join(s.storage.get("containers", []))
    )
    metadata_line = "{} table(s): {}".format(len(tables), ", ".join(tables))
    pipeline_line = "{} included: {}".format(len(included), ", ".join(included))
    pat_line = "available" if s.databricks_pat_available else "missing"

    lines = [
        f"- Source system: {source_line}",
        f"- Target cloud: {s.cloud_provider or 'not selected'}",
        f"- Region: {(s.cloud_region or 'not selected') + region_display}",
        f"- Resource group: {s.resource_group}",
        f"- Azure credentials: {azure_line}" if s.cloud_provider in (None, "azure") else "",
        f"- Secrets vault: {yes_no(s.vault_ready, f'{s.vault_name} ({s.vault_uri})')}",
        f"- Source credentials: {source_creds_line}",
        f"- Storage: {yes_no(s.storage_ready, storage_line)}",
        f"- Metadata: {yes_no(s.metadata_ready, metadata_line)}",
        f"- Databricks: {yes_no(s.databricks_ready, str(s.databricks.get('workspace_url')))}; PAT {pat_line}",
        f"- Notebooks uploaded: {yes_no(s.notebooks_ready, str(s.databricks.get('notebooks_uploaded_count')))}",
        f"- Metadata notebook run: {yes_no(s.metadata_loaded, 'confirmed by user')}",
        f"- Pipeline tables: {yes_no(s.pipeline_configured, pipeline_line)}",
        f"- Data Factory: {yes_no(s.adf_ready, str(s.adf.get('factory_name')))}",
    ]
    return "\n".join(line for line in lines if line)
