from __future__ import annotations

"""
Tools the chat agent uses to read and change migration state.

Every decision the user makes is recorded through a tool, and every cloud
resource is created through a tool gated by a human approval (LangGraph
interrupt). Tools return plain-text results the LLM relays to the user.
"""

import re
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import interrupt

from app.agents.migration_progress import (
    AZURE_REGIONS,
    REGIONS_BY_PROVIDER,
    normalize_region,
)
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.agents.migration_orchestrator import MigrationOrchestratorService

logger = get_logger("migration.tools")

_AFFIRMATIVE = re.compile(
    r"^(y|yes|yeah|yep|yup|ok|okay|sure|approve[d]?|confirm(ed)?|go|go ahead|proceed|do it|create it|"
    r"looks good|lgtm|yes,? please|yes,? (go ahead|proceed|do it|create it|approve))[\s.!]*$",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"^(n|no|nope|cancel|stop|reject(ed)?|abort|decline[d]?|don'?t|do not|not now|no,? thanks?)[\s.!]*$",
    re.IGNORECASE,
)
_RG_NAME = re.compile(r"^[A-Za-z0-9_\-.()]{1,90}$")


def _session_id(config: RunnableConfig) -> str:
    return config["configurable"]["thread_id"]


def _request_approval(
    action: str, title: str, details: dict[str, Any], resource_type: str
) -> tuple[bool, str]:
    """Pause the graph for user approval. Returns (approved, free_text_reply)."""
    lines = "\n".join(f"- **{k}:** `{v}`" for k, v in details.items())
    decision = interrupt({
        "action": action,
        "summary": title,
        "details": details,
        "cloud_resource_type": resource_type,
        "message": f"**Approval needed: {title}**\n\n{lines}\n\nUse **Approve** / **Reject** below, or tell me what to change.",
    })
    if isinstance(decision, dict) and decision.get("approved") is not None:
        return bool(decision["approved"]), ""
    reply = (decision.get("reply") if isinstance(decision, dict) else str(decision or "")) or ""
    first_line = reply.strip().splitlines()[0] if reply.strip() else ""
    if _AFFIRMATIVE.match(first_line):
        return True, ""
    if _NEGATIVE.match(first_line):
        return False, ""
    return False, reply


def _not_approved(reply: str, tool_name: str) -> str:
    if reply:
        return (
            f"NOT EXECUTED. Instead of approving, the user replied: \"{reply}\". "
            f"Handle their reply (e.g. update settings with the right tool or answer the question). "
            f"Call {tool_name} again only if they still want it."
        )
    return "NOT EXECUTED. The user declined. Acknowledge it and ask what they would like to do instead."


def _explain_azure_error(exc: Exception) -> str:
    text = str(exc)
    if "AADSTS7000215" in text or "invalid client secret" in text.lower():
        return (
            "The client secret is invalid. It looks like the Secret **ID** was used instead of the secret "
            "**Value** (App registrations → Certificates & secrets → copy the *Value* column). "
            "Ask the user to paste `client_secret: <value>` again."
        )
    if "AADSTS700016" in text:
        return "The client_id (application ID) was not found in this tenant. Ask the user to double-check client_id and tenant_id."
    if "AADSTS90002" in text or "AADSTS900023" in text:
        return "The tenant_id is not valid. Ask the user to re-check the Directory (tenant) ID."
    if "AuthorizationFailed" in text:
        return (
            "The service principal lacks permission. It needs the **Contributor** role on the subscription "
            "(Subscriptions → Access control (IAM) → Add role assignment)."
        )
    if "ResourceGroupNotFound" in text:
        return "The resource group does not exist and could not be created; the service principal likely needs Contributor on the subscription."
    if "VaultAlreadyExists" in text or "already in use" in text.lower():
        return "That vault name is already taken globally."
    return text[:500]


def build_migration_tools(svc: "MigrationOrchestratorService") -> list:
    repo = svc._session_repo
    registry = svc._source_registry

    def session(sid: str) -> dict[str, Any]:
        return repo.get_session(sid) or {"phase_data": {}, "migration_phase": "INIT"}

    def patch_vault_config(sid: str, patch: dict[str, Any]) -> None:
        vault_config = dict(session(sid).get("phase_data", {}).get("vault_config", {}) or {})
        vault_config.update(patch)
        repo.patch_phase_data(sid, {"vault_config": vault_config})

    def current_phase(sid: str) -> str:
        return session(sid).get("migration_phase", "INIT")

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    @tool
    def select_source_system(source_id: Literal["sap_s4", "oracle"], config: RunnableConfig) -> str:
        """Record the source system to migrate from. Use 'sap_s4' for SAP S/4HANA or ECC, 'oracle' for Oracle Database."""
        sid = _session_id(config)
        snap = svc.snapshot(sid)
        if snap.metadata_ready and snap.source_system and snap.source_system != source_id:
            return (
                f"ERROR: metadata has already been extracted from {snap.source_display}. "
                "Changing the source requires a new migration session."
            )
        plugin = registry.get_plugin(source_id)
        repo.update_phase(sid, current_phase(sid), phase_data_patch={"source_system": source_id}, source_system=source_id)
        fields = ", ".join(f"{f.key} ({f.label}{', secret' if f.secret else ''})" for f in plugin.required_cred_fields)
        return f"Saved source system: {plugin.display_name}. Connection fields needed later: {fields}."

    @tool
    def set_cloud_target(
        provider: Literal["azure", "aws", "gcp"], config: RunnableConfig, region: str | None = None
    ) -> str:
        """Record the target cloud provider and (when known) the region.

        Call it as soon as the user names a provider, even before they pick a region.
        region must be a region code, e.g. Azure 'centralus', 'eastus', 'westeurope';
        AWS 'us-east-1'; GCP 'us-central1'. Display names like 'Central US' are also accepted.
        """
        sid = _session_id(config)
        snap = svc.snapshot(sid)
        if region is None:
            if snap.vault_ready and snap.cloud_provider != provider:
                return f"ERROR: resources already exist in {snap.cloud_provider}; the provider cannot be changed now."
            patch = {"cloud_region": None} if snap.cloud_provider != provider else None
            repo.update_phase(sid, current_phase(sid), phase_data_patch=patch, cloud_provider=provider)
            return f"Saved provider: {provider.upper()}. Region not chosen yet."
        code = normalize_region(provider, region)
        if code is None:
            valid = ", ".join(REGIONS_BY_PROVIDER[provider].keys())
            return f"ERROR: '{region}' is not a known {provider} region. Valid codes: {valid}. Ask the user to pick one."
        if snap.vault_ready and (snap.cloud_provider != provider or snap.cloud_region != code):
            return (
                f"ERROR: resources already exist in {snap.cloud_provider}/{snap.cloud_region} "
                f"(vault {snap.vault_name}). All resources must stay in one region; "
                "changing it requires starting a new migration session."
            )
        repo.update_phase(sid, current_phase(sid), phase_data_patch={"cloud_region": code}, cloud_provider=provider)
        display = REGIONS_BY_PROVIDER[provider][code]
        note = "" if provider == "azure" else f" Note: automated provisioning is Azure-only today; {provider.upper()} is not yet automated."
        return f"Saved target: {provider.upper()} in {display} ({code}).{note}"

    @tool
    def set_resource_group(name: str, config: RunnableConfig) -> str:
        """Record the Azure resource group to create all migration resources in (created if it does not exist)."""
        sid = _session_id(config)
        name = name.strip()
        if not _RG_NAME.match(name) or name.endswith("."):
            return "ERROR: invalid resource group name. Use 1-90 letters, digits, '-', '_', '.', '(' or ')', not ending with '.'."
        snap = svc.snapshot(sid)
        if snap.vault_ready and snap.resource_group != name:
            return f"ERROR: the vault already exists in resource group {snap.resource_group}; it cannot be changed now."
        patch_vault_config(sid, {"resource_group": name})
        return f"Saved resource group: {name}."

    # ------------------------------------------------------------------
    # Azure authentication
    # ------------------------------------------------------------------

    @tool
    def verify_azure_credentials(config: RunnableConfig) -> str:
        """Check that the Azure service principal credentials work and can see the subscription."""
        sid = _session_id(config)
        snap = svc.snapshot(sid)
        if snap.azure_creds_missing:
            return f"ERROR: credentials incomplete. Missing: {', '.join(snap.azure_creds_missing)}."
        try:
            from azure.mgmt.subscription import SubscriptionClient

            credential = svc._get_azure_credential(sid)
            sub_id = svc._get_azure_subscription_id(sid)
            sub = SubscriptionClient(credential).subscriptions.get(sub_id)
            patch_vault_config(sid, {"azure_authenticated": True})
            return f"Verified: authenticated to subscription '{sub.display_name}' ({sub_id[:8]}…)."
        except Exception as exc:
            text = str(exc)
            if "AADSTS7000215" in text:
                vault_config = dict(session(sid).get("phase_data", {}).get("vault_config", {}) or {})
                creds = dict(vault_config.get("azure_credentials", {}) or {})
                creds.pop("client_secret", None)
                patch_vault_config(sid, {"azure_credentials": creds, "azure_authenticated": False})
            elif "AADSTS" not in text and "AuthorizationFailed" in text:
                patch_vault_config(sid, {"azure_authenticated": True})
                return "Credentials are valid, but the service principal cannot read the subscription (needs Reader/Contributor). Resource creation may fail until Contributor is granted."
            logger.warning("verify_azure_credentials failed: %s", exc, extra={"phase": "-", "session_id": sid})
            return f"ERROR: {_explain_azure_error(exc)}"

    # ------------------------------------------------------------------
    # Provisioning (approval-gated)
    # ------------------------------------------------------------------

    @tool
    def create_secrets_vault(config: RunnableConfig) -> str:
        """Create the Azure Key Vault that stores all migration credentials. Shows the user an approval card first."""
        sid = _session_id(config)
        snap = svc.snapshot(sid)
        if snap.vault_ready:
            return f"Vault already exists: {snap.vault_name} ({snap.vault_uri})."
        if not snap.cloud_target_set:
            return "ERROR: choose the target cloud and region first."
        if snap.cloud_provider != "azure":
            return f"ERROR: automated vault creation is only available for Azure (current: {snap.cloud_provider})."
        if snap.azure_creds_missing:
            return f"ERROR: Azure credentials missing: {', '.join(snap.azure_creds_missing)}."

        vault_name = f"migration-vault-{sid[:8]}"
        approved, reply = _request_approval(
            "create_secrets_vault",
            "Create Azure Key Vault",
            {"Name": vault_name, "Resource group": snap.resource_group, "Region": AZURE_REGIONS.get(snap.cloud_region, snap.cloud_region)},
            "Azure Key Vault",
        )
        if not approved:
            return _not_approved(reply, "create_secrets_vault")

        try:
            backend = svc._build_secrets_backend("azure", sid)
            result = backend.create_vault(vault_name, snap.cloud_region, resource_group=snap.resource_group)
        except Exception as exc:
            logger.warning("create_secrets_vault failed: %s", exc, extra={"phase": "-", "session_id": sid})
            return f"ERROR: vault creation failed. {_explain_azure_error(exc)}"

        vault_uri = result.get("uri") or f"https://{vault_name}.vault.azure.net/"
        patch_vault_config(sid, {
            "vault_name": vault_name,
            "vault_uri": vault_uri,
            "vault_service": "Azure Key Vault",
            "resource_group": snap.resource_group,
            "azure_authenticated": True,
        })
        stored, failed = svc.flush_pending_secrets(sid)
        extra = f" Stored previously provided secrets: {', '.join(stored)}." if stored else ""
        if failed:
            extra += f" Failed to store: {', '.join(failed)}."
        return f"SUCCESS: Key Vault {vault_name} created in {snap.resource_group} ({snap.cloud_region}). URI: {vault_uri}.{extra}"

    @tool
    def provision_storage(config: RunnableConfig) -> str:
        """Create the Azure storage account with landing, metadata, bronze and silver containers. Shows an approval card first."""
        sid = _session_id(config)
        snap = svc.snapshot(sid)
        if snap.storage_ready:
            return f"Storage already provisioned: {snap.storage.get('storage_account')}."
        if not snap.vault_ready:
            return "ERROR: create the Key Vault first (storage keys are stored there)."

        account = re.sub(r"[^a-z0-9]", "", f"migstore{sid[:6]}".lower())[:24]
        containers = ["landing", "metadata", "bronze", "silver"]
        approved, reply = _request_approval(
            "provision_storage",
            "Create Azure Storage account",
            {
                "Storage account": account,
                "Containers": ", ".join(containers),
                "Resource group": snap.resource_group,
                "Region": AZURE_REGIONS.get(snap.cloud_region, snap.cloud_region),
            },
            "Azure Storage Account",
        )
        if not approved:
            return _not_approved(reply, "provision_storage")
        try:
            return svc._provision_storage_real(
                {"resource_group": snap.resource_group, "storage_account": account,
                 "region": snap.cloud_region, "containers": containers},
                sid, current_phase(sid),
            )
        except Exception as exc:
            logger.warning("provision_storage failed: %s", exc, extra={"phase": "-", "session_id": sid})
            return f"ERROR: storage creation failed. {_explain_azure_error(exc)}"

    @tool
    def extract_source_metadata(config: RunnableConfig) -> str:
        """Connect to the source system, list its tables and field definitions, and write metadata JSON to the storage account."""
        sid = _session_id(config)
        snap = svc.snapshot(sid)
        if not snap.source_system:
            return "ERROR: no source system selected."
        if not snap.storage_ready:
            return "ERROR: provision storage first."
        if snap.source_creds_missing:
            return f"ERROR: source credentials missing: {', '.join(snap.source_creds_missing)}. Ask the user for them."
        try:
            return svc._extract_metadata_real(
                {"source_system": snap.source_system, "path_prefix": f"metadata/{snap.source_system}/"},
                sid, current_phase(sid),
            )
        except Exception as exc:
            logger.warning("extract_source_metadata failed: %s", exc, extra={"phase": "-", "session_id": sid})
            return f"ERROR: metadata extraction failed: {str(exc)[:400]}"

    @tool
    def provision_databricks(config: RunnableConfig) -> str:
        """Create the Azure Databricks workspace (or a cluster in the configured workspace). Shows an approval card first."""
        sid = _session_id(config)
        snap = svc.snapshot(sid)
        if snap.databricks_ready:
            return f"Databricks already provisioned: {snap.databricks.get('workspace_url')}."
        if not snap.vault_ready:
            return "ERROR: create the Key Vault first."

        proposed = {
            "workspace_name": f"migration-dbx-{sid[:6]}",
            "cluster_name": "migration-cluster",
            "spark_version": "13.3.x-scala2.12",
            "node_type": "Standard_DS3_v2",
            "num_workers": 2,
        }
        approved, reply = _request_approval(
            "provision_databricks",
            "Create Azure Databricks workspace",
            {
                "Workspace": proposed["workspace_name"],
                "Cluster": f"{proposed['cluster_name']} ({proposed['num_workers']} × {proposed['node_type']})",
                "Resource group": snap.resource_group,
                "Region": AZURE_REGIONS.get(snap.cloud_region, snap.cloud_region),
                "Note": "workspace creation can take several minutes",
            },
            "Azure Databricks",
        )
        if not approved:
            return _not_approved(reply, "provision_databricks")
        try:
            return svc._provision_databricks_real(proposed, sid, current_phase(sid))
        except Exception as exc:
            logger.warning("provision_databricks failed: %s", exc, extra={"phase": "-", "session_id": sid})
            return f"ERROR: Databricks provisioning failed. {_explain_azure_error(exc)}"

    @tool
    def upload_notebooks(config: RunnableConfig) -> str:
        """Upload the migration notebooks (generic + source-specific) to the Databricks workspace."""
        sid = _session_id(config)
        snap = svc.snapshot(sid)
        if not snap.databricks_ready:
            return "ERROR: provision Databricks first."
        if not snap.databricks_pat_available:
            return "ERROR: no Databricks personal access token yet. Ask the user to paste one (starts with 'dapi')."
        try:
            return svc._upload_notebooks_real({"source_system": snap.source_system}, sid, current_phase(sid))
        except Exception as exc:
            logger.warning("upload_notebooks failed: %s", exc, extra={"phase": "-", "session_id": sid})
            return f"ERROR: notebook upload failed: {str(exc)[:400]}"

    @tool
    def confirm_manual_step(step: Literal["metadata_loaded"], config: RunnableConfig) -> str:
        """Record that the user confirmed a step they performed manually.
        'metadata_loaded' = they ran /migrations/generic/01_load_metadata in Databricks successfully."""
        sid = _session_id(config)
        repo.patch_phase_data(sid, {step: True})
        return f"Recorded: {step}."

    @tool
    def configure_pipeline_tables(config: RunnableConfig, include_tables: list[str] | None = None) -> str:
        """Choose which extracted tables the pipeline will migrate. Omit include_tables to include all of them."""
        sid = _session_id(config)
        tables = list(session(sid).get("phase_data", {}).get("pipeline_tables", []) or [])
        if not tables:
            return "ERROR: no tables yet; extract metadata first."
        known = {str(t["table_name"]).upper(): t for t in tables}
        if include_tables:
            wanted = {name.strip().upper() for name in include_tables}
            unknown = sorted(wanted - known.keys())
            if unknown:
                return f"ERROR: unknown table(s): {', '.join(unknown)}. Available: {', '.join(known)}."
        else:
            wanted = set(known)
        updated = [{**t, "load_flag": str(t["table_name"]).upper() in wanted} for t in tables]
        repo.patch_phase_data(sid, {"pipeline_tables": updated, "pipeline_configured": True})
        return f"Saved pipeline selection: {len(wanted)} of {len(tables)} tables ({', '.join(sorted(wanted))})."

    @tool
    def provision_adf(config: RunnableConfig) -> str:
        """Create the Azure Data Factory that orchestrates the landing → bronze → silver notebooks. Shows an approval card first."""
        sid = _session_id(config)
        snap = svc.snapshot(sid)
        if snap.adf_ready:
            return f"Data Factory already exists: {snap.adf.get('factory_name')}."
        if not snap.vault_ready:
            return "ERROR: create the Key Vault first."
        factory = f"migration-adf-{sid[:6]}"
        approved, reply = _request_approval(
            "provision_adf",
            "Create Azure Data Factory",
            {"Factory": factory, "Resource group": snap.resource_group,
             "Region": AZURE_REGIONS.get(snap.cloud_region, snap.cloud_region)},
            "Azure Data Factory",
        )
        if not approved:
            return _not_approved(reply, "provision_adf")
        try:
            return svc._provision_adf_real({"factory_name": factory}, sid, current_phase(sid))
        except Exception as exc:
            logger.warning("provision_adf failed: %s", exc, extra={"phase": "-", "session_id": sid})
            return f"ERROR: Data Factory creation failed. {_explain_azure_error(exc)}"

    return [
        select_source_system,
        set_cloud_target,
        set_resource_group,
        verify_azure_credentials,
        create_secrets_vault,
        provision_storage,
        extract_source_metadata,
        provision_databricks,
        upload_notebooks,
        confirm_manual_step,
        configure_pipeline_tables,
        provision_adf,
    ]
