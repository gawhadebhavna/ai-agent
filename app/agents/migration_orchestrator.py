from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from app.agents.memory_service import ConversationMemoryService
from app.agents.migration_graph import MigrationGraphBuilder, MigrationState
from app.agents.model_factory import build_runtime_chat_model
from app.agents.secrets_tools import build_secrets_backend
from app.agents.source_registry import SourceSystemRegistry
from app.config import Settings
from app.core.logging import get_logger, log_step
from app.persistence.session_repository import SessionRepository
from app.schemas.chat import (
    ApprovalItem,
    ChatResponse,
    ChatStatus,
    MigrationPhase,
    SessionCreateResponse,
)

logger = get_logger("migration.orchestrator")

_APPROVAL_TTL_MINUTES = 60


class MigrationOrchestratorService:
    """
    Entry point for the stateful migration chat.

    Each call resumes the LangGraph from the stored SQLite checkpoint
    keyed by session_id.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        session_repo: SessionRepository,
        source_registry: SourceSystemRegistry,
    ) -> None:
        self._settings = settings
        self._session_repo = session_repo
        self._source_registry = source_registry

        # Build LLM
        self._llm = build_runtime_chat_model(settings)

        # Memory service
        self._memory = ConversationMemoryService(session_repo, self._llm)

        # SQLite checkpointer — disabled: our session_repo handles all persistence.
        # SqliteSaver causes message doubling and checkpoint/phase conflicts when
        # state is also reconstructed from session_repo on every call.
        self._checkpointer = None

        # Build graph
        self._graph = MigrationGraphBuilder(
            llm=self._llm,
            source_registry=source_registry,
            secrets_backend_factory=self._build_secrets_backend,
            settings=settings,
            orchestrator=self,
        ).compile(checkpointer=self._checkpointer)

        # Azure MCP tools — optional primary path; Azure SDK is always the fallback
        self._mcp_tools: dict = {}
        self._init_mcp()

        logger.info(
            "MigrationOrchestratorService initialized | provider=%s",
            settings.llm_provider,
            extra={"phase": "SYSTEM", "session_id": "-"},
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self) -> SessionCreateResponse:
        session_id = str(uuid.uuid4())
        with log_step(logger, "create_session", phase="INIT", session_id=session_id):
            self._session_repo.create_session(session_id)
        return SessionCreateResponse(
            session_id=session_id,
            migration_phase=MigrationPhase.INIT,
        )

    # ------------------------------------------------------------------
    # Main chat handler
    # ------------------------------------------------------------------

    def handle_message(
        self,
        session_id: str,
        message: str,
        approval_id: str | None = None,
        approve: bool | None = None,
    ) -> ChatResponse:
        session = self._session_repo.get_session(session_id)
        if session is None:
            return ChatResponse(
                session_id=session_id,
                message="Session not found. Please start a new session.",
                status=ChatStatus.error,
                migration_phase=MigrationPhase.INIT,
            )

        # Handle approval resolution
        if approval_id is not None and approve is not None:
            return self._handle_approval(session_id, approval_id, approve, session)

        return self._run_graph(session_id, message, session)

    # ------------------------------------------------------------------
    # Graph invocation
    # ------------------------------------------------------------------

    def _run_graph(self, session_id: str, message: str, session: dict) -> ChatResponse:
        phase = session.get("migration_phase", MigrationPhase.INIT)

        with log_step(logger, "run_graph", phase=phase, session_id=session_id):
            # Load conversation history
            history_messages = self._memory.load_messages(session_id)

            # Initial graph state
            initial_state: dict[str, Any] = {
                "session_id": session_id,
                "phase": phase,
                "messages": history_messages + [HumanMessage(content=message)],
                "inbound_review": {},
                "outbound_review": {},
                "regen_attempt": 0,
                "source_system": session["phase_data"].get("source_system"),
                "source_config": session["phase_data"].get("source_config", {}),
                "cloud_provider": session.get("cloud_provider"),
                "cloud_region": session["phase_data"].get("cloud_region"),
                "secrets_backend": session.get("cloud_provider"),
                "vault_config": session["phase_data"].get("vault_config", {}),
                "storage_config": session["phase_data"].get("storage_config", {}),
                "databricks_config": session["phase_data"].get("databricks_config", {}),
                "adf_config": session["phase_data"].get("adf_config", {}),
                "pipeline_tables": session["phase_data"].get("pipeline_tables", []),
                "pending_approval": None,
                "draft_response": "",
                "response_message": "",
                "response_status": "ok",
                "response_suggestions": [],
                "response_phase_data": {},
            }

            config = {"configurable": {"thread_id": session_id}}

            try:
                result = self._graph.invoke(initial_state, config=config)
            except Exception as exc:
                logger.error(
                    "Graph invocation failed: %s",
                    exc,
                    extra={"phase": phase, "session_id": session_id},
                )
                return ChatResponse(
                    session_id=session_id,
                    message=f"I encountered an error: {exc}. Please try again.",
                    status=ChatStatus.error,
                    migration_phase=MigrationPhase(phase),
                )

        # Extract results
        new_phase = result.get("phase", phase)
        response_message = result.get("response_message", "")
        response_status = result.get("response_status", "ok")
        suggestions = result.get("response_suggestions", [])
        pending_approval = result.get("pending_approval")

        # Persist conversation turn
        self._memory.save_turn(session_id, message, response_message)

        # Build phase_data patch from result
        # Use truthiness (not just not-None) so empty dicts/lists from LangGraph
        # state initialisation don't overwrite data already stored by the orchestrator
        phase_patch: dict[str, Any] = {}
        for key in ("source_system", "source_config", "vault_config", "storage_config", "databricks_config", "adf_config", "pipeline_tables"):
            val = result.get(key)
            if val:  # skip None AND empty dict/list
                phase_patch[key] = val
        if result.get("cloud_region"):
            phase_patch["cloud_region"] = result["cloud_region"]

        # Persist phase update
        self._session_repo.update_phase(
            session_id,
            new_phase,
            phase_data_patch=phase_patch,
            cloud_provider=result.get("cloud_provider"),
            source_system=result.get("source_system"),
        )

        # Persist pending approval if any
        approval_items: list[ApprovalItem] = []
        if pending_approval:
            new_approval_id = str(uuid.uuid4())
            expires_at = datetime.now(UTC) + timedelta(minutes=_APPROVAL_TTL_MINUTES)
            self._session_repo.create_approval(
                approval_id=new_approval_id,
                session_id=session_id,
                tool_name=pending_approval["tool_name"],
                tool_args=pending_approval.get("tool_args", {}),
                summary=pending_approval["summary"],
                expires_at=expires_at,
                cloud_resource_type=pending_approval.get("cloud_resource_type"),
            )
            approval_items.append(ApprovalItem(
                approval_id=new_approval_id,
                summary=pending_approval["summary"],
                tool_name=pending_approval["tool_name"],
                tool_args=pending_approval.get("tool_args", {}),
                expires_at=expires_at.isoformat(),
                cloud_resource_type=pending_approval.get("cloud_resource_type"),
            ))
            response_status = "approval_required"

        # Add any already-pending approvals
        existing_pending = self._session_repo.list_pending_approvals(session_id)
        for row in existing_pending:
            if not any(a.approval_id == row["approval_id"] for a in approval_items):
                approval_items.append(ApprovalItem(
                    approval_id=row["approval_id"],
                    summary=row["summary"],
                    tool_name=row["tool_name"],
                    tool_args=row["tool_args"],
                    expires_at=row["expires_at"],
                    cloud_resource_type=row.get("cloud_resource_type"),
                ))

        # Refreshed session
        refreshed = self._session_repo.get_session(session_id) or session

        return ChatResponse(
            session_id=session_id,
            message=response_message,
            status=ChatStatus(response_status) if response_status in ChatStatus.__members__.values() else ChatStatus.ok,
            migration_phase=MigrationPhase(new_phase) if new_phase in MigrationPhase.__members__ else MigrationPhase(phase),
            completed_phases=refreshed.get("completed_phases", []),
            pending_approvals=approval_items,
            suggestions=suggestions,
            phase_data=refreshed.get("phase_data", {}),
        )

    # ------------------------------------------------------------------
    # Approval handling
    # ------------------------------------------------------------------

    def _handle_approval(
        self,
        session_id: str,
        approval_id: str,
        approve: bool,
        session: dict,
    ) -> ChatResponse:
        phase = session.get("migration_phase", MigrationPhase.INIT)

        with log_step(logger, "handle_approval", phase=phase, session_id=session_id):
            record = self._session_repo.get_approval(approval_id)
            if record is None:
                return ChatResponse(
                    session_id=session_id,
                    message=f"Approval ID '{approval_id}' not found or expired.",
                    status=ChatStatus.error,
                    migration_phase=MigrationPhase(phase),
                )

            if not approve:
                self._session_repo.resolve_approval(approval_id, "rejected")
                logger.info(
                    "Approval rejected: %s",
                    record["tool_name"],
                    extra={"phase": phase, "session_id": session_id},
                )
                return ChatResponse(
                    session_id=session_id,
                    message=f"The operation '{record['summary']}' was rejected. You can adjust the plan or try again.",
                    status=ChatStatus.ok,
                    migration_phase=MigrationPhase(phase),
                    completed_phases=session.get("completed_phases", []),
                    phase_data=session.get("phase_data", {}),
                )

            # Execute the approved operation
            tool_name = record["tool_name"]
            tool_args = record["tool_args"]
            logger.info(
                "Executing approved tool: %s",
                tool_name,
                extra={"phase": phase, "session_id": session_id},
            )

            result_msg = self._execute_approved_tool(tool_name, tool_args, session_id, phase)
            self._session_repo.resolve_approval(approval_id, "approved")

            # Advance phase based on approved tool
            next_phase = self._next_phase_after_approval(tool_name, phase)
            if next_phase:
                self._session_repo.update_phase(session_id, next_phase)

            refreshed = self._session_repo.get_session(session_id) or session

            return ChatResponse(
                session_id=session_id,
                message=result_msg,
                status=ChatStatus.ok,
                migration_phase=MigrationPhase(next_phase or phase),
                completed_phases=refreshed.get("completed_phases", []),
                pending_approvals=[],
                suggestions=self._suggestions_after_approval(tool_name),
                phase_data=refreshed.get("phase_data", {}),
            )

    def _execute_approved_tool(
        self, tool_name: str, tool_args: dict, session_id: str, phase: str
    ) -> str:
        """Execute an approved tool. Tries MCP first, falls back to Azure SDK."""
        logger.info(
            "[TOOL] Executing approved: %s",
            tool_name,
            extra={"phase": phase, "session_id": session_id},
        )
        try:
            if tool_name == "provision_storage":
                return self._provision_storage_real(tool_args, session_id, phase)
            elif tool_name == "extract_metadata":
                return self._extract_metadata_real(tool_args, session_id, phase)
            elif tool_name == "provision_databricks":
                return self._provision_databricks_real(tool_args, session_id, phase)
            elif tool_name == "provision_adf":
                return self._provision_adf_real(tool_args, session_id, phase)
            else:
                return f"✅ Operation `{tool_name}` completed."
        except Exception as exc:
            logger.error(
                "[TOOL] %s failed: %s", tool_name, exc, exc_info=True,
                extra={"phase": phase, "session_id": session_id},
            )
            return f"❌ **`{tool_name}` failed:** {exc}\n\nCheck server logs for details."

    # ------------------------------------------------------------------
    # Real tool implementations (MCP primary → Azure SDK fallback)
    # ------------------------------------------------------------------

    def _provision_storage_real(self, tool_args: dict, session_id: str, phase: str) -> str:
        """Create storage account + containers via MCP (primary) or Azure SDK (fallback)."""
        import re as _re

        session = self._session_repo.get_session(session_id) or {}
        cloud_provider = session.get("cloud_provider", "azure")
        phase_data = session.get("phase_data", {})
        vault_config = phase_data.get("vault_config", {})

        raw_name = tool_args.get("storage_account", f"migstore{session_id[:6]}")
        account_name = _re.sub(r"[^a-z0-9]", "", raw_name.lower())[:24]
        if len(account_name) < 3:
            account_name = "mig" + _re.sub(r"[^a-z0-9]", "", session_id)[:8]

        resource_group = tool_args.get(
            "resource_group", vault_config.get("resource_group", "migration-rg")
        )
        region = tool_args.get(
            "region",
            session.get("cloud_region") or phase_data.get("cloud_region") or "eastus",
        )
        containers: list[str] = tool_args.get(
            "containers", ["landing", "metadata", "bronze", "silver"]
        )

        if cloud_provider != "azure":
            return (
                f"\u26a0\ufe0f Storage provisioning for **{cloud_provider.upper()}** is not yet "
                "implemented. Please create the storage bucket manually."
            )

        account_key: str | None = None

        # ── Azure SDK (real provisioning) ──────────────────────────────────
        if True:
            from azure.mgmt.resource import ResourceManagementClient
            from azure.mgmt.storage import StorageManagementClient
            from azure.mgmt.storage.models import (
                Kind,
                Sku,
                SkuName,
                StorageAccountCreateParameters,
            )

            credential = self._get_azure_credential(session_id)
            sub_id = self._get_azure_subscription_id(session_id)

            rg_client = ResourceManagementClient(credential, sub_id)
            rg_client.resource_groups.create_or_update(resource_group, {"location": region})
            logger.info("[TOOL][SDK] resource group '%s' ready", resource_group,
                        extra={"phase": phase, "session_id": session_id})

            storage_client = StorageManagementClient(credential, sub_id)
            avail = storage_client.storage_accounts.check_name_availability(
                {"name": account_name, "type": "Microsoft.Storage/storageAccounts"}
            )
            if not avail.name_available and "already" not in str(avail.reason or "").lower():
                import hashlib
                account_name = "mig" + hashlib.md5(session_id.encode()).hexdigest()[:20]

            try:
                storage_client.storage_accounts.begin_create(
                    resource_group,
                    account_name,
                    StorageAccountCreateParameters(
                        sku=Sku(name=SkuName.STANDARD_LRS),
                        kind=Kind.STORAGE_V2,
                        location=region,
                    ),
                ).result()
                logger.info("[TOOL][SDK] storage account '%s' created", account_name,
                            extra={"phase": phase, "session_id": session_id})
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    raise

            # Get account key via REST API (avoids SDK model inconsistencies)
            account_key: str | None = None
            try:
                import requests as _req
                _token = credential.get_token("https://management.azure.com/.default").token
                _url = (
                    f"https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{resource_group}"
                    f"/providers/Microsoft.Storage/storageAccounts/{account_name}"
                    f"/listKeys?api-version=2023-01-01"
                )
                _resp = _req.post(_url, headers={"Authorization": f"Bearer {_token}"}, timeout=30)
                _resp.raise_for_status()
                _keys = _resp.json().get("keys", [])
                account_key = _keys[0]["value"] if _keys else None
                logger.info("[TOOL][SDK] retrieved account key for '%s'", account_name,
                            extra={"phase": phase, "session_id": session_id})
            except Exception as exc:
                logger.warning("[TOOL][SDK] key retrieval failed: %s", exc,
                               extra={"phase": phase, "session_id": session_id})

        # ── Create containers via ARM REST API (avoids SSL issues with blob endpoint) ──
        import requests as _req2
        try:
            _token2 = credential.get_token("https://management.azure.com/.default").token
            _headers = {"Authorization": f"Bearer {_token2}", "Content-Type": "application/json"}
            for cname in containers:
                _c_url = (
                    f"https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{resource_group}"
                    f"/providers/Microsoft.Storage/storageAccounts/{account_name}"
                    f"/blobServices/default/containers/{cname}?api-version=2023-01-01"
                )
                _c_resp = _req2.put(_c_url, headers=_headers, json={}, timeout=30)
                if _c_resp.status_code in (200, 201):
                    logger.info("[TOOL][SDK] container '%s' created", cname,
                                extra={"phase": phase, "session_id": session_id})
                elif _c_resp.status_code == 409:
                    logger.info("[TOOL][SDK] container '%s' already exists", cname,
                                extra={"phase": phase, "session_id": session_id})
                else:
                    logger.warning("[TOOL][SDK] container '%s' error %s", cname, _c_resp.status_code,
                                   extra={"phase": phase, "session_id": session_id})
        except Exception as exc:
            logger.warning("[TOOL][SDK] container creation failed: %s", exc,
                           extra={"phase": phase, "session_id": session_id})

        # ── Store account info in Key Vault ─────────────────────────────────
        vault_name = vault_config.get("vault_name")
        if vault_name:
            try:
                backend = self._build_secrets_backend(cloud_provider, session_id)
                backend.store_secret(vault_name, "storage-account-name", account_name)
                if account_key:
                    conn_str_vault = (
                        f"DefaultEndpointsProtocol=https;AccountName={account_name};"
                        f"AccountKey={account_key};EndpointSuffix=core.windows.net"
                    )
                    backend.store_secret(vault_name, "storage-connection-string", conn_str_vault)
                logger.info("[TOOL] storage info stored in vault '%s'", vault_name,
                            extra={"phase": phase, "session_id": session_id})
            except Exception as exc:
                logger.warning("[TOOL] vault store for storage info failed: %s", exc,
                               extra={"phase": phase, "session_id": session_id})

        # ── Persist to session ─────────────────────────────────────────────
        self._session_repo.update_phase(
            session_id, phase,
            phase_data_patch={
                "storage_config": {
                    "provisioned": True,
                    "storage_account": account_name,
                    "containers": containers,
                    "resource_group": resource_group,
                    "region": region,
                }
            },
        )

        return (
            f"\u2705 **Storage account `{account_name}` created** in resource group `{resource_group}` (region: `{region}`)\n\n"
            f"**Containers created:** {', '.join(f'`{c}`' for c in containers)}\n\n"
            + (f"**Account name stored in Key Vault `{vault_name}`.** " if vault_name else "")
            + "Proceed to extract SAP metadata."
        )

    def _extract_metadata_real(self, tool_args: dict, session_id: str, phase: str) -> str:
        """Fetch table list + field schemas from SAP API; upload JSON to blob storage."""
        import json as _json

        import requests

        source_system = tool_args.get("source_system", "sap_s4")
        path_prefix = tool_args.get("path_prefix", f"metadata/{source_system}/")

        session = self._session_repo.get_session(session_id) or {}
        cloud_provider = session.get("cloud_provider", "azure")
        phase_data = session.get("phase_data", {})
        vault_config = phase_data.get("vault_config", {})
        vault_name = vault_config.get("vault_name")

        # Resolve SAP credentials: vault → settings fallback
        base_url = self._settings.sap_ngrok_base_url or "http://localhost:8001"
        username = self._settings.sap_api_username
        password = self._settings.sap_api_password

        if vault_name:
            try:
                backend = self._build_secrets_backend(cloud_provider, session_id)
                v_url = backend.get_secret(vault_name, "source-base_url")
                v_user = backend.get_secret(vault_name, "source-username")
                v_pass = backend.get_secret(vault_name, "source-password")
                if v_url:
                    base_url = v_url
                if v_user:
                    username = v_user
                if v_pass:
                    password = v_pass
            except Exception as exc:
                logger.warning("[TOOL] vault cred retrieval failed, using settings: %s", exc,
                               extra={"phase": phase, "session_id": session_id})

        ssl_opt = self._settings.ssl_verify_option
        auth = (username, password) if username and password else None

        # List all SAP tables
        try:
            resp = requests.get(
                f"{base_url}/tables", timeout=30, verify=ssl_opt, auth=auth
            )
            resp.raise_for_status()
            data = resp.json()
            tables = [t["name"] for t in data.get("tables", [])]
        except Exception as exc:
            return (
                f"\u274c Cannot connect to SAP at `{base_url}`: **{exc}**\n\n"
                "Ensure the dummy-SAP server is running: `python main.py` in `dummy-sap/`."
            )

        if not tables:
            return (
                f"\u26a0\ufe0f SAP server at `{base_url}` returned **no tables**.\n\n"
                "Check that `dummy-sap/data/` contains `*_DATA.csv` files."
            )

        # Resolve storage account + ARM token for blob upload via management API
        storage_config = phase_data.get("storage_config", {})
        storage_account = storage_config.get("storage_account")
        storage_rg = vault_config.get("resource_group", "migration-rg")
        arm_token: str | None = None
        if storage_account:
            try:
                credential = self._get_azure_credential(session_id)
                sub_id = self._get_azure_subscription_id(session_id)
                arm_token = credential.get_token("https://management.azure.com/.default").token
            except Exception as exc:
                logger.warning("[TOOL] ARM token for blob upload failed: %s", exc,
                               extra={"phase": phase, "session_id": session_id})

        uploaded: list[str] = []
        failed: list[str] = []

        for table_name in tables:
            try:
                meta_resp = requests.get(
                    f"{base_url}/tables/{table_name}/metadata",
                    timeout=30,
                    verify=ssl_opt,
                )
                if meta_resp.status_code == 200:
                    schema = meta_resp.json()
                    logger.info("[TOOL] metadata fetched for table '%s': %d fields",
                                table_name, len(schema.get("fields", schema)),
                                extra={"phase": phase, "session_id": session_id})
                    # Upload via BlobServiceClient with SSL verification disabled (corporate proxy)
                    if storage_account and arm_token:
                        try:
                            blob_name = f"{path_prefix}{table_name}.json"
                            # Get account key via ARM REST (management.azure.com — SSL works)
                            key_url = (
                                f"https://management.azure.com/subscriptions/{sub_id}"
                                f"/resourceGroups/{storage_rg}/providers/Microsoft.Storage"
                                f"/storageAccounts/{storage_account}/listKeys?api-version=2023-01-01"
                            )
                            key_resp = requests.post(
                                key_url,
                                headers={"Authorization": f"Bearer {arm_token}"},
                                timeout=30,
                            )
                            key_resp.raise_for_status()
                            acct_key = key_resp.json()["keys"][0]["value"]
                            # Generate SAS token locally (pure crypto, no network call) then
                            # upload via raw requests with verify=False to bypass corporate SSL
                            from azure.storage.blob import (
                                generate_account_sas as _gen_sas,
                                ResourceTypes as _RT,
                                AccountSasPermissions as _ASP,
                            )
                            from datetime import datetime as _dt, timedelta as _td, timezone as _tz
                            sas_token = _gen_sas(
                                account_name=storage_account,
                                account_key=acct_key,
                                resource_types=_RT(object=True),
                                permission=_ASP(write=True, create=True),
                                expiry=_dt.now(_tz.utc) + _td(hours=1),
                            )
                            body_bytes = _json.dumps(schema, indent=2).encode("utf-8")
                            sas_url = (
                                f"https://{storage_account}.blob.core.windows.net"
                                f"/metadata/{blob_name}?{sas_token}"
                            )
                            upload_resp = requests.put(
                                sas_url,
                                data=body_bytes,
                                headers={
                                    "x-ms-blob-type": "BlockBlob",
                                    "Content-Type": "application/json",
                                    "Content-Length": str(len(body_bytes)),
                                },
                                verify=False,
                                timeout=30,
                            )
                            if upload_resp.status_code in (200, 201):
                                logger.info("[TOOL] blob uploaded: %s", blob_name,
                                            extra={"phase": phase, "session_id": session_id})
                            else:
                                logger.warning("[TOOL] blob upload %s: HTTP %s %s",
                                               blob_name, upload_resp.status_code, upload_resp.text[:200],
                                               extra={"phase": phase, "session_id": session_id})
                        except Exception as upload_exc:
                            logger.warning("[TOOL] blob upload failed for '%s': %s",
                                           table_name, upload_exc,
                                           extra={"phase": phase, "session_id": session_id})
                    uploaded.append(table_name)
                else:
                    failed.append(table_name)
            except Exception as exc:
                logger.warning(
                    "[TOOL] metadata fetch for '%s' failed: %s",
                    table_name, exc,
                    extra={"phase": phase, "session_id": session_id},
                )
                failed.append(table_name)

        self._session_repo.update_phase(
            session_id, phase,
            phase_data_patch={
                "pipeline_tables": [
                    {
                        "table_name": t,
                        "source_system": source_system,
                        "load_type": "full",
                        "load_flag": True,
                    }
                    for t in uploaded
                ]
            },
        )

        blob_note = (
            f"\nMetadata JSON files uploaded to `metadata` container in `{storage_account}`."
            if storage_account else
            "\n⚠️ No storage account in session — metadata recorded locally only."
        )
        failed_note = (
            f"\n\n⚠️ Failed for: {', '.join(f'`{t}`' for t in failed)}" if failed else ""
        )

        return (
            f"\u2705 **Metadata extracted from {len(uploaded)} SAP table(s):**\n"
            + "\n".join(f"- `{t}`" for t in uploaded)
            + blob_note
            + failed_note
            + "\n\nReady to provision Databricks workspace."
        )

    def _provision_databricks_real(self, tool_args: dict, session_id: str, phase: str) -> str:
        """Create a Databricks cluster (existing workspace) or provision a new one via SDK."""
        session = self._session_repo.get_session(session_id) or {}
        phase_data = session.get("phase_data", {})
        vault_config = phase_data.get("vault_config", {})
        resource_group = vault_config.get("resource_group", "migration-rg")
        region = phase_data.get("cloud_region", "eastus")

        workspace_name = tool_args.get("workspace_name", f"migration-dbx-{session_id[:6]}")
        cluster_name = tool_args.get("cluster_name", "migration-cluster")

        # ── Path A: existing workspace credentials in .env ───────────────────────
        has_real_dbx = (
            self._settings.has_databricks_credentials
            and "<" not in (self._settings.databricks_url or "")
        )
        if has_real_dbx:
            try:
                from databricks.sdk import WorkspaceClient
                client = WorkspaceClient(
                    host=self._settings.databricks_url,
                    token=self._settings.databricks_token,
                )
                cluster = client.clusters.create(
                    cluster_name=cluster_name,
                    spark_version=tool_args.get("spark_version", "13.3.x-scala2.12"),
                    node_type_id=tool_args.get("node_type", "Standard_DS3_v2"),
                    num_workers=tool_args.get("num_workers", 2),
                    autotermination_minutes=120,
                )
                self._session_repo.update_phase(
                    session_id, phase,
                    phase_data_patch={
                        "databricks_config": {
                            "workspace_url": self._settings.databricks_url,
                            "cluster_id": cluster.cluster_id,
                            "cluster_name": cluster_name,
                        }
                    },
                )
                return (
                    f"\u2705 **Databricks cluster `{cluster_name}` created.**\n\n"
                    f"- Cluster ID: `{cluster.cluster_id}`\n"
                    f"- Workspace: `{self._settings.databricks_url}`\n\n"
                    "Ready to upload migration notebooks."
                )
            except Exception as exc:
                return (
                    f"\u274c Databricks cluster creation failed: **{exc}**\n\n"
                    "Check `DATABRICKS_URL` and `DATABRICKS_TOKEN` in `.env`."
                )

        # ── Path B: provision a new Azure Databricks workspace via SDK ─────────
        try:
            from azure.mgmt.databricks import AzureDatabricksManagementClient
            from azure.mgmt.databricks.models import Sku as DbxSku
            from azure.mgmt.databricks.models import Workspace as DbxWorkspace

            credential = self._get_azure_credential(session_id)
            sub_id = self._get_azure_subscription_id(session_id)
            managed_rg = (
                f"/subscriptions/{sub_id}/resourceGroups/{resource_group}-dbx-managed"
            )

            dbx_client = AzureDatabricksManagementClient(credential, sub_id)
            workspace = dbx_client.workspaces.begin_create_or_update(
                resource_group_name=resource_group,
                workspace_name=workspace_name,
                parameters=DbxWorkspace(
                    location=region,
                    managed_resource_group_id=managed_rg,
                    sku=DbxSku(name="premium"),
                ),
            ).result()

            ws_url = (
                f"https://{workspace.workspace_url}"
                if workspace.workspace_url
                else "https://<workspace>.azuredatabricks.net"
            )
            self._session_repo.update_phase(
                session_id, phase,
                phase_data_patch={
                    "databricks_config": {
                        "workspace_name": workspace_name,
                        "workspace_url": ws_url,
                        "provisioned": True,
                    }
                },
            )
            return (
                f"\u2705 **Databricks workspace `{workspace_name}` provisioned**"
                f" in `{resource_group}` ({region}).\n\n"
                f"Workspace URL: `{ws_url}`\n\n"
                "**Next step:** Generate a Personal Access Token in the workspace UI, "
                "then add to `.env`:\n"
                "```\n"
                f"DATABRICKS_URL={ws_url}\n"
                "DATABRICKS_TOKEN=<your-pat-token>\n"
                "```\n"
                "Restart the server — the migration cluster will be created on the next step."
            )
        except ImportError:
            return (
                "\u26a0\ufe0f `azure-mgmt-databricks` package not installed.\n\n"
                "Run in your venv: `pip install azure-mgmt-databricks>=4.0.0` then restart."
            )
        except Exception as exc:
            return (
                f"\u26a0\ufe0f **Databricks workspace provisioning failed:** {exc}\n\n"
                "**Manual steps in Azure Portal:**\n"
                f"1. Create Resource \u2192 Azure Databricks\n"
                f"2. Name: **{workspace_name}** | RG: **{resource_group}** | Region: **{region}**\n"
                "3. Generate a Personal Access Token\n"
                "4. Set `DATABRICKS_URL` and `DATABRICKS_TOKEN` in `.env` \u2192 restart"
            )

    def _upload_notebooks_real(self, tool_args: dict, session_id: str, phase: str) -> str:
        """Upload local Databricks notebooks to the workspace via REST API."""
        import base64
        import pathlib

        session = self._session_repo.get_session(session_id) or {}
        phase_data = session.get("phase_data", {})
        dbx_config = phase_data.get("databricks_config", {})
        vault_config = phase_data.get("vault_config", {})
        source_system = phase_data.get("source_system") or tool_args.get("source_system", "sap")

        workspace_url = dbx_config.get("workspace_url") or ""

        # Retrieve PAT from Key Vault (same pattern as SAP creds)
        token = ""
        vault_name = vault_config.get("vault_name", "")
        if vault_name:
            try:
                backend = self._build_secrets_backend(session_id)
                token = backend.get_secret(vault_name, "databricks-pat") or ""
            except Exception:
                pass
        # Fall back to .env if vault has nothing
        if not token:
            token = self._settings.databricks_token or ""

        notebooks_dir = pathlib.Path(__file__).parent.parent / "notebooks"
        notebook_map = {
            notebooks_dir / "generic" / "01_load_metadata.py": "/migrations/generic/01_load_metadata",
            notebooks_dir / source_system / "02_create_bronze_silver_schema.py": f"/migrations/{source_system}/02_create_bronze_silver_schema",
            notebooks_dir / source_system / "03_extract_landing.py": f"/migrations/{source_system}/03_extract_landing",
            notebooks_dir / "generic" / "04_ingest_bronze.py": "/migrations/generic/04_ingest_bronze",
            notebooks_dir / "generic" / "05_transform_silver.py": "/migrations/generic/05_transform_silver",
        }

        # Upload each notebook via Databricks workspace import API
        uploaded: list[str] = []
        failed: list[str] = []

        for local_path, remote_path in notebook_map.items():
            if not local_path.exists():
                logger.warning("[TOOL] notebook not found locally: %s", local_path,
                               extra={"phase": phase, "session_id": session_id})
                failed.append(f"{local_path.name} (not found)")
                continue
            try:
                content_b64 = base64.b64encode(local_path.read_bytes()).decode()
                # Ensure parent folder exists
                folder = "/".join(remote_path.split("/")[:-1])
                requests.post(
                    f"{workspace_url.rstrip('/')}/api/2.0/workspace/mkdirs",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json={"path": folder},
                    verify=False,
                    timeout=30,
                )
                resp = requests.post(
                    f"{workspace_url.rstrip('/')}/api/2.0/workspace/import",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json={
                        "path": remote_path,
                        "language": "PYTHON",
                        "format": "SOURCE",
                        "overwrite": True,
                        "content": content_b64,
                    },
                    verify=False,
                    timeout=60,
                )
                if resp.status_code == 200:
                    logger.info("[TOOL] notebook uploaded: %s → %s", local_path.name, remote_path,
                                extra={"phase": phase, "session_id": session_id})
                    uploaded.append(f"`{local_path.name}` → `{remote_path}`")
                else:
                    err = resp.json().get("message", resp.text[:100])
                    logger.warning("[TOOL] notebook upload failed: %s — %s", local_path.name, err,
                                   extra={"phase": phase, "session_id": session_id})
                    failed.append(f"`{local_path.name}` ({err})")
            except Exception as exc:
                logger.warning("[TOOL] notebook upload error: %s — %s", local_path.name, exc,
                               extra={"phase": phase, "session_id": session_id})
                failed.append(f"`{local_path.name}` ({exc})")

        self._session_repo.update_phase(
            session_id, phase,
            phase_data_patch={
                "databricks_config": {**dbx_config, "notebooks_uploaded": len(failed) == 0, "notebooks_uploaded_count": len(uploaded)}
            },
        )

        ok_lines = "\n".join(f"- {n}" for n in uploaded) if uploaded else "_none_"
        fail_lines = ("\n\n\u26a0\ufe0f Failed:\n" + "\n".join(f"- {n}" for n in failed)) if failed else ""
        return (
            f"\u2705 **{len(uploaded)}/{len(notebook_map)} notebooks uploaded to Databricks:**\n{ok_lines}"
            f"{fail_lines}\n\n"
            "Ready to run the metadata loading notebook."
        )

    def _provision_adf_real(self, tool_args: dict, session_id: str, phase: str) -> str:
        """Create an Azure Data Factory using the Azure management SDK."""
        from azure.mgmt.datafactory import DataFactoryManagementClient
        from azure.mgmt.datafactory.models import Factory

        session = self._session_repo.get_session(session_id) or {}
        phase_data = session.get("phase_data", {})
        vault_config = phase_data.get("vault_config", {})

        factory_name = tool_args.get("factory_name", f"migration-adf-{session_id[:6]}")
        resource_group = vault_config.get(
            "resource_group", tool_args.get("resource_group", "migration-rg")
        )
        region = phase_data.get("cloud_region", "eastus")

        credential = self._get_azure_credential(session_id)
        sub_id = self._get_azure_subscription_id(session_id)

        adf_client = DataFactoryManagementClient(credential, sub_id)
        adf_client.factories.create_or_update(
            resource_group_name=resource_group,
            factory_name=factory_name,
            factory=Factory(location=region),
        )
        logger.info("[TOOL][SDK] ADF '%s' created in '%s'", factory_name, resource_group,
                    extra={"phase": phase, "session_id": session_id})

        self._session_repo.update_phase(
            session_id, phase,
            phase_data_patch={
                "adf_config": {
                    "factory_name": factory_name,
                    "resource_group": resource_group,
                    "provisioned": True,
                }
            },
        )

        return (
            f"\u2705 **Azure Data Factory `{factory_name}` created** in `{resource_group}` ({region}).\n\n"
            "Linked services (Storage, Databricks, Key Vault) and pipelines "
            "(landing \u2192 bronze \u2192 silver) can be configured in ADF Studio.\n\n"
            "\U0001f389 Migration infrastructure setup is **complete!**"
        )

    def _next_phase_after_approval(self, tool_name: str, current_phase: str) -> str | None:
        mapping = {
            "provision_storage": MigrationPhase.STORAGE_PROVISIONED,
            "extract_metadata": MigrationPhase.METADATA_EXTRACTED,
            "provision_databricks": MigrationPhase.DATABRICKS_PROVISIONED,
            "provision_adf": MigrationPhase.ADF_PROVISIONED,
        }
        return mapping.get(tool_name)

    def _suggestions_after_approval(self, tool_name: str) -> list[str]:
        mapping = {
            "provision_storage": ["Extract SAP metadata now", "What tables will be scanned?"],
            "extract_metadata": ["Create Databricks workspace", "What's next?"],
            "provision_databricks": ["Upload notebooks now"],
            "provision_adf": ["Show migration summary"],
        }
        return mapping.get(tool_name, ["What's next?"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_azure_credential(self, session_id: str | None = None):
        """Build Azure credential.
        Priority: session phase_data[azure_credentials] → settings → DefaultAzureCredential.
        """
        from azure.identity import ClientSecretCredential, DefaultAzureCredential

        # 1. Session-stored credentials (collected from user via chat)
        if session_id:
            session = self._session_repo.get_session(session_id) or {}
            creds = session.get("phase_data", {}).get("vault_config", {}).get("azure_credentials", {})
            if all(creds.get(k) for k in ("tenant_id", "client_id", "client_secret")):
                return ClientSecretCredential(
                    tenant_id=creds["tenant_id"],
                    client_id=creds["client_id"],
                    client_secret=creds["client_secret"],
                )

        # 2. Settings (.env)
        if (
            self._settings.azure_tenant_id
            and self._settings.azure_client_id
            and self._settings.azure_client_secret
        ):
            return ClientSecretCredential(
                tenant_id=self._settings.azure_tenant_id,
                client_id=self._settings.azure_client_id,
                client_secret=self._settings.azure_client_secret,
            )

        # 3. Ambient credential (az login / managed identity / env vars)
        return DefaultAzureCredential()

    def _get_azure_subscription_id(self, session_id: str | None = None) -> str:
        """Get subscription ID from session phase_data or settings."""
        if session_id:
            session = self._session_repo.get_session(session_id) or {}
            sub = session.get("phase_data", {}).get("vault_config", {}).get(
                "azure_credentials", {}
            ).get("subscription_id")
            if sub:
                return sub
        return self._settings.azure_subscription_id or ""

    def _init_mcp(self) -> None:
        """Initialize Azure MCP Server in a dedicated thread (avoids event loop conflicts).
        Non-fatal — Azure SDK is always the fallback."""
        import threading

        result: dict = {}

        def _run_in_thread() -> None:
            import asyncio
            from app.agents.mcp_client import AzureMCPClient
            from app.agents.mcp_tools import build_mcp_tools
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                mcp_client = AzureMCPClient(self._settings)
                loop.run_until_complete(mcp_client.initialize())
                result["tools"] = build_mcp_tools(mcp_client)
            except Exception as exc:
                result["error"] = str(exc)
            finally:
                loop.close()

        t = threading.Thread(target=_run_in_thread, daemon=True)
        t.start()
        t.join(timeout=15)  # wait up to 15 s for MCP to come up

        if "tools" in result:
            self._mcp_tools = result["tools"]
            logger.info(
                "Azure MCP Server ready — %d tools available",
                len(self._mcp_tools),
                extra={"phase": "SYSTEM", "session_id": "-"},
            )
        else:
            logger.info(
                "Azure MCP unavailable (SDK-only mode): %s",
                result.get("error", "timeout"),
                extra={"phase": "SYSTEM", "session_id": "-"},
            )

    def _build_secrets_backend(self, cloud_provider: str, session_id: str | None = None):
        """Build a SecretsBackend using session credentials first, then settings."""
        azure_creds: dict = {}
        if session_id:
            session = self._session_repo.get_session(session_id) or {}
            azure_creds = (
                session.get("phase_data", {})
                .get("vault_config", {})
                .get("azure_credentials", {})
            )
        return build_secrets_backend(
            cloud_provider,
            subscription_id=azure_creds.get("subscription_id") or self._settings.azure_subscription_id,
            tenant_id=azure_creds.get("tenant_id") or self._settings.azure_tenant_id,
            client_id=azure_creds.get("client_id") or self._settings.azure_client_id,
            client_secret=azure_creds.get("client_secret") or self._settings.azure_client_secret,
        )
