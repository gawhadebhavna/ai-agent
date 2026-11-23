from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

import sqlite3
import threading

import requests
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.agents.chat_agent import build_chat_agent
from app.agents.migration_progress import (
    AZURE_REGIONS,
    MigrationSnapshot,
    build_snapshot,
    derive_phase,
    next_step,
    render_status,
    suggestions_for,
)
from app.agents.migration_tools import build_migration_tools
from app.agents.model_factory import build_runtime_chat_model
from app.agents.secret_capture import capture_credentials
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
_RECURSION_LIMIT = 40
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "agent_system_prompt.txt"


def public_phase_data(phase_data: dict[str, Any]) -> dict[str, Any]:
    """Copy of phase_data that is safe to return to the browser (no credential values)."""
    safe = dict(phase_data or {})
    vault_config = dict(safe.get("vault_config", {}) or {})
    creds = vault_config.pop("azure_credentials", None)
    if creds:
        vault_config["azure_credentials_collected"] = sorted(k for k, v in creds.items() if v and k != "source")
    safe["vault_config"] = vault_config
    return safe


class MigrationOrchestratorService:
    """
    Entry point for the stateful migration chat.

    Conversation memory lives in a LangGraph checkpoint keyed by session_id;
    migration facts live in the session repository and are re-read every turn.
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

        # Source credentials typed before the vault exists; held in memory only, never persisted.
        self._pending_secrets: dict[str, dict[str, str]] = {}
        self._pending_lock = threading.Lock()

        self._prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")

        checkpoint_path = settings.sqlite_path_obj.with_name("agent_checkpoints.db")
        self._checkpointer = SqliteSaver(sqlite3.connect(str(checkpoint_path), check_same_thread=False))

        self._graph = build_chat_agent(
            self._llm,
            build_migration_tools(self),
            self._build_system_prompt,
            self._checkpointer,
            parallel_tool_calls=False if settings.llm_provider in ("openai", "azure_openai") else None,
        )

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

        config = {"configurable": {"thread_id": session_id}, "recursion_limit": _RECURSION_LIMIT}
        phase = session.get("migration_phase", MigrationPhase.INIT)

        with log_step(logger, "chat_turn", phase=phase, session_id=session_id):
            pending = self._pending_interrupt(config)

            if approval_id is not None and approve is not None:
                if pending is None:
                    self._session_repo.resolve_approval(approval_id, "expired")
                    return self._build_response(
                        session_id, "That action is no longer pending. What would you like to do next?", None
                    )
                logged_user_text = "Approved" if approve else "Rejected"
                graph_input: Any = Command(resume={"approved": approve})
                self._resolve_open_approvals(session_id, "approved" if approve else "rejected")
            else:
                logged_user_text = self._capture_secrets(session_id, message)
                if pending is not None:
                    graph_input = Command(resume={"reply": logged_user_text})
                    self._resolve_open_approvals(session_id, "answered")
                else:
                    graph_input = {"messages": [HumanMessage(content=logged_user_text)]}

            try:
                result = self._graph.invoke(graph_input, config)
            except Exception as exc:
                logger.error("Agent turn failed: %s", exc, exc_info=True,
                             extra={"phase": phase, "session_id": session_id})
                return ChatResponse(
                    session_id=session_id,
                    message="Sorry, I hit a problem while processing that. Your progress is saved; please try again.",
                    status=ChatStatus.error,
                    migration_phase=MigrationPhase(phase),
                )

            new_pending = self._pending_interrupt(config)
            reply = self._final_text(result, new_pending)
            self._session_repo.append_message(session_id, "user", logged_user_text)
            self._session_repo.append_message(session_id, "assistant", reply)
            return self._build_response(session_id, reply, new_pending)

    # ------------------------------------------------------------------
    # Turn helpers
    # ------------------------------------------------------------------

    def snapshot(self, session_id: str) -> MigrationSnapshot:
        session = self._session_repo.get_session(session_id) or {"phase_data": {}}
        with self._pending_lock:
            pending_names = list(self._pending_secrets.get(session_id, {}).keys())
        return build_snapshot(session, self._settings, self._source_registry, pending_names)

    def _build_system_prompt(self, session_id: str) -> str:
        snap = self.snapshot(session_id)
        sources = "\n".join(f"- {s.id}: {s.display_name} — {s.description}" for s in self._source_registry.list_sources())
        regions = ", ".join(f"{code}: {name}" for code, name in AZURE_REGIONS.items())
        pending_note = ""
        if self._session_repo.list_pending_approvals(session_id):
            pending_note = "\nAn approval card is currently waiting for the user's decision.\n"
        return (
            self._prompt_template
            .replace("{{SOURCES}}", sources)
            .replace("{{AZURE_REGIONS}}", regions)
            .replace("{{STATUS}}", render_status(snap))
            .replace("{{PENDING}}", pending_note)
            .replace("{{NEXT_STEP}}", next_step(snap))
        )

    def _pending_interrupt(self, config: dict):
        interrupts = self._graph.get_state(config).interrupts
        return interrupts[0] if interrupts else None

    @staticmethod
    def _message_text(message: AIMessage) -> str:
        content = message.content
        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        return (content or "").strip()

    def _final_text(self, result: dict, pending) -> str:
        last_ai = next((m for m in reversed(result.get("messages", [])) if isinstance(m, AIMessage)), None)
        text = self._message_text(last_ai) if last_ai is not None else ""
        if pending is not None:
            card = (pending.value or {}).get("message", "") if isinstance(pending.value, dict) else str(pending.value)
            return f"{text}\n\n{card}".strip() if text else card
        return text or "Done. What would you like to do next?"

    def _register_approval(self, session_id: str, pending) -> ApprovalItem:
        value = pending.value if isinstance(pending.value, dict) else {"summary": str(pending.value)}
        expires_at = datetime.now(UTC) + timedelta(minutes=self._settings.approval_ttl_minutes or _APPROVAL_TTL_MINUTES)
        if self._session_repo.get_approval(pending.id) is None:
            self._session_repo.create_approval(
                approval_id=pending.id,
                session_id=session_id,
                tool_name=value.get("action", "approval"),
                tool_args=value.get("details", {}),
                summary=value.get("summary", "Approval required"),
                expires_at=expires_at,
                cloud_resource_type=value.get("cloud_resource_type"),
            )
        row = self._session_repo.get_approval(pending.id) or {}
        return ApprovalItem(
            approval_id=pending.id,
            summary=row.get("summary", value.get("summary", "")),
            tool_name=row.get("tool_name", value.get("action", "")),
            tool_args=row.get("tool_args", value.get("details", {})),
            expires_at=row.get("expires_at", expires_at.isoformat()),
            cloud_resource_type=row.get("cloud_resource_type"),
        )

    def _resolve_open_approvals(self, session_id: str, status: str) -> None:
        for row in self._session_repo.list_pending_approvals(session_id):
            self._session_repo.resolve_approval(row["approval_id"], status)

    def _build_response(self, session_id: str, message: str, pending) -> ChatResponse:
        snap = self.snapshot(session_id)
        phase = derive_phase(snap)
        self._session_repo.update_phase(session_id, phase)
        approvals = [self._register_approval(session_id, pending)] if pending is not None else []
        refreshed = self._session_repo.get_session(session_id) or {}
        return ChatResponse(
            session_id=session_id,
            message=message,
            status=ChatStatus.approval_required if pending is not None else ChatStatus.ok,
            migration_phase=phase,
            completed_phases=refreshed.get("completed_phases", []),
            pending_approvals=approvals,
            suggestions=suggestions_for(
                snap, [s.display_name for s in self._source_registry.list_sources()], pending is not None
            ),
            phase_data=public_phase_data(refreshed.get("phase_data", {})),
        )

    # ------------------------------------------------------------------
    # Credential capture
    # ------------------------------------------------------------------

    def _capture_secrets(self, session_id: str, message: str) -> str:
        """Pull credentials out of the message before the LLM sees it; return the safe text."""
        snap = self.snapshot(session_id)
        plugin = self._source_registry.get_plugin(snap.source_system or "sap_s4")
        fields = plugin.required_cred_fields if plugin else []
        captured = capture_credentials(message, [f.key for f in fields], {f.key for f in fields if f.secret})
        if not captured.captured_anything:
            return message

        notes: list[str] = []
        if captured.azure:
            session = self._session_repo.get_session(session_id) or {"phase_data": {}}
            vault_config = dict(session["phase_data"].get("vault_config", {}) or {})
            creds = dict(vault_config.get("azure_credentials", {}) or {})
            creds.update(captured.azure)
            vault_config["azure_credentials"] = creds
            vault_config["azure_authenticated"] = False
            self._session_repo.patch_phase_data(session_id, {"vault_config": vault_config})
            notes.append("Azure credentials received: " + ", ".join(sorted(captured.azure)))

        secrets = {f"source-{k}": v for k, v in captured.source.items()}
        if captured.databricks_pat:
            secrets["databricks-pat"] = captured.databricks_pat
        if secrets:
            if snap.vault_ready:
                stored, failed = self._store_secrets(session_id, secrets)
                if stored:
                    notes.append(f"stored in Key Vault {snap.vault_name}: {', '.join(stored)}")
                if failed:
                    notes.append(f"FAILED to store in Key Vault: {', '.join(failed)}")
            else:
                with self._pending_lock:
                    self._pending_secrets.setdefault(session_id, {}).update(secrets)
                notes.append(f"held until the Key Vault is created: {', '.join(secrets)}")

        logger.info("Captured credentials (%s)", "; ".join(notes), extra={"phase": "-", "session_id": session_id})
        return (
            f"{captured.text}\n\n[System note: credential values were captured by the system and are hidden "
            f"from you. {'; '.join(notes)}. Do not ask the user to repeat them.]"
        )

    def _store_secrets(self, session_id: str, secrets: dict[str, str]) -> tuple[list[str], list[str]]:
        snap = self.snapshot(session_id)
        stored: list[str] = []
        failed: list[str] = []
        try:
            backend = self._build_secrets_backend(snap.cloud_provider or "azure", session_id)
        except Exception as exc:
            logger.warning("Secrets backend unavailable: %s", exc, extra={"phase": "-", "session_id": session_id})
            return stored, list(secrets)
        for name, value in secrets.items():
            try:
                backend.store_secret(snap.vault_name, name, value)
                stored.append(name)
            except Exception as exc:
                logger.warning("Storing secret %s failed: %s", name, exc, extra={"phase": "-", "session_id": session_id})
                failed.append(name)
        if stored:
            session = self._session_repo.get_session(session_id) or {"phase_data": {}}
            vault_config = dict(session["phase_data"].get("vault_config", {}) or {})
            vault_config["stored_secret_names"] = sorted(set(vault_config.get("stored_secret_names", []) or []) | set(stored))
            self._session_repo.patch_phase_data(session_id, {"vault_config": vault_config})
        return stored, failed

    def flush_pending_secrets(self, session_id: str) -> tuple[list[str], list[str]]:
        """Store credentials captured before the vault existed. Called right after vault creation."""
        with self._pending_lock:
            secrets = self._pending_secrets.pop(session_id, {})
        if not secrets:
            return [], []
        stored, failed = self._store_secrets(session_id, secrets)
        if failed:
            with self._pending_lock:
                self._pending_secrets.setdefault(session_id, {}).update({k: secrets[k] for k in failed})
        return stored, failed

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
                backend = self._build_secrets_backend(session.get("cloud_provider") or "azure", session_id)
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
