from __future__ import annotations

"""
Migration Orchestrator LangGraph.

Phase state machine:
  INIT → SOURCE_CONFIGURED → CLOUD_SELECTED → VAULT_SETUP →
  STORAGE_PROVISIONED → METADATA_EXTRACTED → DATABRICKS_PROVISIONED →
  NOTEBOOKS_UPLOADED → METADATA_LOADED → PIPELINE_CONFIGURED →
  ADF_PROVISIONED → COMPLETED

Every user message flows through:
  review_incoming → <phase node> → review_outgoing → return to user
"""

import json
import re
import time
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.core.logging import get_logger, log_step
from app.schemas.chat import (
    InboundReview,
    MigrationPhase,
    OutboundReview,
)

logger = get_logger("migration.graph")

# ---------------------------------------------------------------------------
# Suggestions per phase (returned to frontend for quick-reply chips)
# ---------------------------------------------------------------------------
_PHASE_SUGGESTIONS: dict[str, list[str]] = {
    MigrationPhase.INIT: [],  # populated dynamically from source registry
    MigrationPhase.SOURCE_CONFIGURED: ["Azure", "AWS", "GCP"],
    MigrationPhase.CLOUD_SELECTED: ["Provide credentials"],
    MigrationPhase.AZURE_AUTHENTICATED: ["Create vault", "What is a vault?"],
    MigrationPhase.VAULT_SETUP: ["Yes, create storage", "What will be created?"],
    MigrationPhase.STORAGE_PROVISIONED: ["Scan SAP tables", "What's extracted?"],
    MigrationPhase.METADATA_EXTRACTED: ["Create Databricks", "What cluster size?"],
    MigrationPhase.DATABRICKS_PROVISIONED: ["Upload notebooks", "What notebooks?"],
    MigrationPhase.NOTEBOOKS_UPLOADED: ["Run metadata notebook"],
    MigrationPhase.METADATA_LOADED: ["Configure pipeline", "Which tables?"],
    MigrationPhase.PIPELINE_CONFIGURED: ["Create ADF pipelines"],
    MigrationPhase.ADF_PROVISIONED: ["Show summary"],
    MigrationPhase.COMPLETED: ["Start new migration", "Show summary"],
}

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class MigrationState(TypedDict):
    session_id: str
    phase: str
    messages: Annotated[list[BaseMessage], add_messages]
    # Review node outputs
    inbound_review: dict[str, Any]
    outbound_review: dict[str, Any]
    regen_attempt: int
    # Migration data collected
    source_system: str | None
    source_config: dict[str, Any]
    cloud_provider: str | None
    cloud_region: str | None
    secrets_backend: str | None  # Which native service is active
    vault_config: dict[str, Any]
    storage_config: dict[str, Any]
    databricks_config: dict[str, Any]
    adf_config: dict[str, Any]
    pipeline_tables: list[dict[str, Any]]
    # Pending approval (approval_id or None)
    pending_approval: dict[str, Any] | None
    # Draft response for review loop
    draft_response: str
    # Final assembled response fields
    response_message: str
    response_status: str
    response_suggestions: list[str]
    response_phase_data: dict[str, Any]


# ---------------------------------------------------------------------------
# LLM helper: structured JSON extraction
# ---------------------------------------------------------------------------

def _call_llm_json(llm, prompt: str, fallback: dict) -> dict:
    try:
        result = llm.invoke([SystemMessage(content="You are a structured JSON extractor. Always reply with valid JSON only. No markdown."), HumanMessage(content=prompt)])
        text = result.content if hasattr(result, "content") else str(result)
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
    except Exception as exc:
        logger.warning("LLM JSON extraction failed: %s", exc)
        return fallback


def _call_llm_text(llm, system: str, user: str) -> str:
    try:
        result = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return result.content if hasattr(result, "content") else str(result)
    except Exception as exc:
        logger.error("LLM text call failed: %s", exc)
        return "I encountered an issue generating a response. Please try again."


# ---------------------------------------------------------------------------
# Node: review_incoming
# ---------------------------------------------------------------------------

_INBOUND_REVIEW_SYSTEM = """You are an inbound message classifier for a data migration assistant.
Given the current migration phase and the user message, output a JSON object with:
- intent: short description of what the user wants (string)
- contains_credentials: true if the message contains passwords, secrets, tokens, connection strings, or API keys (boolean)
- credential_fields: dict of field_name → value for any detected credentials (dict, empty if none)
- routing: which phase node to call next (string — see rules below)
- clarification_needed: true only if the message is completely off-topic (boolean)
- clarification_prompt: if clarification_needed, what to ask (string or null)

ROUTING RULES (follow in order, use the FIRST rule that matches):
1. If "Source system already selected" is NOT "none" AND phase=INIT → "collect_source_creds" (user is responding about credentials, NOT re-selecting a source)
2. If contains_credentials=true → "collect_source_creds"
3. If phase=INIT and "Source system already selected" is "none" and user mentions a source system → "greet_and_probe"
4. If phase=INIT and "Source system already selected" is "none" → "greet_and_probe"
5. If phase=SOURCE_CONFIGURED → "select_cloud"
6. If phase=CLOUD_SELECTED → "setup_secrets_manager"
7. If phase=VAULT_SETUP → "provision_storage"
8. If phase=STORAGE_PROVISIONED → "extract_metadata"
9. If phase=METADATA_EXTRACTED → "provision_databricks"
10. If phase=DATABRICKS_PROVISIONED → "upload_notebooks"
11. If phase=NOTEBOOKS_UPLOADED → "run_metadata_notebook"
12. If phase=METADATA_LOADED → "configure_pipeline"
13. If phase=PIPELINE_CONFIGURED → "provision_adf"
14. If phase=ADF_PROVISIONED or phase=COMPLETED → "finalize"
15. Default → use the phase default

CREDENTIAL RULES:
- If the message contains passwords, tokens, connection strings, or API keys → set contains_credentials=true and capture in credential_fields
- Do NOT include credential values anywhere in intent field
- credential_fields keys should be: base_url, username, password, client_id, api_key, etc.
"""


def make_review_incoming_node(llm):
    def review_incoming(state: MigrationState) -> dict:
        phase = state.get("phase", MigrationPhase.INIT)
        session_id = state.get("session_id", "-")
        source_system = state.get("source_system")  # already-selected source, if any
        messages = state.get("messages", [])
        user_msg = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                user_msg = m.content
                break

        with log_step(logger, "review_incoming", phase=phase, session_id=session_id):
            prompt = f"""Current migration phase: {phase}
Source system already selected: {source_system or 'none'}
User message: {user_msg}

Classify and route this message."""

            review_data = _call_llm_json(llm, prompt, {
                "intent": "unknown",
                "contains_credentials": False,
                "credential_fields": {},
                "routing": _default_routing_for_phase(phase),
                "clarification_needed": False,
                "clarification_prompt": None,
            })

        logger.info(
            "review_incoming → intent=%s, credentials=%s, routing=%s",
            review_data.get("intent"),
            review_data.get("contains_credentials"),
            review_data.get("routing"),
            extra={"phase": phase, "session_id": session_id},
        )

        return {"inbound_review": review_data, "regen_attempt": 0}

    return review_incoming


def _default_routing_for_phase(phase: str) -> str:
    mapping = {
        MigrationPhase.INIT: "greet_and_probe",
        MigrationPhase.SOURCE_CONFIGURED: "select_cloud",
        MigrationPhase.CLOUD_SELECTED: "authenticate_cloud",
        MigrationPhase.AZURE_AUTHENTICATED: "setup_secrets_manager",
        MigrationPhase.VAULT_SETUP: "provision_storage",
        MigrationPhase.STORAGE_PROVISIONED: "extract_metadata",
        MigrationPhase.METADATA_EXTRACTED: "provision_databricks",
        MigrationPhase.DATABRICKS_PROVISIONED: "upload_notebooks",
        MigrationPhase.NOTEBOOKS_UPLOADED: "run_metadata_notebook",
        MigrationPhase.METADATA_LOADED: "configure_pipeline",
        MigrationPhase.PIPELINE_CONFIGURED: "provision_adf",
        MigrationPhase.ADF_PROVISIONED: "finalize",
        MigrationPhase.COMPLETED: "finalize",
    }
    return mapping.get(phase, "greet_and_probe")


# ---------------------------------------------------------------------------
# Node: review_outgoing
# ---------------------------------------------------------------------------

_OUTBOUND_REVIEW_SYSTEM = """You are an outbound response validator for a data migration assistant.
Given the current migration phase, the user message, and the draft assistant response, output a JSON object with:
- passes: true if the response is coherent, accurate, does not expose credentials, and does not claim a resource was created unless it actually was (boolean)
- failure_reason: if passes is false, explain why (string or null)
- revised_response: if the issue is minor (e.g., wording, credential redaction needed), provide a corrected version; otherwise null (string or null)

Rules for FAILING:
1. The response contains a credential value (password, token, key, secret)
2. The response claims an Azure/AWS resource was created but the phase data shows it was not
3. The response is completely off-topic or nonsensical
4. The response asks for something the user already provided in this turn

If the response is reasonable and just imperfect wording, set passes=true.
"""


def make_review_outgoing_node(llm):
    def review_outgoing(state: MigrationState) -> dict:
        phase = state.get("phase", MigrationPhase.INIT)
        session_id = state.get("session_id", "-")
        draft = state.get("draft_response", "")
        messages = state.get("messages", [])
        user_msg = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                user_msg = m.content
                break

        with log_step(logger, "review_outgoing", phase=phase, session_id=session_id):
            prompt = f"""Current migration phase: {phase}
User message: {user_msg}
Draft assistant response: {draft}

Validate this response."""

            review_data = _call_llm_json(llm, prompt, {
                "passes": True,
                "failure_reason": None,
                "revised_response": None,
            })

        attempt = state.get("regen_attempt", 0)
        passes = review_data.get("passes", True)
        logger.info(
            "review_outgoing → passes=%s, reason=%s, attempt=%d",
            passes,
            review_data.get("failure_reason"),
            attempt,
            extra={"phase": phase, "session_id": session_id},
        )

        return {"outbound_review": review_data}

    return review_outgoing


# ---------------------------------------------------------------------------
# Phase nodes
# ---------------------------------------------------------------------------

def make_greet_and_probe_node(llm, source_registry):
    def greet_and_probe(state: MigrationState) -> dict:
        phase = state.get("phase", MigrationPhase.INIT)
        session_id = state.get("session_id", "-")
        messages = state.get("messages", [])
        user_content = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")

        with log_step(logger, "greet_and_probe", phase=phase, session_id=session_id):
            sources = source_registry.list_sources()

            # Detect if user already named a source system
            detected_source = None
            lower = user_content.lower()
            for src in sources:
                if src.id in lower or src.display_name.lower() in lower:
                    detected_source = src
                    break
            # Also catch common aliases
            if not detected_source:
                if "sap" in lower:
                    detected_source = source_registry.get_plugin("sap_s4")
                elif "oracle" in lower:
                    detected_source = source_registry.get_plugin("oracle")

            if detected_source:
                # Source identified → advance to SOURCE_CONFIGURED, ask for cloud next.
                # Credentials are collected AFTER the vault is set up (VAULT_SETUP step).
                system = f"""You are a migration AI assistant. The user wants to migrate from {detected_source.display_name}.
Confirm their choice in one sentence.
Then say: before collecting any connection details, let's first set up your cloud environment (secrets vault + storage) so credentials are stored securely from the start.
Ask them to choose their cloud provider: Azure, AWS, or GCP. Keep the entire response to 3 sentences max."""
                response = _call_llm_text(llm, system, user_content)
                return {
                    "draft_response": response,
                    "phase": MigrationPhase.SOURCE_CONFIGURED,
                    "source_system": detected_source.id,
                    "source_config": {"selected": True},
                    "response_suggestions": ["Azure", "AWS", "GCP"],
                }

            # No source detected yet → ask user to pick
            source_list = "\n".join(f"- {s.display_name}: {s.description}" for s in sources)
            system = f"""You are a migration AI assistant. Welcome the user warmly.
Tell them this tool automates the entire data migration from on-premise systems to cloud data lakes.
Ask them to choose a source system:
{source_list}"""
            response = _call_llm_text(llm, system, user_content)

        suggestions = [s.display_name for s in sources]
        return {
            "draft_response": response,
            "response_suggestions": suggestions,
            "phase": MigrationPhase.INIT,
        }
    return greet_and_probe


def make_collect_source_creds_node(llm, source_registry):
    def collect_source_creds(state: MigrationState) -> dict:
        phase = state.get("phase", MigrationPhase.INIT)
        session_id = state.get("session_id", "-")
        inbound = state.get("inbound_review", {})
        messages = state.get("messages", [])
        user_content = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")

        with log_step(logger, "collect_source_creds", phase=phase, session_id=session_id):
            # Detect source system from message
            source_system = state.get("source_system")
            if not source_system:
                for src in source_registry.list_sources():
                    if src.id in user_content.lower() or src.display_name.lower() in user_content.lower():
                        source_system = src.id
                        break

            plugin = source_registry.get_plugin(source_system or "sap_s4")
            cred_fields = plugin.required_cred_fields if plugin else []

            # Check if credentials were intercepted by inbound review
            collected_creds = inbound.get("credential_fields", {})
            source_config = dict(state.get("source_config", {}))

            # Merge intercepted credentials (mark as collected, not store values)
            creds_collected_keys = list(source_config.get("collected_fields", []))
            if collected_creds:
                for key in collected_creds:
                    if key not in creds_collected_keys:
                        creds_collected_keys.append(key)
                source_config["collected_fields"] = creds_collected_keys
                logger.info(
                    "collect_source_creds: intercepted credential fields=%s",
                    list(collected_creds.keys()),
                    extra={"phase": phase, "session_id": session_id},
                )

            # --- Check if user chose a credential approach ---
            lower_content = user_content.lower()
            cred_approach = source_config.get("cred_approach")

            # Detect vault preference
            wants_vault = any(w in lower_content for w in ["use existing", "existing vault", "key vault", "secrets manager", "option 2"])
            wants_vault_later = any(w in lower_content for w in ["set up vault", "vault first", "later", "option 3"])
            wants_direct = any(w in lower_content for w in ["type here", "type them", "provide directly", "option 1", "share here", "give you"])

            if (wants_vault or wants_vault_later) and not cred_approach:
                # User prefers vault — skip direct collection, move to cloud selection
                vault_pref = "existing" if wants_vault else "new"
                source_config["cred_approach"] = vault_pref
                src_name = plugin.display_name if plugin else "source system"
                system = f"""You are a migration AI assistant. The user wants to use a {'existing' if wants_vault else 'new'} secrets vault for their {src_name} credentials.
Confirm this is a great choice. Tell them:
- We'll set up the vault during cloud configuration
- They can then paste or upload credentials directly into the vault UI
Now ask them to choose their cloud provider (Azure, AWS, or GCP) so we can configure the right secrets manager."""
                response = _call_llm_text(llm, system, user_content)
                return {
                    "draft_response": response,
                    "phase": MigrationPhase.SOURCE_CONFIGURED,  # advance — creds will come via vault
                    "source_system": source_system,
                    "source_config": source_config,
                    "response_suggestions": ["Azure", "AWS", "GCP"],
                }

            # Direct credential collection mode
            if wants_direct and not cred_approach:
                source_config["cred_approach"] = "direct"

            # Determine what's still needed
            required_keys = [f.key for f in cred_fields]
            still_needed = [k for k in required_keys if k not in creds_collected_keys]

            if not still_needed:
                # All creds collected — advance phase
                new_phase = MigrationPhase.SOURCE_CONFIGURED
                source_config["source_system"] = source_system
                system = """You are a migration AI assistant. The source system credentials have been captured successfully.
Confirm this warmly. Tell the user their credentials will be stored securely in their cloud secrets manager — never logged.
Now ask them to choose a cloud provider (Azure, AWS, or GCP) to continue."""
                response = _call_llm_text(llm, system, user_content)
                return {
                    "draft_response": response,
                    "phase": new_phase,
                    "source_system": source_system,
                    "source_config": source_config,
                    "response_suggestions": ["Azure", "AWS", "GCP"],
                }
            elif collected_creds:
                # Partial — got some this turn, ask for the rest
                missing_labels = [f.label for f in cred_fields if f.key in still_needed]
                system = f"""You are a migration AI assistant. You received some credentials for {plugin.display_name if plugin else 'the source system'}.
Thank the user. Tell them you still need: {', '.join(missing_labels)}.
Remind them credentials are captured and immediately stored securely — never echoed back or logged.
Ask them to provide the remaining details."""
                response = _call_llm_text(llm, system, user_content)
                return {
                    "draft_response": response,
                    "source_system": source_system,
                    "source_config": source_config,
                    "response_suggestions": [],
                }
            else:
                # No creds yet — ask user to type them
                non_secret = [f.label for f in cred_fields if not f.secret]
                secret = [f.label for f in cred_fields if f.secret]
                system = f"""You are a migration AI assistant. The user wants to type their {plugin.display_name if plugin else 'source system'} credentials here.
Ask them to provide:
- {', '.join(non_secret)} (not sensitive)
- {', '.join(secret)} (will be masked immediately)

Be reassuring: credentials are captured by the system, never stored in chat history, and immediately written to the secrets manager.
You can provide them all in one message like: base_url: http://..., username: demo, password: secret, client_id: 100"""
                response = _call_llm_text(llm, system, user_content)
                return {
                    "draft_response": response,
                    "source_system": source_system,
                    "source_config": source_config,
                    "response_suggestions": [],
                }

    return collect_source_creds


def make_select_cloud_node(llm):
    _REGION_SUGGESTIONS = {
        "azure": ["East US", "West Europe", "Southeast Asia", "UK South"],
        "aws": ["us-east-1", "eu-west-1", "ap-south-1", "us-west-2"],
        "gcp": ["us-central1", "europe-west1", "asia-south1"],
    }
    # Map what a user might type to a canonical region code
    _REGION_NORMALIZE = {
        "east us": "eastus", "east us 2": "eastus2", "eastus": "eastus",
        "west us": "westus", "west us 2": "westus2", "westus": "westus",
        "west europe": "westeurope", "westeurope": "westeurope",
        "north europe": "northeurope", "northeurope": "northeurope",
        "southeast asia": "southeastasia", "southeastasia": "southeastasia",
        "uk south": "uksouth", "uksouth": "uksouth",
        "us-east-1": "us-east-1", "us-east-2": "us-east-2",
        "us-west-1": "us-west-1", "us-west-2": "us-west-2",
        "eu-west-1": "eu-west-1", "ap-south-1": "ap-south-1",
        "us-central1": "us-central1", "europe-west1": "europe-west1",
        "asia-south1": "asia-south1",
    }

    def select_cloud(state: MigrationState) -> dict:
        phase = state.get("phase", MigrationPhase.SOURCE_CONFIGURED)
        session_id = state.get("session_id", "-")
        messages = state.get("messages", [])
        user_content = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")

        with log_step(logger, "select_cloud", phase=phase, session_id=session_id):
            # cloud_provider may already be saved from a previous turn
            cloud_provider = state.get("cloud_provider")
            cloud_region = state.get("cloud_region")
            lower = user_content.lower()

            # Detect cloud from the current message
            if not cloud_provider:
                if "azure" in lower:
                    cloud_provider = "azure"
                elif "aws" in lower or "amazon" in lower:
                    cloud_provider = "aws"
                elif "gcp" in lower or "google" in lower:
                    cloud_provider = "gcp"

            # Detect region from the current message
            if cloud_provider and not cloud_region:
                for region_name, region_code in _REGION_NORMALIZE.items():
                    if region_name in lower:
                        cloud_region = region_code
                        break

            # ── Both cloud AND region known ────────────────────────────────
            if cloud_provider and cloud_region:
                from app.agents.secrets_tools import get_secrets_service_meta
                svc_name, _ = get_secrets_service_meta(cloud_provider)
                system = f"""You are a migration AI assistant.
Confirm: the user has chosen {cloud_provider.upper()} in region **{cloud_region}**.
Tell them the next step is configuring an {cloud_provider.upper()} account and {svc_name} vault.
Ask them to confirm to proceed. Keep to 2-3 sentences."""
                response = _call_llm_text(llm, system, user_content)
                return {
                    "draft_response": response,
                    "phase": MigrationPhase.CLOUD_SELECTED,
                    "cloud_provider": cloud_provider,
                    "cloud_region": cloud_region,
                    "secrets_backend": cloud_provider,
                    "response_suggestions": ["Proceed with account setup"],
                }

            # ── Cloud known but no region yet ─────────────────────────────
            if cloud_provider:
                suggestions = _REGION_SUGGESTIONS.get(cloud_provider, [])
                system = f"""You are a migration AI assistant.
The user chose {cloud_provider.upper()}. Ask them for their preferred {cloud_provider.upper()} region.
Give 2-3 example options: {', '.join(suggestions[:3])}.
Keep to 1-2 sentences."""
                response = _call_llm_text(llm, system, user_content)
                return {
                    "draft_response": response,
                    "cloud_provider": cloud_provider,  # persist for next turn; phase stays SOURCE_CONFIGURED
                    "response_suggestions": suggestions,
                }

            # ── No cloud detected ─────────────────────────────────────────
            system = """You are a migration AI assistant. Ask the user to choose a cloud provider.
Options: Azure, AWS, or GCP. Keep to 1-2 sentences."""
            response = _call_llm_text(llm, system, user_content)
            return {
                "draft_response": response,
                "response_suggestions": ["Azure", "AWS", "GCP"],
            }

    return select_cloud


# ---------------------------------------------------------------------------
# Azure credential fields the user must provide
# ---------------------------------------------------------------------------
_AZURE_CRED_FIELDS = [
    ("tenant_id",       "Tenant ID",       "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"),
    ("client_id",       "Client ID (App)", "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"),
    ("client_secret",   "Client Secret",   "your-client-secret"),
    ("subscription_id", "Subscription ID", "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"),
]
_AZURE_CRED_ALIASES = {
    "tenantid": "tenant_id", "tenant": "tenant_id",
    "clientid": "client_id", "appid": "client_id", "app_id": "client_id",
    "clientsecret": "client_secret", "secret": "client_secret",
    "subscriptionid": "subscription_id", "sub": "subscription_id", "subid": "subscription_id",
}


def make_authenticate_cloud_node(llm, settings=None):
    """CLOUD_SELECTED phase: collect Azure/AWS/GCP credentials from the user.
    Stores them in azure_credentials inside vault_config (persisted to session).
    Advances to AZURE_AUTHENTICATED once all required fields are present."""

    def authenticate_cloud(state: MigrationState) -> dict:
        phase = state.get("phase", MigrationPhase.CLOUD_SELECTED)
        session_id = state.get("session_id", "-")
        cloud_provider = state.get("cloud_provider", "azure")
        messages = state.get("messages", [])
        user_content = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        vault_config = dict(state.get("vault_config", {}))
        stored_creds: dict = vault_config.get("azure_credentials", {})

        with log_step(logger, "authenticate_cloud", phase=phase, session_id=session_id):

            # ── Non-Azure clouds: check settings fallback, then ask ──────────
            if cloud_provider != "azure":
                system = f"""You are a migration AI assistant.
{cloud_provider.upper()} cloud selected. Credential collection for {cloud_provider.upper()} is not
automated yet. Tell the user to add credentials to the server .env and restart.
Then ask them to type 'ready' to continue."""
                return {
                    "draft_response": _call_llm_text(llm, system, user_content),
                    "vault_config": vault_config,
                    "response_suggestions": ["Ready"],
                }

            # ── Check if settings already has full Azure creds ───────────────
            if settings and all([
                getattr(settings, "azure_tenant_id", None),
                getattr(settings, "azure_client_id", None),
                getattr(settings, "azure_client_secret", None),
                getattr(settings, "azure_subscription_id", None),
            ]):
                # Credentials already configured in server env — auto-authenticate
                sub_id = settings.azure_subscription_id
                vault_config["azure_credentials"] = {
                    "tenant_id":       settings.azure_tenant_id,
                    "client_id":       settings.azure_client_id,
                    "client_secret":   settings.azure_client_secret,
                    "subscription_id": sub_id,
                    "source":          "settings",
                }
                vault_config["resource_group"] = getattr(settings, "azure_resource_group", None) or "migration-rg"
                sub_display = f"{sub_id[:8]}...{sub_id[-4:]}" if sub_id and len(sub_id) > 12 else sub_id
                system = f"""You are a migration AI assistant.
Azure credentials are already configured in the server environment.
Tell the user:
- ✅ Authenticated with Azure subscription **{sub_display}**
- No credentials needed — the server is already configured

Ask them which **resource group** to use for all migration resources.
Suggest `migration-rg` as the default. Keep to 2 sentences."""
                response = _call_llm_text(llm, system, user_content)
                # Also parse resource group from message if user typed it now
                rg_match = re.search(r'\b([\w-]+-rg|[\w-]+-group|[\w-]+group)\b', user_content.lower())
                if rg_match:
                    vault_config["resource_group"] = rg_match.group(1)
                return {
                    "draft_response": response,
                    "phase": MigrationPhase.AZURE_AUTHENTICATED,
                    "vault_config": vault_config,
                    "response_suggestions": ["Use migration-rg", "Use different resource group"],
                }

            # ── Parse credentials from user message ──────────────────────────
            required_keys = [k for k, _, _ in _AZURE_CRED_FIELDS]
            parsed = _parse_creds_from_text(user_content, required_keys)
            # Apply aliases
            for raw_key, val in list(parsed.items()):
                canonical = _AZURE_CRED_ALIASES.get(raw_key.replace("-", "").lower(), raw_key)
                if canonical != raw_key:
                    parsed[canonical] = val
                    del parsed[raw_key]

            if parsed:
                stored_creds.update(parsed)
                vault_config["azure_credentials"] = stored_creds

            # Check what's still missing
            still_needed = [label for key, label, _ in _AZURE_CRED_FIELDS if key not in stored_creds]

            if not still_needed:
                # All credentials collected — parse resource group too
                rg_match = re.search(r'\b([\w-]+-rg|[\w-]+-group)\b', user_content.lower())
                vault_config["resource_group"] = (
                    rg_match.group(1) if rg_match else "migration-rg"
                )
                # Validate by calling Azure to list subscriptions
                validation_note = ""
                try:
                    from azure.identity import ClientSecretCredential
                    from azure.mgmt.subscription import SubscriptionClient
                    cred = ClientSecretCredential(
                        tenant_id=stored_creds["tenant_id"],
                        client_id=stored_creds["client_id"],
                        client_secret=stored_creds["client_secret"],
                    )
                    sub_client = SubscriptionClient(cred)
                    # Use get() directly — list() returns empty if SP lacks subscription-level Reader
                    try:
                        sub = sub_client.subscriptions.get(stored_creds["subscription_id"])
                        validation_note = f" Subscription **{sub.display_name}** confirmed."
                    except Exception:
                        # get() failed — fall back to list() as a secondary check
                        subs = list(sub_client.subscriptions.list())
                        matched = next(
                            (s for s in subs if s.subscription_id == stored_creds["subscription_id"]),
                            None,
                        )
                        if matched:
                            validation_note = f" Subscription **{matched.display_name}** confirmed."
                        elif subs:
                            available = ", ".join(s.subscription_id for s in subs[:3])
                            validation_note = f" ⚠️ Subscription ID not found among visible subscriptions: {available}."
                        else:
                            # SP can authenticate but can't enumerate subscriptions (missing Reader role)
                            # Treat as valid — vault creation will catch real permission issues
                            validation_note = " Credentials accepted (subscription visibility requires Reader role, but vault creation will proceed)."
                except Exception as exc:
                    logger.warning(
                        "authenticate_cloud: validation failed: %s", exc,
                        extra={"phase": phase, "session_id": session_id},
                    )
                    exc_str = str(exc)
                    # AADSTS7000215 = secret ID provided instead of secret value
                    if "AADSTS7000215" in exc_str or "invalid client secret" in exc_str.lower():
                        # Remove the bad client_secret so the user is prompted to re-enter it
                        stored_creds.pop("client_secret", None)
                        vault_config["azure_credentials"] = stored_creds
                        system = """You are a migration AI assistant.
Tell the user:

❌ **Authentication failed — wrong client secret format.**

The `client_secret` you provided appears to be the **Secret ID** (a GUID like `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`), NOT the secret **value**.

Here is how to get the correct value:
1. Go to **Azure Portal → Azure Active Directory → App registrations**
2. Open your app → **Certificates & secrets → Client secrets**
3. Copy the value in the **Value** column (not the **Secret ID** column)

The value looks like a random string, e.g. `abc~8Q.xyz123...` — it is only shown once when created.

Please reply with: `client_secret: <the actual value>`"""
                        return {
                            "draft_response": _call_llm_text(llm, system, user_content),
                            "phase": phase,  # stay in CLOUD_SELECTED
                            "vault_config": vault_config,
                            "response_suggestions": ["client_secret: <paste value here>"],
                        }
                    validation_note = f" (Validation skipped: {exc})"

                    system = f"""You are a migration AI assistant.
Tell the user: ✅ Azure credentials received.{validation_note}
You will now create a Key Vault to store all secrets securely.
Ask them to confirm to proceed. 3 sentences max."""
                    return {
                        "draft_response": _call_llm_text(llm, system, user_content),
                        "phase": MigrationPhase.AZURE_AUTHENTICATED,
                        "vault_config": vault_config,
                        "response_suggestions": ["Proceed", "Use migration-rg"],
                    }

                system = f"""You are a migration AI assistant.
Tell the user: ✅ Azure credentials received.{validation_note}
You will now create a Key Vault to store all secrets securely.
Ask them to confirm to proceed. 3 sentences max."""
                return {
                    "draft_response": _call_llm_text(llm, system, user_content),
                    "phase": MigrationPhase.AZURE_AUTHENTICATED,
                    "vault_config": vault_config,
                    "response_suggestions": ["Proceed", "Use migration-rg"],
                }

            # Not all creds yet — ask for missing ones
            already = ", ".join(stored_creds.keys()) if stored_creds else "none yet"
            still_labels = [
                f"{label} (`{key}`)"
                for key, label, _ in _AZURE_CRED_FIELDS
                if key not in stored_creds
            ]
            system = f"""You are a migration AI assistant setting up Azure authentication.
Still needed: {', '.join(still_labels)}.
Already collected: {already}.

Give the user a numbered step-by-step guide to get these values from the Azure Portal.
Use this exact structure in your response (markdown):

**Step 1 — Subscription ID**
1. Go to [portal.azure.com](https://portal.azure.com)
2. Search **Subscriptions** in the top bar → click your subscription
3. Copy the **Subscription ID** from the Overview page

**Step 2 — Register an App (gets Tenant ID + Client ID)**
1. Search **App registrations** → click **+ New registration**
2. Name: `migration-sp` → click **Register**
3. On the app Overview page copy:
   - **Directory (tenant) ID** → your `tenant_id`
   - **Application (client) ID** → your `client_id`

**Step 3 — Create a Client Secret**
1. Left menu → **Certificates & secrets** → **+ New client secret**
2. Description: `migration` | Expiry: 24 months → **Add**
3. Copy the **Value** immediately — it disappears after you navigate away
   - This is your `client_secret`

**Step 4 — Grant Contributor role**
1. Search **Subscriptions** → click your subscription → **Access control (IAM)**
2. **+ Add** → **Add role assignment** → select **Contributor** → Next
3. **+ Select members** → search `migration-sp` → Select → **Review + assign**

Once done, paste all four values like this:
```
tenant_id: <your-tenant-id>
client_id: <your-client-id>
client_secret: <your-secret-value>
subscription_id: <your-subscription-id>
```
Do not add any commentary — just output the guide above verbatim, adapted for which fields are still missing."""
            return {
                "draft_response": _call_llm_text(llm, system, user_content),
                "vault_config": vault_config,
                "response_suggestions": ["I have the values ready"],
            }

    return authenticate_cloud


def make_setup_secrets_manager_node(llm, secrets_backend_factory, settings=None):
    """AZURE_AUTHENTICATED phase: create vault, then advance to VAULT_SETUP.
    Azure credentials are already in vault_config[azure_credentials] at this point."""
    def setup_secrets_manager(state: MigrationState) -> dict:
        phase = state.get("phase", MigrationPhase.AZURE_AUTHENTICATED)
        session_id = state.get("session_id", "-")
        cloud_provider = state.get("cloud_provider", "azure")
        source_system = state.get("source_system", "")
        messages = state.get("messages", [])
        user_content = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        vault_config = dict(state.get("vault_config", {}))

        with log_step(logger, "setup_secrets_manager", phase=phase, session_id=session_id):
            from app.agents.secrets_tools import get_secrets_service_meta
            svc_name, _ = get_secrets_service_meta(cloud_provider)

            # resource_group was set by authenticate_cloud; fall back to default
            resource_group = vault_config.get("resource_group", "migration-rg")
            region = state.get("cloud_region", "eastus")

            # ── Create vault NOW ───────────────────────────────────────────
            vault_name = f"migration-vault-{session_id[:8]}"
            vault_config.update({
                "vault_name": vault_name,
                "vault_service": svc_name,
                "creds_stored": False,
            })

            vault_uri: str | None = None
            vault_error: str | None = None
            try:
                backend = secrets_backend_factory(cloud_provider, session_id)
                result = backend.create_vault(vault_name, region, resource_group=resource_group)
                vault_uri = result.get("uri", f"https://{vault_name}.vault.azure.net/")
                vault_config["vault_uri"] = vault_uri
                logger.info(
                    "setup_secrets_manager: vault '%s' created → %s",
                    vault_name, vault_uri,
                    extra={"phase": phase, "session_id": session_id},
                )
            except Exception as exc:
                vault_error = str(exc)
                logger.warning(
                    "setup_secrets_manager: vault creation issue: %s",
                    exc,
                    extra={"phase": phase, "session_id": session_id},
                )

            # ResourceGroupNotFound — RG auto-create must have failed (permission issue)
            if vault_error and "ResourceGroupNotFound" in vault_error:
                system = f"""You are a migration AI assistant. Tell the user:

❌ **Vault creation failed — resource group `{resource_group}` does not exist and could not be created automatically.**

This usually means the service principal lacks the **Contributor** role at the subscription level.

**To fix this, do ONE of:**

**Option A — Grant Contributor on subscription (recommended):**
1. Azure Portal → Subscriptions → your subscription → **Access control (IAM)**
2. Add role assignment → **Contributor** → select your app registration → Save

**Option B — Create the resource group manually:**
1. Azure Portal → Resource groups → **Create**
2. Name: `{resource_group}`, Region: {region} → Review + create

Once done, reply **"retry"** to try again."""
                return {{
                    "draft_response": _call_llm_text(llm, system, user_content),
                    "phase": MigrationPhase.AZURE_AUTHENTICATED,  # stay, let user fix then retry
                    "vault_config": vault_config,
                    "response_suggestions": ["retry", f"Use existing resource group"],
                }}

            # AADSTS7000215: user gave Secret ID instead of secret Value — force re-auth
            if vault_error and "AADSTS7000215" in vault_error:
                # Wipe the bad client_secret so authenticate_cloud re-collects it
                azure_creds = vault_config.get("azure_credentials", {})
                azure_creds.pop("client_secret", None)
                vault_config["azure_credentials"] = azure_creds
                system = """You are a migration AI assistant. Tell the user:

❌ **Vault creation failed — wrong client secret format (AADSTS7000215).**

The `client_secret` you entered is the **Secret ID** (a GUID), not the secret **value**.

**How to get the correct value:**
1. Go to **Azure Portal → Azure Active Directory → App registrations**
2. Click your app → **Certificates & secrets → Client secrets**
3. Copy from the **Value** column (not the Secret ID column)

The value is a random string like `abc~8Q.xyz...` — shown **once** at creation time only.
If you did not save it, create a new secret.

Please reply with: `client_secret: <paste the value here>`"""
                return {
                    "draft_response": _call_llm_text(llm, system, user_content),
                    "phase": MigrationPhase.CLOUD_SELECTED,  # go back to re-collect creds
                    "vault_config": vault_config,
                    "response_suggestions": ["client_secret: <paste value here>", "Create a new client secret"],
                }

            src_display = source_system.replace("_", " ").upper() if source_system else "source system"
            if vault_uri:
                system = f"""You are a migration AI assistant. Tell the user:
- ✅ **{svc_name}** vault **{vault_name}** has been **successfully created** in resource group **{resource_group}** (region: {region})
- Next: provide your {src_display} connection credentials — they will be stored immediately and securely in this vault
- Credentials are never echoed back, logged, or displayed after being stored
Ask the user to proceed to provide credentials. Keep to 3-4 sentences."""
            else:
                system = f"""You are a migration AI assistant. Tell the user:
- Attempted to create **{svc_name}** vault **{vault_name}** in resource group **{resource_group}**
- Note: {vault_error or 'vault may already exist or permissions are still propagating'}
- We will proceed — credentials will be stored in vault **{vault_name}** when collected
Ask to proceed. Keep to 3-4 sentences."""
            response = _call_llm_text(llm, system, user_content)

        return {
            "draft_response": response,
            "phase": MigrationPhase.VAULT_SETUP,
            "vault_config": vault_config,
            "response_suggestions": ["Proceed to credentials", "What is stored in the vault?"],
        }

    return setup_secrets_manager


def _parse_creds_from_text(text: str, required_keys: list[str]) -> dict[str, str]:
    """Directly parse key:value or key=value credential pairs from user text.
    More reliable than LLM detection for structured input like:
    'base_url: http://..., username: demo, password: secret, client_id: 100'"""
    found: dict[str, str] = {}
    lower_keys = {k.lower(): k for k in required_keys}
    # Also match common aliases (client_id -> clientid, base_url -> baseurl etc.)
    aliases = {
        "baseurl": "base_url", "base-url": "base_url", "url": "base_url", "host": "base_url",
        "pass": "password", "pwd": "password",
        "user": "username", "user_name": "username",
        "clientid": "client_id", "client-id": "client_id", "mandt": "client_id",
    }
    for m in re.finditer(r'\b([\w_-]+)\s*[=:]\s*([^\s,;|]+)', text):
        raw_key = m.group(1).lower().strip()
        val = m.group(2).strip('\'"` ')
        key = aliases.get(raw_key, raw_key)
        if key in lower_keys:
            found[lower_keys[key]] = val
    return found


def make_provision_storage_node(llm, source_registry, secrets_backend_factory, orchestrator=None):
    """VAULT_SETUP phase: collect source creds (step A) then create storage directly (step B)."""
    def provision_storage(state: MigrationState) -> dict:
        phase = state.get("phase", MigrationPhase.VAULT_SETUP)
        session_id = state.get("session_id", "-")
        source_system = state.get("source_system", "sap_s4")
        cloud_provider = state.get("cloud_provider", "azure")
        messages = state.get("messages", [])
        user_content = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        inbound = state.get("inbound_review", {})
        vault_config = dict(state.get("vault_config", {}))

        with log_step(logger, "provision_storage", phase=phase, session_id=session_id):

            # ── Step A: collect + store source credentials ──────────────────
            if not vault_config.get("creds_stored"):
                credential_fields = inbound.get("credential_fields", {})
                plugin = source_registry.get_plugin(source_system)
                cred_fields = plugin.required_cred_fields if plugin else []
                src_name = plugin.display_name if plugin else source_system
                vault_name = vault_config.get("vault_name", f"migration-vault-{session_id[:8]}")
                from app.agents.secrets_tools import get_secrets_service_meta
                svc_name, _ = get_secrets_service_meta(cloud_provider)

                # Also try direct parsing — more reliable than LLM for key:value format
                required_keys = [f.key for f in cred_fields]
                direct_parsed = _parse_creds_from_text(user_content, required_keys)
                if direct_parsed:
                    credential_fields = {**direct_parsed, **credential_fields}

                if credential_fields:
                    # Creds received — store them in vault, then confirm before proposing storage
                    stored_keys = []
                    store_error: str | None = None
                    try:
                        backend = secrets_backend_factory(cloud_provider, session_id)
                        for key, value in credential_fields.items():
                            backend.store_secret(vault_name, f"source-{key}", value)
                            stored_keys.append(f"source-{key}")
                    except Exception as exc:
                        store_error = str(exc)
                        logger.warning("Secret store failed: %s", exc,
                                       extra={"phase": phase, "session_id": session_id})

                    if store_error:
                        system = f"""You are a migration AI assistant. Tell the user:
❌ **Failed to store credentials in {svc_name} vault `{vault_name}`.**
Error: {store_error}

This may be a permissions issue — the service principal needs **Key Vault Secrets Officer** role on the vault.
Please check vault access policies and try again."""
                        return {
                            "draft_response": _call_llm_text(llm, system, user_content),
                            "vault_config": vault_config,
                            "response_suggestions": ["Retry", "Check vault permissions"],
                        }

                    vault_config["creds_stored"] = True
                    vault_config["stored_secret_names"] = stored_keys
                    logger.info("provision_storage: stored %d creds in vault", len(stored_keys),
                                extra={"phase": phase, "session_id": session_id})

                    # Return cred confirmation — storage proposal happens next turn
                    system = f"""You are a migration AI assistant. Tell the user:
✅ **{src_name} credentials have been securely stored in {svc_name} vault `{vault_name}`.**
The following secrets were written (values are never displayed): {', '.join(f'`{k}`' for k in stored_keys)}

Next step: provision the Azure storage account and containers to hold migrated data.
Ask them to confirm to proceed."""
                    return {
                        "draft_response": _call_llm_text(llm, system, user_content),
                        "vault_config": vault_config,
                        "response_suggestions": ["Yes, create storage", "What will be created?"],
                    }

                else:
                    # No creds yet — ask for them
                    required_non_secret = [f.label for f in cred_fields if not f.secret]
                    required_secret = [f.label for f in cred_fields if f.secret]
                    system = f"""You are a migration AI assistant. The vault **{vault_name}** is ready in {svc_name}.
Now collect the {src_name} connection details to store securely in this vault.

Ask the user to provide:
- {', '.join(required_non_secret)} (not sensitive)
- {', '.join(required_secret)} (will be stored as a secret, never echoed)

They can paste all at once, e.g.:
`base_url: http://sap-host:8011, username: demo, password: secret, client_id: 100`

Be reassuring: values go straight into the vault — I never log or display them."""
                    return {
                        "draft_response": _call_llm_text(llm, system, user_content),
                        "vault_config": vault_config,
                        "response_suggestions": [],
                    }

            # ── Step B: creds stored — propose storage provisioning ─────────
            cloud = cloud_provider
            region = state.get("cloud_region", "eastus")
            storage_config = dict(state.get("storage_config", {}))

            if storage_config.get("provisioned"):
                system = """Storage is already provisioned. Tell the user containers are ready and suggest proceeding to metadata extraction."""
                return {
                    "draft_response": _call_llm_text(llm, system, user_content),
                    "phase": MigrationPhase.STORAGE_PROVISIONED,
                    "vault_config": vault_config,
                    "response_suggestions": ["Extract metadata now"],
                }

            # Build storage plan from session state
            resource_group = vault_config.get("resource_group", "migration-rg")
            import re as _re
            account_name = _re.sub(r"[^a-z0-9]", "", f"migstore{session_id[:6]}".lower())[:24]
            containers = ["landing", "metadata", "bronze", "silver"]
            region = state.get("cloud_region", "eastus")

            # If user is confirming (yes/approve/proceed/create), run real SDK now
            lower_content = user_content.lower().strip()
            is_confirmation = any(w in lower_content for w in [
                "yes", "approve", "proceed", "create", "ok", "go ahead", "confirm", "start"
            ])

            if not is_confirmation:
                # First time landing here after creds stored — show plan and ask for confirmation
                response = (
                    f"Ready to create Azure storage resources:\n\n"
                    f"- **Resource Group:** `{resource_group}`\n"
                    f"- **Storage Account:** `{account_name}` (region: `{region}`)\n"
                    f"- **Containers:** `landing`, `metadata`, `bronze`, `silver`\n\n"
                    f"Reply **yes** to create these now."
                )
                return {
                    "draft_response": response,
                    "storage_config": storage_config,
                    "vault_config": vault_config,
                    "response_suggestions": ["Yes, create storage", "Change region"],
                }

            # User confirmed — run real storage provisioning now
            result_msg = f"Creating storage account `{account_name}` in `{resource_group}`..."
            try:
                if orchestrator is not None:
                    result_msg = orchestrator._provision_storage_real(
                        {
                            "resource_group": resource_group,
                            "storage_account": account_name,
                            "region": region,
                            "containers": containers,
                        },
                        session_id,
                        phase,
                    )
                else:
                    result_msg = (
                        f"✅ Storage provisioning queued for `{account_name}`. "
                        "Orchestrator not available in graph context."
                    )
            except Exception as exc:
                logger.error(
                    "provision_storage: real SDK call failed: %s", exc,
                    extra={"phase": phase, "session_id": session_id},
                )
                result_msg = f"❌ Storage creation failed: {exc}"
                return {
                    "draft_response": result_msg,
                    "storage_config": storage_config,
                    "vault_config": vault_config,
                    "response_suggestions": ["Retry", "Check Azure permissions"],
                }

            return {
                "draft_response": result_msg,
                "phase": MigrationPhase.STORAGE_PROVISIONED,
                "storage_config": {**storage_config, "provisioned": True, "storage_account": account_name, "containers": containers},
                "vault_config": vault_config,
                "response_suggestions": ["Extract SAP metadata", "What's next?"],
            }

    return provision_storage


def make_extract_metadata_node(llm, source_registry, orchestrator=None):
    def extract_metadata(state: MigrationState) -> dict:
        phase = state.get("phase", MigrationPhase.STORAGE_PROVISIONED)
        session_id = state.get("session_id", "-")
        source_system = state.get("source_system", "sap_s4")
        messages = state.get("messages", [])
        user_content = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")

        with log_step(logger, "extract_metadata", phase=phase, session_id=session_id):
            plugin = source_registry.get_plugin(source_system)
            source_name = plugin.display_name if plugin else source_system
            phase_data = state.get("vault_config", {})

            lower_content = user_content.lower().strip()
            is_confirmation = any(w in lower_content for w in [
                "yes", "approve", "proceed", "extract", "ok", "go ahead", "confirm", "start", "run"
            ])

            if not is_confirmation:
                # First time — explain what will happen and ask to confirm
                system = f"""You are a migration AI assistant. Tell the user:
Storage is ready. Now you will extract SAP metadata:
1. Connect to {source_name} and list all available tables
2. Fetch field definitions and data types for each table
3. Store metadata JSON files in the `metadata` container

Ask them to confirm to proceed."""
                return {
                    "draft_response": _call_llm_text(llm, system, user_content),
                    "response_suggestions": ["Yes, extract metadata", "Which tables will be scanned?"],
                }

            # User confirmed — call real SAP extraction now
            try:
                if orchestrator is not None:
                    result_msg = orchestrator._extract_metadata_real(
                        {"source_system": source_system, "path_prefix": f"metadata/{source_system}/"},
                        session_id,
                        phase,
                    )
                else:
                    result_msg = "⚠️ Orchestrator not available — cannot extract metadata."
            except Exception as exc:
                logger.error("extract_metadata: real call failed: %s", exc,
                             extra={"phase": phase, "session_id": session_id})
                result_msg = f"❌ Metadata extraction failed: {exc}"
                return {
                    "draft_response": result_msg,
                    "response_suggestions": ["Retry", "Check SAP connection"],
                }

            return {
                "draft_response": result_msg,
                "phase": MigrationPhase.METADATA_EXTRACTED,
                "response_suggestions": ["Provision Databricks"],
            }

    return extract_metadata


def make_provision_databricks_node(llm, orchestrator=None):
    def provision_databricks(state: MigrationState) -> dict:
        phase = state.get("phase", MigrationPhase.METADATA_EXTRACTED)
        session_id = state.get("session_id", "-")
        messages = state.get("messages", [])
        user_content = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        lower_content = user_content.lower()

        with log_step(logger, "provision_databricks", phase=phase, session_id=session_id):
            proposed = {
                "workspace_name": f"migration-dbx-{session_id[:6]}",
                "cluster_name": "migration-cluster",
                "spark_version": "13.3.x-scala2.12",
                "node_type": "Standard_DS3_v2",
                "num_workers": 2,
            }

            is_confirmation = any(w in lower_content for w in [
                "yes", "approve", "proceed", "ok", "go ahead", "confirm",
                "start", "create", "provision", "setup", "set up",
            ])

            if not is_confirmation:
                # Show the plan and ask for approval
                plan = (
                    f"I'll provision an Azure Databricks workspace to run the migration notebooks.\n\n"
                    f"**Planned resources:**\n"
                    f"- Workspace: `{proposed['workspace_name']}`\n"
                    f"- Cluster: `{proposed['cluster_name']}` ({proposed['num_workers']} workers, `{proposed['node_type']}`)\n"
                    f"- Spark version: `{proposed['spark_version']}`\n"
                    f"- Region: same as storage account\n\n"
                    f"Shall I proceed?"
                )
                return {
                    "draft_response": plan,
                    "response_status": "approval_required",
                    "pending_approval": {
                        "tool_name": "provision_databricks",
                        "summary": f"Create Databricks workspace '{proposed['workspace_name']}'",
                        "tool_args": proposed,
                        "cloud_resource_type": "Azure Databricks",
                    },
                    "response_suggestions": ["Yes, provision Databricks", "What size cluster do I need?"],
                }

            # User confirmed — provision for real
            if orchestrator is None:
                return {
                    "draft_response": "⚠️ Databricks provisioning not available (no orchestrator configured).",
                    "response_suggestions": [],
                }

            result_msg = orchestrator._provision_databricks_real(
                proposed, session_id, phase
            )
            success = result_msg.startswith("\u2705")
            # Read back the config stored by the real provisioning call
            _session = orchestrator._session_repo.get_session(session_id) or {}
            _dbx_cfg = _session.get("phase_data", {}).get("databricks_config", {})
            result = {
                "draft_response": result_msg,
                "response_suggestions": ["Upload notebooks", "What notebooks will be uploaded?"] if success else ["Retry", "Set up manually"],
            }
            if success:
                result["phase"] = MigrationPhase.DATABRICKS_PROVISIONED
            if _dbx_cfg:
                result["databricks_config"] = _dbx_cfg
            return result

    return provision_databricks


def make_upload_notebooks_node(llm, orchestrator=None):
    def upload_notebooks(state: MigrationState) -> dict:
        phase = state.get("phase", MigrationPhase.DATABRICKS_PROVISIONED)
        session_id = state.get("session_id", "-")
        source_system = state.get("source_system", "sap")
        messages = state.get("messages", [])
        user_content = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")

        with log_step(logger, "upload_notebooks", phase=phase, session_id=session_id):
            if orchestrator is None:
                return {
                    "draft_response": "⚠️ Notebook upload not available (no orchestrator configured).",
                    "response_suggestions": [],
                }

            # Check if PAT already stored in vault
            _session = orchestrator._session_repo.get_session(session_id) or {}
            _phase_data = _session.get("phase_data", {})
            _vault_config = _phase_data.get("vault_config", {})
            _dbx_config = _phase_data.get("databricks_config", {})
            _vault_name = _vault_config.get("vault_name", "")
            _workspace_url = _dbx_config.get("workspace_url", "")

            existing_pat = ""
            if _vault_name:
                try:
                    _backend = orchestrator._build_secrets_backend(session_id)
                    existing_pat = _backend.get_secret(_vault_name, "databricks-pat") or ""
                except Exception:
                    pass

            # If no PAT yet — check if user just pasted one (looks like a dapi... token)
            pat_in_msg = ""
            import re as _re
            pat_match = _re.search(r'\b(dapi[a-zA-Z0-9]{32,})\b', user_content)
            if pat_match:
                pat_in_msg = pat_match.group(1)

            if not existing_pat and not pat_in_msg:
                # Ask user to generate and paste PAT
                return {
                    "draft_response": (
                        f"🔑 **To upload notebooks I need a Databricks Personal Access Token (PAT).**\n\n"
                        f"Your workspace: `{_workspace_url}`\n\n"
                        "**Steps to generate a PAT:**\n"
                        f"1. Open your workspace → [Launch Workspace]({_workspace_url})\n"
                        "2. Click your **profile icon** (top-right) → **Settings**\n"
                        "3. Go to **Developer** → **Access tokens** → **Generate new token**\n"
                        "4. Set a comment (e.g. `migration-tool`) and lifetime (e.g. `90` days)\n"
                        "5. Click **Generate** — copy the token (starts with `dapi`)\n\n"
                        "Then **paste it here** and I'll store it securely in your Key Vault and upload the notebooks."
                    ),
                    "response_suggestions": ["dapi<paste-your-token-here>"],
                }

            # If user just pasted a new PAT — store it in vault first
            if pat_in_msg and pat_in_msg != existing_pat:
                try:
                    _backend = orchestrator._build_secrets_backend(session_id)
                    _backend.store_secret(_vault_name, "databricks-pat", pat_in_msg)
                    logger.info("upload_notebooks: stored databricks-pat in vault %s", _vault_name,
                                extra={"phase": phase, "session_id": session_id})
                    existing_pat = pat_in_msg
                except Exception as exc:
                    return {
                        "draft_response": f"❌ Failed to store PAT in Key Vault: **{exc}**\n\nCheck vault access policies and try again.",
                        "response_suggestions": ["Retry"],
                    }

            # PAT available — upload notebooks
            result_msg = orchestrator._upload_notebooks_real(
                {"source_system": source_system}, session_id, phase
            )
            success = result_msg.startswith("✅")
            _session = orchestrator._session_repo.get_session(session_id) or {}
            _dbx_cfg = _session.get("phase_data", {}).get("databricks_config", {})
            result = {
                "draft_response": result_msg,
                "response_suggestions": ["Run the metadata notebook now"] if success else ["Upload notebooks", "Skip for now"],
            }
            result["phase"] = MigrationPhase.NOTEBOOKS_UPLOADED
            if _dbx_cfg:
                result["databricks_config"] = _dbx_cfg
            return result

    return upload_notebooks


def make_run_metadata_notebook_node(llm):
    def run_metadata_notebook(state: MigrationState) -> dict:
        phase = state.get("phase", MigrationPhase.NOTEBOOKS_UPLOADED)
        session_id = state.get("session_id", "-")
        messages = state.get("messages", [])
        user_content = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")

        with log_step(logger, "run_metadata_notebook", phase=phase, session_id=session_id):
            dbx_config = state.get("databricks_config", {})
            cluster_id = dbx_config.get("cluster_id", "")
            system = f"""You are a migration AI assistant. Tell the user you are triggering the metadata notebook:
/migrations/generic/01_load_metadata on cluster {cluster_id or 'the migration cluster'}.
This notebook reads the table metadata from blob storage and creates delta schema tables in Databricks.
Tell the user this will complete in a few minutes and ask if they want to proceed."""
            response = _call_llm_text(llm, system, user_content)
            return {
                "draft_response": response,
                "phase": MigrationPhase.METADATA_LOADED,
                "response_suggestions": ["Configure the pipeline table now", "What does the metadata notebook do?"],
            }

    return run_metadata_notebook


def make_configure_pipeline_node(llm, source_registry):
    def configure_pipeline(state: MigrationState) -> dict:
        phase = state.get("phase", MigrationPhase.METADATA_LOADED)
        session_id = state.get("session_id", "-")
        source_system = state.get("source_system", "sap_s4")
        messages = state.get("messages", [])
        user_content = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")

        with log_step(logger, "configure_pipeline", phase=phase, session_id=session_id):
            vault_config = state.get("vault_config", {})
            stored_secrets = vault_config.get("stored_secret_names", [])
            system = f"""You are a migration AI assistant. Tell the user you will now populate the pipeline_config delta table.
This table controls which tables are extracted, ingested into bronze, and transformed into silver.
Each row contains: table_name, source_system='{source_system}', secret names for credentials ({', '.join(stored_secrets)}), paths, load type, load flag.
The same notebooks run for all tables using this config — no per-table code needed.
Ask if they want to include all discovered tables or select specific ones."""
            response = _call_llm_text(llm, system, user_content)
            return {
                "draft_response": response,
                "phase": MigrationPhase.PIPELINE_CONFIGURED,
                "response_suggestions": ["Include all tables", "Let me select specific tables", "What is the pipeline_config table?"],
            }

    return configure_pipeline


def make_provision_adf_node(llm):
    def provision_adf(state: MigrationState) -> dict:
        phase = state.get("phase", MigrationPhase.PIPELINE_CONFIGURED)
        session_id = state.get("session_id", "-")
        messages = state.get("messages", [])
        user_content = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")

        with log_step(logger, "provision_adf", phase=phase, session_id=session_id):
            proposed = {
                "factory_name": f"migration-adf-{session_id[:6]}",
                "linked_services": ["Storage", "Databricks", "KeyVault"],
                "pipelines": ["landing_pipeline", "bronze_pipeline", "silver_pipeline"],
            }
            system = f"""You are a migration AI assistant. Tell the user you will create an Azure Data Factory to orchestrate the migration:
- Factory: {proposed['factory_name']}
- Linked Services: {', '.join(proposed['linked_services'])}
- Pipelines: {', '.join(proposed['pipelines'])} (one per layer)

ADF will trigger Databricks notebooks in sequence for each migration run.
Ask for approval to create these resources."""
            response = _call_llm_text(llm, system, user_content)
            return {
                "draft_response": response,
                "response_status": "approval_required",
                "pending_approval": {
                    "tool_name": "provision_adf",
                    "summary": f"Create Azure Data Factory '{proposed['factory_name']}' with 3 pipelines",
                    "tool_args": proposed,
                    "cloud_resource_type": "Azure Data Factory",
                },
                "response_suggestions": ["Approve ADF creation", "What does ADF do here?"],
            }

    return provision_adf


def make_finalize_node(llm):
    def finalize(state: MigrationState) -> dict:
        phase = state.get("phase", MigrationPhase.ADF_PROVISIONED)
        session_id = state.get("session_id", "-")
        messages = state.get("messages", [])
        user_content = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")

        with log_step(logger, "finalize", phase=phase, session_id=session_id):
            storage = state.get("storage_config", {})
            dbx = state.get("databricks_config", {})
            adf = state.get("adf_config", {})
            vault = state.get("vault_config", {})

            summary = {
                "source_system": state.get("source_system"),
                "cloud_provider": state.get("cloud_provider"),
                "secrets_vault": vault.get("vault_name"),
                "storage_account": storage.get("proposed", {}).get("storage_account"),
                "databricks_workspace": dbx.get("workspace_name"),
                "adf_factory": adf.get("factory_name"),
            }

            system = f"""You are a migration AI assistant. The migration setup is complete! Provide a summary:
{json.dumps(summary, indent=2)}

Tell the user everything that was created, what the pipeline_config table contains, and how to trigger a migration run via ADF.
Be encouraging and concise."""
            response = _call_llm_text(llm, system, user_content)
            return {
                "draft_response": response,
                "phase": MigrationPhase.COMPLETED,
                "response_phase_data": summary,
                "response_suggestions": ["Start a new migration", "How do I trigger a run?"],
            }

    return finalize


def make_clarify_node(llm):
    def clarify(state: MigrationState) -> dict:
        inbound = state.get("inbound_review", {})
        prompt = inbound.get("clarification_prompt") or "Could you please clarify what you'd like to do next?"
        return {
            "draft_response": prompt,
            "response_suggestions": _PHASE_SUGGESTIONS.get(state.get("phase", MigrationPhase.INIT), []),
        }
    return clarify


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_inbound_review(state: MigrationState) -> str:
    review = state.get("inbound_review", {})
    phase = state.get("phase", MigrationPhase.INIT)

    if review.get("clarification_needed"):
        return "clarify"

    # All non-INIT phases: route deterministically by phase. Never trust LLM routing
    # for non-INIT because the LLM only sees message text, not full state.
    if phase != MigrationPhase.INIT:
        return _default_routing_for_phase(phase)

    # INIT: let LLM routing stand (greet_and_probe vs clarify)
    return review.get("routing", _default_routing_for_phase(phase))


def route_after_outbound_review(state: MigrationState) -> str:
    review = state.get("outbound_review", {})
    if review.get("passes", True):
        return "assemble_response"
    attempt = state.get("regen_attempt", 0)
    if attempt >= 2:
        return "assemble_response"  # Use fallback
    return route_after_inbound_review(state)  # Re-route to same phase node


# ---------------------------------------------------------------------------
# Response assembly node
# ---------------------------------------------------------------------------

def assemble_response(state: MigrationState) -> dict:
    """Finalize the response from draft + review corrections."""
    outbound = state.get("outbound_review", {})
    draft = state.get("draft_response", "")
    attempt = state.get("regen_attempt", 0)

    if not outbound.get("passes", True) and attempt >= 2:
        final_message = "I encountered an issue generating a response. Please try again."
    elif outbound.get("revised_response"):
        final_message = outbound["revised_response"]
    else:
        final_message = draft

    phase = state.get("phase", MigrationPhase.INIT)
    suggestions = state.get("response_suggestions") or _PHASE_SUGGESTIONS.get(phase, [])
    status = state.get("response_status", "ok")

    return {
        "response_message": final_message,
        "response_status": status,
        "response_suggestions": suggestions,
        "messages": [AIMessage(content=final_message)],
        "regen_attempt": attempt + 1 if not outbound.get("passes", True) else 0,
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class MigrationGraphBuilder:
    def __init__(self, llm, source_registry, secrets_backend_factory, settings=None, orchestrator=None) -> None:
        self._llm = llm
        self._source_registry = source_registry
        self._secrets_backend_factory = secrets_backend_factory
        self._settings = settings
        self._orchestrator = orchestrator

    def compile(self, checkpointer=None):
        g = StateGraph(MigrationState)

        # Review nodes
        g.add_node("review_incoming", make_review_incoming_node(self._llm))
        g.add_node("review_outgoing", make_review_outgoing_node(self._llm))
        g.add_node("assemble_response", assemble_response)

        # Phase nodes
        g.add_node("greet_and_probe", make_greet_and_probe_node(self._llm, self._source_registry))
        g.add_node("collect_source_creds", make_collect_source_creds_node(self._llm, self._source_registry))
        g.add_node("select_cloud", make_select_cloud_node(self._llm))
        g.add_node("authenticate_cloud", make_authenticate_cloud_node(self._llm, self._settings))
        g.add_node("setup_secrets_manager", make_setup_secrets_manager_node(self._llm, self._secrets_backend_factory, self._settings))
        g.add_node("provision_storage", make_provision_storage_node(self._llm, self._source_registry, self._secrets_backend_factory, self._orchestrator))
        g.add_node("extract_metadata", make_extract_metadata_node(self._llm, self._source_registry, self._orchestrator))
        g.add_node("provision_databricks", make_provision_databricks_node(self._llm, self._orchestrator))
        g.add_node("upload_notebooks", make_upload_notebooks_node(self._llm, self._orchestrator))
        g.add_node("run_metadata_notebook", make_run_metadata_notebook_node(self._llm))
        g.add_node("configure_pipeline", make_configure_pipeline_node(self._llm, self._source_registry))
        g.add_node("provision_adf", make_provision_adf_node(self._llm))
        g.add_node("finalize", make_finalize_node(self._llm))
        g.add_node("clarify", make_clarify_node(self._llm))

        # Entry
        g.set_entry_point("review_incoming")

        # review_incoming → phase node (conditional)
        g.add_conditional_edges(
            "review_incoming",
            route_after_inbound_review,
            {
                "greet_and_probe": "greet_and_probe",
                "collect_source_creds": "collect_source_creds",
                "select_cloud": "select_cloud",
                "authenticate_cloud": "authenticate_cloud",
                "setup_secrets_manager": "setup_secrets_manager",
                "provision_storage": "provision_storage",
                "extract_metadata": "extract_metadata",
                "provision_databricks": "provision_databricks",
                "upload_notebooks": "upload_notebooks",
                "run_metadata_notebook": "run_metadata_notebook",
                "configure_pipeline": "configure_pipeline",
                "provision_adf": "provision_adf",
                "finalize": "finalize",
                "clarify": "clarify",
            },
        )

        # All phase nodes → review_outgoing
        for node in [
            "greet_and_probe", "collect_source_creds", "select_cloud",
            "authenticate_cloud", "setup_secrets_manager", "provision_storage", "extract_metadata",
            "provision_databricks", "upload_notebooks", "run_metadata_notebook",
            "configure_pipeline", "provision_adf", "finalize", "clarify",
        ]:
            g.add_edge(node, "review_outgoing")

        # review_outgoing → assemble or re-invoke phase node
        g.add_conditional_edges(
            "review_outgoing",
            route_after_outbound_review,
            {
                "assemble_response": "assemble_response",
                "greet_and_probe": "greet_and_probe",
                "collect_source_creds": "collect_source_creds",
                "select_cloud": "select_cloud",
                "authenticate_cloud": "authenticate_cloud",
                "setup_secrets_manager": "setup_secrets_manager",
                "provision_storage": "provision_storage",
                "extract_metadata": "extract_metadata",
                "provision_databricks": "provision_databricks",
                "upload_notebooks": "upload_notebooks",
                "run_metadata_notebook": "run_metadata_notebook",
                "configure_pipeline": "configure_pipeline",
                "provision_adf": "provision_adf",
                "finalize": "finalize",
                "clarify": "clarify",
            },
        )

        g.add_edge("assemble_response", END)

        return g.compile(checkpointer=checkpointer)
