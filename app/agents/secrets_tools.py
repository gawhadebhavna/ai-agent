from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.core.logging import get_logger, log_step

logger = get_logger("migration.secrets")


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------

class SecretsBackend(ABC):
    """Cloud-agnostic secrets management interface."""

    @abstractmethod
    def list_vaults(self) -> list[dict[str, Any]]:
        """List available vaults/namespaces."""
        ...

    @abstractmethod
    def create_vault(self, name: str, region: str, **kwargs: Any) -> dict[str, Any]:
        """Create a new vault/namespace. Raises on failure."""
        ...

    @abstractmethod
    def store_secret(self, vault_ref: str, key: str, value: str) -> dict[str, Any]:
        """Store a secret in the vault. Returns info dict (no value echoed)."""
        ...

    @abstractmethod
    def get_secret_names(self, vault_ref: str) -> list[str]:
        """List secret names (never values) in a vault."""
        ...

    @abstractmethod
    def get_secret(self, vault_ref: str, key: str) -> str | None:
        """Retrieve a single secret value by name. Returns None if not found."""
        ...


# ---------------------------------------------------------------------------
# Azure Key Vault backend (MCP first, SDK fallback)
# ---------------------------------------------------------------------------

class AzureKeyVaultBackend(SecretsBackend):
    def __init__(
        self,
        mcp_tools: dict | None = None,
        *,
        subscription_id: str | None = None,
        tenant_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        self._mcp_tools = mcp_tools or {}
        self._subscription_id = subscription_id
        self._tenant_id = tenant_id
        self._client_id = client_id
        self._client_secret = client_secret

    # --- MCP helpers --------------------------------------------------------

    def _try_mcp(self, tool_name: str, args: dict) -> dict[str, Any] | None:
        tool = self._mcp_tools.get(tool_name)
        if tool is None:
            return None
        try:
            result = tool.invoke(args)
            return result if isinstance(result, dict) else {"result": result}
        except Exception as exc:
            logger.warning("[MCP] %s failed (%s), falling back to SDK", tool_name, exc)
            return None

    # --- Secret name sanitizer -------------------------------------------
    # Azure Key Vault only allows alphanumeric characters and dashes.
    # Underscores and other characters must be replaced.
    @staticmethod
    def _sanitize_secret_name(name: str) -> str:
        import re as _re
        return _re.sub(r"[^a-zA-Z0-9-]", "-", name)

    # --- SDK credential helper ---------------------------------------------

    def _get_credential(self):
        from azure.identity import ClientSecretCredential, DefaultAzureCredential
        if self._tenant_id and self._client_id and self._client_secret:
            return ClientSecretCredential(self._tenant_id, self._client_id, self._client_secret)
        return DefaultAzureCredential()

    # --- Interface ---------------------------------------------------------

    def list_vaults(self) -> list[dict[str, Any]]:
        logger.info("[SECRETS] list_vaults → Azure Key Vault", extra={"phase": "VAULT_SETUP", "session_id": "-"})
        mcp_result = self._try_mcp("list_keyvaults", {})
        if mcp_result is not None:
            vaults = mcp_result.get("vaults") or mcp_result.get("result") or []
            return vaults if isinstance(vaults, list) else []

        # SDK fallback
        from azure.mgmt.keyvault import KeyVaultManagementClient
        credential = self._get_credential()
        client = KeyVaultManagementClient(credential, self._subscription_id or "")
        return [
            {"name": v.name, "location": v.location, "resource_group": v.id.split("/")[4]}
            for v in client.vaults.list()
        ]

    def create_vault(self, name: str, region: str, resource_group: str = "migration-rg", **kwargs: Any) -> dict[str, Any]:
        logger.info("[SECRETS] create_vault %s → Azure Key Vault", name, extra={"phase": "VAULT_SETUP", "session_id": "-"})
        mcp_result = self._try_mcp("create_keyvault", {"vault_name": name, "resource_group": resource_group, "location": region})
        if mcp_result is not None:
            return mcp_result

        # SDK fallback
        from azure.mgmt.keyvault import KeyVaultManagementClient
        from azure.mgmt.keyvault.models import (
            Sku,
            SkuName,
            VaultCreateOrUpdateParameters,
            VaultProperties,
        )
        try:
            credential = self._get_credential()
        except Exception as exc:
            exc_str = str(exc)
            if "AADSTS7000215" in exc_str or "invalid client secret" in exc_str.lower():
                raise ValueError(
                    "AADSTS7000215: Invalid client secret. You provided the Secret ID (a GUID) "
                    "instead of the secret Value. Go to Azure Portal → App registrations → "
                    "your app → Certificates & secrets → Client secrets, and copy the Value column."
                ) from exc
            raise

        # Ensure resource group exists before creating the vault
        try:
            from azure.mgmt.resource import ResourceManagementClient
            rg_client = ResourceManagementClient(credential, self._subscription_id or "")
            try:
                rg_client.resource_groups.get(resource_group)
            except Exception:
                logger.info(
                    "[SECRETS] Resource group '%s' not found — creating in %s",
                    resource_group, region, extra={"phase": "VAULT_SETUP", "session_id": "-"},
                )
                rg_client.resource_groups.create_or_update(resource_group, {"location": region})
                logger.info(
                    "[SECRETS] Resource group '%s' created",
                    resource_group, extra={"phase": "VAULT_SETUP", "session_id": "-"},
                )
        except Exception as rg_exc:
            logger.warning(
                "[SECRETS] Could not ensure resource group '%s': %s — proceeding anyway",
                resource_group, rg_exc, extra={"phase": "VAULT_SETUP", "session_id": "-"},
            )

        client = KeyVaultManagementClient(credential, self._subscription_id or "")

        # Decode the service-principal OID from the JWT token.
        # The OID (not client_id) is required for vault access policies.
        sp_oid: str | None = None
        try:
            import base64 as _b64
            import json as _j
            tok = credential.get_token("https://management.azure.com/.default")
            parts = tok.token.split(".")
            if len(parts) >= 2:
                padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
                claims = _j.loads(_b64.urlsafe_b64decode(padded))
                sp_oid = claims.get("oid")
        except Exception as _exc:
            exc_str = str(_exc)
            if "AADSTS7000215" in exc_str or "invalid client secret" in exc_str.lower():
                raise ValueError(
                    "AADSTS7000215: Invalid client secret. You provided the Secret ID (a GUID) "
                    "instead of the secret Value. Go to Azure Portal → App registrations → "
                    "your app → Certificates & secrets → Client secrets, and copy the Value column."
                ) from _exc
            logger.warning(
                "[SECRETS] OID extraction failed (%s) — vault created without access policy",
                _exc, extra={"phase": "VAULT_SETUP", "session_id": "-"},
            )

        access_policies: list = []
        if sp_oid and self._tenant_id:
            from azure.mgmt.keyvault.models import (
                AccessPolicyEntry,
                Permissions,
                SecretPermissions,
            )
            access_policies.append(
                AccessPolicyEntry(
                    tenant_id=self._tenant_id,
                    object_id=sp_oid,
                    permissions=Permissions(
                        secrets=[
                            SecretPermissions.GET,
                            SecretPermissions.LIST,
                            SecretPermissions.SET,
                            SecretPermissions.DELETE,
                        ]
                    ),
                )
            )

        props = VaultProperties(
            tenant_id=self._tenant_id or "",
            sku=Sku(family="A", name=SkuName.standard),
            # Use access policy mode (default) — does NOT require roleAssignments/write.
            # RBAC mode requires roleAssignments/write to assign roles, which Contributor lacks.
            enable_rbac_authorization=False,
            access_policies=access_policies,
        )
        params = VaultCreateOrUpdateParameters(location=region, properties=props)
        poller = client.vaults.begin_create_or_update(resource_group, name, params)
        vault = poller.result()

        # Optionally try RBAC role assignment as an enhancement (requires Owner/UAA on subscription).
        # If it fails the access policy above is already sufficient.
        if sp_oid and self._subscription_id:
            try:
                import uuid as _uuid
                from azure.mgmt.authorization import AuthorizationManagementClient
                from azure.mgmt.authorization.models import RoleAssignmentCreateParameters

                KV_SECRETS_OFFICER = "b86a8fe4-44ce-4948-aee5-eccb2c155cd7"
                vault_resource_id = vault.id
                role_def_id = (
                    f"/subscriptions/{self._subscription_id}/providers/Microsoft.Authorization"
                    f"/roleDefinitions/{KV_SECRETS_OFFICER}"
                )
                auth_client = AuthorizationManagementClient(credential, self._subscription_id)
                auth_client.role_assignments.create(
                    scope=vault_resource_id,
                    role_assignment_name=str(_uuid.uuid4()),
                    parameters=RoleAssignmentCreateParameters(
                        role_definition_id=role_def_id,
                        principal_id=sp_oid,
                        principal_type="ServicePrincipal",
                    ),
                )
                logger.info(
                    "[SECRETS] Key Vault Secrets Officer RBAC role also assigned to SP '%s'",
                    sp_oid, extra={"phase": "VAULT_SETUP", "session_id": "-"},
                )
            except Exception as rbac_exc:
                rbac_err = str(rbac_exc)
                if "RoleAssignmentExists" not in rbac_err and "AuthorizationFailed" not in rbac_err and "InsufficientPermissions" not in rbac_err:
                    logger.warning(
                        "[SECRETS] Optional RBAC assignment failed: %s",
                        rbac_err, extra={"phase": "VAULT_SETUP", "session_id": "-"},
                    )
                # Access policy mode is already sufficient — this is just a bonus

        return {"name": vault.name, "uri": vault.properties.vault_uri, "location": vault.location}

    def store_secret(self, vault_ref: str, key: str, value: str) -> dict[str, Any]:
        safe_key = self._sanitize_secret_name(key)
        logger.info("[SECRETS] store_secret %s (as '%s') \u2192 Azure Key Vault %s", key, safe_key, vault_ref, extra={"phase": "VAULT_SETUP", "session_id": "-"})
        mcp_result = self._try_mcp("keyvault_set_secret", {"vault_name": vault_ref, "secret_name": safe_key, "secret_value": value})
        if mcp_result is not None:
            return {"vault": vault_ref, "secret_name": safe_key, "status": "stored"}

        # SDK fallback
        from azure.keyvault.secrets import SecretClient
        vault_url = vault_ref if vault_ref.startswith("https://") else f"https://{vault_ref}.vault.azure.net"
        credential = self._get_credential()
        client = SecretClient(vault_url=vault_url, credential=credential)
        client.set_secret(safe_key, value)
        return {"vault": vault_ref, "secret_name": safe_key, "status": "stored"}

    def get_secret_names(self, vault_ref: str) -> list[str]:
        vault_url = vault_ref if vault_ref.startswith("https://") else f"https://{vault_ref}.vault.azure.net"
        credential = self._get_credential()
        from azure.keyvault.secrets import SecretClient
        client = SecretClient(vault_url=vault_url, credential=credential)
        return [s.name for s in client.list_properties_of_secrets()]

    def get_secret(self, vault_ref: str, key: str) -> str | None:
        safe_key = self._sanitize_secret_name(key)
        logger.info("[SECRETS] get_secret '%s' (as '%s') from %s", key, safe_key, vault_ref,
                    extra={"phase": "-", "session_id": "-"})
        mcp_result = self._try_mcp("keyvault_get_secret", {"vault_name": vault_ref, "secret_name": safe_key})
        if mcp_result is not None:
            return mcp_result.get("value") or mcp_result.get("secret_value")
        try:
            from azure.keyvault.secrets import SecretClient
            vault_url = (
                vault_ref if vault_ref.startswith("https://")
                else f"https://{vault_ref}.vault.azure.net"
            )
            credential = self._get_credential()
            client = SecretClient(vault_url=vault_url, credential=credential)
            return client.get_secret(safe_key).value
        except Exception as exc:
            logger.warning("[SECRETS] get_secret '%s/%s' failed: %s", vault_ref, safe_key, exc,
                           extra={"phase": "-", "session_id": "-"})
            return None


# ---------------------------------------------------------------------------
# AWS Secrets Manager backend
# ---------------------------------------------------------------------------

class AwsSecretsManagerBackend(SecretsBackend):
    def __init__(self, region: str | None = None, **boto_kwargs: Any) -> None:
        self._region = region or "us-east-1"
        self._boto_kwargs = boto_kwargs

    def _client(self):
        import boto3
        return boto3.client("secretsmanager", region_name=self._region, **self._boto_kwargs)

    def list_vaults(self) -> list[dict[str, Any]]:
        logger.info("[SECRETS] list_vaults → AWS Secrets Manager", extra={"phase": "VAULT_SETUP", "session_id": "-"})
        paginator = self._client().get_paginator("list_secrets")
        vaults: list[dict[str, Any]] = []
        for page in paginator.paginate():
            for s in page.get("SecretList", []):
                vaults.append({"name": s["Name"], "arn": s.get("ARN", ""), "description": s.get("Description", "")})
        return vaults

    def create_vault(self, name: str, region: str = "", **kwargs: Any) -> dict[str, Any]:
        logger.info("[SECRETS] create_vault %s → AWS Secrets Manager", name, extra={"phase": "VAULT_SETUP", "session_id": "-"})
        # AWS Secrets Manager doesn't have "vaults"; we create a namespace prefix secret placeholder
        client = self._client()
        resp = client.create_secret(Name=f"{name}/placeholder", SecretString='{"_init":"true"}')
        return {"name": name, "arn": resp.get("ARN", ""), "region": self._region}

    def store_secret(self, vault_ref: str, key: str, value: str) -> dict[str, Any]:
        logger.info("[SECRETS] store_secret %s/%s → AWS Secrets Manager", vault_ref, key, extra={"phase": "VAULT_SETUP", "session_id": "-"})
        secret_id = f"{vault_ref}/{key}"
        client = self._client()
        try:
            client.put_secret_value(SecretId=secret_id, SecretString=value)
        except client.exceptions.ResourceNotFoundException:
            client.create_secret(Name=secret_id, SecretString=value)
        return {"vault": vault_ref, "secret_name": key, "secret_id": secret_id, "status": "stored"}

    def get_secret_names(self, vault_ref: str) -> list[str]:
        paginator = self._client().get_paginator("list_secrets")
        names: list[str] = []
        for page in paginator.paginate(Filters=[{"Key": "name", "Values": [f"{vault_ref}/"]}]):
            for s in page.get("SecretList", []):
                short = s["Name"].removeprefix(f"{vault_ref}/")
                names.append(short)
        return names

    def get_secret(self, vault_ref: str, key: str) -> str | None:
        try:
            resp = self._client().get_secret_value(SecretId=f"{vault_ref}/{key}")
            return resp.get("SecretString")
        except Exception as exc:
            logger.warning("[SECRETS] AWS get_secret '%s/%s' failed: %s", vault_ref, key, exc,
                           extra={"phase": "-", "session_id": "-"})
            return None


# ---------------------------------------------------------------------------
# GCP Secret Manager (stub)
# ---------------------------------------------------------------------------

class GcpSecretManagerBackend(SecretsBackend):
    def list_vaults(self) -> list[dict[str, Any]]:
        raise NotImplementedError("GCP Secret Manager is not yet supported.")

    def create_vault(self, name: str, region: str = "", **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("GCP Secret Manager is not yet supported.")

    def store_secret(self, vault_ref: str, key: str, value: str) -> dict[str, Any]:
        raise NotImplementedError("GCP Secret Manager is not yet supported.")

    def get_secret_names(self, vault_ref: str) -> list[str]:
        raise NotImplementedError("GCP Secret Manager is not yet supported.")

    def get_secret(self, vault_ref: str, key: str) -> str | None:
        raise NotImplementedError("GCP Secret Manager is not yet supported.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_CLOUD_SECRETS_SERVICE_NAMES: dict[str, tuple[str, str]] = {
    "azure": ("Azure Key Vault", "keyvault"),
    "aws": ("AWS Secrets Manager", "aws_secrets"),
    "gcp": ("GCP Secret Manager", "gcp_secrets"),
}


def get_secrets_service_meta(cloud_provider: str) -> tuple[str, str]:
    """Return (service_name, icon_key) for a cloud provider."""
    return _CLOUD_SECRETS_SERVICE_NAMES.get(cloud_provider.lower(), ("Unknown Secrets Service", "secrets"))


def build_secrets_backend(cloud_provider: str, **kwargs: Any) -> SecretsBackend:
    """Factory: return the correct SecretsBackend for the given cloud."""
    provider = cloud_provider.lower()
    if provider == "azure":
        return AzureKeyVaultBackend(**kwargs)
    if provider == "aws":
        return AwsSecretsManagerBackend(**kwargs)
    if provider == "gcp":
        return GcpSecretManagerBackend()
    raise ValueError(f"Unsupported cloud provider for secrets: '{cloud_provider}'")
