from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.tools import StructuredTool

from app.schemas.chat import CredentialField, SourceSystemMeta


class SourceSystemPlugin(ABC):
    """Abstract base for a source system integration."""

    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @property
    @abstractmethod
    def icon_key(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def required_cred_fields(self) -> list[CredentialField]: ...

    @abstractmethod
    def build_tools(self, **kwargs: Any) -> dict[str, StructuredTool]:
        """Return a dict of tool_name → StructuredTool for this source system."""
        ...

    def to_meta(self) -> SourceSystemMeta:
        return SourceSystemMeta(
            id=self.id,
            display_name=self.display_name,
            icon_key=self.icon_key,
            description=self.description,
            required_cred_fields=self.required_cred_fields,
        )


# ---------------------------------------------------------------------------
# SAP plugin
# ---------------------------------------------------------------------------

class SapSourcePlugin(SourceSystemPlugin):
    @property
    def id(self) -> str:
        return "sap_s4"

    @property
    def display_name(self) -> str:
        return "SAP S/4HANA"

    @property
    def icon_key(self) -> str:
        return "sap"

    @property
    def description(self) -> str:
        return "OData and API-based extraction from SAP S/4HANA or ECC systems."

    @property
    def required_cred_fields(self) -> list[CredentialField]:
        return [
            CredentialField(key="base_url", label="SAP API Base URL", placeholder="http://your-sap-host:8011", secret=False),
            CredentialField(key="username", label="Username", placeholder="demo", secret=False),
            CredentialField(key="password", label="Password", placeholder="••••••••", secret=True),
            CredentialField(key="client_id", label="Client ID", placeholder="100", secret=False),
        ]

    def build_tools(self, **kwargs: Any) -> dict[str, StructuredTool]:
        from app.agents.sap_tools import build_sap_tools
        from app.config import Settings

        settings: Settings = kwargs["settings"]
        return build_sap_tools(settings)


# ---------------------------------------------------------------------------
# Future stub — Oracle
# ---------------------------------------------------------------------------

class OracleSourcePlugin(SourceSystemPlugin):
    @property
    def id(self) -> str:
        return "oracle"

    @property
    def display_name(self) -> str:
        return "Oracle Database"

    @property
    def icon_key(self) -> str:
        return "oracle"

    @property
    def description(self) -> str:
        return "Schema and table-based extraction from Oracle Database."

    @property
    def required_cred_fields(self) -> list[CredentialField]:
        return [
            CredentialField(key="host", label="Host", placeholder="oracle-host.example.com", secret=False),
            CredentialField(key="port", label="Port", placeholder="1521", secret=False),
            CredentialField(key="service_name", label="Service Name", placeholder="ORCL", secret=False),
            CredentialField(key="username", label="Username", placeholder="schema_user", secret=False),
            CredentialField(key="password", label="Password", placeholder="••••••••", secret=True),
        ]

    def build_tools(self, **kwargs: Any) -> dict[str, StructuredTool]:
        raise NotImplementedError("Oracle source plugin tools are not yet implemented.")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SourceSystemRegistry:
    """Registry of all available source system plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, SourceSystemPlugin] = {}
        # Register built-in plugins
        self._register(SapSourcePlugin())
        self._register(OracleSourcePlugin())

    def _register(self, plugin: SourceSystemPlugin) -> None:
        self._plugins[plugin.id] = plugin

    def get_plugin(self, source_id: str) -> SourceSystemPlugin | None:
        return self._plugins.get(source_id)

    def list_sources(self) -> list[SourceSystemMeta]:
        return [p.to_meta() for p in self._plugins.values()]

    def get_tools(self, source_id: str, **kwargs: Any) -> dict[str, StructuredTool]:
        plugin = self.get_plugin(source_id)
        if plugin is None:
            raise ValueError(f"Unknown source system '{source_id}'. Available: {list(self._plugins.keys())}")
        return plugin.build_tools(**kwargs)
