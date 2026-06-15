from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Migration AI Tool"
    app_env: str = "development"

    # AWS (kept for potential future AWS paths)
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    aws_profile: str | None = None
    aws_secrets_manager_region: str | None = None
    sqlite_path: str = "./var/app.db"
    llm_provider: str = "azure_openai"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    openai_model: str = "gpt-4.1-mini"
    openai_api_key: str | None = None
    approval_ttl_minutes: int = Field(default=60, ge=1)

    # Conversation memory
    max_conversation_history: int = 20

    # Notebooks
    notebooks_dir: str = "app/notebooks"

    # SAP API settings
    sap_ngrok_base_url: str | None = None
    sap_api_username: str | None = None
    sap_api_password: str | None = None

    # Azure OpenAI settings
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-02-15-preview"

    # Azure Blob Storage settings
    azure_storage_connection_string: str | None = None
    azure_container_name: str = "migration"

    # Azure Authentication (AAD/Entra ID)
    azure_token_credentials: str | None = None
    azure_tenant_id: str | None = None
    azure_client_id: str | None = None
    azure_client_secret: str | None = None
    azure_subscription_id: str | None = None

    # Azure resources
    azure_resource_group: str | None = None
    azure_location: str = "eastus"
    adf_factory_name: str | None = None

    # Databricks settings
    databricks_url: str | None = None
    databricks_token: str | None = None

    # TLS / SSL
    ssl_verify: bool = True
    ssl_ca_bundle: str | None = None

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def sqlite_path_obj(self) -> Path:
        return Path(self.sqlite_path).expanduser().resolve()

    @property
    def notebooks_dir_obj(self):
        from pathlib import Path as _Path
        return _Path(self.notebooks_dir).expanduser().resolve()

    @property
    def has_explicit_aws_credentials(self) -> bool:
        return bool(self.aws_access_key_id and self.aws_secret_access_key)

    @property
    def has_azure_openai_credentials(self) -> bool:
        return bool(self.azure_openai_api_key and self.azure_openai_endpoint)

    @property
    def has_azure_blob_credentials(self) -> bool:
        return bool(self.azure_storage_connection_string)

    @property
    def has_databricks_credentials(self) -> bool:
        return bool(self.databricks_url and self.databricks_token)

    @property
    def has_sap_api_credentials(self) -> bool:
        return bool(self.sap_ngrok_base_url and self.sap_api_username and self.sap_api_password)

    @property
    def has_azure_service_principal(self) -> bool:
        return bool(self.azure_tenant_id and self.azure_client_id and self.azure_client_secret)

    @property
    def ssl_verify_option(self) -> bool | str:
        if self.ssl_ca_bundle:
            return str(Path(self.ssl_ca_bundle).expanduser())
        return self.ssl_verify


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.sqlite_path_obj.parent.mkdir(parents=True, exist_ok=True)
    return settings
