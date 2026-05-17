from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Migration Automation API"
    app_env: str = "development"
    aws_region: str = "ap-south-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    aws_profile: str | None = None
    allowed_buckets: str = "agentic-ai-migration-bkt"
    allowed_prefixes_json: str = '{"agentic-ai-migration-bkt":["landing/"]}'
    sqlite_path: str = "./var/app.db"
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gpt-oss:20b"
    openai_model: str = "gpt-4.1-mini"
    openai_api_key: str | None = None
    approval_ttl_minutes: int = Field(default=30, ge=1)

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
    azure_container_name: str = "sap-api"

    # Databricks settings
    databricks_url: str | None = None
    databricks_token: str | None = None
    databricks_job_id: int = 123

    # TLS / SSL
    # Set to false on networks with corporate SSL inspection certificates
    ssl_verify: bool = True
    # Optional custom CA bundle path for TLS verification (recommended on corporate proxies).
    ssl_ca_bundle: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def allowed_bucket_list(self) -> list[str]:
        return [item.strip() for item in self.allowed_buckets.split(",") if item.strip()]

    @property
    def allowed_prefixes(self) -> dict[str, list[str]]:
        raw = json.loads(self.allowed_prefixes_json or "{}")
        result: dict[str, list[str]] = {}
        for bucket, prefixes in raw.items():
            if isinstance(prefixes, str):
                result[bucket] = [prefixes]
            else:
                result[bucket] = [str(item) for item in prefixes]
        return result

    @property
    def sqlite_path_obj(self) -> Path:
        return Path(self.sqlite_path).expanduser().resolve()

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
    def ssl_verify_option(self) -> bool | str:
        if self.ssl_ca_bundle:
            return str(Path(self.ssl_ca_bundle).expanduser())
        return self.ssl_verify


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.sqlite_path_obj.parent.mkdir(parents=True, exist_ok=True)
    return settings
