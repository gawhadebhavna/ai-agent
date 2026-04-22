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


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.sqlite_path_obj.parent.mkdir(parents=True, exist_ok=True)
    return settings
