from __future__ import annotations

import httpx

from app.config import Settings
from app.core.exceptions import ProviderConfigurationError


def _looks_like_openai_key(api_key: str) -> bool:
    return api_key.strip().startswith("sk-")


def _build_ollama_chat_model(*, settings: Settings, verify: bool | str):
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ProviderConfigurationError(
            "langchain-ollama is not installed. Install project dependencies first."
        ) from exc
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"verify": verify},
        temperature=0,
    )


def _build_openai_chat_model(*, settings: Settings, verify: bool | str):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ProviderConfigurationError(
            "langchain-openai is not installed. Install project dependencies first."
        ) from exc

    openai_api_key = (settings.openai_api_key or "").strip()
    if not openai_api_key:
        raise ProviderConfigurationError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

    if not _looks_like_openai_key(openai_api_key):
        if settings.has_azure_openai_credentials:
            return _build_azure_openai_chat_model(
                settings=settings,
                verify=verify,
                fallback_api_key=openai_api_key,
            )
        raise ProviderConfigurationError(
            "OPENAI_API_KEY does not look like a valid OpenAI key (expected prefix 'sk-'). "
            "If you are using Azure OpenAI credentials, set LLM_PROVIDER=azure_openai "
            "and configure AZURE_OPENAI_* values."
        )

    sync_client = httpx.Client(verify=verify)
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=openai_api_key,
        http_client=sync_client,
        temperature=0,
    )


def _build_azure_openai_chat_model(
    *,
    settings: Settings,
    verify: bool | str,
    fallback_api_key: str | None = None,
):
    try:
        from langchain_openai import AzureChatOpenAI
    except ImportError as exc:
        raise ProviderConfigurationError(
            "langchain-openai is not installed. Install project dependencies first."
        ) from exc

    azure_endpoint = (settings.azure_openai_endpoint or "").strip()
    azure_api_key = (settings.azure_openai_api_key or fallback_api_key or "").strip()
    if not azure_api_key or not azure_endpoint:
        raise ProviderConfigurationError(
            "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are required when "
            "LLM_PROVIDER=azure_openai."
        )

    sync_client = httpx.Client(verify=verify)
    return AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_deployment,
        http_client=sync_client,
        temperature=0,
    )


def build_runtime_chat_model(settings: Settings):
    """Build a runtime chat model from the LLM_PROVIDER toggle."""
    verify = settings.ssl_verify_option
    provider = settings.llm_provider.lower().strip()
    if provider == "ollama":
        return _build_ollama_chat_model(settings=settings, verify=verify)

    if provider == "openai":
        return _build_openai_chat_model(settings=settings, verify=verify)

    if provider in {"azure_openai", "azure"}:
        return _build_azure_openai_chat_model(settings=settings, verify=verify)

    raise ProviderConfigurationError(
        f"Unsupported LLM provider '{settings.llm_provider}'. Use 'ollama', 'openai', or 'azure_openai'."
    )
