from __future__ import annotations

"""
Deterministic credential capture.

Runs on every inbound user message BEFORE it reaches the LLM or the conversation
checkpoint. Recognised `key: value` / `key=value` pairs and Databricks tokens are
extracted and secret values are replaced with a placeholder, so they are never sent
to the model, logged, or persisted in chat history.
"""

import re
from dataclasses import dataclass, field

_KV_RE = re.compile(
    r"(?P<key>\b[A-Za-z][\w\-]{1,40})\s*[:=]\s*(?P<val>\"[^\"]*\"|'[^']*'|`[^`]*`|[^\s,;|]+)"
)
_PAT_RE = re.compile(r"\bdapi[a-zA-Z0-9\-]{20,}\b")
_GUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

_AZURE_ALIASES = {
    "tenant": "tenant_id", "tenantid": "tenant_id", "tenant_id": "tenant_id",
    "directory_id": "tenant_id", "directoryid": "tenant_id",
    "subscription": "subscription_id", "subscriptionid": "subscription_id",
    "subscription_id": "subscription_id", "sub_id": "subscription_id", "subid": "subscription_id",
    "client_secret": "client_secret", "clientsecret": "client_secret", "secret_value": "client_secret",
    "app_id": "client_id", "appid": "client_id", "application_id": "client_id",
}
_AZURE_ONLY_KEYS = {"tenant_id", "subscription_id", "client_secret"}

_SOURCE_ALIASES = {
    "baseurl": "base_url", "base-url": "base_url", "url": "base_url",
    "user": "username", "user_name": "username", "login": "username",
    "pass": "password", "pwd": "password", "passwd": "password",
    "clientid": "client_id", "client-id": "client_id", "mandt": "client_id",
    "servicename": "service_name", "service": "service_name",
}

_SECRET_KEYS = {"client_secret", "password", "token", "api_key", "apikey", "secret"}
PLACEHOLDER = "[captured]"


@dataclass
class CaptureResult:
    text: str
    azure: dict[str, str] = field(default_factory=dict)
    source: dict[str, str] = field(default_factory=dict)
    databricks_pat: str | None = None

    @property
    def captured_anything(self) -> bool:
        return bool(self.azure or self.source or self.databricks_pat)


def _clean(value: str) -> str:
    return value.strip().strip("'\"`").strip()


def capture_credentials(message: str, source_keys: list[str], source_secret_keys: set[str]) -> CaptureResult:
    """Extract credentials from a user message and return a redacted copy of it."""
    pairs: list[tuple[re.Match, str, str]] = []
    for m in _KV_RE.finditer(message):
        raw_key = m.group("key").lower().replace("-", "_")
        pairs.append((m, raw_key, _clean(m.group("val"))))

    azure_context = any(_AZURE_ALIASES.get(k) in _AZURE_ONLY_KEYS for _, k, _ in pairs)
    result = CaptureResult(text=message)
    redact_spans: list[tuple[int, int]] = []

    for m, raw_key, value in pairs:
        if not value:
            continue
        target: str | None = None
        key: str | None = None

        azure_key = _AZURE_ALIASES.get(raw_key)
        if azure_key:
            target, key = "azure", azure_key
        elif raw_key == "client_id" and (azure_context or _GUID_RE.match(value)):
            target, key = "azure", "client_id"
        else:
            source_key = raw_key if raw_key in source_keys else _SOURCE_ALIASES.get(raw_key)
            if source_key in source_keys:
                target, key = "source", source_key

        if target is None or key is None:
            continue
        if target == "azure":
            result.azure[key] = value
        else:
            result.source[key] = value
        if key in _SECRET_KEYS or key in source_secret_keys:
            redact_spans.append(m.span("val"))

    pat = _PAT_RE.search(message)
    if pat:
        result.databricks_pat = pat.group(0)
        if not any(s <= pat.start() < e for s, e in redact_spans):
            redact_spans.append(pat.span())

    if redact_spans:
        text = message
        for start, end in sorted(set(redact_spans), reverse=True):
            text = text[:start] + PLACEHOLDER + text[end:]
        result.text = text
    return result
