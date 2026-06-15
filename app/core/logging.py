from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from typing import Any

# Patterns that indicate sensitive credential data
_CREDENTIAL_PATTERNS = [
    re.compile(r"(?i)(password|passwd|pwd|secret|api_key|apikey|token|client_secret)\s*[=:]\s*\S+"),
    re.compile(r"(?i)(Authorization:\s*Bearer\s+)\S+"),
]

_REDACT_REPLACE = "[REDACTED]"


def redact_credentials(text: str) -> str:
    """Replace credential values in a string with [REDACTED]."""
    for pattern in _CREDENTIAL_PATTERNS:
        text = pattern.sub(
            lambda m: m.group(0).rsplit(m.group(0).split()[-1], 1)[0] + _REDACT_REPLACE
            if " " in m.group(0)
            else re.sub(r"([=:]\s*)\S+", r"\1" + _REDACT_REPLACE, m.group(0)),
            text,
        )
    return text


def contains_credentials(text: str) -> bool:
    """Return True if the text appears to contain credential values."""
    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(text):
            return True
    # Also check for raw password-like patterns: long random strings next to known field names
    simple = re.compile(
        r"(?i)\b(password|passwd|secret|token|key|credential)\b.{0,30}[\'\"]?([A-Za-z0-9+/=!@#$%^&*]{12,})",
        re.DOTALL,
    )
    return bool(simple.search(text))


class MigrationFormatter(logging.Formatter):
    """Custom formatter that adds step prefix and redacts credentials."""

    _step_counter: int = 0

    def format(self, record: logging.LogRecord) -> str:
        # Increment step counter per log record
        MigrationFormatter._step_counter += 1
        step = f"[STEP {MigrationFormatter._step_counter:03d}]"

        phase = getattr(record, "phase", "SYSTEM")
        session = getattr(record, "session_id", "-")
        session_label = f"[session:{session[:8]}]" if session and session != "-" else "[system]"
        phase_label = f"[{phase}]"

        original = super().format(record)
        # Redact any credential patterns that sneak into logs
        original = redact_credentials(original)

        return f"{step} {phase_label} {session_label} {original}"


def get_logger(name: str = "migration") -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root migration logger with MigrationFormatter."""
    MigrationFormatter._step_counter = 0  # reset on startup
    handler = logging.StreamHandler()
    handler.setFormatter(MigrationFormatter(fmt="%(name)s | %(levelname)s | %(message)s"))
    root = logging.getLogger("migration")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False


@contextmanager
def log_step(
    logger: logging.Logger,
    operation: str,
    *,
    phase: str = "SYSTEM",
    session_id: str = "-",
    extra: dict[str, Any] | None = None,
):
    """Context manager that logs START and OK/FAIL with elapsed time."""
    ctx = {"phase": phase, "session_id": session_id, **(extra or {})}
    logger.info("%s → START", operation, extra=ctx)
    t0 = time.perf_counter()
    try:
        yield
        elapsed = time.perf_counter() - t0
        logger.info("%s → OK (%.2fs)", operation, elapsed, extra=ctx)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("%s → FAIL (%.2fs): %s", operation, elapsed, exc, extra=ctx)
        raise


def log_tool_call(
    logger: logging.Logger,
    tool_name: str,
    args: dict[str, Any],
    *,
    source: str = "SDK",
    phase: str = "SYSTEM",
    session_id: str = "-",
) -> None:
    """Log a tool invocation, redacting credential values from args."""
    safe_args = {k: _REDACT_REPLACE if contains_credentials(f"{k}={v}") else v for k, v in args.items()}
    logger.info(
        "[TOOL] %s → using %s | args=%s",
        tool_name,
        source,
        safe_args,
        extra={"phase": phase, "session_id": session_id},
    )
