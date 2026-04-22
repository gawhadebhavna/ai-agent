from __future__ import annotations


class AppError(Exception):
    def __init__(self, message: str, *, status_code: int = 400, code: str = "app_error") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class AccessDeniedError(AppError):
    def __init__(self, message: str = "Access denied.") -> None:
        super().__init__(message, status_code=403, code="access_denied")


class NotFoundError(AppError):
    def __init__(self, message: str = "Resource not found.") -> None:
        super().__init__(message, status_code=404, code="not_found")


class ProviderConfigurationError(AppError):
    def __init__(self, message: str = "LLM provider is not configured correctly.") -> None:
        super().__init__(message, status_code=500, code="provider_configuration_error")


class ApprovalError(AppError):
    def __init__(self, message: str = "Approval request is invalid.") -> None:
        super().__init__(message, status_code=409, code="approval_error")
