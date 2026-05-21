from __future__ import annotations

import boto3
from botocore.client import BaseClient

from app.config import Settings


class AWSClientFactory:
    def __init__(self, settings: Settings) -> None:
        self._session = boto3.session.Session(**self._build_session_kwargs(settings))

    def s3(self) -> BaseClient:
        return self._session.client("s3")

    @staticmethod
    def _build_session_kwargs(settings: Settings) -> dict[str, str]:
        kwargs: dict[str, str] = {"region_name": settings.aws_region}
        if settings.has_explicit_aws_credentials:
            kwargs["aws_access_key_id"] = settings.aws_access_key_id or ""
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key or ""
            if settings.aws_session_token:
                kwargs["aws_session_token"] = settings.aws_session_token
            return kwargs
        if settings.aws_profile:
            kwargs["profile_name"] = settings.aws_profile
        return kwargs
