from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from app.config import get_settings
from app.services.aws import AWSClientFactory


AWS_ENV_KEYS = {
    "AWS_ACCESS_KEY_ID": "",
    "AWS_SECRET_ACCESS_KEY": "",
    "AWS_SESSION_TOKEN": "",
    "AWS_PROFILE": "",
}


def test_settings_read_explicit_aws_credentials_from_env_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "AWS_REGION=ap-south-1",
                "AWS_ACCESS_KEY_ID=test-access-key",
                "AWS_SECRET_ACCESS_KEY=test-secret-key",
                "AWS_SESSION_TOKEN=test-session-token",
                "ALLOWED_BUCKETS=agentic-ai-migration-bkt",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.aws_region == "ap-south-1"
    assert settings.aws_access_key_id == "test-access-key"
    assert settings.aws_secret_access_key == "test-secret-key"
    assert settings.aws_session_token == "test-session-token"
    assert settings.has_explicit_aws_credentials is True


def test_settings_allow_empty_aws_credentials():
    settings = Settings(
        aws_access_key_id=None,
        aws_secret_access_key=None,
        aws_session_token=None,
        aws_profile=None,
    )

    assert settings.has_explicit_aws_credentials is False


def test_aws_client_factory_prefers_explicit_credentials():
    with patch.dict(os.environ, AWS_ENV_KEYS, clear=False):
        settings = Settings(
            aws_region="ap-south-1",
            aws_access_key_id="test-access-key",
            aws_secret_access_key="test-secret-key",
            aws_session_token="test-session-token",
        )

        with patch("app.services.aws.boto3.session.Session") as session_ctor:
            AWSClientFactory(settings)

        session_ctor.assert_called_once_with(
            region_name="ap-south-1",
            aws_access_key_id="test-access-key",
            aws_secret_access_key="test-secret-key",
            aws_session_token="test-session-token",
        )


def test_aws_client_factory_uses_profile_when_keys_are_absent():
    with patch.dict(os.environ, AWS_ENV_KEYS, clear=False):
        settings = Settings(
            aws_region="ap-south-1",
            aws_profile="local-dev",
        )

        with patch("app.services.aws.boto3.session.Session") as session_ctor:
            AWSClientFactory(settings)

        session_ctor.assert_called_once_with(
            region_name="ap-south-1",
            profile_name="local-dev",
        )


def test_aws_client_factory_falls_back_to_default_boto3_chain():
    with patch.dict(os.environ, AWS_ENV_KEYS, clear=False):
        settings = Settings(
            aws_region="ap-south-1",
        )

        with patch("app.services.aws.boto3.session.Session") as session_ctor:
            AWSClientFactory(settings)

        session_ctor.assert_called_once_with(region_name="ap-south-1")
