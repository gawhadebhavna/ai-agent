from __future__ import annotations

from azure.storage.blob import BlobServiceClient

from app.config import get_settings
from app.core.exceptions import ProviderConfigurationError


class AzureBlobClientFactory:
    """Creates Azure Blob Storage clients from settings.

    Mirrors the AWSClientFactory pattern.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.has_azure_blob_credentials:
            raise ProviderConfigurationError(
                "AZURE_STORAGE_CONNECTION_STRING is required for Azure Blob Storage."
            )
        self._connection_string = settings.azure_storage_connection_string
        self._ssl_verify = settings.ssl_verify

    def blob_service(self) -> BlobServiceClient:
        return BlobServiceClient.from_connection_string(
            self._connection_string,
            connection_verify=self._ssl_verify,
        )
