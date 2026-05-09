import requests

from app.schemas.metadata import SAPMetadata


class SAPMetadataService:

    def __init__(
        self,
        base_url,
        username,
        password
    ):

        self.base_url = base_url
        self.username = username
        self.password = password

    def fetch_metadata(self, metadata_endpoint: str):

        url = f"{self.base_url}{metadata_endpoint}"

        response = requests.get(
            url,
            auth=(self.username, self.password)
        )

        response.raise_for_status()

        data = response.json()

        return SAPMetadata(**data)