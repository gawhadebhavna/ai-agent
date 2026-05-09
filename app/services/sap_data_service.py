import requests

from requests.auth import HTTPBasicAuth


class SAPDataService:

    def __init__(
        self,
        base_url,
        username,
        password
    ):

        self.base_url = base_url
        self.username = username
        self.password = password

    def fetch_data(
        self,
        endpoint: str,
        params: dict | None = None
    ):

        url = f"{self.base_url}{endpoint}"

        response = requests.get(
            url,
            params=params,
            auth=HTTPBasicAuth(
                self.username,
                self.password
            ),
            headers={
                "ngrok-skip-browser-warning": "true"
            },
            timeout=30
        )

        response.raise_for_status()

        return response.json()