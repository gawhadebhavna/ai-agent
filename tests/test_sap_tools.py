from __future__ import annotations

from unittest.mock import patch

from app.agents.sap_tools import build_sap_tools, call_sap_api, get_sap_table_metadata
from app.config import Settings


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def build_settings() -> Settings:
    return Settings(
        sap_ngrok_base_url="https://dummy-sap.example",
        sap_api_username="user",
        sap_api_password="pass",
    )


def test_call_sap_api_reads_table_results_without_d_wrapper():
    payload = {"table": "KNA1", "count": 1, "results": [{"KUNNR": "100001"}]}

    with patch("app.agents.sap_tools.requests.get", return_value=FakeResponse(payload)):
        rows = call_sap_api("/tables/KNA1/data", settings=build_settings())

    assert rows == [{"KUNNR": "100001"}]


def test_get_sap_table_data_tool_uses_table_endpoint_and_query_aliases():
    payload = {"table": "KNA1", "count": 1, "results": [{"KUNNR": "100001", "ORT01": "DXB"}]}

    with patch("app.agents.sap_tools.requests.get", return_value=FakeResponse(payload)) as request_get:
        result = build_sap_tools(build_settings())["get_sap_table_data"].invoke(
            {
                "table_name": "customers",
                "top": 2,
                "search": "ENOC",
                "filter_query": "ORT01 eq 'DXB'",
                "select_query": "KUNNR,ORT01",
            }
        )

    assert result == payload
    _, kwargs = request_get.call_args
    assert request_get.call_args.args[0] == "https://dummy-sap.example/tables/KNA1/data"
    assert kwargs["params"] == {
        "$filter": "ORT01 eq 'DXB'",
        "$select": "KUNNR,ORT01",
        "$top": 2,
        "search": "ENOC",
    }


def test_get_sap_table_metadata_returns_metadata_payload():
    payload = {"table": "MARA", "field_count": 1, "fields": [{"FIELDNAME": "MATNR"}]}

    with patch("app.agents.sap_tools.requests.get", return_value=FakeResponse(payload)) as request_get:
        result = get_sap_table_metadata("products", settings=build_settings())

    assert result == payload
    assert request_get.call_args.args[0] == "https://dummy-sap.example/tables/MARA/metadata"
