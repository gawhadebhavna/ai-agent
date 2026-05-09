from app.services.metadata_registry import ENTITY_REGISTRY
from app.services.sap_data_service import SAPDataService


class DataOrchestrator:

    def __init__(
        self,
        sap_base_url,
        sap_username,
        sap_password
    ):

        self.registry = ENTITY_REGISTRY

        self.sap_service = SAPDataService(
            sap_base_url,
            sap_username,
            sap_password
        )

    def process_prompt(self, entity_name: str):

        entity = entity_name.lower()

        if entity not in self.registry:
            raise Exception(f"Unsupported entity: {entity}")

        config = self.registry[entity]

        endpoint = config["data_endpoint"]

        response = self.sap_service.fetch_data(endpoint)

        return {
            "status": "success",
            "entity": entity,
            "records": response
        }