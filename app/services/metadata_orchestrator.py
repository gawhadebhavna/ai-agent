from app.services.metadata_registry import ENTITY_REGISTRY
from app.services.sap_metadata_service import SAPMetadataService
from app.services.metadata_delta_service import MetadataDeltaService


class MetadataOrchestrator:

    def __init__(
        self,
        sap_base_url,
        sap_username,
        sap_password
    ):

        self.registry = ENTITY_REGISTRY

        self.sap_service = SAPMetadataService(
            sap_base_url,
            sap_username,
            sap_password
        )

        self.delta_service = MetadataDeltaService()

    def process_prompt(self, entity_name: str):

        entity = entity_name.lower()

        if entity not in self.registry:
            raise Exception(f"Unsupported entity: {entity}")

        config = self.registry[entity]

        metadata = self.sap_service.fetch_metadata(
            config["metadata_endpoint"]
        )

        self.delta_service.create_metadata_tables()

        self.delta_service.insert_metadata(metadata)

        return {
            "status": "success",
            "entity": metadata.entity_name,
            "table": metadata.table_name
        }