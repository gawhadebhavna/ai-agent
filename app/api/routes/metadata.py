from fastapi import APIRouter

from app.services.metadata_orchestrator import MetadataOrchestrator

from app.config import get_settings

router = APIRouter(prefix="/v1/metadata")


@router.post("/register/{entity}")
def register_metadata(entity: str):

    orchestrator = MetadataOrchestrator(
        sap_base_url=settings.SAP_NGROK_BASE_URL,
        sap_username=settings.SAP_API_USERNAME,
        sap_password=settings.SAP_API_PASSWORD
    )

    return orchestrator.process_prompt(entity)