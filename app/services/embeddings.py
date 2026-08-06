from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from app.core.config import Settings
from app.utils.logging import get_logger

logger = get_logger("embeddings")


def build_embeddings(settings: Settings) -> NVIDIAEmbeddings:
    if not settings.nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY is required to build embeddings")
    return NVIDIAEmbeddings(
        model=settings.nvidia_embed_model,
        api_key=settings.nvidia_api_key,
    )
