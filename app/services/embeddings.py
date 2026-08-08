from langchain_ollama import OllamaEmbeddings

from app.core.config import Settings
from app.utils.logging import get_logger

logger = get_logger("embeddings")


def build_embeddings(settings: Settings) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
    )
