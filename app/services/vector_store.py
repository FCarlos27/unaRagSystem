from langchain_chroma import Chroma

from app.core.config import Settings
from app.utils.logging import get_logger

logger = get_logger("vector_store")


def build_vector_store(settings: Settings, embeddings) -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_db_dir,
    )
