from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    nvidia_api_key: str = ""
    nvidia_embed_model: str = "nvidia/Nemotron-3-Embed-1B"
    nvidia_llm_model: str = "deepseek-ai/DeepSeek-R1-Distill-8B"

    chroma_db_dir: str = "data/chroma_db"
    chroma_collection_name: str = "unasucre_documents"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 4

    raw_pdfs_dir: str = "data/raw_pdfs"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
