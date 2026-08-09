from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Kept for compatibility with NVIDIA API deployments (currently unused).
    nvidia_api_key: str = ""

    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_llm_model: str = "llama3.2:3b"

    chroma_db_dir: str = "data/chroma_db"
    chroma_collection_name: str = "unasucre_documents"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 4

    raw_pdfs_dir: str = "data/raw_pdfs"
    json_docs_dir: str = "data/json_docs"

    unanswered_log_path: str = "data/unanswered_queries.jsonl"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
