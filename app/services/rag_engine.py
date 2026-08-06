from functools import lru_cache

from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from app.core.config import Settings, get_settings
from app.services.embeddings import build_embeddings
from app.services.llm import build_llm, generate_answer
from app.services.vector_store import build_vector_store
from app.utils.logging import get_logger

logger = get_logger("rag_engine")


class RagEngine:
    def __init__(
        self,
        settings: Settings,
        embeddings: NVIDIAEmbeddings,
        llm: ChatNVIDIA,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.vector_store = build_vector_store(settings, embeddings)

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        top_k = top_k or self.settings.retrieval_top_k
        return self.vector_store.similarity_search(query, k=top_k)

    def answer(self, query: str) -> tuple[str, list[str], float]:
        docs = self.retrieve(query)
        if not docs:
            return "No pude encontrar información relevante.", [], 0.0

        context = "\n\n".join(doc.page_content for doc in docs)
        sources = sorted({doc.metadata.get("source", "desconocido") for doc in docs})
        answer = generate_answer(self.llm, query, context)
        return answer, sources, 1.0


@lru_cache
def get_rag_engine() -> RagEngine:
    settings = get_settings()
    embeddings = build_embeddings(settings)
    llm = build_llm(settings)
    return RagEngine(settings, embeddings, llm)
