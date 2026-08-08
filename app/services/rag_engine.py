from functools import lru_cache

from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.core.config import Settings, get_settings
from app.services.chat_history import (
    chat_sessions,
    format_history,
    new_session_id,
)
from app.services.embeddings import build_embeddings
from app.services.llm import build_llm, generate_answer, reformulate_query
from app.services.vector_store import build_vector_store
from app.utils.logging import get_logger

logger = get_logger("rag_engine")


class RagEngine:
    def __init__(
        self,
        settings: Settings,
        embeddings: OllamaEmbeddings,
        llm: ChatOllama,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.vector_store = build_vector_store(settings, embeddings)

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        top_k = top_k or self.settings.retrieval_top_k
        return self.vector_store.similarity_search(query, k=top_k)

    def answer(
        self, query: str, session_id: str | None = None
    ) -> tuple[str, list[str], float, str]:
        session_id = session_id or new_session_id()
        history = chat_sessions.get_or_create(session_id)
        history_text = format_history(history)

        search_query = query
        if history_text:
            search_query = reformulate_query(self.llm, query, history_text)

        docs = self.retrieve(search_query)
        if not docs:
            answer_text = "No pude encontrar información relevante."
            sources: list[str] = []
            confidence = 0.0
        else:
            context = "\n\n".join(doc.page_content for doc in docs)
            sources = sorted({doc.metadata.get("source", "desconocido") for doc in docs})
            answer_text = generate_answer(self.llm, query, context, history_text)
            confidence = 1.0

        chat_sessions.add_user_message(session_id, query)
        chat_sessions.add_ai_message(session_id, answer_text)
        return answer_text, sources, confidence, session_id


@lru_cache
def get_rag_engine() -> RagEngine:
    settings = get_settings()
    embeddings = build_embeddings(settings)
    llm = build_llm(settings)
    return RagEngine(settings, embeddings, llm)
