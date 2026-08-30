from functools import lru_cache

from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.core.config import Settings, get_settings
from app.services.chat_history import (
    chat_sessions,
    format_history,
    new_session_id,
)

from app.services.llm import build_chat_llm, generate_answer, reformulate_query, build_embeddings_llm
from app.services.vector_store import build_vector_store
from app.utils.logging import get_logger

logger = get_logger("rag_engine")


def _declined_answer(answer_text: str) -> bool:
    """Detect when the LLM refused to answer despite having retrieved context."""
    declined = (
        "no puedo",
        "no pude",
        "no puedo determinar",
        "no tengo la información",
        "no se proporciona información",
        "no se indica",
        "no dispongo",
    )
    return any(marker in answer_text.lower() for marker in declined)

class RagEngine:
    def __init__(self, settings: Settings, embeddings_llm: OllamaEmbeddings, chat_llm: ChatOllama) -> None:
        self.settings = settings
        self.chat_llm = chat_llm
        self.vector_store = build_vector_store(settings, embeddings_llm)

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Semantic similarity search over the vector store."""
        k = top_k or self.settings.retrieval_top_k
        return self.vector_store.similarity_search(query, k=k)

    def answer(self, query: str, session_id: str | None = None) -> tuple[str, list[str], float, str]:
        """Core RAG execution: Reformulate -> Retrieve -> Generate."""
        session_id = session_id or new_session_id()
        history = chat_sessions.get_or_create(session_id)
        history_text = format_history(history)

        # 1. Query Reformulation
        search_query = reformulate_query(self.chat_llm, query, history_text) if history_text else query

        # 2. Retrieval
        docs = self.retrieve(search_query)
        if not docs:
            self._log_unanswered(query, session_id, reason="no_context")
            return "No pude encontrar información relevante.", [], 0.0, session_id

        sources = sorted({doc.metadata.get("source", "desconocido") for doc in docs})
        context = "\n\n".join(doc.page_content for doc in docs)

        # 3. Generation
        answer_text = generate_answer(self.chat_llm, query, context, history_text)
        
        if _declined_answer(answer_text):
            self._log_unanswered(query, session_id, sources=sources, reason="declined_with_context")

        # 4. History Update
        chat_sessions.add_user_message(session_id, query)
        chat_sessions.add_ai_message(session_id, answer_text)

        return answer_text, sources, 1.0, session_id


@lru_cache
def get_rag_engine() -> RagEngine:
    """Build (and cache) a single RagEngine wired to settings, embeddings_llm, and LLM."""
    settings = get_settings()
    embeddings_llm = build_embeddings_llm(settings)
    chat_llm = build_chat_llm(settings)
    return RagEngine(settings, embeddings_llm, chat_llm)
