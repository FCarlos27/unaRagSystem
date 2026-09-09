from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.core.config import Settings, get_settings
from app.services.chat_history import (
    chat_sessions,
    format_history,
    new_session_id,
)
from app.services.llm import (
    generate_answer,
    reformulate_query,
    build_chat_llm,
    build_embeddings_llm,
)
from app.services.rules import (
    evaluate_deterministic_rules,
    match_heuristic,
    should_skip_reformulation,
)
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
    def __init__(
        self,
        settings: Settings,
        embeddings_llm: OllamaEmbeddings,
        chat_llm: ChatOllama,
    ) -> None:
        self.settings = settings
        self.llm = chat_llm
        self.embeddings = embeddings_llm
        self.vector_store = build_vector_store(settings, embeddings_llm)

    def retrieve(
        self, query: str, top_k: int | None = None, where: dict | None = None
    ) -> List[Document]:
        """Semantic similarity search over the vector store."""
        top_k = top_k or self.settings.retrieval_top_k
        return self.vector_store.similarity_search(query, k=top_k, filter=where)

    def answer(
        self, query: str, session_id: str | None = None
    ) -> tuple[str, List[str], float, str]:
        """Waterfall Router: Heuristics -> Deterministic -> Semantic RAG."""
        session_id = session_id or new_session_id()
        history = chat_sessions.get_or_create(session_id)
        history_messages = history.messages
        history_text = format_history(history)

        # --- Tier 1: Heuristics (fast-path, <1ms) ---
        heuristic_response = match_heuristic(query)
        if heuristic_response:
            logger.info("Tier 1 (heuristic) matched for query: %s", query[:50])
            chat_sessions.add_user_message(session_id, query)
            chat_sessions.add_ai_message(session_id, heuristic_response)
            return heuristic_response, [], 1.0, session_id

        # --- Tier 2: Deterministic (exact metadata lookups with targeted filters) ---
        deterministic_response, sources = evaluate_deterministic_rules(
            query=query,
            retrieve_fn=lambda query, filter: self.retrieve(query, where=filter),
        )
        if deterministic_response:
            logger.info("Tier 2 (deterministic) matched for query: %s", query[:50])
            chat_sessions.add_user_message(session_id, query)
            chat_sessions.add_ai_message(session_id, deterministic_response)
            return deterministic_response, sources or [], 1.0, session_id

        # --- Tier 3: Semantic RAG (LLM generation, ~500-1500ms) ---
        search_query = query
        docs = self.retrieve(search_query)

        # Check if reformulation is needed
        if history_text and not should_skip_reformulation(query, history_messages):
            reformulated = reformulate_query(self.llm, query, history_text)
            if reformulated != query:
                search_query = reformulated
                # Re-retrieve vector docs using the expanded search query
                docs = self.retrieve(search_query)

        if not docs:
            logger.info("Tier 3 (semantic): No documents retrieved for query: %s", query[:50])
            fallback = "No pude encontrar información relevante."
            chat_sessions.add_user_message(session_id, query)
            chat_sessions.add_ai_message(session_id, fallback)
            return fallback, [], 0.0, session_id

        sources = sorted({doc.metadata.get("source", "desconocido") for doc in docs})
        context = "\n\n".join(doc.page_content for doc in docs)

        answer_text = generate_answer(self.llm, search_query, context, history_text)

        if _declined_answer(answer_text):
            logger.warning("LLM declined to answer despite context: %s", search_query[:50])

        chat_sessions.add_user_message(session_id, query)
        chat_sessions.add_ai_message(session_id, answer_text)

        logger.info("Tier 3 (semantic) generated answer for query: %s", query[:50])
        return answer_text, sources, 0.95, session_id


@lru_cache
def get_rag_engine() -> RagEngine:
    """Build (and cache) a single RagEngine wired to settings, embeddings_llm, and LLM."""
    settings = get_settings()
    embeddings_llm = build_embeddings_llm(settings)
    chat_llm = build_chat_llm(settings)
    return RagEngine(settings, embeddings_llm, chat_llm)
