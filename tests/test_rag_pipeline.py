"""Pytest suite for the RAG microservice pipeline components.

All external dependencies (Ollama, ChromaDB server) are mocked so the
suite runs instantly and deterministically in any environment.
"""

from typing import List, Generator
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings() -> MagicMock:
    """Return a mocked Settings instance with typical defaults."""
    settings = MagicMock()
    settings.retrieval_top_k = 4
    settings.chunk_size = 1000
    settings.chunk_overlap = 180
    return settings


@pytest.fixture
def mock_llm() -> MagicMock:
    """Return a mock ChatOllama that echoes predictable content."""
    llm = MagicMock()
    default_response = MagicMock()
    default_response.content = "Respuesta predeterminada del LLM"
    llm.invoke.return_value = default_response
    return llm


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Return a mock vector store whose similarity_search yields controlled docs."""
    store = MagicMock()
    return store


@pytest.fixture
def mock_rag_engine(mock_settings, mock_llm, mock_vector_store) -> Generator[MagicMock, None, None]:
    """Patch RagEngine dependencies and return a fully mocked instance."""
    import app.services.rag_engine as rag_engine_mod

    with (
        patch.object(rag_engine_mod, "build_vector_store", return_value=mock_vector_store),
        patch.object(rag_engine_mod, "chat_sessions") as mock_sessions,
    ):
        RagEngineClass = rag_engine_mod.RagEngine

        engine = RagEngineClass(mock_settings, MagicMock(), mock_llm)
        engine.vector_store = mock_vector_store
        engine._mock_sessions = mock_sessions
        engine._log_unanswered = MagicMock()
        yield engine


# ---------------------------------------------------------------------------
# reformulate_query tests
# ---------------------------------------------------------------------------

def test_reformulate_query_standalone(mock_llm: MagicMock) -> None:
    """When history is empty, reformulate_query returns the original question."""
    from app.services.llm import reformulate_query

    result = reformulate_query(mock_llm, "¿Cuál es mi horario?", "")
    assert result == "¿Cuál es mi horario?"


def test_reformulate_query_with_history(mock_llm: MagicMock) -> None:
    """With chat history, LLM is invoked and the reformulated string is returned."""
    from app.services.llm import reformulate_query

    mock_response = MagicMock()
    mock_response.content = "horario de clases"
    mock_llm.invoke.return_value = mock_response

    history = "Usuario: ¿Dónde estudio?\nAsistente: En la UNASUCRE."
    result = reformulate_query(mock_llm, "¿Y cuál es mi horario?", history)

    assert result == "horario de clases"
    mock_llm.invoke.assert_called_once()


def test_reformulate_query_fallback_on_exception(mock_llm: MagicMock) -> None:
    """If the LLM raises, reformulate_query falls back to the raw question."""
    from app.services.llm import reformulate_query

    mock_llm.invoke.side_effect = RuntimeError("LLM exploded")

    history = "Usuario: ¿Qué es la UNASUCRE?"
    result = reformulate_query(mock_llm, "¿Y el horario?", history)

    assert result == "¿Y el horario?"


# ---------------------------------------------------------------------------
# RagEngine.answer tests
# ---------------------------------------------------------------------------

def test_rag_engine_answer_with_context(mock_rag_engine, mock_llm) -> None:
    """Full RAG flow when documents are retrieved: context built, LLM invoked."""
    docs: List[Document] = [
        Document(
            page_content="La UNASUCRE ofrece cursos por semestre.",
            metadata={"source": "instructivo.docx", "chunk_index": 0},
        ),
        Document(
            page_content="Los horarios se publican en la web.",
            metadata={"source": "instructivo.docx", "chunk_index": 1},
        ),
    ]
    mock_rag_engine.vector_store.similarity_search.return_value = docs

    mock_response = MagicMock()
    mock_response.content = "La UNASUCRE publica horarios en su portal web."
    mock_llm.invoke.return_value = mock_response

    mock_sessions = mock_rag_engine._mock_sessions
    mock_history = MagicMock()
    mock_history.messages = []
    mock_sessions.get_or_create.return_value = mock_history
    mock_sessions.add_user_message = MagicMock()
    mock_sessions.add_ai_message = MagicMock()

    answer, sources, confidence, session_id = mock_rag_engine.answer(
        "¿Dónde encuentro mis horarios?", session_id="test-session"
    )

    assert answer == "La UNASUCRE publica horarios en su portal web."
    assert sources == ["instructivo.docx"]
    assert confidence == 0.95
    assert session_id == "test-session"

    mock_rag_engine.vector_store.similarity_search.assert_called_once()
    mock_llm.invoke.assert_called_once()
    mock_sessions.add_user_message.assert_called_once_with(
        "test-session", "¿Dónde encuentro mis horarios?"
    )
    mock_sessions.add_ai_message.assert_called_once_with(
        "test-session", "La UNASUCRE publica horarios en su portal web."
    )


def test_rag_engine_answer_no_context(mock_rag_engine) -> None:
    """When vector search returns [], fallback answer is returned."""
    mock_rag_engine.vector_store.similarity_search.return_value = []
    mock_rag_engine._mock_sessions.get_or_create.return_value = MagicMock(messages=[])

    answer, sources, confidence, session_id = mock_rag_engine.answer(
        "¿Qué no está en los documentos?"
    )

    assert answer == "No pude encontrar información relevante."
    assert sources == []
    assert confidence == 0.0
    assert session_id is not None


# ---------------------------------------------------------------------------
# Session history persistence test
# ---------------------------------------------------------------------------

def test_session_history_persistence() -> None:
    """Multi-turn user/AI messages are appended to ChatSessionStore correctly."""
    from app.services.chat_history import ChatSessionStore

    store = ChatSessionStore(max_sessions=10)

    session_id = "unit-test-session"
    history = store.get_or_create(session_id)

    assert len(history.messages) == 0

    store.add_user_message(session_id, "Hola, ¿qué carreras ofrece?")
    store.add_ai_message(session_id, "Ofertamos Ingeniería y Diseño.")

    assert len(history.messages) == 2
    assert history.messages[0].content == "Hola, ¿qué carreras ofrece?"
    assert history.messages[1].content == "Ofertamos Ingeniería y Diseño."
