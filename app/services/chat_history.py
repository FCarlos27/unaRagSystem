import threading
import uuid

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from app.utils.logging import get_logger

logger = get_logger("chat_history")

MAX_SESSIONS = 100


class ChatSessionStore:
    """In-memory store of per-session chat histories with LRU-style eviction."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._sessions: dict[str, InMemoryChatMessageHistory] = {}
        self._lock = threading.Lock()
        self._max_sessions = max_sessions

    def get_or_create(self, session_id: str) -> InMemoryChatMessageHistory:
        """Return a session's history, creating it (and evicting the oldest) if needed."""
        with self._lock:
            if session_id not in self._sessions:
                if len(self._sessions) >= self._max_sessions:
                    oldest = next(iter(self._sessions))
                    self._sessions.pop(oldest)
                    logger.info("Evicted oldest session %s", oldest)
                self._sessions[session_id] = InMemoryChatMessageHistory()
            return self._sessions[session_id]

    def add_user_message(self, session_id: str, content: str) -> None:
        """Append a user message to the session."""
        self.get_or_create(session_id).add_user_message(content)

    def add_ai_message(self, session_id: str, content: str) -> None:
        """Append an assistant message to the session."""
        self.get_or_create(session_id).add_ai_message(content)


chat_sessions = ChatSessionStore()


def new_session_id() -> str:
    """Generate a fresh unique session identifier."""
    return uuid.uuid4().hex


def format_history(history: InMemoryChatMessageHistory) -> str:
    """Render a chat history as plain text for the LLM prompt."""
    lines = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"Usuario: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Asistente: {msg.content}")
    return "\n".join(lines)
