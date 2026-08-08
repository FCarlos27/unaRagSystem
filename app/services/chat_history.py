import threading
import uuid

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from app.utils.logging import get_logger

logger = get_logger("chat_history")

MAX_SESSIONS = 100


class ChatSessionStore:
    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._sessions: dict[str, InMemoryChatMessageHistory] = {}
        self._lock = threading.Lock()
        self._max_sessions = max_sessions

    def get_or_create(self, session_id: str) -> InMemoryChatMessageHistory:
        with self._lock:
            if session_id not in self._sessions:
                if len(self._sessions) >= self._max_sessions:
                    oldest = next(iter(self._sessions))
                    self._sessions.pop(oldest)
                    logger.info("Evicted oldest session %s", oldest)
                self._sessions[session_id] = InMemoryChatMessageHistory()
            return self._sessions[session_id]

    def add_user_message(self, session_id: str, content: str) -> None:
        self.get_or_create(session_id).add_user_message(content)

    def add_ai_message(self, session_id: str, content: str) -> None:
        self.get_or_create(session_id).add_ai_message(content)


chat_sessions = ChatSessionStore()


def new_session_id() -> str:
    return uuid.uuid4().hex


def format_history(history: InMemoryChatMessageHistory) -> str:
    lines = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"Usuario: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Asistente: {msg.content}")
    return "\n".join(lines)
