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
from app.utils.unanswered_log import ensure_log_dir, log_unanswered

logger = get_logger("rag_engine")


def _declined_answer(answer_text: str) -> bool:
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


def _contact_score(value: str) -> tuple[int, int, int]:
    has_domain = 0 if any(marker in value for marker in ("@", ".com", ".org", ".edu", ".net")) else 1
    return (has_domain, value.count(" "), -len(value))


ROLE_KEYWORDS = ("coordinador", "jefe de registro", "registro y control")
CENTRO_ALIASES = {
    "sucre": "Sucre",
    "carabobo": "Carabobo",
    "metropolitano": "Metropolitano",
    "nueva esparta": "Nueva Esparta",
    "anzoategui": "Anzoátegui",
    "apure": "Apure",
    "aragua": "Aragua",
    "barinas": "Barinas",
    "bolivar": "Bolívar",
    "cojedes": "Cojedes",
    "falcon": "Falcón",
    "guarico": "Guárico",
    "lara": "Lara",
    "merida": "Mérida",
    "monagas": "Monagas",
    "portuguesa": "Portuguesa",
    "tachira": "Táchira",
    "trujillo": "Trujillo",
    "yaracuy": "Yaracuy",
    "zulia": "Zulia",
}


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

    def _contact_table(self) -> dict[str, dict[str, str]]:
        docs = self.vector_store.similarity_search(
            "coordinador centro local correo",
            k=1000,
            filter={"table": "contactos"},
        )
        table: dict[str, dict[str, str]] = {}
        for doc in docs:
            lines = doc.page_content.splitlines()
            centro_entry = next(
                (ln.split(":", 1)[1].strip().rstrip(".") for ln in lines
                 if ln.lower().startswith("centro local:")),
                None,
            )
            if not centro_entry:
                continue
            row = table.setdefault(centro_entry.lower(), {})
            for label in ("Coordinador(a)", "Jefe de Registro y Control de Estudios"):
                value = next(
                    (ln.split(":", 1)[1].strip().rstrip(".") for ln in lines
                     if ln.startswith(label)),
                    None,
                )
                if value and (
                    label not in row
                    or _contact_score(value) < _contact_score(row[label])
                ):
                    row[label] = value
        return table

    def _direct_contact_answer(self, query: str) -> str | None:
        low = query.lower()
        is_role_query = any(k in low for k in ROLE_KEYWORDS)
        if not is_role_query:
            return None
        centro_asked = next(
            (label for alias, label in CENTRO_ALIASES.items() if alias in low),
            None,
        )
        if not centro_asked:
            return None

        want_jefe = "jefe" in low
        label = "Jefe de Registro y Control de Estudios" if want_jefe else "Coordinador(a)"
        row = self._contact_table().get(centro_asked.lower())
        value = row.get(label) if row else None
        if value:
            return (
                f"Según la tabla de contactos de la Secretaría, el {label} "
                f"del Centro Local {centro_asked} es {value}. "
                "El documento indica el correo de contacto, pero no el nombre."
            )
        return None

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
            self._log_unanswered(query, session_id, reason="no_context")
        else:
            sources = sorted({doc.metadata.get("source", "desconocido") for doc in docs})
            direct = self._direct_contact_answer(query)
            if direct:
                answer_text = direct
                confidence = 0.9
            else:
                context = "\n\n".join(doc.page_content for doc in docs)
                answer_text = generate_answer(self.llm, query, context, history_text)
                confidence = 1.0
                if _declined_answer(answer_text):
                    self._log_unanswered(
                        query, session_id, sources=sources, reason="declined_with_context"
                    )

        chat_sessions.add_user_message(session_id, query)
        chat_sessions.add_ai_message(session_id, answer_text)
        return answer_text, sources, confidence, session_id

    def _log_unanswered(
        self,
        query: str,
        session_id: str,
        sources: list[str] | None = None,
        reason: str = "no_context",
    ) -> None:
        path = self.settings.unanswered_log_path
        if not path:
            return
        ensure_log_dir(path)
        log_unanswered(path, query, session_id, sources, reason)


@lru_cache
def get_rag_engine() -> RagEngine:
    settings = get_settings()
    embeddings = build_embeddings(settings)
    llm = build_llm(settings)
    return RagEngine(settings, embeddings, llm)
