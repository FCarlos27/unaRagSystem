from langchain_ollama import ChatOllama

from app.core.config import Settings
from app.utils.logging import get_logger

logger = get_logger("llm")

SYSTEM_PROMPT = (
    "Eres un asistente de la Universidad Nacional (UNASUCRE/UNASEC). "
    "Responde en español basándote ÚNICAMENTE en el contexto proporcionado. "
    "Si el contexto no contiene la información necesaria, responde que no puedes "
    "responder con certeza y no inventes información."
)

REFORMULATE_PROMPT = (
    "Convierte la pregunta actual en una pregunta autónoma e independiente, "
    "resolviendo pronombres y referencias ambiguas usando la conversación anterior. "
    "Responde solo con la pregunta reformulada, sin texto adicional."
)


def build_llm(settings: Settings) -> ChatOllama:
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_llm_model,
        temperature=0.1,
    )


def generate_answer(llm, question: str, context: str, history: str = "") -> str:
    preamble = f"Conversación anterior:\n{history}\n\n" if history else ""
    messages = [
        ("system", SYSTEM_PROMPT),
        (
            "user",
            f"{preamble}Contexto:\n{context}\n\nPregunta:\n{question}",
        ),
    ]
    response = llm.invoke(messages)
    return response.content


def reformulate_query(llm, question: str, history: str) -> str:
    messages = [
        ("system", REFORMULATE_PROMPT),
        (
            "user",
            f"Conversación anterior:\n{history}\n\nPregunta actual:\n{question}\n\n"
            "Pregunta reformulada:",
        ),
    ]
    response = llm.invoke(messages)
    return response.content.strip()
