from langchain_ollama import ChatOllama

from app.core.config import Settings
from app.utils.logging import get_logger

logger = get_logger("llm")

SYSTEM_PROMPT = (
    "Eres un asistente de la Universidad Nacional (UNASUCRE/UNASEC). "
    "Responde en español basándote ÚNICAMENTE en el contexto proporcionado. "
    "Si el contexto no contiene la información necesaria, responde que no puedes "
    "responder con certeza y no inventes información. "
    "Si el contexto incluye una tabla de contactos con una columna 'Coordinador(a)' "
    "o 'Jefe de Registro y Control de Estudios', reporta el valor de esa columna "
    "para el Centro Local consultado tal como aparece, aunque sea solo un correo "
    "o parezca un buzón genérico; aclara que no se indica el nombre si ese es el caso."
)

REFORMULATE_PROMPT = (
    "Convierte la pregunta actual en una pregunta autónoma e independiente, "
    "resolviendo pronombres y referencias ambiguas usando la conversación anterior. "
    "Responde solo con la pregunta reformulada, sin texto adicional."
)


def build_llm(settings: Settings) -> ChatOllama:
    """Create the Ollama chat model configured from settings."""
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_llm_model,
        temperature=0.1,
    )


def generate_answer(llm, question: str, context: str, history: str = "") -> str:
    """Generate an answer grounded strictly on the retrieved context."""
    preamble = f"Conversación anterior:\n{history}\n\n" if history else ""
    instruction = (
        "Instrucción: si el contexto contiene una tabla de contactos con columnas "
        "'Coordinador(a)' o 'Jefe de Registro y Control de Estudios' para el Centro "
        "Local consultado, indica el valor de la columna relevante tal como aparece "
        "(puede ser un correo sin nombre). No lo omitas."
    )
    messages = [
        ("system", SYSTEM_PROMPT),
        (
            "user",
            f"{preamble}{instruction}\n\nContexto:\n{context}\n\nPregunta:\n{question}",
        ),
    ]
    response = llm.invoke(messages)
    return response.content


def reformulate_query(llm, question: str, history: str) -> str:
    """Reframe a follow-up question into a standalone query using chat history."""
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
