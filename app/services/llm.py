from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.core.config import Settings
from app.utils.logging import get_logger

logger = get_logger("llm")

SYSTEM_PROMPT = (
    "Eres un asistente de la Universidad Nacional (UNASUCRE/UNASEC). "
    "Responde en español basándote ÚNICAMENTE en el contexto proporcionado. "
    "Si el contexto no contiene la información necesaria, responde que no puedes "
    "responder con certeza y no inventes información."
)


def build_llm(settings: Settings) -> ChatNVIDIA:
    if not settings.nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY is required to build the LLM")
    return ChatNVIDIA(
        model=settings.nvidia_llm_model,
        api_key=settings.nvidia_api_key,
        temperature=0.1,
    )


def generate_answer(llm, question: str, context: str) -> str:
    messages = [
        ("system", SYSTEM_PROMPT),
        (
            "user",
            f"Contexto:\n{context}\n\nPregunta:\n{question}",
        ),
    ]
    response = llm.invoke(messages)
    return response.content
