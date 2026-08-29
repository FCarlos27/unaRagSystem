from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.core.config import Settings
from app.utils.logging import get_logger

logger = get_logger("llm")

SYSTEM_PROMPT = ("""Eres un asistente de IA preciso y útil para el Centro Local Sucre (UNASUCRE) de la Universidad Nacional Abierta 
Responde a la pregunta del usuario utilizando ÚNICAMENTE el contexto proporcionado.

Reglas de respuesta:
1. Si la respuesta se encuentra en tablas, listas o directorios de contacto (como correos, coordinadores o jefes de departamento), extrae y presenta la información de forma explícita sin omitir detalles.
2. Si la información no está presente en el contexto, indica amablemente que no dispones de esos datos.
3. Sé conciso, directo y mantén un tono profesional.""")

REFORMULATE_PROMPT = (
    "Dada una conversación anterior y una pregunta actual del usuario, "
    "reformula la pregunta actual para que sea una consulta autónoma e independiente "
    "en español, resolviendo todos los pronombres y referencias implícitas. "
    "NO respondas a la pregunta; simplemente reformula la consulta si es necesario. "
    "Responde ÚNICAMENTE con la pregunta reformulada, sin explicaciones ni texto adicional."
)


def build_llm(settings: Settings) -> ChatOllama:
    """Create the Ollama chat model configured from settings."""
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_llm_model,
        temperature=0.1,
    )

def build_embeddings(settings: Settings) -> OllamaEmbeddings:
    """Create the Ollama embedding model configured from settings."""
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
    )


def generate_answer(llm, question: str, context: str, history: str = "") -> str:
    """Generate an answer grounded strictly on the retrieved context."""
    user_content_parts = []
    
    if history:
        user_content_parts.append(f"Conversación anterior:\n{history}")
        
    user_content_parts.append(f"Contexto:\n{context}")
    user_content_parts.append(f"Pregunta:\n{question}")
    
    user_message = "\n\n".join(user_content_parts)
    
    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", user_message),
    ]
    
    response = llm.invoke(messages)
    return response.content


def reformulate_query(llm, question: str, history: str) -> str:
    """Reframe a follow-up question into a standalone query using chat history."""
    if not history or not history.strip():
        return question.strip()
    
    messages = [
        ("system", REFORMULATE_PROMPT),
        (
            "user",
            f"Conversación anterior:\n{history}\n\n"
            f"Pregunta actual:\n{question}\n\n"
            "Pregunta reformulada:",
        ),
    ]
    
    try:
        response = llm.invoke(messages)
        standalone_query = response.content.strip()
        logger.info("Reformulated query: '%s' -> '%s'", question, standalone_query)
        return standalone_query
    except Exception as e:
        logger.warning("Query reformulation failed (%s); falling back to raw question", e)
        return question.strip()
