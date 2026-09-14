"""Fast-path evaluation module: handles conversational heuristics and deterministic data lookups."""

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

# =====================================================================
# TIER 1: HEURISTICS (Conversational Pattern Matching)
# =====================================================================

GREETING_PATTERN = re.compile(
    r"^(hola|buenas|buenos días|buenas tardes|buenas noches|hey|saludos|qué tal)[\s!.]*$",
    re.IGNORECASE | re.UNICODE,
)
THANK_PATTERN = re.compile(
    r"^(gracias|muchas gracias|mil gracias|te agradezco|vale gracias|ok gracias)[\s!.]*$",
    re.IGNORECASE | re.UNICODE,
)
FAREWELL_PATTERN = re.compile(
    r"^(adiós|hasta luego|nos vemos|hasta pronto|chao|chau)[\s!.]*$",
    re.IGNORECASE | re.UNICODE,
)

def match_heuristic(query: str) -> Optional[str]:
    """Pattern-match casual conversational inputs to bypass vector DB entirely."""
    stripped = query.strip()
    if len(stripped.split()) > 6:
        return None

    if GREETING_PATTERN.match(stripped):
        return "¡Hola! Soy tu asistente de la UNASUCRE. ¿En qué puedo ayudarte hoy?"
    if THANK_PATTERN.match(stripped):
        return "¡Con gusto! ¿Hay algo más en lo que pueda asistirte?"
    if FAREWELL_PATTERN.match(stripped):
        return "¡Que tengas un día excelente! Vuelve cuando me necesites."
    return None


def should_skip_reformulation(query: str, history_messages: list[BaseMessage]) -> bool:
    """Check if query reformulation should be skipped."""
    if not history_messages or match_heuristic(query) is not None:
        return True
    return False


# =====================================================================
# TIER 2: DETERMINISTIC (Exact Entity checks — Centro Local Sucre only)
# =====================================================================

DIRECTORY_SOURCE = "Directorio_centro_local_sucre.md"
MASTER_SOURCE = "Instructivo_general_inscripcciones_y_servicios.md"

def _normalize(text: str) -> str:
    """Strip diacritics and lowercase (Táchira/táchira -> tachira)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    ).lower()


def _strip_markdown(line: str) -> str:
    """Remove markdown syntax (*, _, `, >) and leading list bullets from a text line."""
    line = re.sub(r"[*_`>]", "", line)
    return re.sub(r"^[\s\-•]+", "", line).strip()


BANK_PATTERN = re.compile(
    _normalize(r"\b(banco|bancaria|cuenta|cuentas|arancel|pago|transferencia)\b"),
    re.IGNORECASE,
)
DIRECTORY_PATTERN = re.compile(
    _normalize(
        r"\b(coordinador|jefe|registro|secretaría|correo|contacto|"
        r"ubicación|dirección|dónde queda|dónde está|"
        r"teléfono|teléfonos|fax|código|queda)\b"
    ),
    re.IGNORECASE,
)

# Other national centers (NOT Sucre) that should trigger the redirect notice.
OTRO_CENTRO_PATTERN = re.compile(
    _normalize(
        r"\b(metropolitano|anzoátegui|apure|aragua|barinas|bolívar|"
        r"carabobo|cojedes|falcón|guárico|lara|mérida|monagas|"
        r"nueva esparta|portuguesa|táchira|trujillo|yaracuy|zulia|"
        r"delta amacuro|amazonas|caucagua|valles del tuy|vargas|"
        r"puerto cabello|punto fijo|el tigre|anaco|guasdualito|"
        r"tovar|boconó|carora|mantecal)\b"
    ),
    re.IGNORECASE,
)

# The only entities we serve, keyed by normalized name.
SUCRE_ENTITIES = {
    "sucre": "Centro Local Sucre",
    "carupano": "Unidad de Apoyo Carúpano",
    "güiria": "Unidad de Apoyo Güiria",
    "guiria": "Unidad de Apoyo Güiria",
    "cariaco": "Unidad de Apoyo Cariaco",
}


def _query_entity(query: str) -> Optional[str]:
    """Return the SUCRE_ENTITIES name explicitly asked for, if any."""
    norm_query = _normalize(query)
    
    # 1. Highest Priority: Specific Unidades de Apoyo
    for name in ("carupano", "güiria", "guiria", "cariaco"):
        if _normalize(name) in norm_query:
            return _normalize(name)
            
    # 2. Fallback Priority: Main Center (Cumaná)
    for alias in ("centro local sucre", "centro local", "sucre"):
        if alias in norm_query:
            return "sucre"
            
    return None


def _doc_entity(doc: Document) -> Optional[Tuple[str, bool]]:
    """Return (entity title, whether it is an Unidad de Apoyo) from doc headers."""
    title = doc.metadata.get("h3") or doc.metadata.get("h2")
    if not title:
        return None
    return title.strip(), "unidad de apoyo" in title.lower()


def match_other_centro(query: str) -> Optional[Tuple[str, List[str]]]:
    """Redirect requests for centers outside Centro Local Sucre."""
    if OTRO_CENTRO_PATTERN.search(_normalize(query)):
        message = (
            "Este asistente solo dispone de información del Centro Local Sucre "
            "(Cumaná) y sus Unidades de Apoyo. Para el directorio completo de "
            "centros locales de la UNA, consulta el sitio oficial: www.unasec.com."
        )
        return message, []
    return None


def match_banks(norm_query: str, docs: List[Document]) -> Tuple[Optional[str], Optional[List[str]]]:
    """Return all authorized bank accounts and payment guidelines."""
    if not BANK_PATTERN.search(norm_query):
        return None, None

    for doc in docs:
        # Select the exact "Pagos, Bancos..." section, not any chunk mentioning "banco"
        if doc.metadata.get("h2") == "2. Pagos, Bancos y Ajustes Financieros":
            lines = ["Cuentas Bancarias Autorizadas y Pagos (UNA):"]
            for line in doc.page_content.splitlines():
                clean_line = _strip_markdown(line)
                if not clean_line or clean_line.startswith("#"):
                    continue
                lines.append(f"- {clean_line}")

            return "\n".join(lines), [MASTER_SOURCE]

    return None, None


def match_directory(norm_query: str, docs: List[Document]) -> Tuple[Optional[str], Optional[List[str]]]:
    """Returns all location, telephone, and email details for the requested Sucre entity."""
    if not DIRECTORY_PATTERN.search(norm_query):
        return None, None

    target = _query_entity(norm_query) or "sucre"
    unidades = ("carupano", "güiria", "guiria", "cariaco")

    for doc in docs:
        entity = _doc_entity(doc)
        if not entity:
            continue
        title, is_unidad = entity
        norm_title = _normalize(title)

        # Match entity target to document header
        if target in unidades and _normalize(target) not in norm_title:
            continue
        if target == "sucre" and (is_unidad or "sucre" not in norm_title):
            continue

        # Extract all content lines from the matched entity chunk
        lines = [f"Información oficial de {title}:"]
        for line in doc.page_content.splitlines():
            clean_line = _strip_markdown(line)
            if clean_line.startswith("#") or not clean_line:
                continue
            lines.append(f"- {clean_line}")

        if len(lines) > 1:
            return "\n".join(lines), [DIRECTORY_SOURCE]

    return None, None


@dataclass
class DeterministicRule:
    """Binds a regex trigger to a specific DB filter and execution function."""
    pattern: re.Pattern
    db_filter: Dict[str, str]
    resolver: Callable[[str, List[Document]], Tuple[Optional[str], Optional[List[str]]]]

# The Registry: Order matters (evaluated top to bottom)
DETERMINISTIC_RULES = [
    DeterministicRule(
        pattern=BANK_PATTERN,
        db_filter={"source": MASTER_SOURCE},
        resolver=match_banks
    ),
    DeterministicRule(
        pattern=DIRECTORY_PATTERN,
        db_filter={"source": DIRECTORY_SOURCE},
        resolver=match_directory
    ),
]

def evaluate_deterministic_rules(
    query: str, retrieve_fn: Callable[[str, Dict[str, str]], List[Document]]
) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Evaluates rules and selectively queries the DB only when a pattern matches.
    Delegates the actual DB call back to the engine via retrieve_fn.
    """
    # Redirect anything that names a center outside Sucre without touching the DB.
    other_centro = match_other_centro(query)
    if other_centro:
        return other_centro

    norm_query = _normalize(query)
    for rule in DETERMINISTIC_RULES:
        if rule.pattern.search(norm_query):
            # Only fetch documents from the exact source file required by this rule
            docs = retrieve_fn(query, rule.db_filter)

            response, sources = rule.resolver(norm_query, docs)
            if response:
                return response, sources

    return None, None
