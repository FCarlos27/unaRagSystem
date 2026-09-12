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

BANK_PATTERN = re.compile(r"\b(banco|bancaria|cuenta|cuentas|arancel|pago|transferencia)\b", re.IGNORECASE)
CONTACT_PATTERN = re.compile(r"\b(coordinador|jefe|registro|secretaría|correo|contacto)\b", re.IGNORECASE)
LOCATION_PATTERN = re.compile(
    r"\b(?:ubicación|ubicacion|dirección|direccion|donde queda|dónde queda|"
    r"donde esta|dónde esta|dónde está|"
    r"teléfono|telefono|teléfonos|telefonos|fax|código|codigo|queda)\b",
    re.IGNORECASE | re.UNICODE,
)

# Other national centers (NOT Sucre) that should trigger the redirect notice.
OTRO_CENTRO_PATTERN = re.compile(
    r"\b(metropolitano|anzoategui|anzoátegui|apure|aragua|barinas|bolivar|bolívar|"
    r"carabobo|cojedes|falcon|falcón|guarico|guárico|lara|merida|mérida|monagas|"
    r"nueva esparta|portuguesa|tachira|táchira|trujillo|yaracuy|zulia|delta amacuro|"
    r"amazonas|caucagua|valles del tuy|vargas|puerto cabello|punto fijo|el tigre|"
    r"anaco|guasdualito|tovar|bocono|boconó|carora|mantecal)\b",
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


def _normalize(text: str) -> str:
    """Strip diacritics and lowercase (Táchira/táchira -> tachira)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    ).lower()


def _strip_markdown(line: str) -> str:
    """Remove markdown syntax (*, _, `, >) from a text line."""
    return re.sub(r"[*_`>]", "", line).strip()


def _query_entity(query: str) -> Optional[str]:
    """Return the SUCRE_ENTITIES name explicitly asked for, if any."""
    norm_query = _normalize(query)
    for alias in ("centro local sucre", "centro local", "sucre"):
        if alias in norm_query:
            return "sucre"
    for name in ("carupano", "güiria", "guiria", "cariaco"):
        if _normalize(name) in norm_query:
            return _normalize(name)
    return None


def _doc_entity(doc: Document) -> Optional[Tuple[str, bool]]:
    """Return (entity title, whether it is an Unidad de Apoyo) from doc headers."""
    title = doc.metadata.get("h3") or doc.metadata.get("h2")
    if not title:
        return None
    return title.strip(), "unidad de apoyo" in title.lower()


def _extract_field(text: str, label: str) -> Optional[str]:
    """Return the value for a labelled field like 'Dirección:', 'Teléfonos:'."""
    for line in text.splitlines():
        clean = _strip_markdown(line)
        match = re.search(rf"^{label}\s*:\s*(.+)$", clean, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def match_other_centro(query: str) -> Optional[Tuple[str, List[str]]]:
    """Redirect requests for centers outside Centro Local Sucre."""
    if OTRO_CENTRO_PATTERN.search(query):
        message = (
            "Este asistente solo dispone de información del Centro Local Sucre "
            "(Cumaná) y sus Unidades de Apoyo. Para el directorio completo de "
            "centros locales de la UNA, consulta el sitio oficial: www.unasec.com."
        )
        return message, []
    return None


def match_banks(query: str, docs: List[Document]) -> Tuple[Optional[str], Optional[List[str]]]:
    """Return the authorized bank accounts read from the master guide text."""
    if not BANK_PATTERN.search(query):
        return None, None

    accounts = []
    account_pat = re.compile(r"\b(\d{4}-\d{4}-\d{4}-\d{4}-\d{4})\b")
    for doc in docs:
        for line in doc.page_content.splitlines():
            number_match = account_pat.search(line)
            if not number_match:
                continue
            number = number_match.group(1)
            bank = _strip_markdown(line.split(number_match.group(0))[0])
            bank = re.sub(r"^[\s\-•*]+|[:–-]+$", "", bank).strip()
            if bank and (bank, number) not in accounts:
                accounts.append((bank, number))

    if not accounts:
        return None, None

    lines = [f"- {bank}: `{number}`" for bank, number in accounts]
    response = "Las cuentas bancarias autorizadas para pagar aranceles son:\n" + "\n".join(lines)
    return response, [MASTER_SOURCE]


def match_contacts(query: str, docs: List[Document]) -> Tuple[Optional[str], Optional[List[str]]]:
    """Return the Registro or Coordinación email for Centro Local Sucre."""
    if not CONTACT_PATTERN.search(query):
        return None, None

    is_registro = any(term in query.lower() for term in ["registro", "jefe", "control"])
    label = "Registro y Control de Estudios" if is_registro else "Coordinación"

    for doc in docs:
        for line in doc.page_content.splitlines():
            clean = _strip_markdown(line).lower()
            if is_registro and "registro" not in clean:
                continue
            if not is_registro and "coordinaci" not in clean:
                continue
            email_match = re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", clean)
            if email_match:
                response = (
                    f"El correo electrónico para la oficina de **{label}** "
                    f"del Centro Local Sucre es: `{email_match.group(0)}`."
                )
                return response, [DIRECTORY_SOURCE]

    return None, None


def match_directory_info(query: str, docs: List[Document]) -> Tuple[Optional[str], Optional[List[str]]]:
    """Return physical address, phone numbers, or codes for Sucre entities."""
    if not LOCATION_PATTERN.search(query):
        return None, None

    wants_unidad = re.search(r"unidad de apoyo|apoyo", query, re.IGNORECASE) is not None
    target = _query_entity(query)

    for doc in docs:
        entity = _doc_entity(doc)
        if not entity:
            continue
        title, is_unidad = entity

        if wants_unidad and not is_unidad:
            continue
        if not wants_unidad and is_unidad and target not in ("carupano", "güiria", "guiria", "cariaco"):
            continue

        direccion = _extract_field(doc.page_content, "Dirección")
        telefonos = _extract_field(doc.page_content, "Teléfonos")
        codigo = _extract_field(doc.page_content, "Código")
        fax = _extract_field(doc.page_content, "Fax")

        lines = [f"**{title}**" + (f" (Código: `{codigo}`)" if codigo else "")]
        if direccion:
            lines.append(f"- **Dirección:** {direccion}")
        if telefonos:
            lines.append(f"- **Teléfonos:** {telefonos}")
        if fax:
            lines.append(f"- **Fax:** {fax}")

        if len(lines) > 1:
            response = "Información de contacto oficial:\n" + "\n".join(lines)
            return response, [DIRECTORY_SOURCE]

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
        pattern=LOCATION_PATTERN,
        db_filter={"source": DIRECTORY_SOURCE},
        resolver=match_directory_info
    ),
    DeterministicRule(
        pattern=BANK_PATTERN,
        db_filter={"source": MASTER_SOURCE},
        resolver=match_banks
    ),
    DeterministicRule(
        pattern=CONTACT_PATTERN,
        db_filter={"source": DIRECTORY_SOURCE},
        resolver=match_contacts
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

    for rule in DETERMINISTIC_RULES:
        if rule.pattern.search(query):
            # Only fetch documents from the exact source file required by this rule
            docs = retrieve_fn(query, rule.db_filter)

            response, sources = rule.resolver(query, docs)
            if response:
                return response, sources

    return None, None
