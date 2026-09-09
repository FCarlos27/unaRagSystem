"""Fast-path evaluation module: handles conversational heuristics and deterministic data lookups."""

import re
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
# TIER 2: DETERMINISTIC (Exact Entity & JSON Metadata Lookups)
# =====================================================================

BANK_PATTERN = re.compile(r"\b(banco|bancaria|cuenta|cuentas|arancel|pago|transferencia)\b", re.IGNORECASE)
CONTACT_PATTERN = re.compile(r"\b(coordinador|jefe|registro|secretaría|correo|contacto)\b", re.IGNORECASE)
CENTRO_REGEX = re.compile(r"\b(sucre|carabobo|metropolitano|nueva esparta|anzoátegui|táchira|mérida|zulia)\b", re.IGNORECASE)
LOCATION_PATTERN = re.compile(
    r"\b(?:ubicación|ubicacion|dirección|direccion|donde queda|dónde queda|"
    r"donde esta|dónde esta|dónde está|"
    r"teléfono|telefono|teléfonos|telefonos|fax|código|codigo|queda)\b",
    re.IGNORECASE | re.UNICODE,
)

def _extract_centro_from_query(query: str) -> Optional[str]:
    """Return the centro local mentioned in the query, if any."""
    match = CENTRO_REGEX.search(query)
    return match.group(1).lower() if match else None


def _doc_matches_centro(metadata, target_centro: Optional[str], key: str) -> Optional[str]:
    """Return centro name if present and (optionally) matching the query target."""
    centro_name = metadata.get(key)
    if not centro_name:
        return None
    if target_centro and target_centro not in centro_name.lower():
        return None
    return centro_name


def _matches_query_name(query: str, nombre: Optional[str]) -> bool:
    """True if the doc's name appears in the query (case-insensitive)."""
    return bool(nombre) and re.search(re.escape(nombre), query, re.IGNORECASE) is not None


def match_banks(query: str, docs: List[Document]) -> Tuple[Optional[str], Optional[List[str]]]:
    """Return a formatted banking response and specific source if matched."""
    if not BANK_PATTERN.search(query):
        return None, None

    lines = []
    sources = set()
    
    for doc in docs:
        if doc.metadata.get("table") != "contactos" and doc.metadata.get("entidad_bancaria"):
            bank = doc.metadata.get("entidad_bancaria")
            account_type = doc.metadata.get("tipo_cuenta", "")
            account_number = doc.metadata.get("numero_cuenta", "")
            
            if bank and account_number:
                label = f"{bank} ({account_type})" if account_type else bank
                lines.append(f"- {label}: {account_number}")
                sources.add(doc.metadata.get("source", "unknown"))

    if not lines:
        return None, None
        
    response = "Las cuentas bancarias autorizadas para pagar aranceles son:\n" + "\n".join(lines)
    return response, sorted(list(sources))

def match_contacts(query: str, docs: List[Document]) -> Tuple[Optional[str], Optional[List[str]]]:
    """Return a formatted contact response directly from document metadata."""
    if not CONTACT_PATTERN.search(query):
        return None, None

    # Extract target location from query (e.g., "cumana" -> "sucre", or explicit match)
    target_centro = _extract_centro_from_query(query)

    for doc in docs:
        metadata = doc.metadata

        # Check if document matches the contact directory schema
        centro_name = _doc_matches_centro(metadata, target_centro, "centro_local")
        if not centro_name:
            continue

        # Determine if the query targets Registro vs Coordinación
        is_registro = any(term in query.lower() for term in ["registro", "jefe", "control"])
        
        if is_registro:
            email = metadata.get("email_registro")
            role_title = "Registro y Control de Estudios"
        else:
            email = metadata.get("email_coordinacion")
            role_title = "Coordinación"

        if email:
            response = (
                f"El correo electrónico para la oficina de **{role_title}** "
                f"del Centro Local {centro_name} es: `{email}`."
            )
            source = metadata.get("source", "directorio_contactos.json")
            return response, [source]

    return None, None

def match_directory_info(query: str, docs: List[Document]) -> Tuple[Optional[str], Optional[List[str]]]:
    """Return physical address, phone numbers, or codes for a Centro Local or Unidad
    de Apoyo directly from metadata."""
    if not LOCATION_PATTERN.search(query):
        return None, None

    wants_unidad = re.search(r"unidad de apoyo|apoyo", query, re.IGNORECASE) is not None

    for doc in docs:
        metadata = doc.metadata
        tipo = (metadata.get("tipo") or "CENTRO LOCAL").upper()
        centro_name = metadata.get("nombre") or metadata.get("centro_local")

        if not centro_name:
            continue

        # Prefer the entity type the query asks for, but fall back to any type
        if wants_unidad:
            if tipo != "UNIDAD DE APOYO":
                continue
        elif tipo == "UNIDAD DE APOYO":
            # A 'centro local' query should not match an unidad de apoyo unless the
            # name is explicitly asked (e.g. 'donde queda caucagua?').
            match = _extract_centro_from_query(query)
            if match and not _matches_query_name(query, centro_name):
                continue
        elif not _matches_query_name(query, centro_name):
            continue

        # Extract fields matching your JSON schema
        direccion = metadata.get("direccion")
        telefonos = metadata.get("telefonos")
        codigo = metadata.get("codigo")
        fax = metadata.get("fax")

        label = f"**Unidad de Apoyo {centro_name}**" if tipo == "UNIDAD DE APOYO" else f"**Centro Local {centro_name}**"
        lines = [f"{label} (Código: `{codigo}`)" if codigo else label]

        if direccion:
            lines.append(f"- **Dirección:** {direccion}")

        if telefonos:
            phone_str = ", ".join(telefonos) if isinstance(telefonos, list) else str(telefonos)
            lines.append(f"- **Teléfonos:** {phone_str}")

        if fax:
            lines.append(f"- **Fax:** {fax}")

        if len(lines) > 1:
            source = metadata.get("source", "directorio_oficial.json")
            response = "Información de contacto oficial:\n" + "\n".join(lines)
            return response, [source]

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
        db_filter={"source": "Directorio_centros_locales.json"},
        resolver=match_directory_info
    ),
    DeterministicRule(
        pattern=BANK_PATTERN,
        db_filter={"source": "Bancos_autorizados.json"},
        resolver=match_banks
    ),
    DeterministicRule(
        pattern=CONTACT_PATTERN,
        db_filter={"source": "directorio_registro_y_coordinacion.json"},
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
    for rule in DETERMINISTIC_RULES:
        if rule.pattern.search(query):
            # Only fetch documents from the exact source file required by this rule
            docs = retrieve_fn(query, rule.db_filter)

            response, sources = rule.resolver(query, docs)
            if response:
                return response, sources

    return None, None
