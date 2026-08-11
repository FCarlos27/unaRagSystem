import re

from app.utils.logging import get_logger

logger = get_logger("table_extractor")

CENTRO_LOCALES = [
    "NUEVA ESPARTA",
    "ANZOÁTEGUI",
    "ANZOATEGUI",
    "PORTUGUESA",
    "METROPOLITANO",
    "APURE",
    "ARAGUA",
    "BARINAS",
    "BOLÍVAR",
    "BOLIVAR",
    "CARABOBO",
    "COJEDES",
    "FALCÓN",
    "FALCON",
    "GUÁRICO",
    "GUARICO",
    "LARA",
    "MÉRIDA",
    "MERIDA",
    "MONAGAS",
    "SUCRE",
    "TÁCHIRA",
    "TACHIRA",
    "TRUJILLO",
    "YARACUY",
    "ZULIA",
]

TOKEN = re.compile(r"\S+", re.UNICODE)
SEPARATORS = {"|", "—", "-"}

EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+(?:\s*[Q0O@])?\s*(?:gmail|hotmail|yahoo)\.com"
    r"|[A-Za-z0-9._%+\-]*(?:mail|hotmail|una\.edu)[A-Za-z0-9._%+\-]*",
    re.I,
)


def _has_email(token: str) -> bool:
    """Return True if a token looks like an email fragment."""
    low = token.lower()
    return any(k in low for k in ("mail", "hotmail", "una.edu", ".com"))


def _find_centro(line: str) -> str | None:
    """Return the centro local name present in a line, if any."""
    upper = line.upper()
    for name in CENTRO_LOCALES:
        if name in upper:
            return name
    return None


def _row_tokens(line: str) -> list[str]:
    """Split a table row into meaningful tokens, dropping separator glyphs."""
    return [t for t in TOKEN.findall(line) if t not in SEPARATORS]


def _parse_row(line: str, centro: str) -> dict | None:
    """Parse a contact row into centro, jefe, and coordinator entries."""
    emails = [m.strip() for m in EMAIL_RE.findall(line)]
    if not emails:
        return None

    coordinator = emails[-1]
    coordinator_tokens = set(_row_tokens(coordinator))
    name_parts = centro.upper().split()
    rest = [
        t
        for t in _row_tokens(line)
        if t.upper() not in name_parts and t not in coordinator_tokens
    ]
    jefe = " ".join(rest) or None
    return {"centro": centro, "jefe": jefe, "coordinador": coordinator}


def extract_contact_table_entries(text: str, source: str) -> list[dict]:
    """Rebuild clean contact-table chunks from the OCR'd Secretaría table."""
    entries = []
    upper = text.upper()
    marker = upper.find("CORREOS ELECTRONICOS")
    if marker == -1:
        return entries
    header = upper.find("COORDINADOR", marker)
    if header == -1:
        return entries

    seen: set[str] = set()
    for line in text[header:header + 6000].splitlines():
        if len(line.strip()) < 4:
            continue
        centro = _find_centro(line)
        if not centro or centro in seen:
            continue
        row = _parse_row(line, centro)
        if not row:
            continue
        seen.add(centro)

        content = (
            f"Centro Local: {centro}.\n"
            f"Jefe de Registro y Control de Estudios: {row['jefe']}.\n"
            f"Coordinador(a): {row['coordinador']}.\n"
            "Fuente: tabla de correos electrónicos de la Secretaría, "
            "Dirección de Registro y Control de Estudios."
        )
        entries.append(
            {
                "page_content": content,
                "metadata": {"source": source, "table": "contactos"},
            }
        )
        logger.info("Extracted contact entry for %s", centro)

    return entries
