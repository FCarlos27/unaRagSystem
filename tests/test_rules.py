"""Unit tests for the Waterfall Router fast-path rules (Heuristics & Deterministic)."""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from app.services.rules import (
    match_heuristic,
    should_skip_reformulation,
    resolve_deterministic,
)

# =====================================================================
# MOCK DATA FIXTURES
# =====================================================================

@pytest.fixture
def bank_docs():
    return [
        Document(
            page_content="Datos del banco Banesco",
            metadata={
                "entidad_bancaria": "Banesco",
                "numero_cuenta": "0134-1234-5678",
                "tipo_cuenta": "Corriente",
                "source": "bancos.json"
            }
        ),
        Document(
            page_content="Datos del banco Venezuela",
            metadata={
                "entidad_bancaria": "Banco de Venezuela",
                "numero_cuenta": "0102-9876-5432",
                "tipo_cuenta": "",  # Testing missing account type
                "source": "bancos.json"
            }
        )
    ]

@pytest.fixture
def contact_docs():
    return [
        Document(
            page_content="Directorio Sucre",
            metadata={
                "table": "contactos",
                "centro_local": "Sucre",
                "email_coordinacion": "coord.sucre@gmail.com",
                "email_registro": "registro.sucre@gmail.com",
                "source": "directorio.json"
            }
        ),
        Document(
            page_content="Directorio Metropolitano",
            metadata={
                "table": "contactos",
                "centro_local": "Metropolitano",
                "email_coordinacion": "coord.metro@gmail.com",
                "email_registro": "registro.metro@gmail.com",
                "source": "directorio.json"
            }
        )
    ]

# =====================================================================
# TIER 1: HEURISTICS TESTS
# =====================================================================

def test_match_heuristic_greetings():
    assert "¡Hola!" in match_heuristic("hola")
    assert "¡Hola!" in match_heuristic("Buenas tardes.")
    assert "¡Hola!" in match_heuristic("  QUÉ TAL  ")

def test_match_heuristic_thanks():
    assert "¡Con gusto!" in match_heuristic("muchas gracias")
    assert "¡Con gusto!" in match_heuristic("vale gracias!")

def test_match_heuristic_farewells():
    assert "excelente" in match_heuristic("hasta luego")
    assert "excelente" in match_heuristic("chao.")

def test_match_heuristic_fallthrough():
    # Real questions should return None
    assert match_heuristic("¿Cuáles son los requisitos de inscripción?") is None
    # Long conversational phrases (>6 words) should return None
    assert match_heuristic("Hola muy buenos días estimado asistente, tengo una duda") is None

def test_should_skip_reformulation():
    # Case 1: Empty history -> Skip
    assert should_skip_reformulation("¿Dónde queda?", []) is True
    
    # Case 2: History exists, but query is a heuristic match -> Skip
    history = [HumanMessage(content="hola"), AIMessage(content="¡Hola!")]
    assert should_skip_reformulation("gracias", history) is True

    # Case 3: History exists, query is a real question -> Do not skip
    assert should_skip_reformulation("¿Y cuál es el horario?", history) is False

# =====================================================================
# TIER 2: DETERMINISTIC TESTS
# =====================================================================

def test_resolve_deterministic_banks(bank_docs):
    # Tests matching bank keywords and formatting output
    query = "¿A qué cuenta bancaria debo pagar el arancel?"
    response, sources = resolve_deterministic(query, bank_docs)
    
    assert response is not None
    assert "Banesco (Corriente): 0134-1234-5678" in response
    assert "Banco de Venezuela: 0102-9876-5432" in response
    assert sources == ["bancos.json"]

def test_resolve_deterministic_contacts_coordinador(contact_docs):
    # Tests matching coordinator role and specific location
    query = "¿Cuál es el correo del coordinador metropolitano?"
    response, sources = resolve_deterministic(query, contact_docs)
    
    assert response is not None
    assert "coord.metro@gmail.com" in response
    assert "Registro y Control" not in response
    assert sources == ["directorio.json"]

def test_resolve_deterministic_contacts_registro(contact_docs):
    # Tests matching "jefe" or "registro" to pull the correct email
    query = "Necesito hablar con el jefe de registro en sucre"
    response, sources = resolve_deterministic(query, contact_docs)
    
    assert response is not None
    assert "registro.sucre@gmail.com" in response
    assert "Registro y Control de Estudios" in response
    assert sources == ["directorio.json"]

def test_resolve_deterministic_contacts_no_location(contact_docs):
    # Tests fallback when no specific center is mentioned (returns first match)
    query = "¿Cuál es el correo de control de estudios?"
    response, sources = resolve_deterministic(query, contact_docs)
    
    assert response is not None
    # Assuming the loop finds Sucre first in the list
    assert "registro.sucre@gmail.com" in response

def test_resolve_deterministic_no_match(bank_docs, contact_docs):
    # Tests that unrelated queries safely fall through to Tier 3
    query = "¿Cómo me inscribo en la universidad?"
    response, sources = resolve_deterministic(query, bank_docs + contact_docs)
    
    assert response is None
    assert sources is None