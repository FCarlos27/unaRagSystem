"""Unit tests for the Waterfall Router fast-path rules (Heuristics & Deterministic)."""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from app.services.rules import (
    DETERMINISTIC_RULES,
    evaluate_deterministic_rules,
    match_heuristic,
    should_skip_reformulation,
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
                "source": "Bancos_autorizados.json",
            }
        ),
        Document(
            page_content="Datos del banco Venezuela",
            metadata={
                "entidad_bancaria": "Banco de Venezuela",
                "numero_cuenta": "0102-9876-5432",
                "tipo_cuenta": "",  # Testing missing account type
                "source": "Bancos_autorizados.json",
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
                "source": "directorio_registro_y_coordinacion.json",
            }
        ),
        Document(
            page_content="Directorio Metropolitano",
            metadata={
                "table": "contactos",
                "centro_local": "Metropolitano",
                "email_coordinacion": "coord.metro@gmail.com",
                "email_registro": "registro.metro@gmail.com",
                "source": "directorio_registro_y_coordinacion.json",
            }
        )
    ]

@pytest.fixture
def centro_docs():
    return [
        Document(
            page_content="Centro Local Sucre",
            metadata={
                "nombre": "Sucre",
                "direccion": "Calle Sucre, Parroquia Santa Inés",
                "telefonos": "(0293) 4333358",
                "codigo": "1700",
                "fax": "(0293) 4312677",
                "source": "Directorio_centros_locales.json",
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
# TIER 2: DETERMINISTIC REGISTRY TESTS
# =====================================================================

def _fake_retrieve(docs_by_filter):
    """Build a retrieve_fn that returns the docs registered for each db_filter."""
    def retrieve_fn(query, db_filter):
        return docs_by_filter.get(tuple(sorted(db_filter.items())), [])
    return retrieve_fn


def test_registry_maps_patterns_to_source_filters():
    """Each rule binds the expected pattern to the correct source filter."""
    assert DETERMINISTIC_RULES[0].db_filter == {"source": "Directorio_centros_locales.json"}
    assert DETERMINISTIC_RULES[1].db_filter == {"source": "Bancos_autorizados.json"}
    assert DETERMINISTIC_RULES[2].db_filter == {"source": "directorio_registro_y_coordinacion.json"}


def test_evaluate_banks(bank_docs):
    """Bank query retrieves the bancos source and formats the accounts."""
    docs_by_filter = {
        (("source", "Bancos_autorizados.json"),): bank_docs,
    }
    retrieve_fn = _fake_retrieve(docs_by_filter)

    response, sources = evaluate_deterministic_rules(
        "¿A qué cuenta bancaria debo pagar el arancel?", retrieve_fn
    )

    assert response is not None
    assert "Banesco (Corriente): 0134-1234-5678" in response
    assert "Banco de Venezuela: 0102-9876-5432" in response
    assert sources == ["Bancos_autorizados.json"]


def test_evaluate_contacts_coordinador(contact_docs):
    """Contact query maps to the coordinator email for the specific location."""
    docs_by_filter = {
        (("source", "directorio_registro_y_coordinacion.json"),): contact_docs,
    }
    retrieve_fn = _fake_retrieve(docs_by_filter)

    response, sources = evaluate_deterministic_rules(
        "¿Cuál es el correo del coordinador metropolitano?", retrieve_fn
    )

    assert response is not None
    assert "coord.metro@gmail.com" in response
    assert "Registro y Control" not in response
    assert sources == ["directorio_registro_y_coordinacion.json"]


def test_evaluate_contacts_registro(contact_docs):
    """'jefe'/'registro' keywords pull the registro email for the right center."""
    docs_by_filter = {
        (("source", "directorio_registro_y_coordinacion.json"),): contact_docs,
    }
    retrieve_fn = _fake_retrieve(docs_by_filter)

    response, sources = evaluate_deterministic_rules(
        "Necesito hablar con el jefe de registro en sucre", retrieve_fn
    )

    assert response is not None
    assert "registro.sucre@gmail.com" in response
    assert "Registro y Control de Estudios" in response
    assert sources == ["directorio_registro_y_coordinacion.json"]


def test_evaluate_contacts_no_location(contact_docs):
    """Without a location in the query, the first matching contact is returned."""
    docs_by_filter = {
        (("source", "directorio_registro_y_coordinacion.json"),): contact_docs,
    }
    retrieve_fn = _fake_retrieve(docs_by_filter)

    response, sources = evaluate_deterministic_rules(
        "¿Cuál es el correo de control de estudios?", retrieve_fn
    )

    assert response is not None
    # Assuming the loop finds Sucre first in the list
    assert "registro.sucre@gmail.com" in response


def test_evaluate_directory_info(centro_docs):
    """Location query retrieves the centros locales source and returns the address."""
    docs_by_filter = {
        (("source", "Directorio_centros_locales.json"),): centro_docs,
    }
    retrieve_fn = _fake_retrieve(docs_by_filter)

    response, sources = evaluate_deterministic_rules(
        "Donde queda el centro local sucre?", retrieve_fn
    )

    assert response is not None
    assert "Calle Sucre" in response
    assert "Código: `1700`" in response
    assert sources == ["Directorio_centros_locales.json"]


def test_evaluate_no_match_skips_retrieval():
    """Unrelated queries never call retrieve_fn and fall through to Tier 3."""
    calls = []

    def retrieve_fn(query, db_filter):
        calls.append(db_filter)
        return []

    response, sources = evaluate_deterministic_rules(
        "¿Cómo me inscribo en la universidad?", retrieve_fn
    )

    assert response is None
    assert sources is None
    assert calls == []


def test_evaluate_empty_docs_falls_through():
    """A matching rule with no filtered docs returns None (falls to Tier 3)."""
    retrieve_fn = _fake_retrieve({})

    response, sources = evaluate_deterministic_rules(
        "Donde queda el centro local sucre?", retrieve_fn
    )

    assert response is None
    assert sources is None