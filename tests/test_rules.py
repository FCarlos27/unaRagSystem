"""Unit tests for the Waterfall Router fast-path rules (Heuristics & Deterministic).

Scoped to Centro Local Sucre: directory chunks are plain markdown text, not JSON.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from app.services.rules import (
    DETERMINISTIC_RULES,
    DIRECTORY_SOURCE,
    MASTER_SOURCE,
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
            page_content="""## 2. Pagos, Bancos y Ajustes Financieros
- **Banco de Venezuela:** 0102-0104-7300-0032-3062
- **Banesco:** 0134-0380-5638-0100-5054""",
            metadata={"source": MASTER_SOURCE, "h2": "2. Pagos, Bancos y Ajustes Financieros"},
        ),
    ]

@pytest.fixture
def contact_docs():
    return [
        Document(
            page_content="""## Centro Local Sucre (Cumaná)
- Código: 1700
- Ubicación/Dirección: Calle Sucre, Parroquia Santa Inés. Cumaná, Estado Sucre.
- Teléfonos: (0293) 4333358, (0293) 4314890
- Fax: (0293) 4312677
- Registro y Control de Estudios: profsergiosalazar20@gmail.com
- Coordinación: angelamaiz@gmail.com
- Orientadores:
  - Elizabeth Nuez: elinuez01@gmail.com""",
            metadata={"h2": "Centro Local Sucre (Cumaná)", "source": DIRECTORY_SOURCE},
        ),
    ]

@pytest.fixture
def centro_docs():
    return [
        Document(
            page_content="""## Centro Local Sucre
Dirección: Calle Sucre, Parroquia Santa Inés, Cumaná
Teléfonos: (0293) 4333358, (0293) 4314890
Código: 1700
Fax: (0293) 4312677""",
            metadata={"h2": "Centro Local Sucre", "source": DIRECTORY_SOURCE},
        ),
        Document(
            page_content="""## Unidad de Apoyo Carúpano
Dirección: Av. Independencia, Edif. Tawil, Carúpano
Código: 1701
Fax: (0294) 3310855""",
            metadata={"h2": "Unidad de Apoyo Carúpano", "source": DIRECTORY_SOURCE},
        ),
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
    """Each rule binds the expected pattern to the correct source file."""
    assert DETERMINISTIC_RULES[0].db_filter == {"source": MASTER_SOURCE}
    assert DETERMINISTIC_RULES[1].db_filter == {"source": DIRECTORY_SOURCE}


def test_evaluate_banks(bank_docs):
    """Bank query retrieves the master guide source and formats the accounts."""
    docs_by_filter = {
        (("source", MASTER_SOURCE),): bank_docs,
    }
    retrieve_fn = _fake_retrieve(docs_by_filter)

    response, sources = evaluate_deterministic_rules(
        "¿A qué cuenta bancaria debo pagar el arancel?", retrieve_fn
    )

    assert response is not None
    assert "- Banco de Venezuela: 0102-0104-7300-0032-3062" in response
    assert "- Banesco: 0134-0380-5638-0100-5054" in response
    assert sources == [MASTER_SOURCE]


def test_evaluate_directory(contact_docs):
    """Contact query maps to the directory source and returns the relevant emails."""
    docs_by_filter = {
        (("source", DIRECTORY_SOURCE),): contact_docs,
    }
    retrieve_fn = _fake_retrieve(docs_by_filter)

    response, sources = evaluate_deterministic_rules(
        "¿Cuál es el correo del coordinador de Centro Local Sucre?", retrieve_fn
    )

    assert response is not None
    assert "Ubicación/Dirección: " in response
    assert "Código" in response
    assert "Orientadores" in response
    assert sources == [DIRECTORY_SOURCE]


def test_evaluate_directory_info_unidad(centro_docs):
    """A query mentioning 'unidad de apoyo' resolves to Carúpano, not the CL."""
    docs_by_filter = {
        (("source", DIRECTORY_SOURCE),): centro_docs,
    }
    retrieve_fn = _fake_retrieve(docs_by_filter)

    response, sources = evaluate_deterministic_rules(
        "Donde queda la unidad de apoyo carupano?", retrieve_fn
    )

    assert response is not None
    assert "Unidad de Apoyo Carúpano" in response
    assert "Código: 1701" in response
    assert sources == [DIRECTORY_SOURCE]


def test_evaluate_other_centro_redirects():
    """Asking about another center returns the Sucre-only notice without retrieval."""
    calls = []

    def retrieve_fn(query, db_filter):
        calls.append(db_filter)
        return []

    response, sources = evaluate_deterministic_rules(
        "Donde queda el centro local carabobo?", retrieve_fn
    )

    assert response is not None
    assert "www.unasec.com" in response
    assert sources == []
    assert calls == []


def test_partial_centro_names_still_redirect():
    """Accent-insensitive redirect for another center's name."""
    response, _ = evaluate_deterministic_rules(
        "Cuál es la dirección del centro de tachira?", lambda q, f: []
    )
    assert response is not None
    assert "www.unasec.com" in response


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