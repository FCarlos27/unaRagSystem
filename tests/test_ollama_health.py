"""Unit tests for the Ollama startup health gate."""

import json
import urllib.error

import pytest

from app.services import ollama_health
from app.services.ollama_health import verify_ollama_ready


class _FakeSettings:
    def __init__(self, base_url="http://localhost:11434", embed="nomic-embed-text", llm="llama3.2:3b"):
        self.ollama_base_url = base_url
        self.ollama_embed_model = embed
        self.ollama_llm_model = llm


class _FakeResponse:
    def __init__(self, payload):
        self.status = 200
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _mock_urlopen(monkeypatch, endpoints):
    """Serve payloads by endpoint path; raise URLError for anything else."""
    def fake(path, timeout=None):
        if path in endpoints:
            return _FakeResponse(endpoints[path])
        raise urllib.error.URLError("unreachable")

    monkeypatch.setattr(ollama_health.urllib.request, "urlopen", fake)


def test_verify_passes_when_server_and_models_ready(monkeypatch):
    _mock_urlopen(
        monkeypatch,
        {
            "http://localhost:11434/api/version": {"version": "0.5.1"},
            "http://localhost:11434/api/tags": {
                "models": [
                    {"name": "nomic-embed-text:latest"},
                    {"name": "llama3.2:3b"},
                ]
            },
        },
    )
    verify_ollama_ready(_FakeSettings())  # must not raise


def test_verify_exits_when_model_missing(monkeypatch):
    _mock_urlopen(
        monkeypatch,
        {
            "http://localhost:11434/api/version": {"version": "0.5.1"},
            "http://localhost:11434/api/tags": {
                "models": [{"name": "nomic-embed-text:latest"}]
            },
        },
    )
    with pytest.raises(SystemExit) as exc:
        verify_ollama_ready(_FakeSettings())
    assert "not installed" in str(exc.value)
    assert "ollama pull llama3.2:3b" in str(exc.value)
    assert "ollama pull nomic-embed-text" not in str(exc.value)


def test_verify_exits_when_server_unreachable(monkeypatch):
    _mock_urlopen(monkeypatch, {})
    with pytest.raises(SystemExit) as exc:
        verify_ollama_ready(_FakeSettings())
    assert "Ollama is not available" in str(exc.value)
    assert "ollama serve" in str(exc.value)


def test_verify_follows_configured_models(monkeypatch):
    _mock_urlopen(
        monkeypatch,
        {
            "http://localhost:11434/api/version": {"version": "0.5.1"},
            "http://localhost:11434/api/tags": {
                "models": [
                    {"name": "nomic-embed-text:latest"},
                    {"name": "llama3.2:3b"},
                ]
            },
        },
    )
    settings = _FakeSettings(llm="llama3.1:8b")
    with pytest.raises(SystemExit) as exc:
        verify_ollama_ready(settings)
    assert "llama3.1:8b" in str(exc.value)
    assert "ollama pull llama3.1:8b" in str(exc.value)


def test_ollama_running_false_on_network_error(monkeypatch):
    _mock_urlopen(monkeypatch, {})
    assert ollama_health.ollama_running("http://localhost:11434") is False
    assert ollama_health.installed_models("http://localhost:11434") == []