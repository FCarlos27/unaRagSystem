"""Startup gating for Ollama: verify the server is reachable and the required models exist.

We never auto-start the server nor auto-pull models; we only surface the exact
commands the user must run and exit non-zero when the environment is not ready.
"""

import json
import urllib.error
import urllib.request

from app.core.config import Settings
from app.utils.logging import get_logger

logger = get_logger("ollama_health")

CHECK_TIMEOUT_SECONDS = 3


def _get_json(base_url: str, endpoint: str, timeout: float) -> dict:
    """Fetch and parse a JSON endpoint from the Ollama server."""
    with urllib.request.urlopen(
        f"{base_url.rstrip('/')}/{endpoint}", timeout=timeout
    ) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_running(base_url: str, timeout: float = CHECK_TIMEOUT_SECONDS) -> bool:
    """Return True when the Ollama server answers at base_url."""
    try:
        _get_json(base_url, "api/version", timeout)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError):
        return False


def installed_models(base_url: str, timeout: float = CHECK_TIMEOUT_SECONDS) -> list[str]:
    """Return the base names of models installed on the Ollama server."""
    try:
        payload = _get_json(base_url, "api/tags", timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError):
        return []
    return [_normalize_model_name(model.get("name", "")) for model in payload.get("models", [])]


def _normalize_model_name(name: str) -> str:
    """'nomic-embed-text:latest' -> 'nomic-embed-text'."""
    return name.split(":", 1)[0] if name else ""


def _missing_models(required: list[str], installed: list[str]) -> list[str]:
    """Models configured but not present on the server."""
    return [model for model in required if _normalize_model_name(model) not in installed]


def verify_ollama_ready(settings: Settings) -> None:
    """Check Ollama is reachable and the configured models are installed.

    Raises SystemExit(1) with the exact commands to fix the environment when
    the server is down or a required model is missing.
    """
    base_url = settings.ollama_base_url.rstrip("/")

    if not ollama_running(base_url):
        raise SystemExit(
            f"Ollama is not available at {base_url}.\n"
            "Start it with:\n"
            "   ollama serve"
        )

    required = [settings.ollama_embed_model, settings.ollama_llm_model]
    missing = _missing_models(required, installed_models(base_url))
    if missing:
        commands = "\n".join(f"   ollama pull {model}" for model in missing)
        raise SystemExit(
            "The following required models are not installed:\n"
            + "\n".join(f"   - {model}" for model in missing)
            + "\nDownload each one while Ollama is running:\n"
            + commands
        )

    logger.info(
        "Ollama is running at %s with the required models installed.",
        base_url,
    )