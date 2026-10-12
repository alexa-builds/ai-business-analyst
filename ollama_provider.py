"""Optional, loopback-only adapter for a locally running Ollama server."""

from __future__ import annotations

import http.client
import json
import re
from typing import Any


OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
_NUMBER_WORDS = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"hundred|thousand|million|billion|first|second|third|fourth|fifth|twice|thrice)\b",
    re.IGNORECASE,
)
_CAUSAL_WORDS = re.compile(
    r"\b(?:cause|causes|caused|causing|because|due to|drives|driven by|leads to|"
    r"resulted in|results in|resulting in|impact(?:s|ed)?|therefore|contributes to)\b",
    re.IGNORECASE,
)


class OllamaUnavailable(RuntimeError):
    """Raised when the optional loopback Ollama service cannot be used."""


class UnsafeInterpretation(ValueError):
    """Raised when model text adds numbers or causal claims."""


def _local_request(method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 1.0) -> dict[str, Any]:
    """Talk only to the fixed loopback endpoint (no proxy or redirect support)."""
    connection = None
    try:
        connection = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=timeout)
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            raise OllamaUnavailable(f"Local Ollama returned HTTP {response.status}.")
        return json.loads(response.read().decode("utf-8"))
    except OllamaUnavailable:
        raise
    except (OSError, TimeoutError, http.client.HTTPException, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OllamaUnavailable("Local Ollama is unavailable. The Python analysis remains available.") from exc
    finally:
        if connection is not None:
            connection.close()


def list_local_models(timeout: float = 0.6) -> list[str]:
    """List installed local models, excluding models explicitly tagged as cloud."""
    response = _local_request("GET", "/api/tags", timeout=timeout)
    model_names = []
    for model in response.get("models", []):
        name = str(model.get("name", "")).strip()
        if name and not re.search(r"(?:^|[-_:])cloud(?:$|[-_:])", name, re.IGNORECASE):
            model_names.append(name)
    return sorted(set(model_names))


def interpretation_context(result: dict[str, Any]) -> dict[str, Any]:
    """Keep model context limited to the question's calculated evidence."""
    return {
        "question": str(result.get("question", ""))[:400],
        "python_finding": str(result.get("headline", ""))[:600],
        "supporting_calculations": result.get("facts", [])[:8],
        "calculation_method": str(result.get("calculation", ""))[:500],
        "limitations": result.get("limitations", [])[:4],
    }


def validate_interpretation(text: str) -> str:
    """Reject numeric or causal claims so Python evidence remains authoritative."""
    cleaned = " ".join(str(text).split()).strip()
    if not cleaned:
        raise UnsafeInterpretation("The local model returned an empty interpretation.")
    if re.search(r"\d", cleaned) or _NUMBER_WORDS.search(cleaned):
        raise UnsafeInterpretation("The local model added a numeric claim; use the Python-calculated figures above.")
    if _CAUSAL_WORDS.search(cleaned):
        raise UnsafeInterpretation("The local model made a causal claim that this dataset cannot establish.")
    if len(cleaned) > 900:
        raise UnsafeInterpretation("The local model response exceeded the interpretation length limit.")
    return cleaned


def generate_interpretation(model: str, result: dict[str, Any], timeout: float = 45.0) -> str:
    """Ask a local model for brief qualitative wording, never for calculations."""
    if not model:
        raise OllamaUnavailable("Choose an installed local Ollama model first.")
    if model not in list_local_models(timeout=min(timeout, 1.0)):
        raise OllamaUnavailable("The selected model is not available as a local Ollama model.")
    context = interpretation_context(result)
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 140},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a concise business analyst explaining a Python-calculated finding. "
                    "Use only the supplied finding, calculations, method, and limitations. "
                    "Explain the finding cautiously without naming groups or adding figures. "
                    "Do not calculate or repeat figures. Do not write digits or number words. "
                    "Do not invent categories or facts. Do not claim causes, effects, drivers, or predictions. "
                    "If evidence is limited, say so. Return one plain-language interpretation under seventy words."
                ),
            },
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
    }
    response = _local_request("POST", "/api/chat", payload, timeout=timeout)
    try:
        content = response["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise OllamaUnavailable("Local Ollama returned an unexpected chat response.") from exc
    return validate_interpretation(content)
