"""Thin client for the local Ollama server (http://localhost:11434).

Ollama exposes a plain REST/JSON API, so we call it with `requests`:
- POST /api/embed    -> embedding vectors for a list of texts
- POST /api/generate -> a text completion from the chat model
"""

import os

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.1:8b")


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


def _post(path: str, payload: dict, timeout: float) -> dict:
    try:
        response = requests.post(f"{OLLAMA_URL}{path}", json=payload, timeout=timeout)
    except requests.ConnectionError as err:
        raise OllamaError(
            f"Cannot reach Ollama at {OLLAMA_URL}. Is the Ollama app running?"
        ) from err
    if response.status_code != 200:
        raise OllamaError(f"Ollama {path} returned {response.status_code}: {response.text[:200]}")
    return response.json()


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per input text, in order."""
    data = _post("/api/embed", {"model": EMBED_MODEL, "input": texts}, timeout=120)
    return data["embeddings"]


def generate(prompt: str, temperature: float = 0.0) -> str:
    """Single-turn completion. temperature=0 makes answers repeatable, which evals need."""
    data = _post(
        "/api/generate",
        {
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=300,
    )
    return data["response"].strip()
