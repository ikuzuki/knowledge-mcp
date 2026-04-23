"""Embedding providers for knowledge-mcp.

Currently provides an Ollama-backed implementation of the
:class:`~knowledge_mcp.types.EmbeddingProvider` protocol.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BACKOFF_SECONDS: tuple[float, ...] = (0.5, 1.0, 2.0)


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider fails terminally."""


class OllamaProvider:
    """Embedding provider backed by a local Ollama server.

    Ollama's ``/api/embeddings`` endpoint accepts a single ``prompt`` per call,
    so batch embedding is implemented by issuing one request per input text
    while reusing the same :class:`httpx.Client` for connection pooling.
    """

    def __init__(
        self,
        *,
        endpoint: str = "http://localhost:11434/api/embeddings",
        model_version: str = "nomic-embed-text:v1.5",
        dim: int = 768,
        timeout_s: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        self.endpoint = endpoint
        self.model_version = model_version
        self.dim = dim
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def _post(self, client: httpx.Client, text: str) -> list[float]:
        """Post a single prompt to Ollama with retry/backoff, return its vector."""
        payload: dict[str, Any] = {"model": self.model_version, "prompt": text}
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = client.post(self.endpoint, json=payload)
            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_exc = exc
                logger.warning(
                    "Ollama embed transport error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                self._sleep_backoff(attempt)
                continue

            status = response.status_code
            if 500 <= status < 600:
                last_exc = EmbeddingError(
                    f"Ollama returned {status}: {response.text[:200]}"
                )
                logger.warning(
                    "Ollama 5xx response (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    status,
                )
                self._sleep_backoff(attempt)
                continue

            if 400 <= status < 500:
                raise EmbeddingError(
                    f"Ollama returned client error {status}: {response.text[:200]}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise EmbeddingError("Ollama returned non-JSON response") from exc

            vector = data.get("embedding")
            if not isinstance(vector, list):
                raise EmbeddingError(
                    "Ollama response missing 'embedding' list"
                )

            if len(vector) != self.dim:
                raise ValueError(
                    f"Ollama returned vector of length {len(vector)}, "
                    f"expected {self.dim}"
                )

            return [float(v) for v in vector]

        raise EmbeddingError(
            f"Ollama embedding failed after {self.max_retries} attempts"
        ) from last_exc

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        if attempt >= len(_BACKOFF_SECONDS):
            delay = _BACKOFF_SECONDS[-1]
        else:
            delay = _BACKOFF_SECONDS[attempt]
        time.sleep(delay)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, preserving order."""
        if not texts:
            return []

        vectors: list[list[float]] = []
        with httpx.Client(timeout=self.timeout_s) as client:
            for text in texts:
                vectors.append(self._post(client, text))
        return vectors
