"""Tests for the Ollama embedding provider."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from knowledge_mcp.embed import EmbeddingError, OllamaProvider
from knowledge_mcp.types import EmbeddingProvider


DIM = 4


def _vec(seed: float, dim: int = DIM) -> list[float]:
    return [seed + i for i in range(dim)]


def _mock_response(status_code: int, json_data: Any | None = None, text: str = "") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    if json_data is None:
        resp.json.side_effect = ValueError("no json")
    else:
        resp.json.return_value = json_data
    return resp


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip real backoff delays across all tests."""
    monkeypatch.setattr("knowledge_mcp.embed.time.sleep", lambda _s: None)


@pytest.fixture
def provider() -> OllamaProvider:
    return OllamaProvider(dim=DIM, max_retries=3, timeout_s=1.0)


def _patch_client(responses_or_exceptions: list[Any]) -> Any:
    """Return a patch context manager for httpx.Client whose .post() cycles through
    the provided responses or raises when an element is an Exception."""
    client_instance = MagicMock()

    def _post(*_args: Any, **_kwargs: Any) -> MagicMock:
        item = responses_or_exceptions.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client_instance.post.side_effect = _post

    ctx = MagicMock()
    ctx.__enter__.return_value = client_instance
    ctx.__exit__.return_value = False

    return patch("knowledge_mcp.embed.httpx.Client", return_value=ctx), client_instance


def test_empty_input_returns_empty_no_http(provider: OllamaProvider) -> None:
    with patch("knowledge_mcp.embed.httpx.Client") as client_cls:
        assert provider.embed([]) == []
        client_cls.assert_not_called()


def test_three_inputs_returns_three_vectors_in_order(provider: OllamaProvider) -> None:
    responses = [
        _mock_response(200, {"embedding": _vec(0)}),
        _mock_response(200, {"embedding": _vec(10)}),
        _mock_response(200, {"embedding": _vec(20)}),
    ]
    patcher, client = _patch_client(responses)
    with patcher:
        out = provider.embed(["a", "b", "c"])

    assert out == [_vec(0), _vec(10), _vec(20)]
    assert all(len(v) == DIM for v in out)
    assert client.post.call_count == 3


def test_4xx_raises_embedding_error_no_retry(provider: OllamaProvider) -> None:
    responses = [_mock_response(400, text="bad request")]
    patcher, client = _patch_client(responses)
    with patcher, pytest.raises(EmbeddingError):
        provider.embed(["x"])

    assert client.post.call_count == 1


def test_transient_503_retries_and_succeeds(provider: OllamaProvider) -> None:
    responses = [
        _mock_response(503, text="busy"),
        _mock_response(200, {"embedding": _vec(1)}),
    ]
    patcher, client = _patch_client(responses)
    with patcher:
        out = provider.embed(["hello"])

    assert out == [_vec(1)]
    assert client.post.call_count == 2


def test_connection_error_exhausts_retries(provider: OllamaProvider) -> None:
    responses: list[Any] = [
        httpx.ConnectError("boom"),
        httpx.ConnectError("boom"),
        httpx.ConnectError("boom"),
    ]
    patcher, client = _patch_client(responses)
    with patcher, pytest.raises(EmbeddingError):
        provider.embed(["hello"])

    assert client.post.call_count == 3


def test_wrong_dim_raises_value_error(provider: OllamaProvider) -> None:
    responses = [_mock_response(200, {"embedding": [0.1, 0.2]})]  # dim=2, expected 4
    patcher, _client = _patch_client(responses)
    with patcher, pytest.raises(ValueError):
        provider.embed(["hello"])


def test_satisfies_embedding_provider_protocol(provider: OllamaProvider) -> None:
    assert isinstance(provider, EmbeddingProvider)
    assert provider.model_version
    assert provider.dim == DIM
