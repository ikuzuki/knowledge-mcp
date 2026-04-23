"""Tests for the hybrid search module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from knowledge_mcp.search import HybridSearch, reciprocal_rank_fusion
from knowledge_mcp.storage.fts import FTSStore
from knowledge_mcp.types import Chunk, SearchHit


def _hit(path: str, chunk_index: int = 0, score: float = 0.0, content: str = "") -> SearchHit:
    return SearchHit(
        path=path,
        heading_path=[],
        content=content or f"{path}#{chunk_index}",
        score=score,
        chunk_index=chunk_index,
    )


class FakeVectorStore:
    def __init__(self, hits: list[SearchHit] | Exception | None = None) -> None:
        self._hits = hits if hits is not None else []
        self.search = MagicMock(side_effect=self._search)

    def _search(self, query_embedding: list[float], limit: int) -> list[SearchHit]:
        if isinstance(self._hits, Exception):
            raise self._hits
        return list(self._hits)[:limit]


class FakeEmbedder:
    model_version = "fake-v1"
    dim = 3

    def __init__(self, raise_exc: Exception | None = None) -> None:
        self._raise = raise_exc
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if self._raise is not None:
            raise self._raise
        return [[0.1, 0.2, 0.3] for _ in texts]


# ------------------------- RRF ------------------------------------------------


def test_rrf_overlapping_scores_formula() -> None:
    a = _hit("a.md", 0)
    b = _hit("b.md", 0)
    c = _hit("c.md", 0)
    bm25 = [a, b, c]          # ranks 1,2,3
    vector = [c, a]           # ranks 1,2  (overlap a and c)

    fused = reciprocal_rank_fusion([bm25, vector], k=60, limit=5)

    fused_map = {(h.path, h.chunk_index): h.score for h in fused}
    # a: 1/61 + 1/62
    assert fused_map[("a.md", 0)] == pytest.approx(1 / 61 + 1 / 62)
    # b: 1/62 only
    assert fused_map[("b.md", 0)] == pytest.approx(1 / 62)
    # c: 1/63 + 1/61
    assert fused_map[("c.md", 0)] == pytest.approx(1 / 63 + 1 / 61)

    # Ordering: a > c > b (a has top rank in bm25 + rank 2 in vector)
    assert [h.path for h in fused] == ["a.md", "c.md", "b.md"]


def test_rrf_disjoint_lists_all_present() -> None:
    bm25 = [_hit("a.md"), _hit("b.md")]
    vector = [_hit("c.md"), _hit("d.md")]
    fused = reciprocal_rank_fusion([bm25, vector], k=60, limit=10)
    paths = {h.path for h in fused}
    assert paths == {"a.md", "b.md", "c.md", "d.md"}
    # a and c both rank-1, so they should tie and come before b and d.
    top_two = {fused[0].path, fused[1].path}
    assert top_two == {"a.md", "c.md"}


def test_rrf_limit_truncates() -> None:
    bm25 = [_hit(f"f{i}.md") for i in range(10)]
    vector = [_hit(f"g{i}.md") for i in range(10)]
    fused = reciprocal_rank_fusion([bm25, vector], limit=3)
    assert len(fused) == 3


def test_rrf_keeps_first_occurrence_content() -> None:
    # Same key from both lists — BM25 version should be retained.
    bm25 = [_hit("a.md", 0, content="from-bm25")]
    vector = [_hit("a.md", 0, content="from-vector")]
    fused = reciprocal_rank_fusion([bm25, vector], limit=5)
    assert len(fused) == 1
    assert fused[0].content == "from-bm25"


# ------------------------- HybridSearch --------------------------------------


@pytest.fixture
def fts_store(tmp_path: Path) -> FTSStore:
    store = FTSStore(tmp_path / "fts.db")
    store.upsert_chunks(
        "alpha.md",
        [Chunk(text="alpha apple banana", chunk_index=0)],
    )
    store.upsert_chunks(
        "beta.md",
        [Chunk(text="beta banana cherry", chunk_index=0)],
    )
    return store


def test_bm25_mode_does_not_touch_vectors(fts_store: FTSStore) -> None:
    vectors = FakeVectorStore([_hit("unused.md")])
    embedder = FakeEmbedder()
    hs = HybridSearch(fts_store, vectors, embedder)

    results = hs.search("banana", limit=5, mode="bm25")

    assert len(results) >= 1
    vectors.search.assert_not_called()
    assert embedder.calls == []


def test_vector_mode_missing_providers_raises(fts_store: FTSStore) -> None:
    hs = HybridSearch(fts_store, vectors=None, embeddings=None)
    with pytest.raises(ValueError, match="vector mode requires"):
        hs.search("banana", mode="vector")


def test_hybrid_missing_providers_falls_back_to_bm25(fts_store: FTSStore) -> None:
    hs = HybridSearch(fts_store, vectors=None, embeddings=None)
    results = hs.search("banana", limit=5, mode="hybrid")
    # Should be BM25 results, no exception raised.
    assert len(results) >= 1
    assert all(isinstance(h, SearchHit) for h in results)


def test_hybrid_embedder_raises_falls_back_to_bm25(fts_store: FTSStore) -> None:
    vectors = FakeVectorStore([_hit("vec.md")])
    embedder = FakeEmbedder(raise_exc=RuntimeError("ollama down"))
    hs = HybridSearch(fts_store, vectors, embedder)

    results = hs.search("banana", limit=5, mode="hybrid")

    # No raise; results come from BM25 only.
    paths = {h.path for h in results}
    assert "vec.md" not in paths
    assert len(results) >= 1


def test_hybrid_fuses_both_branches(fts_store: FTSStore) -> None:
    # vector store surfaces a hit that BM25 also finds (overlap) plus a novel one.
    vector_hits = [
        _hit("beta.md", 0, score=0.9),
        _hit("gamma.md", 0, score=0.5),
    ]
    vectors = FakeVectorStore(vector_hits)
    embedder = FakeEmbedder()
    hs = HybridSearch(fts_store, vectors, embedder)

    results = hs.search("banana", limit=5, mode="hybrid")
    paths = {h.path for h in results}

    # Union of BM25 ({alpha, beta}) and vector ({beta, gamma}).
    assert "gamma.md" in paths  # came only from vector
    assert "beta.md" in paths  # appeared in both
    # Embedder was invoked exactly once for the query.
    assert len(embedder.calls) == 1
    assert embedder.calls[0] == ["banana"]
    # All scores are RRF fused scores (small floats).
    for h in results:
        assert 0 < h.score < 1


def test_empty_query_returns_empty(fts_store: FTSStore) -> None:
    vectors = FakeVectorStore([_hit("x.md")])
    embedder = FakeEmbedder()
    hs = HybridSearch(fts_store, vectors, embedder)

    assert hs.search("", mode="hybrid") == []
    assert hs.search("   ", mode="bm25") == []
    assert hs.search("", mode="vector") == []
    # Nothing downstream should have been called.
    vectors.search.assert_not_called()
    assert embedder.calls == []
