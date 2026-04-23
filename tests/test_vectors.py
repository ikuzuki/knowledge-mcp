"""Tests for ``knowledge_mcp.storage.vectors.VectorStore``."""

from __future__ import annotations

import numpy as np
import pytest

from knowledge_mcp.storage.vectors import VectorStore
from knowledge_mcp.types import Chunk

DIM = 16


def _random_embeddings(n: int, dim: int = DIM, seed: int = 0) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype("float32").tolist()


def _mk_chunk(idx: int, text: str, headings: list[str] | None = None) -> Chunk:
    return Chunk(
        text=text,
        heading_path=headings or [],
        chunk_index=idx,
        token_count=len(text.split()),
    )


@pytest.fixture()
def store(tmp_path):
    s = VectorStore(tmp_path / "vectors.lance", dim=DIM)
    yield s
    s.close()


def test_upsert_and_search_returns_both_chunks(store):
    chunks = [_mk_chunk(0, "alpha text"), _mk_chunk(1, "beta text")]
    embs = _random_embeddings(2)
    store.upsert_chunks("foo.md", chunks, embs, "v1")

    hits = store.search(embs[0], limit=10)
    assert len(hits) == 2
    for h in hits:
        assert h.path == "foo.md"
        assert isinstance(h.score, float)


def test_upsert_replaces_same_file(store):
    store.upsert_chunks(
        "foo.md",
        [_mk_chunk(0, "first"), _mk_chunk(1, "second")],
        _random_embeddings(2, seed=1),
        "v1",
    )
    store.upsert_chunks(
        "foo.md",
        [_mk_chunk(0, "replaced"), _mk_chunk(1, "replaced2")],
        _random_embeddings(2, seed=2),
        "v1",
    )
    hits = store.search(_random_embeddings(1, seed=3)[0], limit=50)
    contents = [h.content for h in hits]
    assert sorted(contents) == ["replaced", "replaced2"]
    # No duplicate ids ==> exactly two rows.
    assert len(hits) == 2


def test_delete_file_removes_rows(store):
    store.upsert_chunks(
        "a.md", [_mk_chunk(0, "apple")], _random_embeddings(1, seed=1), "v1"
    )
    store.upsert_chunks(
        "b.md", [_mk_chunk(0, "banana")], _random_embeddings(1, seed=2), "v1"
    )
    store.delete_file("a.md")
    hits = store.search(_random_embeddings(1, seed=4)[0], limit=50)
    assert all(h.path != "a.md" for h in hits)
    assert any(h.path == "b.md" for h in hits)


def test_model_versions_present(store):
    store.upsert_chunks(
        "a.md", [_mk_chunk(0, "x")], _random_embeddings(1, seed=1), "v1"
    )
    store.upsert_chunks(
        "b.md", [_mk_chunk(0, "y")], _random_embeddings(1, seed=2), "v2"
    )
    store.upsert_chunks(
        "c.md", [_mk_chunk(0, "z")], _random_embeddings(1, seed=3), "v1"
    )
    assert store.model_versions_present() == {"v1", "v2"}


def test_clear_empties_table(store):
    store.upsert_chunks(
        "a.md", [_mk_chunk(0, "x")], _random_embeddings(1, seed=1), "v1"
    )
    store.clear()
    assert store.model_versions_present() == set()
    assert store.search(_random_embeddings(1, seed=4)[0]) == []


def test_mismatched_lengths_raises(store):
    with pytest.raises(ValueError):
        store.upsert_chunks(
            "a.md",
            [_mk_chunk(0, "x"), _mk_chunk(1, "y")],
            _random_embeddings(1),
            "v1",
        )


def test_wrong_embedding_dim_raises(store):
    bad = [[0.0] * (DIM + 1)]
    with pytest.raises(ValueError):
        store.upsert_chunks("a.md", [_mk_chunk(0, "x")], bad, "v1")


def test_search_results_ordered_by_score_desc(store):
    chunks = [_mk_chunk(i, f"chunk{i}") for i in range(5)]
    embs = _random_embeddings(5, seed=7)
    store.upsert_chunks("foo.md", chunks, embs, "v1")

    hits = store.search(embs[0], limit=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_empty_table_returns_empty_list(store):
    assert store.search([0.0] * DIM) == []


def test_heading_path_roundtrips(store):
    headings = ["Top", "Sub", "Leaf"]
    store.upsert_chunks(
        "doc.md",
        [_mk_chunk(0, "content here", headings)],
        _random_embeddings(1, seed=1),
        "v1",
    )
    hits = store.search(_random_embeddings(1, seed=1)[0], limit=1)
    assert len(hits) == 1
    assert hits[0].heading_path == headings
