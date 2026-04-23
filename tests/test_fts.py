"""Tests for knowledge_mcp.storage.fts.FTSStore."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from knowledge_mcp.storage.fts import FTSStore
from knowledge_mcp.types import Chunk


@pytest.fixture
def store(tmp_path: Path) -> FTSStore:
    s = FTSStore(tmp_path / "db" / "test.db")
    yield s
    s.close()


def _chunk(text: str, idx: int = 0, headings: list[str] | None = None) -> Chunk:
    return Chunk(
        text=text,
        heading_path=headings or [],
        chunk_index=idx,
        token_count=len(text.split()),
    )


def test_upsert_and_search_returns_finite_scores(store: FTSStore) -> None:
    store.upsert_chunks(
        "notes/a.md",
        [
            _chunk("the quick brown fox jumps", 0, ["Animals"]),
            _chunk("brown bears love honey", 1, ["Animals", "Bears"]),
        ],
    )
    hits = store.search("brown")
    assert len(hits) == 2
    for h in hits:
        assert math.isfinite(h.score)
        assert h.path == "notes/a.md"


def test_upsert_replaces_prior_rows(store: FTSStore) -> None:
    store.upsert_chunks("f.md", [_chunk("alpha beta", 0), _chunk("gamma delta", 1)])
    store.upsert_chunks("f.md", [_chunk("alpha only", 0)])
    hits = store.search("alpha")
    assert len(hits) == 1
    assert hits[0].content == "alpha only"
    # prior terms gone
    assert store.search("gamma") == []


def test_delete_file_removes_rows(store: FTSStore) -> None:
    store.upsert_chunks("f.md", [_chunk("unique term xyzzy", 0)])
    assert len(store.search("xyzzy")) == 1
    store.delete_file("f.md")
    assert store.search("xyzzy") == []


def test_search_only_returns_remaining_file_after_delete(store: FTSStore) -> None:
    store.upsert_chunks("a.md", [_chunk("shared keyword", 0)])
    store.upsert_chunks("b.md", [_chunk("shared keyword", 0)])
    store.delete_file("a.md")
    hits = store.search("shared")
    assert len(hits) == 1
    assert hits[0].path == "b.md"


def test_search_ranks_more_relevant_higher(store: FTSStore) -> None:
    store.upsert_chunks(
        "doc.md",
        [
            _chunk("python python python testing framework", 0),
            _chunk("a brief mention of python in a long text about cats dogs birds fish", 1),
        ],
    )
    hits = store.search("python")
    assert len(hits) == 2
    # higher score = better match (we invert bm25)
    assert hits[0].score >= hits[1].score
    assert hits[0].chunk_index == 0


def test_search_handles_special_characters(store: FTSStore) -> None:
    store.upsert_chunks("f.md", [_chunk("hello world", 0)])
    # Must not raise on punctuation/operators in the query
    hits = store.search('"hello!?"')
    assert len(hits) == 1
    # More exotic characters
    assert store.search("(foo AND bar) OR *:") == []
    assert store.search("--NEAR(x y)") == []


def test_empty_query_returns_empty(store: FTSStore) -> None:
    store.upsert_chunks("f.md", [_chunk("something", 0)])
    assert store.search("") == []
    assert store.search("   ") == []


def test_heading_path_roundtrip(store: FTSStore) -> None:
    headings = ["Top", "Section Two", "Sub: with punctuation!"]
    store.upsert_chunks("f.md", [_chunk("body text", 0, headings)])
    hits = store.search("body")
    assert len(hits) == 1
    assert hits[0].heading_path == headings
    assert hits[0].chunk_index == 0


def test_empty_chunks_list_just_deletes(store: FTSStore) -> None:
    store.upsert_chunks("f.md", [_chunk("will be gone", 0)])
    store.upsert_chunks("f.md", [])
    assert store.search("gone") == []
