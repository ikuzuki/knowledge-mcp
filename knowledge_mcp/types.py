"""Shared dataclasses used across chunking, storage, search, and indexing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """A chunk of a markdown document, ready for indexing.

    Attributes:
        text: The chunk's textual content (post-chunking, pre-embedding).
        heading_path: Ordered list of ancestor markdown headings (h1 → h3).
            Empty list for content preceding the first heading.
        chunk_index: 0-based position of this chunk within its source file.
        token_count: Approximate token count (via tiktoken cl100k_base).
    """

    text: str
    heading_path: list[str] = field(default_factory=list)
    chunk_index: int = 0
    token_count: int = 0


@dataclass(frozen=True)
class SearchHit:
    """A single result returned by FTS, vector, or hybrid search."""

    path: str
    heading_path: list[str]
    content: str
    score: float
    chunk_index: int = 0
