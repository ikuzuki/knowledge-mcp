"""Hybrid search: BM25 (FTS) + dense vector retrieval fused via Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal

from knowledge_mcp.storage.fts import FTSStore
from knowledge_mcp.types import EmbeddingProvider, SearchHit

if TYPE_CHECKING:
    from knowledge_mcp.storage.vectors import VectorStore

logger = logging.getLogger(__name__)

SearchMode = Literal["hybrid", "bm25", "vector"]


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchHit]],
    *,
    k: int = 60,
    limit: int = 5,
) -> list[SearchHit]:
    """Fuse multiple ranked hit lists into one using Reciprocal Rank Fusion.

    For each hit at 1-indexed rank `r` in a list, we add `1 / (k + r)` to its
    fused score. Hits are keyed by ``(path, chunk_index)`` so the same chunk
    surfacing in multiple lists accumulates contributions.

    Design choice: when the same chunk appears in multiple lists we keep the
    content/heading_path/chunk_index from the FIRST occurrence encountered while
    iterating ``ranked_lists`` in order. Callers pass BM25 first, so the text
    representation is stable and favours the lexical match.
    """
    scores: dict[tuple[str, int], float] = {}
    first_hit: dict[tuple[str, int], SearchHit] = {}

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            key = (hit.path, hit.chunk_index)
            contribution = 1.0 / (k + rank)
            scores[key] = scores.get(key, 0.0) + contribution
            if key not in first_hit:
                first_hit[key] = hit

    fused = [
        SearchHit(
            path=first_hit[key].path,
            heading_path=first_hit[key].heading_path,
            content=first_hit[key].content,
            score=score,
            chunk_index=first_hit[key].chunk_index,
        )
        for key, score in scores.items()
    ]
    fused.sort(key=lambda h: h.score, reverse=True)
    return fused[:limit]


class HybridSearch:
    """Orchestrates BM25, vector, and hybrid (RRF-fused) search modes."""

    def __init__(
        self,
        fts: FTSStore,
        vectors: "VectorStore | None",
        embeddings: EmbeddingProvider | None,
        *,
        rrf_k: int = 60,
        per_source_limit: int = 20,
    ) -> None:
        self.fts = fts
        self.vectors = vectors
        self.embeddings = embeddings
        self.rrf_k = rrf_k
        self.per_source_limit = per_source_limit
        self._warned_no_vectors = False

    def _vector_search(self, query: str, limit: int) -> list[SearchHit]:
        assert self.embeddings is not None and self.vectors is not None
        embedding = self.embeddings.embed([query])[0]
        return self.vectors.search(embedding, limit)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        mode: SearchMode = "hybrid",
    ) -> list[SearchHit]:
        if not query or not query.strip():
            return []

        if mode == "bm25":
            hits = self.fts.search(query, limit)
            logger.debug("bm25 search returned %d hits", len(hits))
            return hits

        if mode == "vector":
            if self.vectors is None or self.embeddings is None:
                raise ValueError("vector mode requires embeddings and vectors store")
            hits = self._vector_search(query, limit)
            logger.debug("vector search returned %d hits", len(hits))
            return hits

        # mode == "hybrid"
        if self.vectors is None or self.embeddings is None:
            if not self._warned_no_vectors:
                logger.warning(
                    "hybrid mode requested but vectors/embeddings unavailable; "
                    "falling back to BM25-only"
                )
                self._warned_no_vectors = True
            return self.fts.search(query, limit)

        # Run the vector branch (blocking I/O to Ollama + LanceDB) on a worker
        # thread while BM25 runs on the calling thread. We cannot submit the
        # BM25 call to the pool because SQLite connections are thread-bound.
        with ThreadPoolExecutor(max_workers=1) as pool:
            vector_future = pool.submit(self._vector_search, query, self.per_source_limit)

            try:
                bm25_hits = self.fts.search(query, self.per_source_limit)
            except Exception:
                logger.exception("BM25 branch failed during hybrid search")
                bm25_hits = []

            try:
                vector_hits = vector_future.result()
            except Exception:
                logger.exception(
                    "Vector branch failed during hybrid search; falling back to BM25 only"
                )
                logger.debug("hybrid fallback: bm25=%d vector=0", len(bm25_hits))
                return bm25_hits[:limit]

        logger.debug(
            "hybrid search: bm25=%d vector=%d", len(bm25_hits), len(vector_hits)
        )
        fused = reciprocal_rank_fusion(
            [bm25_hits, vector_hits], k=self.rrf_k, limit=limit
        )
        logger.debug("hybrid fused count=%d", len(fused))
        return fused
