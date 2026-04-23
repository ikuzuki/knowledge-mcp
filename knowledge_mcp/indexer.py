"""Indexer: chunk files and keep the FTS + vector stores in sync with the vault."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import blake3
import frontmatter

from knowledge_mcp.chunking import chunk_markdown
from knowledge_mcp.storage.fts import FTSStore
from knowledge_mcp.types import EmbeddingProvider

if TYPE_CHECKING:
    from knowledge_mcp.storage.vectors import VectorStore

logger = logging.getLogger(__name__)


def _is_hidden(rel: Path) -> bool:
    return any(part.startswith(".") for part in rel.parts)


class Indexer:
    """Coordinates chunking, FTS upserts, and vector upserts for a vault.

    The vector path is optional — if ``vectors`` or ``embeddings`` is None,
    indexing degrades gracefully to BM25-only. This is the same policy
    ``HybridSearch`` enforces at query time, keeping the two halves consistent
    when Ollama is down.
    """

    def __init__(
        self,
        vault_path: Path,
        fts: FTSStore,
        *,
        vectors: "VectorStore | None" = None,
        embeddings: EmbeddingProvider | None = None,
    ) -> None:
        self.vault_path = vault_path
        self.fts = fts
        self.vectors = vectors
        self.embeddings = embeddings
        self._hashes: dict[str, str] = {}
        self._lock = threading.Lock()

    def _hash_bytes(self, data: bytes) -> str:
        return blake3.blake3(data).hexdigest()

    def _embed_and_upsert_vectors(self, rel_path: str, chunks) -> None:
        if self.vectors is None or self.embeddings is None:
            return
        if not chunks:
            self.vectors.delete_file(rel_path)
            return
        try:
            vectors = self.embeddings.embed([c.text for c in chunks])
        except Exception:
            logger.exception("embedding failed for %s; vector branch skipped", rel_path)
            return
        try:
            self.vectors.upsert_chunks(
                rel_path, chunks, vectors, self.embeddings.model_version
            )
        except Exception:
            logger.exception("vector upsert failed for %s", rel_path)

    def reindex_file(self, rel_path: str) -> bool:
        abs_path = (self.vault_path / rel_path).resolve()
        if not abs_path.exists() or not abs_path.is_file():
            self.delete_file(rel_path)
            return True

        data = abs_path.read_bytes()
        digest = self._hash_bytes(data)
        with self._lock:
            if self._hashes.get(rel_path) == digest:
                return False
            self._hashes[rel_path] = digest

        try:
            post = frontmatter.loads(data.decode("utf-8", errors="replace"))
            chunks = chunk_markdown(post.content, dict(post.metadata))
        except Exception:
            logger.exception("chunking failed for %s", rel_path)
            return False

        self.fts.upsert_chunks(rel_path, chunks)
        self._embed_and_upsert_vectors(rel_path, chunks)
        logger.info("indexed %s (%d chunks)", rel_path, len(chunks))
        return True

    def delete_file(self, rel_path: str) -> None:
        self.fts.delete_file(rel_path)
        if self.vectors is not None:
            try:
                self.vectors.delete_file(rel_path)
            except Exception:
                logger.exception("vector delete failed for %s", rel_path)
        with self._lock:
            self._hashes.pop(rel_path, None)
        logger.info("removed %s from index", rel_path)

    def reindex_all(self) -> int:
        count = 0
        for md in self.vault_path.rglob("*.md"):
            rel = md.relative_to(self.vault_path)
            if _is_hidden(rel):
                continue
            if self.reindex_file(rel.as_posix()):
                count += 1
        logger.info("reindex_all: %d files indexed", count)
        return count

    def reindex_all_embeddings(self) -> int:
        """Force a full re-embed of every chunk in the vault.

        Used after changing the embedding model. Clears the vector store and
        walks the vault, re-embedding and upserting everything. FTS is left
        untouched since its content doesn't depend on the model.
        """
        if self.vectors is None or self.embeddings is None:
            raise RuntimeError(
                "reindex_all_embeddings requires both a vector store and an embedder"
            )
        self.vectors.clear()
        with self._lock:
            self._hashes.clear()  # force reindex_file to run chunking again too
        return self.reindex_all()
