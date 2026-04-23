"""Indexer: chunk files and keep the FTS store in sync with the vault."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import blake3
import frontmatter

from knowledge_mcp.chunking import chunk_markdown
from knowledge_mcp.storage.fts import FTSStore

logger = logging.getLogger(__name__)


def _is_hidden(rel: Path) -> bool:
    return any(part.startswith(".") for part in rel.parts)


def _vault_rel(vault_root: Path, absolute: Path) -> str:
    return absolute.relative_to(vault_root).as_posix()


class Indexer:
    """Coordinates chunking and FTS upserts for a vault.

    The in-memory ``_hashes`` cache lets ``reindex_file`` short-circuit no-op
    updates; the watcher also dedups by hash independently, but we guard here
    too since ``reindex_all`` and manual calls bypass the watcher.
    """

    def __init__(self, vault_path: Path, fts: FTSStore) -> None:
        self.vault_path = vault_path
        self.fts = fts
        self._hashes: dict[str, str] = {}
        self._lock = threading.Lock()

    def _hash_bytes(self, data: bytes) -> str:
        return blake3.blake3(data).hexdigest()

    def reindex_file(self, rel_path: str) -> bool:
        """Reindex a single file. Returns True if the index changed."""
        abs_path = (self.vault_path / rel_path).resolve()
        if not abs_path.exists() or not abs_path.is_file():
            # Caller probably meant delete; be defensive.
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
        logger.info("indexed %s (%d chunks)", rel_path, len(chunks))
        return True

    def delete_file(self, rel_path: str) -> None:
        self.fts.delete_file(rel_path)
        with self._lock:
            self._hashes.pop(rel_path, None)
        logger.info("removed %s from index", rel_path)

    def reindex_all(self) -> int:
        """Walk the vault and reindex every markdown file. Returns count."""
        count = 0
        for md in self.vault_path.rglob("*.md"):
            rel = md.relative_to(self.vault_path)
            if _is_hidden(rel):
                continue
            rel_posix = rel.as_posix()
            if self.reindex_file(rel_posix):
                count += 1
        logger.info("reindex_all: %d files indexed", count)
        return count
