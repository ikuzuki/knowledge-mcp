"""MCP server entry point.

v2 surface: list_documents, get_document, search (hybrid), create_document,
update_document, reindex_all_embeddings.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import frontmatter
from mcp.server.fastmcp import FastMCP

from knowledge_mcp.config import Settings, load_settings
from knowledge_mcp.embed import OllamaProvider
from knowledge_mcp.indexer import Indexer
from knowledge_mcp.search import HybridSearch
from knowledge_mcp.storage.fts import FTSStore
from knowledge_mcp.storage.vectors import VectorStore
from knowledge_mcp.vault import list_markdown, read_markdown
from knowledge_mcp.watcher import VaultWatcher

logger = logging.getLogger("knowledge_mcp")


def _resolve_inside_vault(vault_root: Path, rel: str) -> Path:
    candidate = (vault_root / rel).resolve()
    try:
        candidate.relative_to(vault_root)
    except ValueError as e:
        raise ValueError(f"path escapes vault root: {rel}") from e
    return candidate


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_md_path(rel: str) -> None:
    if not rel or rel.startswith("/") or rel.startswith("\\"):
        raise ValueError("path must be vault-relative")
    if not rel.endswith(".md"):
        raise ValueError("path must end in .md")


def _check_embedding_drift(vectors: VectorStore, expected: str) -> None:
    present = vectors.model_versions_present()
    if not present:
        return
    drift = present - {expected}
    if drift:
        logger.warning(
            "embedding drift: vector store contains model_version=%s but config "
            "expects %s — call reindex_all_embeddings to reconcile",
            sorted(drift),
            expected,
        )


def build_server(
    settings: Settings | None = None,
    *,
    start_indexing: bool = True,
    enable_vectors: bool = True,
) -> tuple[FastMCP, "ServerDeps"]:
    if settings is None:
        settings = load_settings()

    index_dir = settings.vault_path / ".index"
    index_dir.mkdir(parents=True, exist_ok=True)
    fts = FTSStore(index_dir / "fts.db")

    vectors: VectorStore | None = None
    embeddings: OllamaProvider | None = None
    if enable_vectors:
        try:
            embeddings = OllamaProvider(
                endpoint=settings.embedding_endpoint,
                model_version=settings.embedding_model,
            )
            vectors = VectorStore(index_dir / "vectors.lance", dim=embeddings.dim)
            _check_embedding_drift(vectors, embeddings.model_version)
        except Exception:
            logger.exception("vector subsystem failed to initialise; degrading to BM25-only")
            vectors = None
            embeddings = None

    indexer = Indexer(
        settings.vault_path, fts, vectors=vectors, embeddings=embeddings
    )
    hybrid = HybridSearch(
        fts=fts, vectors=vectors, embeddings=embeddings, rrf_k=settings.rrf_k
    )

    watcher: VaultWatcher | None = None

    if start_indexing:
        import threading
        threading.Thread(
            target=indexer.reindex_all, name="reindex-all", daemon=True
        ).start()
        watcher = VaultWatcher(
            vault_path=settings.vault_path,
            on_upsert=indexer.reindex_file,
            on_delete=indexer.delete_file,
        )
        watcher.start()

    deps = ServerDeps(
        settings=settings,
        fts=fts,
        vectors=vectors,
        indexer=indexer,
        watcher=watcher,
    )

    mcp = FastMCP("knowledge-mcp")

    @mcp.tool()
    def list_documents(prefix: str = "") -> list[dict[str, str]]:
        """List markdown documents in the vault. Optional path-prefix filter."""
        try:
            summaries = list_markdown(settings.vault_path, prefix=prefix)
            return [{"path": s.path, "title": s.title} for s in summaries]
        except Exception as e:
            logger.exception("list_documents failed")
            return [{"error": str(e)}]

    @mcp.tool()
    def get_document(path: str) -> dict[str, Any]:
        """Read a markdown document from the vault."""
        try:
            doc = read_markdown(settings.vault_path, path)
            return {
                "path": doc.path,
                "title": doc.title,
                "frontmatter": doc.frontmatter,
                "content": doc.content,
            }
        except Exception as e:
            logger.exception("get_document failed")
            return {"error": str(e)}

    @mcp.tool()
    def search(
        query: str,
        limit: int = 5,
        mode: Literal["hybrid", "bm25", "vector"] = "hybrid",
    ) -> list[dict[str, Any]]:
        """Search the vault. Modes: hybrid (default, BM25+vector+RRF), bm25, vector."""
        try:
            hits = hybrid.search(query, limit=limit, mode=mode)
            return [
                {
                    "path": h.path,
                    "heading_path": list(h.heading_path),
                    "content": h.content,
                    "score": h.score,
                    "chunk_index": h.chunk_index,
                }
                for h in hits
            ]
        except Exception as e:
            logger.exception("search failed")
            return [{"error": str(e)}]

    @mcp.tool()
    def create_document(
        path: str, content: str, frontmatter_data: dict | None = None
    ) -> dict[str, Any]:
        """Create a new markdown document in the vault. Errors if it exists."""
        try:
            _validate_md_path(path)
            target = _resolve_inside_vault(settings.vault_path, path)
            if target.exists():
                return {"error": f"document already exists: {path}"}
            target.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_markdown(target, content, frontmatter_data or {})
            return {"path": path, "created_at": _now_iso()}
        except Exception as e:
            logger.exception("create_document failed")
            return {"error": str(e)}

    @mcp.tool()
    def update_document(
        path: str, content: str, frontmatter_data: dict | None = None
    ) -> dict[str, Any]:
        """Overwrite a markdown document atomically. Errors if missing."""
        try:
            _validate_md_path(path)
            target = _resolve_inside_vault(settings.vault_path, path)
            if not target.exists():
                return {"error": f"document does not exist: {path}"}
            _atomic_write_markdown(target, content, frontmatter_data or {})
            return {"path": path, "updated_at": _now_iso()}
        except Exception as e:
            logger.exception("update_document failed")
            return {"error": str(e)}

    @mcp.tool()
    def reindex_all_embeddings() -> dict[str, Any]:
        """Wipe and rebuild every chunk's embedding (run after changing the model)."""
        try:
            count = indexer.reindex_all_embeddings()
            return {"reindexed_files": count, "completed_at": _now_iso()}
        except Exception as e:
            logger.exception("reindex_all_embeddings failed")
            return {"error": str(e)}

    return mcp, deps


class ServerDeps:
    def __init__(
        self,
        settings: Settings,
        fts: FTSStore,
        vectors: VectorStore | None,
        indexer: Indexer,
        watcher: VaultWatcher | None,
    ) -> None:
        self.settings = settings
        self.fts = fts
        self.vectors = vectors
        self.indexer = indexer
        self.watcher = watcher

    def shutdown(self) -> None:
        if self.watcher is not None:
            try:
                self.watcher.stop()
            except Exception:
                logger.exception("watcher.stop failed")
        try:
            self.fts.close()
        except Exception:
            logger.exception("fts.close failed")
        if self.vectors is not None:
            try:
                self.vectors.close()
            except Exception:
                logger.exception("vectors.close failed")


def _atomic_write_markdown(target: Path, content: str, fm: dict) -> None:
    if fm:
        post = frontmatter.Post(content, **fm)
        serialised = frontmatter.dumps(post)
    else:
        serialised = content
    if not serialised.endswith("\n"):
        serialised += "\n"
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(serialised, encoding="utf-8", newline="\n")
    os.replace(tmp, target)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    server, deps = build_server()
    try:
        server.run()
    finally:
        deps.shutdown()


if __name__ == "__main__":
    main()
