"""MCP server entry point.

v1 surface: list_documents, get_document, search, create_document, update_document.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import frontmatter
from mcp.server.fastmcp import FastMCP

from knowledge_mcp.config import Settings, load_settings
from knowledge_mcp.indexer import Indexer
from knowledge_mcp.storage.fts import FTSStore
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


def build_server(
    settings: Settings | None = None,
    *,
    start_indexing: bool = True,
) -> tuple[FastMCP, "ServerDeps"]:
    """Construct a configured FastMCP instance.

    Returns the MCP server plus a ``ServerDeps`` handle so tests and ``main``
    can inspect or shut down the background machinery.
    """
    if settings is None:
        settings = load_settings()

    index_dir = settings.vault_path / ".index"
    index_dir.mkdir(parents=True, exist_ok=True)
    fts = FTSStore(index_dir / "fts.db")
    indexer = Indexer(settings.vault_path, fts)

    watcher: VaultWatcher | None = None
    reindex_thread: threading.Thread | None = None

    if start_indexing:
        reindex_thread = threading.Thread(
            target=indexer.reindex_all, name="reindex-all", daemon=True
        )
        reindex_thread.start()
        watcher = VaultWatcher(
            vault_path=settings.vault_path,
            on_upsert=indexer.reindex_file,
            on_delete=indexer.delete_file,
        )
        watcher.start()

    deps = ServerDeps(settings=settings, fts=fts, indexer=indexer, watcher=watcher)

    mcp = FastMCP("knowledge-mcp")

    @mcp.tool()
    def list_documents(prefix: str = "") -> list[dict[str, str]]:
        """List markdown documents in the vault.

        Args:
            prefix: Optional vault-relative path prefix filter.
        """
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
    def search(query: str, limit: int = 5) -> list[dict[str, Any]]:
        """BM25 keyword search over the vault (hybrid retrieval arrives in v2).

        Args:
            query: Free-text query.
            limit: Max results to return.
        """
        try:
            hits = fts.search(query, limit=limit)
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

    return mcp, deps


class ServerDeps:
    """Handle on background machinery so main() and tests can shut down cleanly."""

    def __init__(
        self,
        settings: Settings,
        fts: FTSStore,
        indexer: Indexer,
        watcher: VaultWatcher | None,
    ) -> None:
        self.settings = settings
        self.fts = fts
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
