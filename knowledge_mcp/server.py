"""MCP server entry point.

v0 surface: ``list_documents`` and ``get_document``. More tools land in v1/v2.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from knowledge_mcp.config import Settings, load_settings
from knowledge_mcp.vault import list_markdown, read_markdown

logger = logging.getLogger("knowledge_mcp")


def build_server(settings: Settings | None = None) -> FastMCP:
    """Construct and return a configured FastMCP instance.

    Injection-friendly so tests can pass a settings object pointing at a tmp vault.
    """
    if settings is None:
        settings = load_settings()

    mcp = FastMCP("knowledge-mcp")

    @mcp.tool()
    def list_documents(prefix: str = "") -> list[dict[str, str]]:
        """List markdown documents in the vault.

        Args:
            prefix: Optional vault-relative path prefix filter (e.g. ``"strategy/"``).

        Returns:
            A list of ``{path, title}`` entries, sorted by path.
        """
        try:
            summaries = list_markdown(settings.vault_path, prefix=prefix)
            return [{"path": s.path, "title": s.title} for s in summaries]
        except Exception as e:
            logger.exception("list_documents failed")
            return [{"error": str(e)}]

    @mcp.tool()
    def get_document(path: str) -> dict[str, Any]:
        """Read a markdown document from the vault.

        Args:
            path: Vault-relative path to a ``.md`` file.

        Returns:
            ``{path, title, frontmatter, content}`` or ``{error}`` on failure.
        """
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

    return mcp


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
