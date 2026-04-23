"""v0 smoke tests: list_documents and get_document against a tmp vault."""

from __future__ import annotations

import pytest

from knowledge_mcp.server import build_server


@pytest.mark.asyncio
async def test_list_documents_returns_both(settings):
    server = build_server(settings)
    tool = await _call_tool(server, "list_documents", {})
    paths = {entry["path"] for entry in tool}
    assert paths == {"beta.md", "notes/alpha.md"}


@pytest.mark.asyncio
async def test_list_documents_prefix_filter(settings):
    server = build_server(settings)
    tool = await _call_tool(server, "list_documents", {"prefix": "notes/"})
    assert [e["path"] for e in tool] == ["notes/alpha.md"]


@pytest.mark.asyncio
async def test_list_documents_title_extraction(settings):
    server = build_server(settings)
    tool = await _call_tool(server, "list_documents", {})
    titles = {e["path"]: e["title"] for e in tool}
    assert titles["notes/alpha.md"] == "Alpha Note"
    assert titles["beta.md"] == "Beta"


@pytest.mark.asyncio
async def test_get_document_parses_frontmatter(settings):
    server = build_server(settings)
    doc = await _call_tool(server, "get_document", {"path": "notes/alpha.md"})
    assert doc["title"] == "Alpha Note"
    assert doc["frontmatter"] == {"tags": ["alpha"]}
    assert "First body paragraph." in doc["content"]


@pytest.mark.asyncio
async def test_get_document_missing_returns_error(settings):
    server = build_server(settings)
    doc = await _call_tool(server, "get_document", {"path": "does-not-exist.md"})
    assert "error" in doc


@pytest.mark.asyncio
async def test_get_document_rejects_traversal(settings):
    server = build_server(settings)
    doc = await _call_tool(server, "get_document", {"path": "../outside.md"})
    assert "error" in doc


async def _call_tool(server, name: str, arguments: dict):
    """Invoke a FastMCP tool by name via the low-level call_tool API.

    FastMCP returns a list of content blocks; for JSON-returning tools the
    structured result is attached to the block via ``.structuredContent`` or
    parseable text. We prefer the raw Python return by looking it up through
    the registered tool callable.
    """
    # The simplest path: call the underlying function directly via the
    # FastMCP tool registry. This avoids round-tripping through the MCP
    # content-block machinery for unit tests.
    tool = server._tool_manager.get_tool(name)  # pyright: ignore[reportPrivateUsage]
    return tool.fn(**arguments)
