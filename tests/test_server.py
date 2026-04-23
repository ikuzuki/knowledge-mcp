"""Server smoke tests covering v0 and v1 tool surfaces."""

from __future__ import annotations

import pytest

from knowledge_mcp.server import build_server


def _tool(server, name: str):
    return server._tool_manager.get_tool(name).fn  # pyright: ignore[reportPrivateUsage]


# ---------- v0 ----------


def test_list_documents_returns_both(settings):
    server, deps = build_server(settings, start_indexing=False)
    try:
        result = _tool(server, "list_documents")()
        paths = {entry["path"] for entry in result}
        assert paths == {"beta.md", "notes/alpha.md"}
    finally:
        deps.shutdown()


def test_list_documents_prefix_filter(settings):
    server, deps = build_server(settings, start_indexing=False)
    try:
        result = _tool(server, "list_documents")(prefix="notes/")
        assert [e["path"] for e in result] == ["notes/alpha.md"]
    finally:
        deps.shutdown()


def test_list_documents_title_extraction(settings):
    server, deps = build_server(settings, start_indexing=False)
    try:
        result = _tool(server, "list_documents")()
        titles = {e["path"]: e["title"] for e in result}
        assert titles["notes/alpha.md"] == "Alpha Note"
        assert titles["beta.md"] == "Beta"
    finally:
        deps.shutdown()


def test_get_document_parses_frontmatter(settings):
    server, deps = build_server(settings, start_indexing=False)
    try:
        doc = _tool(server, "get_document")(path="notes/alpha.md")
        assert doc["title"] == "Alpha Note"
        assert doc["frontmatter"] == {"tags": ["alpha"]}
        assert "First body paragraph." in doc["content"]
    finally:
        deps.shutdown()


def test_get_document_missing_returns_error(settings):
    server, deps = build_server(settings, start_indexing=False)
    try:
        doc = _tool(server, "get_document")(path="does-not-exist.md")
        assert "error" in doc
    finally:
        deps.shutdown()


def test_get_document_rejects_traversal(settings):
    server, deps = build_server(settings, start_indexing=False)
    try:
        doc = _tool(server, "get_document")(path="../outside.md")
        assert "error" in doc
    finally:
        deps.shutdown()


# ---------- v1 ----------


def test_create_document_writes_and_errors_on_duplicate(settings):
    server, deps = build_server(settings, start_indexing=False)
    try:
        create = _tool(server, "create_document")
        res = create(path="new/fresh.md", content="# Fresh\n\nBody.")
        assert res.get("path") == "new/fresh.md"
        assert "created_at" in res
        on_disk = (settings.vault_path / "new" / "fresh.md").read_text(encoding="utf-8")
        assert "# Fresh" in on_disk

        # Second create must fail.
        dup = create(path="new/fresh.md", content="# Dup")
        assert "error" in dup
    finally:
        deps.shutdown()


def test_update_document_atomic_and_requires_existing(settings):
    server, deps = build_server(settings, start_indexing=False)
    try:
        update = _tool(server, "update_document")
        missing = update(path="nope.md", content="x")
        assert "error" in missing

        res = update(path="beta.md", content="# Beta v2\n\nNew body.")
        assert res.get("path") == "beta.md"
        content = (settings.vault_path / "beta.md").read_text(encoding="utf-8")
        assert "Beta v2" in content
        # No stray .tmp file left behind.
        assert not (settings.vault_path / "beta.md.tmp").exists()
    finally:
        deps.shutdown()


def test_search_returns_bm25_hits(settings):
    server, deps = build_server(settings, start_indexing=False)
    try:
        # Seed the FTS index directly via the indexer.
        deps.indexer.reindex_all()
        hits = _tool(server, "search")(query="paragraph", limit=5)
        assert isinstance(hits, list)
        assert hits, "expected at least one hit"
        assert "path" in hits[0]
        assert "score" in hits[0]
    finally:
        deps.shutdown()


@pytest.mark.parametrize("bad", ["../outside.md", "/abs.md", "no-suffix"])
def test_create_document_rejects_bad_paths(settings, bad):
    server, deps = build_server(settings, start_indexing=False)
    try:
        res = _tool(server, "create_document")(path=bad, content="x")
        assert "error" in res
    finally:
        deps.shutdown()
