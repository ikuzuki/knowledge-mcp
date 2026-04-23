"""Tests for `knowledge_mcp.chunking`."""

from __future__ import annotations

from knowledge_mcp.chunking import chunk_markdown


def test_two_h2_sections_produce_two_chunks() -> None:
    md = (
        "## Alpha\n"
        "Alpha body content.\n\n"
        "## Beta\n"
        "Beta body content.\n"
    )
    chunks = chunk_markdown(md)
    assert len(chunks) == 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].heading_path == ["Alpha"]
    assert chunks[1].heading_path == ["Beta"]
    assert all(c.token_count > 0 for c in chunks)


def test_nested_headings_preserve_full_chain() -> None:
    md = (
        "# Top\n"
        "Top intro.\n\n"
        "## Middle\n"
        "Middle intro.\n\n"
        "### Deep\n"
        "Deep content goes here.\n"
    )
    chunks = chunk_markdown(md)
    deep_chunks = [c for c in chunks if c.text.startswith("Deep content")]
    assert len(deep_chunks) == 1
    assert deep_chunks[0].heading_path == ["Top", "Middle", "Deep"]


def test_content_before_first_heading_has_empty_heading_path() -> None:
    md = (
        "Preamble text before any heading.\n\n"
        "## Section\n"
        "Section body.\n"
    )
    chunks = chunk_markdown(md)
    assert chunks[0].heading_path == []
    assert chunks[0].chunk_index == 0
    assert "Preamble" in chunks[0].text


def test_oversized_section_triggers_fallback_splitter() -> None:
    # ~2000 tokens of filler — "word " repeats produce ~1 token each.
    filler = ("word " * 2000).strip()
    md = f"## Big\n{filler}\n"
    chunks = chunk_markdown(md, max_tokens=256, overlap_tokens=20)
    assert len(chunks) > 1
    assert all(c.heading_path == ["Big"] for c in chunks)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))
    assert all(c.token_count <= 256 for c in chunks)


def test_empty_string_returns_empty_list() -> None:
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  ") == []
