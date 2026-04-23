from __future__ import annotations

from pathlib import Path

import pytest

from knowledge_mcp.config import Settings


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    """A vault directory with two sample markdown files."""
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "alpha.md").write_text(
        "---\ntags: [alpha]\n---\n\n# Alpha Note\n\nFirst body paragraph.\n",
        encoding="utf-8",
    )
    (tmp_path / "beta.md").write_text(
        "# Beta\n\nNo frontmatter here.\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def settings(tmp_vault: Path) -> Settings:
    return Settings(vault_path=tmp_vault)
