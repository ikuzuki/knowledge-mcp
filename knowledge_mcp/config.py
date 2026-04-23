"""Configuration loaded from env vars and optional knowledge_mcp.toml."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KNOWLEDGE_MCP_",
        env_file=".env",
        extra="ignore",
    )

    vault_path: Path = Field(..., description="Absolute path to the markdown vault root.")
    embedding_model: str = Field(default="nomic-embed-text:v1.5")
    embedding_endpoint: str = Field(default="http://localhost:11434/api/embeddings")
    chunk_size: int = Field(default=512, ge=64)
    chunk_overlap: int = Field(default=50, ge=0)
    rrf_k: int = Field(default=60, ge=1)

    @field_validator("vault_path")
    @classmethod
    def _vault_must_exist(cls, v: Path) -> Path:
        v = v.expanduser().resolve()
        if not v.exists() or not v.is_dir():
            raise ValueError(f"vault_path does not exist or is not a directory: {v}")
        return v


def load_settings() -> Settings:
    """Entry point for loading settings.

    Env vars take the form ``KNOWLEDGE_MCP_VAULT_PATH`` etc. A ``.env`` file in the
    working directory is also honoured. A TOML file at
    ``$KNOWLEDGE_MCP_CONFIG`` (or ``./knowledge_mcp.toml``) is loaded if present and
    its keys are merged under env precedence.
    """
    import os
    import tomllib

    toml_path_str = os.environ.get("KNOWLEDGE_MCP_CONFIG", "knowledge_mcp.toml")
    toml_path = Path(toml_path_str)
    overrides: dict = {}
    if toml_path.exists():
        with toml_path.open("rb") as f:
            overrides = tomllib.load(f)
    return Settings(**overrides)
