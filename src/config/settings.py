"""Application settings loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"


class Settings(BaseSettings):
    """Runtime settings for MedGemma assistant."""

    # Ollama / model configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MEDGEMMA_MODEL: str = "MedAIBase/MedGemma1.5:4b"
    MEDGEMMA_IMAGE_MODEL: str = "MedAIBase/MedGemma1.5:4b"
    MEDGEMMA_REFLECT_MODEL: str = "MedAIBase/MedGemma1.5:4b"
    MEDGEMMA_TEMPERATURE: float = 0.7
    MEDGEMMA_IMAGE_TEMPERATURE: float = 0.0
    MEDGEMMA_NUM_CTX: int = 4096

    # Workflow defaults
    MAX_REFLECT_LOOPS: int = 2
    USE_LLM_REFLECTION: bool = True

    # External APIs
    TAVILY_API_KEY: str | None = None

    # Tavily tool defaults
    TAVILY_MAX_RESULTS: int = 5
    TAVILY_SEARCH_DEPTH: str = "basic"

    # Image explain tool defaults
    IMAGE_EXPLAIN_MAX_BYTES: int = 10_000_000

    # Tool node execution defaults
    TOOL_NODE_MAX_CONCURRENCY: int = 3

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
