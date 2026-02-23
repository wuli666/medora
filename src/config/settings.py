"""
Application-wide settings using pydantic-settings.
All runtime env access in src/ should go through this module.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

_DASHSCOPE_COMPAT_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class Settings(BaseSettings):
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    LOG_DIR: str = "./logs"
    LOG_FILE_NAME: str = "app.debug.log"
    LOG_FILE_WHEN: str = "midnight"
    LOG_FILE_INTERVAL: int = 1
    LOG_FILE_BACKUP_COUNT: int = 7
    LOG_FILE_ENCODING: str = "utf-8"
    LOG_FILE_LEVEL: str = "DEBUG"

    # Global LLM settings
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    DASHSCOPE_API_KEY: str = ""
    DASHSCOPE_BASE_URL: str = _DASHSCOPE_COMPAT_DEFAULT_BASE_URL
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Runtime
    AGENT_LOG_TRUNCATE: int = 600

    # Search tools
    TAVILY_API_KEY: str = ""
    CHROMA_DIR: str = "./data/chroma"
    WEB_SEARCH_MAX_RESULTS: int = 5
    RAG_TOP_K: int = 3

    # Database
    DB_PATH: str = "./data/patients.db"

    # Agent-specific overrides
    SUPERVISOR_PROVIDER: str = ""
    SUPERVISOR_MODEL: str = ""

    TOOLER_TEXT_PROVIDER: str = ""
    TOOLER_TEXT_MODEL: str = ""

    TOOLER_IMAGE_PROVIDER: str = ""
    TOOLER_IMAGE_MODEL: str = ""

    TOOLER_MERGE_PROVIDER: str = ""
    TOOLER_MERGE_MODEL: str = ""

    SEARCHER_PROVIDER: str = ""
    SEARCHER_MODEL: str = ""

    PLANNER_PROVIDER: str = ""
    PLANNER_MODEL: str = ""

    REFLECTOR_PROVIDER: str = ""
    REFLECTOR_MODEL: str = ""

    SUMMARIZER_PROVIDER: str = ""
    SUMMARIZER_MODEL: str = ""

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    def _agent_value(self, agent_key: str, suffix: str) -> str:
        key = (agent_key or "").strip().upper()
        if not key:
            return ""
        return str(getattr(self, f"{key}_{suffix}", "") or "").strip()

    def get_agent_model(self, agent_key: str, default_model: str) -> str:
        return self._agent_value(agent_key, "MODEL") or default_model

    def get_agent_provider(self, agent_key: str) -> str:
        return self._agent_value(agent_key, "PROVIDER")

    def get_agent_api_key(self, agent_key: str) -> str:
        _ = agent_key
        return (
            self.OPENAI_API_KEY
            or self.DASHSCOPE_API_KEY
        )

    def get_agent_base_url(self, agent_key: str, provider_hint: str = "") -> str:
        _ = agent_key
        hint = (provider_hint or "").strip().lower()
        if hint == "ollama":
            return self.OLLAMA_BASE_URL
        return (
            self.OPENAI_BASE_URL
            or self.DASHSCOPE_BASE_URL
            or _DASHSCOPE_COMPAT_DEFAULT_BASE_URL
        )

    def has_openai_like_creds(self, agent_key: str) -> bool:
        return bool(self.get_agent_api_key(agent_key))


settings = Settings()
