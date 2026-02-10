import json
import os
from abc import ABC, abstractmethod
from typing import Any
from urllib import error, request


def _normalize_model_name(model: str) -> str:
    return (model or "").strip().lower()


def _has_openai_creds(agent_key: str) -> bool:
    key = agent_key.upper()
    return bool(
        os.getenv(f"{key}_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )


class BaseModelProvider(ABC):
    """Abstract provider contract for chat model creation."""

    name: str = "base"

    @abstractmethod
    def is_available(self, agent_key: str, model: str) -> bool:
        """Whether this provider can serve the given agent/model."""

    @abstractmethod
    def create(
        self,
        agent_key: str,
        model: str,
        temperature: float,
        num_ctx: int,
    ) -> Any:
        """Create provider-specific langchain chat model instance."""


class OpenAIProvider(BaseModelProvider):
    name = "openai"

    def is_available(self, agent_key: str, model: str) -> bool:
        return _has_openai_creds(agent_key)

    def create(
        self,
        agent_key: str,
        model: str,
        temperature: float,
        num_ctx: int,
    ) -> Any:
        from langchain_openai import ChatOpenAI

        key = agent_key.upper()
        api_key = (
            os.getenv(f"{key}_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY", "")
        )
        base_url = (
            os.getenv(f"{key}_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "temperature": temperature,
            "max_retries": 1,
        }
        return ChatOpenAI(
            **kwargs
        )


class OllamaProvider(BaseModelProvider):
    name = "ollama"

    def _base_url(self, agent_key: str) -> str:
        key = agent_key.upper()
        return (
            os.getenv(f"{key}_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )

    def _model_exists(self, base_url: str, model: str) -> bool:
        tags_url = f"{base_url.rstrip('/')}/api/tags"
        try:
            with request.urlopen(tags_url, timeout=1.5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (error.URLError, error.HTTPError, TimeoutError, ValueError):
            return False

        names = {
            _normalize_model_name(item.get("name", ""))
            for item in payload.get("models", [])
        }
        wanted = _normalize_model_name(model)
        if wanted in names:
            return True
        return ":" not in wanted and f"{wanted}:latest" in names

    def is_available(self, agent_key: str, model: str) -> bool:
        return self._model_exists(self._base_url(agent_key), model)

    def create(
        self,
        agent_key: str,
        model: str,
        temperature: float,
        num_ctx: int,
    ) -> Any:
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = {
            "model": model,
            "base_url": self._base_url(agent_key),
            "temperature": temperature,
            "num_ctx": num_ctx,
        }
        return ChatOllama(**kwargs)


class ModelFactory:
    """Provider registry + resolution strategy."""

    def __init__(self) -> None:
        self.providers: dict[str, BaseModelProvider] = {
            OpenAIProvider.name: OpenAIProvider(),
            OllamaProvider.name: OllamaProvider(),
        }

    def _resolve_provider(
        self,
        agent_key: str,
        model: str,
        explicit_provider: str,
    ) -> BaseModelProvider:
        provider_name = (explicit_provider or "").strip().lower()
        if provider_name and provider_name != "auto":
            provider = self.providers.get(provider_name)
            if provider is None:
                raise ValueError(f"Unknown provider: {provider_name}")
            return provider

        # Auto strategy: prefer local Ollama if model exists, else OpenAI-compatible.
        ollama = self.providers["ollama"]
        if ollama.is_available(agent_key, model):
            return ollama
        if self.providers["openai"].is_available(agent_key, model):
            return self.providers["openai"]
        # Final fallback keeps previous behavior.
        return ollama

    def create_chat_model(
        self,
        agent_key: str,
        default_model: str,
        temperature: float,
        num_ctx: int,
    ) -> Any:
        key = agent_key.upper()
        model = os.getenv(f"{key}_MODEL") or default_model
        explicit_provider = os.getenv(f"{key}_PROVIDER") or ""
        provider = self._resolve_provider(
            agent_key=agent_key,
            model=model,
            explicit_provider=explicit_provider,
        )
        return provider.create(
            agent_key=agent_key,
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
        )


_FACTORY = ModelFactory()


def get_chat_model(
    agent_key: str,
    default_model: str = "qwen-plus",
    temperature: float = 0.3,
    num_ctx: int = 8192,
) -> Any | None:
    """Backward-compatible entrypoint."""
    try:
        return _FACTORY.create_chat_model(
            agent_key=agent_key,
            default_model=default_model,
            temperature=temperature,
            num_ctx=num_ctx,
        )
    except Exception:
        return None
