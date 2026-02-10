"""LLM module."""

try:
    from src.llm.llm import OllamaLLM, get_llm
except Exception:  # pragma: no cover - optional runtime dependency
    OllamaLLM = None
    get_llm = None

__all__ = ["OllamaLLM", "get_llm"]
