"""LLM wrapper module backed by model factory."""

from typing import Optional, List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from src.config.settings import settings
from src.llm.model_factory import get_chat_model


class _UnavailableLLM:
    def __init__(self, reason: str):
        self.reason = reason

    def invoke(self, _messages):
        raise RuntimeError(self.reason)

    async def ainvoke(self, _messages):
        raise RuntimeError(self.reason)

    def stream(self, _messages):
        raise RuntimeError(self.reason)

    async def astream(self, _messages):
        raise RuntimeError(self.reason)


class OllamaLLM:
    """Backward-compatible chat wrapper.

    Despite the historical name, this wrapper now supports both `ollama`
    and OpenAI-compatible providers. By default it resolves model instances
    through `src.llm.model_factory.get_chat_model`.
    """

    def __init__(
        self,
        model: str = "qwen-plus",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        num_ctx: int = 4096,
        agent_key: str = "GENERIC",
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.agent_key = agent_key

        # If explicit overrides are provided, build model directly.
        if provider or base_url or api_key:
            self._llm = self._build_from_overrides(
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                num_ctx=num_ctx,
                **kwargs,
            )
        else:
            self._llm = get_chat_model(
                agent_key=agent_key,
                default_model=model,
                temperature=temperature,
                num_ctx=num_ctx,
            )

        if self._llm is None:
            self._llm = _UnavailableLLM(
                f"LLM initialization failed for agent '{agent_key}'. "
                "Please check provider/model env config."
            )

    def _build_from_overrides(
        self,
        provider: Optional[str],
        model: str,
        base_url: Optional[str],
        api_key: Optional[str],
        temperature: float,
        num_ctx: int,
        **kwargs: Any,
    ) -> Any:
        resolved_provider = (provider or "").lower()
        if not resolved_provider:
            resolved_provider = (
                "openai"
                if (api_key or settings.has_openai_like_creds(self.agent_key))
                else "ollama"
            )

        if resolved_provider == "openai":
            from langchain_openai import ChatOpenAI

            resolved_api_key = (
                api_key
                or settings.get_agent_api_key(self.agent_key)
            )
            resolved_base_url = (
                base_url
                or settings.get_agent_base_url(self.agent_key, provider_hint="openai")
            )
            return ChatOpenAI(
                model=model,
                api_key=resolved_api_key,
                base_url=resolved_base_url,
                temperature=temperature,
                **kwargs,
            )

        from langchain_ollama import ChatOllama

        resolved_base_url = base_url or settings.get_agent_base_url(
            self.agent_key, provider_hint="ollama"
        )
        return ChatOllama(
            model=model,
            base_url=resolved_base_url,
            temperature=temperature,
            num_ctx=num_ctx,
            **kwargs,
        )

    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = self._build_messages(prompt, system_prompt)
        response = self._llm.invoke(messages)
        return response.content

    async def ainvoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = self._build_messages(prompt, system_prompt)
        response = await self._llm.ainvoke(messages)
        return response.content

    def chat(self, messages: List[Dict[str, str]]) -> str:
        langchain_messages = self._convert_messages(messages)
        response = self._llm.invoke(langchain_messages)
        return response.content

    async def achat(self, messages: List[Dict[str, str]]) -> str:
        langchain_messages = self._convert_messages(messages)
        response = await self._llm.ainvoke(langchain_messages)
        return response.content

    def stream(self, prompt: str, system_prompt: Optional[str] = None):
        messages = self._build_messages(prompt, system_prompt)
        for chunk in self._llm.stream(messages):
            yield chunk.content

    async def astream(self, prompt: str, system_prompt: Optional[str] = None):
        messages = self._build_messages(prompt, system_prompt)
        async for chunk in self._llm.astream(messages):
            yield chunk.content

    def _build_messages(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> List[BaseMessage]:
        messages: List[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        return messages

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        langchain_messages: List[BaseMessage] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))
        return langchain_messages

    @property
    def llm(self) -> Any:
        return self._llm


def get_llm(
    model: str = "qwen-plus",
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    agent_key: str = "GENERIC",
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> OllamaLLM:
    """Factory function to create provider-agnostic LLM wrapper."""
    return OllamaLLM(
        model=model,
        base_url=base_url,
        temperature=temperature,
        agent_key=agent_key,
        provider=provider,
        api_key=api_key,
        **kwargs,
    )
