"""LLM module for Ollama integration."""

from typing import Optional, List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage


class OllamaLLM:
    """Wrapper for Ollama LLM using LangChain."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        num_ctx: int = 4096,
        **kwargs: Any,
    ):
        """Initialize Ollama LLM.

        Args:
            model: Model name to use (e.g., "llama3", "mistral", "medgemma").
            base_url: Ollama server URL.
            temperature: Sampling temperature (0.0 to 1.0).
            num_ctx: Context window size.
            **kwargs: Additional parameters passed to ChatOllama.
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.num_ctx = num_ctx

        self._llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_ctx=num_ctx,
            **kwargs,
        )

    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Invoke LLM with a prompt.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.

        Returns:
            Generated response text.
        """
        messages = self._build_messages(prompt, system_prompt)
        response = self._llm.invoke(messages)
        return response.content

    async def ainvoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Async invoke LLM with a prompt.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.

        Returns:
            Generated response text.
        """
        messages = self._build_messages(prompt, system_prompt)
        response = await self._llm.ainvoke(messages)
        return response.content

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat with message history.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                     Roles can be "system", "user", or "assistant".

        Returns:
            Generated response text.
        """
        langchain_messages = self._convert_messages(messages)
        response = self._llm.invoke(langchain_messages)
        return response.content

    async def achat(self, messages: List[Dict[str, str]]) -> str:
        """Async chat with message history.

        Args:
            messages: List of message dicts with "role" and "content" keys.

        Returns:
            Generated response text.
        """
        langchain_messages = self._convert_messages(messages)
        response = await self._llm.ainvoke(langchain_messages)
        return response.content

    def stream(self, prompt: str, system_prompt: Optional[str] = None):
        """Stream LLM response.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.

        Yields:
            Response chunks.
        """
        messages = self._build_messages(prompt, system_prompt)
        for chunk in self._llm.stream(messages):
            yield chunk.content

    async def astream(self, prompt: str, system_prompt: Optional[str] = None):
        """Async stream LLM response.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.

        Yields:
            Response chunks.
        """
        messages = self._build_messages(prompt, system_prompt)
        async for chunk in self._llm.astream(messages):
            yield chunk.content

    def _build_messages(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> List[BaseMessage]:
        """Build message list from prompt and optional system prompt."""
        messages: List[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        return messages

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        """Convert dict messages to LangChain message objects."""
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
    def llm(self) -> ChatOllama:
        """Get underlying ChatOllama instance."""
        return self._llm


def get_llm(
    model: str = "llama3",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    **kwargs: Any,
) -> OllamaLLM:
    """Factory function to create OllamaLLM instance.

    Args:
        model: Model name.
        base_url: Ollama server URL.
        temperature: Sampling temperature.
        **kwargs: Additional parameters.

    Returns:
        OllamaLLM instance.
    """
    return OllamaLLM(model=model, base_url=base_url, temperature=temperature, **kwargs)
