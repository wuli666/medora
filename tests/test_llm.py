"""Tests for OllamaLLM module."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.llm.llm import OllamaLLM, get_llm


class TestOllamaLLM:
    """Test cases for OllamaLLM class."""

    @patch("src.llm.llm.ChatOllama")
    def test_init_default_params(self, mock_chat_ollama):
        """Test initialization with default parameters."""
        llm = OllamaLLM()

        assert llm.model == "llama3"
        assert llm.base_url == "http://localhost:11434"
        assert llm.temperature == 0.7
        assert llm.num_ctx == 4096

        mock_chat_ollama.assert_called_once_with(
            model="llama3",
            base_url="http://localhost:11434",
            temperature=0.7,
            num_ctx=4096,
        )

    @patch("src.llm.llm.ChatOllama")
    def test_init_custom_params(self, mock_chat_ollama):
        """Test initialization with custom parameters."""
        llm = OllamaLLM(
            model="medgemma",
            base_url="http://192.168.1.100:11434",
            temperature=0.5,
            num_ctx=8192,
        )

        assert llm.model == "medgemma"
        assert llm.base_url == "http://192.168.1.100:11434"
        assert llm.temperature == 0.5
        assert llm.num_ctx == 8192

    @patch("src.llm.llm.ChatOllama")
    def test_invoke_simple(self, mock_chat_ollama):
        """Test simple invoke without system prompt."""
        mock_response = MagicMock()
        mock_response.content = "Hello, how can I help you?"
        mock_chat_ollama.return_value.invoke.return_value = mock_response

        llm = OllamaLLM()
        result = llm.invoke("Hello")

        assert result == "Hello, how can I help you?"
        mock_chat_ollama.return_value.invoke.assert_called_once()

        call_args = mock_chat_ollama.return_value.invoke.call_args[0][0]
        assert len(call_args) == 1
        assert isinstance(call_args[0], HumanMessage)
        assert call_args[0].content == "Hello"

    @patch("src.llm.llm.ChatOllama")
    def test_invoke_with_system_prompt(self, mock_chat_ollama):
        """Test invoke with system prompt."""
        mock_response = MagicMock()
        mock_response.content = "I'm a medical assistant."
        mock_chat_ollama.return_value.invoke.return_value = mock_response

        llm = OllamaLLM()
        result = llm.invoke("Who are you?", system_prompt="You are a medical assistant.")

        assert result == "I'm a medical assistant."

        call_args = mock_chat_ollama.return_value.invoke.call_args[0][0]
        assert len(call_args) == 2
        assert isinstance(call_args[0], SystemMessage)
        assert call_args[0].content == "You are a medical assistant."
        assert isinstance(call_args[1], HumanMessage)
        assert call_args[1].content == "Who are you?"

    @patch("src.llm.llm.ChatOllama")
    def test_chat_with_history(self, mock_chat_ollama):
        """Test chat with message history."""
        mock_response = MagicMock()
        mock_response.content = "You should rest and drink water."
        mock_chat_ollama.return_value.invoke.return_value = mock_response

        llm = OllamaLLM()
        messages = [
            {"role": "system", "content": "You are a doctor."},
            {"role": "user", "content": "I have a headache."},
            {"role": "assistant", "content": "How long have you had it?"},
            {"role": "user", "content": "Since this morning."},
        ]
        result = llm.chat(messages)

        assert result == "You should rest and drink water."

        call_args = mock_chat_ollama.return_value.invoke.call_args[0][0]
        assert len(call_args) == 4
        assert isinstance(call_args[0], SystemMessage)
        assert isinstance(call_args[1], HumanMessage)
        assert isinstance(call_args[2], AIMessage)
        assert isinstance(call_args[3], HumanMessage)

    @patch("src.llm.llm.ChatOllama")
    def test_stream(self, mock_chat_ollama):
        """Test streaming response."""
        chunk1 = MagicMock()
        chunk1.content = "Hello"
        chunk2 = MagicMock()
        chunk2.content = " world"
        chunk3 = MagicMock()
        chunk3.content = "!"

        mock_chat_ollama.return_value.stream.return_value = iter([chunk1, chunk2, chunk3])

        llm = OllamaLLM()
        chunks = list(llm.stream("Hi"))

        assert chunks == ["Hello", " world", "!"]

    @patch("src.llm.llm.ChatOllama")
    def test_llm_property(self, mock_chat_ollama):
        """Test llm property returns underlying ChatOllama instance."""
        llm = OllamaLLM()
        assert llm.llm == mock_chat_ollama.return_value


class TestOllamaLLMAsync:
    """Async test cases for OllamaLLM class."""

    @patch("src.llm.llm.ChatOllama")
    def test_ainvoke(self, mock_chat_ollama):
        """Test async invoke."""
        import asyncio

        mock_response = MagicMock()
        mock_response.content = "Async response"
        mock_chat_ollama.return_value.ainvoke = AsyncMock(return_value=mock_response)

        llm = OllamaLLM()
        result = asyncio.run(llm.ainvoke("Hello"))

        assert result == "Async response"
        mock_chat_ollama.return_value.ainvoke.assert_called_once()

    @patch("src.llm.llm.ChatOllama")
    def test_achat(self, mock_chat_ollama):
        """Test async chat."""
        import asyncio

        mock_response = MagicMock()
        mock_response.content = "Async chat response"
        mock_chat_ollama.return_value.ainvoke = AsyncMock(return_value=mock_response)

        llm = OllamaLLM()
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = asyncio.run(llm.achat(messages))

        assert result == "Async chat response"

    @patch("src.llm.llm.ChatOllama")
    def test_astream(self, mock_chat_ollama):
        """Test async streaming."""
        import asyncio

        chunk1 = MagicMock()
        chunk1.content = "Async"
        chunk2 = MagicMock()
        chunk2.content = " stream"

        async def async_gen():
            for chunk in [chunk1, chunk2]:
                yield chunk

        mock_chat_ollama.return_value.astream = MagicMock(return_value=async_gen())

        async def run_test():
            llm = OllamaLLM()
            chunks = []
            async for chunk in llm.astream("Test"):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run_test())
        assert chunks == ["Async", " stream"]


class TestGetLLM:
    """Test cases for get_llm factory function."""

    @patch("src.llm.llm.ChatOllama")
    def test_get_llm_default(self, mock_chat_ollama):
        """Test get_llm with default parameters."""
        llm = get_llm()

        assert isinstance(llm, OllamaLLM)
        assert llm.model == "llama3"

    @patch("src.llm.llm.ChatOllama")
    def test_get_llm_custom(self, mock_chat_ollama):
        """Test get_llm with custom parameters."""
        llm = get_llm(model="medgemma", temperature=0.3)

        assert isinstance(llm, OllamaLLM)
        assert llm.model == "medgemma"
        assert llm.temperature == 0.3


class TestMessageConversion:
    """Test cases for message conversion methods."""

    @patch("src.llm.llm.ChatOllama")
    def test_convert_messages_all_roles(self, mock_chat_ollama):
        """Test conversion of all message roles."""
        llm = OllamaLLM()
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant message"},
        ]

        result = llm._convert_messages(messages)

        assert len(result) == 3
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "System message"
        assert isinstance(result[1], HumanMessage)
        assert result[1].content == "User message"
        assert isinstance(result[2], AIMessage)
        assert result[2].content == "Assistant message"

    @patch("src.llm.llm.ChatOllama")
    def test_convert_messages_unknown_role(self, mock_chat_ollama):
        """Test conversion of unknown role defaults to HumanMessage."""
        llm = OllamaLLM()
        messages = [
            {"role": "unknown", "content": "Unknown role message"},
        ]

        result = llm._convert_messages(messages)

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)

    @patch("src.llm.llm.ChatOllama")
    def test_convert_messages_missing_role(self, mock_chat_ollama):
        """Test conversion when role is missing defaults to user."""
        llm = OllamaLLM()
        messages = [
            {"content": "No role message"},
        ]

        result = llm._convert_messages(messages)

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "No role message"

    @patch("src.llm.llm.ChatOllama")
    def test_build_messages_without_system(self, mock_chat_ollama):
        """Test building messages without system prompt."""
        llm = OllamaLLM()
        result = llm._build_messages("Hello")

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)

    @patch("src.llm.llm.ChatOllama")
    def test_build_messages_with_system(self, mock_chat_ollama):
        """Test building messages with system prompt."""
        llm = OllamaLLM()
        result = llm._build_messages("Hello", system_prompt="Be helpful")

        assert len(result) == 2
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
