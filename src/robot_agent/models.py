"""Model provider boundary reused from the project's existing demo configuration."""

from __future__ import annotations

from typing import Any


def load_chat_model(streaming: bool = False) -> Any:
    """Reuse the existing OpenAI/Azure/Anthropic/Ollama provider selection."""
    from turtle_agent.scripts.llm import get_llm

    return get_llm(streaming=streaming)
