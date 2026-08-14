"""Self-contained chat-model provider boundary for the robot agent."""

from __future__ import annotations

import os
from typing import Any

import dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI


def load_chat_model(streaming: bool = False) -> Any:
    """Build the configured OpenAI, Azure, Anthropic, or Ollama chat model."""
    dotenv.load_dotenv(dotenv.find_dotenv())
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "openai":
        return ChatOpenAI(
            api_key=get_env_variable("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            streaming=streaming,
        )
    if provider == "azure":
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
        if api_version:
            return AzureChatOpenAI(
                api_key=get_env_variable("AZURE_OPENAI_API_KEY"),
                azure_endpoint=get_env_variable("AZURE_OPENAI_ENDPOINT"),
                api_version=api_version,
                azure_deployment=get_azure_deployment_name(),
                streaming=streaming,
            )
        return ChatOpenAI(
            api_key=get_env_variable("AZURE_OPENAI_API_KEY"),
            base_url=get_azure_v1_base_url(),
            model=get_azure_deployment_name(),
            streaming=streaming,
        )
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "LLM_PROVIDER=anthropic requires langchain-anthropic"
            ) from exc
        return ChatAnthropic(
            api_key=get_env_variable("ANTHROPIC_API_KEY"),
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            streaming=streaming,
        )
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "LLM_PROVIDER=ollama requires langchain-ollama"
            ) from exc
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            streaming=streaming,
        )
    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. "
        "Expected openai, azure, anthropic, or ollama."
    )


def get_azure_deployment_name() -> str:
    """Resolve the Azure deployment from Azure-specific or shared model vars."""
    deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_MODEL")
        or os.getenv("OPENAI_MODEL")
    )
    if deployment:
        return deployment
    raise ValueError(
        "Azure OpenAI requires AZURE_OPENAI_DEPLOYMENT, "
        "AZURE_OPENAI_MODEL, or OPENAI_MODEL."
    )


def get_azure_v1_base_url() -> str:
    """Normalize an Azure resource endpoint to its OpenAI-compatible v1 URL."""
    endpoint = get_env_variable("AZURE_OPENAI_ENDPOINT").rstrip("/")
    if endpoint.endswith("/openai/v1"):
        return endpoint
    return f"{endpoint}/openai/v1/"


def get_env_variable(name: str, allow_empty: bool = False) -> str:
    """Return a required environment variable with consistent validation."""
    value = os.getenv(name)
    if value is None or (not allow_empty and not value.strip()):
        raise ValueError(f"Environment variable {name} is not set.")
    return value
