"""
Centralised LLM configuration for all agents.

Returns a ChatOllama instance per agent with sensible defaults.
"""

from __future__ import annotations

import os

from langchain_ollama import ChatOllama

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("AGENT_LLM_MODEL", "gemma2:2b")

_AGENT_OVERRIDES: dict[str, dict] = {
    "decision_agent": {"temperature": 0.1, "model": "llama3.2"},
}


class AgentLLM:

    def get_llm(self, agent_name: str) -> ChatOllama:
        overrides = _AGENT_OVERRIDES.get(agent_name, {})
        return ChatOllama(
            model=overrides.get("model", DEFAULT_MODEL),
            base_url=OLLAMA_BASE_URL,
            temperature=overrides.get("temperature", 0.1),
        )
