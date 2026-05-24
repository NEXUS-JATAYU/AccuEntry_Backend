"""
Centralised LLM configuration for all agents.

Supports Gemini (production / GCP) and Ollama (local dev fallback).
"""

from __future__ import annotations

import os
from typing import Any

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("AGENT_LLM_MODEL", "llama3.2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GOOGLE_API_KEY = (
    os.getenv("BACKEND_GOOGLE_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or ""
).strip()

_AGENT_OVERRIDES: dict[str, dict] = {
    "decision_agent": {"temperature": 0.1, "model": "llama3.2"},
    "data_capture": {"temperature": 0, "model": os.getenv("OLLAMA_MODEL", "gemma2:2b")},
    "faq": {"temperature": 0},
    "aml": {"temperature": 0.1},
}


def _resolve_model(agent_name: str) -> str:
    overrides = _AGENT_OVERRIDES.get(agent_name, {})
    if LLM_PROVIDER == "gemini":
        return GEMINI_MODEL
    return overrides.get("model", DEFAULT_MODEL)


def _resolve_temperature(agent_name: str) -> float:
    overrides = _AGENT_OVERRIDES.get(agent_name, {})
    return float(overrides.get("temperature", 0.1))


class AgentLLM:

    def get_llm(self, agent_name: str) -> Any:
        model = _resolve_model(agent_name)
        temperature = _resolve_temperature(agent_name)

        if LLM_PROVIDER == "gemini":
            if not GOOGLE_API_KEY:
                raise RuntimeError("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=GOOGLE_API_KEY,
                temperature=temperature,
            )

        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
        )


def generate_text(prompt: str, *, model: str | None = None, timeout: int = 30) -> str:
    """
    Non-streaming text generation for fraud check and similar callers.
    Uses Gemini or Ollama depending on LLM_PROVIDER.
    """
    resolved = model or os.getenv("FRAUD_LLM_MODEL") or (
        GEMINI_MODEL if LLM_PROVIDER == "gemini" else os.getenv("OLLAMA_MODEL", "gemma2:2b")
    )

    if LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")
        import google.generativeai as genai

        genai.configure(api_key=GOOGLE_API_KEY)
        gemini = genai.GenerativeModel(resolved)
        response = gemini.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 256,
            },
        )
        return (response.text or "").strip()

    import json
    import urllib.request

    payload = json.dumps({
        "model": resolved,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 256},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body["response"]
