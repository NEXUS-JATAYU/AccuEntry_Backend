"""LLM provider configuration tests."""

import os
import sys
from unittest.mock import MagicMock, patch

# Ensure backend root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_agent_llm_ollama_default():
    with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}, clear=False):
        import importlib
        import llm_config

        importlib.reload(llm_config)
        llm = llm_config.AgentLLM().get_llm("decision_agent")
        assert llm.__class__.__name__ == "ChatOllama"


def test_agent_llm_gemini():
    with patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "gemini",
            "GOOGLE_API_KEY": "test-key",
            "GEMINI_MODEL": "gemini-2.0-flash",
        },
        clear=False,
    ):
        mock_genai = MagicMock()
        mock_cls = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "langchain_google_genai": MagicMock(ChatGoogleGenerativeAI=mock_cls),
            },
        ):
            import importlib
            import llm_config

            importlib.reload(llm_config)
            llm_config.AgentLLM().get_llm("decision_agent")
            mock_cls.assert_called_once()
