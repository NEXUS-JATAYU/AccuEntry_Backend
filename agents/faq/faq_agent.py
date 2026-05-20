# agents/faq/faq_agent.py
from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from rag_service import retrieve_as_context
from state import OnboardingState
from prompts import SYSTEM_PROMPT
import os

# _FAQ_SYSTEM = """
# You are a helpful onboarding assistant for AccuEntry bank.
# Answer the user's question using ONLY the policy excerpts provided.
# If the answer is not in the excerpts, say you'll connect them with support.
# Keep answers under 3 sentences.
# """.strip()

async def faq_node(state: OnboardingState) -> dict:
    user_text = next(
        (m["text"] for m in reversed(state.get("messages", []))
         if m.get("role") == "user"), ""
    )
    if not user_text:
        return {}

    policy_chunks = retrieve_as_context(user_text, top_k=4)
    
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "gemma2:2b"), temperature=0)
    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Policy excerpts:\n{policy_chunks}\n\nUser question: {user_text}"),
    ])
    return {
        "messages": [{"role": "assistant", "text": response.content}]
    }