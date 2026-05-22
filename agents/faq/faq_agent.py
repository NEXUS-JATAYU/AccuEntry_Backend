# agents/faq/faq_agent.py
from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from rag_service import build_faq_retrieval_query, retrieve_as_context
from state import OnboardingState
from prompts import SYSTEM_PROMPT
import os
import re

# Shown after decision/activation when the user may ask process questions via RAG.
POST_PROCESS_FAQ_INVITE = (
    "If you have any questions about this onboarding process, timelines, "
    "KYC/AML checks, or required documents, type your question here and I will "
    "answer from our policy guide."
)

# _FAQ_SYSTEM = """
# You are a helpful onboarding assistant for AccuEntry bank.
# Answer the user's question using ONLY the policy excerpts provided.
# If the answer is not in the excerpts, say you'll connect them with support.
# Keep answers under 3 sentences.
# """.strip()

def _extract_best_knowledge_answer(context: str, user_text: str) -> str:
    if not context:
        return ""

    blocks = [block.strip() for block in context.split("\n\n---\n\n") if block.strip()]
    user_tokens = set(re.findall(r"[a-z0-9]+", user_text.lower()))

    best_block = ""
    best_score = -1
    for block in blocks:
        block_tokens = set(re.findall(r"[a-z0-9]+", block.lower()))
        score = len(user_tokens & block_tokens)
        if score > best_score:
            best_score = score
            best_block = block

    if not best_block:
        return ""

    answer_lines = [line.strip() for line in best_block.splitlines() if line.strip()]
    for line in answer_lines:
        if line.upper().startswith("A:"):
            return line[2:].strip()

    return re.sub(r"^Q\d+:\s*", "", answer_lines[0]).strip()


async def faq_node(state: OnboardingState) -> dict:
    user_text = next(
        (m["text"] for m in reversed(state.get("messages", []))
         if m.get("role") == "user"), ""
    )
    if not user_text:
        return {}

    rag_query = build_faq_retrieval_query(
        user_text,
        stage=state.get("stage"),
        decision_action=state.get("decision_action"),
        aml_status=state.get("aml_status"),
    )
    policy_chunks = retrieve_as_context(rag_query, top_k=4)
    if not policy_chunks:
        return {
            "messages": [{
                "role": "assistant",
                "text": "I'm sorry, I couldn't find that information in the knowledge base. Please contact support or try rephrasing your question.",
            }]
        }

    formatted_prompt = SYSTEM_PROMPT.format(context=policy_chunks, question=user_text)
    
    try:
        llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "gemma2:2b"), temperature=0)
        response = await llm.ainvoke([
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=user_text),
        ])
        answer = str(getattr(response, "content", "") or "").strip()
    except Exception:
        answer = ""

    if not answer:
        answer = _extract_best_knowledge_answer(policy_chunks, user_text)

    if not answer:
        answer = "I'm sorry, I couldn't find that information in the knowledge base. Please contact support or try rephrasing your question."

    return {
        "messages": [{"role": "assistant", "text": answer}]
    }