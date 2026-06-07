# agents/faq/faq_agent.py
from __future__ import annotations
import re

from rag_service import build_faq_retrieval_query, retrieve_as_context
from state import OnboardingState

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


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 2}


def _extract_qa_pairs(context: str) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    current_question = ""
    current_answer_lines: list[str] = []

    def flush_current_pair() -> None:
        if current_question and current_answer_lines:
            pairs.append({
                "question": current_question.strip(),
                "answer": " ".join(current_answer_lines).strip(),
            })

    for raw_line in context.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line == "---":
            continue

        if re.match(r"^Q\d+:\s*", line, flags=re.IGNORECASE):
            flush_current_pair()
            current_question = re.sub(r"^Q\d+:\s*", "", line, flags=re.IGNORECASE).strip()
            current_answer_lines = []
            continue

        if re.match(r"^A:\s*", line, flags=re.IGNORECASE):
            answer_line = re.sub(r"^A:\s*", "", line, flags=re.IGNORECASE).strip()
            if current_question:
                current_answer_lines.append(answer_line)
            continue

        if current_question and current_answer_lines:
            current_answer_lines.append(line)

    flush_current_pair()
    return pairs


def _score_pair(user_tokens: set[str], user_text: str, pair: dict[str, str]) -> int:
    def _opening_phrase(text: str) -> str:
        words = re.findall(r"[a-z0-9]+", (text or "").lower())
        return " ".join(words[:2])

    question_tokens = _tokenize(pair.get("question", ""))
    answer_tokens = _tokenize(pair.get("answer", ""))
    score = (len(user_tokens & question_tokens) * 4) + len(user_tokens & answer_tokens)
    if _opening_phrase(pair.get("question", "")) == _opening_phrase(user_text):
        score += 2
    return score


def _build_grounded_answer(context: str, user_text: str, *, status_info: str) -> str:
    pairs = _extract_qa_pairs(context)
    if not pairs:
        return ""

    user_tokens = _tokenize(user_text)
    scored_pairs = [
        (_score_pair(user_tokens, user_text, pair), pair)
        for pair in pairs
    ]
    scored_pairs = [item for item in scored_pairs if item[0] > 0]
    if not scored_pairs:
        return ""

    scored_pairs.sort(key=lambda item: (item[0], len(item[1].get("answer", ""))), reverse=True)

    selected_pairs: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    for _, pair in scored_pairs:
        question_key = pair.get("question", "").lower()
        if question_key in seen_questions:
            continue
        seen_questions.add(question_key)
        selected_pairs.append(pair)
        if len(selected_pairs) == 3:
            break

    if not selected_pairs:
        return ""

    primary_pair = selected_pairs[0]
    lines = [
        "### Relevant guidance",
        f"- {primary_pair['answer']}",
    ]

    if len(selected_pairs) > 1:
        lines.extend([
            "",
            "### Related policy notes",
        ])
        for extra_pair in selected_pairs[1:]:
            lines.append(f"- {extra_pair['question']}: {extra_pair['answer']}")

    lines.extend([
        "",
        "### Next step",
        f"- Based on your current status, {status_info.split('(', 1)[0].strip().lower()}. If you want, send the exact step or document name and I can narrow it down further.",
    ])
    return "\n".join(lines).strip()

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
    policy_chunks = retrieve_as_context(rag_query, top_k=8)
    if not policy_chunks:
        return {
            "messages": [{
                "role": "assistant",
                "text": "I'm sorry, I couldn't find that information in the knowledge base. Please contact support or try rephrasing your question.",
            }]
        }

    stage = state.get("stage", "complete")
    if stage == "complete":
        status_info = "COMPLETED / ACTIVATED (Your account is active and ready for use. Celebrate this with the user and provide details on next steps, app download, card delivery, net banking, etc.)"
    elif stage == "rejected":
        status_info = "REJECTED (Your application was declined following compliance/KYC/AML checks. Offer advice empathetically, explain potential reasons like name mismatch or blurry uploads, and guide them on how to reapply or appeal.)"
    elif stage == "manual_review":
        status_info = "PENDING / MANUAL REVIEW (Your application is under manual review by a compliance officer. Usually takes 24-48 hours. Advise them to wait patiently.)"
    elif stage == "pending_docs":
        status_info = "PENDING DOCUMENTS (Your application is pending clean document uploads. Advise them on document requirements and clarity.)"
    elif stage == "escalated":
        status_info = "ESCALATED (Your application is escalated for advanced compliance review. Advise them on typical wait times.)"
    else:
        status_info = f"IN PROGRESS (Current stage: {stage})"

    answer = _build_grounded_answer(policy_chunks, user_text, status_info=status_info)

    if not answer:
        answer = _extract_best_knowledge_answer(policy_chunks, user_text)

    if not answer:
        answer = "I'm sorry, I couldn't find that information in the knowledge base. Please contact support or try rephrasing your question."

    return {
        "messages": [{"role": "assistant", "text": answer}]
    }