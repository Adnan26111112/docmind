import os
import re
from typing import List, Dict, Tuple
from groq import Groq

_NOT_FOUND_SIGNAL = "NOT_IN_DOCUMENT"

SYSTEM_PROMPT = f"""You are DocMind, a precise document-question-answering assistant.

Your job is to answer the user's question using ONLY the context passages provided.

Rules:
1. Base your answer strictly on the provided context.
2. If the context does not contain enough information to answer the question,
   start your response with the exact token: {_NOT_FOUND_SIGNAL}
   followed by a brief honest note that the answer isn't in the document.
3. Be concise and factual. Cite relevant details from the context when useful.
4. Never fabricate information that isn't in the context without flagging it.
5. Maintain a helpful, professional tone.
6. Never use HTML tags in your response. Plain text only.
"""


def _build_messages(
    question: str,
    context_chunks: List[str],
    chat_history: List[Dict],
) -> List[Dict]:
    context_text = "\n\n---\n\n".join(context_chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in chat_history[-6:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    user_content = (
        f"Context from the document:\n\n{context_text}\n\n"
        f"Question: {question}"
    )
    messages.append({"role": "user", "content": user_content})
    return messages


def ask_question(
    question: str,
    context_chunks: List[str],
    chat_history: List[Dict],
) -> Tuple[str, bool]:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "⚠ No Groq API key found. Please add it in the sidebar.", False

    client = Groq(api_key=api_key)
    messages = _build_messages(question, context_chunks, chat_history)

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            return "⚠ Groq rate limit reached. Please wait 1 minute and try again.", False
        return f"⚠ Error: {str(e)[:200]}", False

    raw_answer: str = response.choices[0].message.content.strip()
    raw_answer = re.sub(r'<[^>]+>', '', raw_answer).strip()

    not_in_doc = raw_answer.startswith(_NOT_FOUND_SIGNAL)
    if not_in_doc:
        clean_answer = raw_answer[len(_NOT_FOUND_SIGNAL):].strip(" :\n")
    else:
        clean_answer = raw_answer

    return clean_answer, not_in_doc
