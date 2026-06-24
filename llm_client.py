"""
llm_client.py
Wraps the Groq API (llama-3.1-8b-instant) to answer questions grounded
in retrieved document chunks.  Returns both the answer text and a boolean
flag indicating whether the answer was found in the document.
"""

import os
from typing import List, Dict, Tuple

from groq import Groq

# Sentinel phrase the model uses when the answer is not in the context
_NOT_FOUND_SIGNAL = "NOT_IN_DOCUMENT"

SYSTEM_PROMPT = f"""You are DocMind, a precise document-question-answering assistant.

Your job is to answer the user's question using ONLY the context passages provided.

Rules:
1. Base your answer strictly on the provided context.
2. If the context does not contain enough information to answer the question,
   start your response with the exact token: {_NOT_FOUND_SIGNAL}
   followed by a brief honest note that the answer isn't in the document,
   then optionally provide a short general-knowledge answer labelled
   "General knowledge:".
3. Be concise and factual. Cite relevant details from the context when useful.
4. Never fabricate information that isn't in the context without flagging it.
5. Maintain a helpful, professional tone.
"""


def _build_messages(
    question: str,
    context_chunks: List[str],
    chat_history: List[Dict],
) -> List[Dict]:
    """
    Construct the full message list for the Groq API call, including
    conversation history (last 6 turns to stay within token limits).
    """
    context_text = "\n\n---\n\n".join(context_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include up to 6 recent turns of history
    for turn in chat_history[-6:]:
        messages.append({"role": "user",      "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    # Current question with injected context
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
    """
    Call the Groq LLM and return (answer_text, not_in_doc_flag).

    not_in_doc_flag=True means the model signalled the answer wasn't in
    the document context.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "⚠ No Groq API key found. Please add it in the sidebar.", False

    client = Groq(api_key=api_key)

    messages = _build_messages(question, context_chunks, chat_history)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )

    raw_answer: str = response.choices[0].message.content.strip()

    not_in_doc = raw_answer.startswith(_NOT_FOUND_SIGNAL)
    if not_in_doc:
        # Strip the sentinel so it doesn't appear verbatim in the UI
        clean_answer = raw_answer[len(_NOT_FOUND_SIGNAL):].strip(" :\n")
    else:
        clean_answer = raw_answer

    return clean_answer, not_in_doc
