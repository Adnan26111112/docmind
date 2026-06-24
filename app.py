import streamlit as st
import os
from utils.pdf_processor import extract_text_from_pdf, chunk_text
from utils.llm_client import ask_question
from utils.vector_store import build_vector_store, retrieve_relevant_chunks

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind – AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Background ── */
  .stApp {
    background: #0d0f14;
    color: #e2e8f0;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #13161e;
    border-right: 1px solid #1e2330;
  }

  /* ── Brand header ── */
  .brand-header {
    padding: 1.2rem 0 0.6rem;
    text-align: center;
  }
  .brand-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.02em;
    margin: 0;
  }
  .brand-header .accent { color: #6ee7b7; }
  .brand-header p {
    font-size: 0.8rem;
    color: #64748b;
    margin: 0.3rem 0 0;
  }

  /* ── Upload area ── */
  [data-testid="stFileUploader"] {
    background: #13161e;
    border: 1px dashed #2d3348;
    border-radius: 10px;
    padding: 0.5rem;
  }

  /* ── Chat bubbles ── */
  .chat-wrap { display: flex; flex-direction: column; gap: 1rem; padding: 0.5rem 0; }

  .bubble-user {
    align-self: flex-end;
    background: #1a2744;
    border: 1px solid #2d3f6b;
    color: #bfdbfe;
    padding: 0.7rem 1rem;
    border-radius: 16px 16px 4px 16px;
    max-width: 80%;
    font-size: 0.9rem;
    line-height: 1.5;
  }

  .bubble-ai {
    align-self: flex-start;
    background: #131a24;
    border: 1px solid #1e2a3a;
    color: #e2e8f0;
    padding: 0.7rem 1rem;
    border-radius: 16px 16px 16px 4px;
    max-width: 85%;
    font-size: 0.9rem;
    line-height: 1.6;
  }

  .bubble-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
  }
  .bubble-user .bubble-label  { color: #60a5fa; }
  .bubble-ai   .bubble-label  { color: #6ee7b7; }

  /* ── Not-found notice ── */
  .not-found {
    background: #1c1508;
    border: 1px solid #3d2e0a;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    color: #fbbf24;
    font-size: 0.82rem;
    margin-top: 0.4rem;
  }

  /* ── Status pill ── */
  .status-pill {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 500;
  }
  .status-ready   { background: #052e16; color: #6ee7b7; border: 1px solid #065f46; }
  .status-waiting { background: #1e1b0f; color: #fbbf24; border: 1px solid #78350f; }

  /* ── Divider ── */
  hr { border-color: #1e2330 !important; margin: 0.8rem 0 !important; }

  /* ── Input box ── */
  .stTextInput input {
    background: #13161e;
    border: 1px solid #2d3348;
    color: #e2e8f0;
    border-radius: 8px;
  }
  .stTextInput input:focus { border-color: #6ee7b7; box-shadow: none; }

  /* ── Button ── */
  .stButton > button {
    background: #6ee7b7;
    color: #052e16;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.45rem 1.2rem;
    transition: background 0.15s;
  }
  .stButton > button:hover { background: #34d399; }

  /* ── Chunk / info text ── */
  .info-text { color: #64748b; font-size: 0.8rem; }

  /* ── Scrollable chat area ── */
  .chat-container {
    max-height: 58vh;
    overflow-y: auto;
    padding-right: 4px;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "chat_history": [],
    "doc_chunks": [],
    "vector_store": None,
    "doc_name": None,
    "doc_processed": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-header">
      <h1>Doc<span class="accent">Mind</span></h1>
      <p>AI-powered document assistant</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # API key input
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com",
    )
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

    st.markdown("---")

    # File uploader
    st.markdown("**Upload a PDF**")
    uploaded_file = st.file_uploader(
        label="",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        if uploaded_file.name != st.session_state.doc_name:
            with st.spinner("Reading & indexing document…"):
                raw_text = extract_text_from_pdf(uploaded_file)
                if raw_text.strip():
                    chunks = chunk_text(raw_text)
                    st.session_state.doc_chunks = chunks
                    st.session_state.vector_store = build_vector_store(chunks)
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.doc_processed = True
                    st.session_state.chat_history = []
                else:
                    st.error("Could not extract text. Is the PDF scanned/image-only?")

    st.markdown("---")

    # Doc status
    if st.session_state.doc_processed:
        st.markdown(
            f'<span class="status-pill status-ready">✓ {st.session_state.doc_name}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="info-text">{len(st.session_state.doc_chunks)} text chunks indexed</p>',
            unsafe_allow_html=True,
        )
        if st.button("Clear document"):
            for k in ["chat_history", "doc_chunks", "vector_store", "doc_name", "doc_processed"]:
                st.session_state[k] = [] if k in ("chat_history", "doc_chunks") else None if k != "doc_processed" else False
            st.rerun()
    else:
        st.markdown(
            '<span class="status-pill status-waiting">⏳ No document loaded</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<p class="info-text">Powered by Groq · LLaMA 3.1 · sentence-transformers</p>', unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## Ask your document anything")

# Render chat history
if st.session_state.chat_history:
    chat_html = '<div class="chat-container"><div class="chat-wrap">'
    for turn in st.session_state.chat_history:
        chat_html += f"""
        <div class="bubble-user">
          <div class="bubble-label">You</div>
          {turn["question"]}
        </div>
        <div class="bubble-ai">
          <div class="bubble-label">DocMind</div>
          {turn["answer"]}
          {"<div class='not-found'>⚠ The answer to this question wasn't found in the document. The response is based on general knowledge.</div>" if turn.get("not_in_doc") else ""}
        </div>
        """
    chat_html += "</div></div>"
    st.markdown(chat_html, unsafe_allow_html=True)
else:
    if st.session_state.doc_processed:
        st.markdown('<p style="color:#64748b; font-size:0.9rem;">Document ready — ask your first question below.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#64748b; font-size:0.9rem;">Upload a PDF in the sidebar to get started.</p>', unsafe_allow_html=True)

st.markdown("---")

# Input row
col1, col2 = st.columns([5, 1])
with col1:
    user_question = st.text_input(
        label="",
        placeholder="e.g. What is the main argument of this paper?",
        label_visibility="collapsed",
        key="user_input",
        disabled=not st.session_state.doc_processed,
    )
with col2:
    send_btn = st.button("Ask →", disabled=not st.session_state.doc_processed)

# Handle submission
if (send_btn or user_question) and user_question.strip():
    if not os.environ.get("GROQ_API_KEY"):
        st.warning("Please enter your Groq API key in the sidebar first.")
    elif not st.session_state.doc_processed:
        st.warning("Please upload a PDF document first.")
    else:
        with st.spinner("Thinking…"):
            # Retrieve relevant chunks
            relevant_chunks = retrieve_relevant_chunks(
                user_question,
                st.session_state.vector_store,
                st.session_state.doc_chunks,
                top_k=4,
            )
            # Build context & call LLM
            answer, not_in_doc = ask_question(
                question=user_question,
                context_chunks=relevant_chunks,
                chat_history=st.session_state.chat_history,
            )
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": answer,
                "not_in_doc": not_in_doc,
            })
        st.rerun()
