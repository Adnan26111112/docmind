# DocMind — AI-Powered Document Assistant

> Ask questions about any PDF. Get grounded, accurate answers — powered by Groq (LLaMA 3.1) and semantic search.




## Live Demo

🔗 https://docmind-mdxnwn98dvnsekjjcynxxs.streamlit.app

---

## Features

| Feature | Detail |
|---|---|
| 📄 PDF upload | Drag-and-drop; text extracted via PyMuPDF |
| 🔍 Semantic search | `all-MiniLM-L6-v2` embeddings + cosine similarity |
| 🤖 LLM answering | Groq API · `llama-3.1-8b-instant` |
| 💬 Chat history | Full multi-turn context sent to the model each turn |
| ⚠ Out-of-scope detection | Model explicitly signals when answer isn't in document |
| 🌑 Dark UI | Custom Streamlit CSS — no white-flash clutter |

---

## Architecture

```
PDF Upload (Streamlit)
      │
      ▼
PyMuPDF – text extraction
      │
      ▼
Chunker (800-char windows, 150-char overlap)
      │
      ▼
sentence-transformers encode → in-memory numpy vector store
      │
      ▼  (at query time)
Cosine similarity retrieval (top-4 chunks)
      │
      ▼
Groq API  (llama-3.1-8b-instant)
  system prompt + retrieved context + chat history + question
      │
      ▼
Answer + not-in-document flag → Streamlit UI
```

---

## Setup (Local)

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### Install

```bash
git clone https://github.com/<your-username>/docmind.git
cd docmind
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.  
Enter your Groq API key in the sidebar, upload a PDF, and start asking questions.

---

## Deploy to Streamlit Community Cloud

1. Fork / push this repo to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, entry file `app.py`.
4. Under **Advanced settings → Secrets**, add:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

5. Click **Deploy**.  The app auto-installs dependencies from `requirements.txt`.

> **Tip:** After adding the secret, users won't need to enter the key manually — the sidebar key input falls back to the environment variable, so you can hide it or leave it as an optional override.

---

## Project Structure

```
docmind/
├── app.py                  # Streamlit UI + session state
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Dark-theme & server config
└── utils/
    ├── __init__.py
    ├── pdf_processor.py    # PDF extraction + chunking
    ├── vector_store.py     # Embedding + cosine retrieval
    └── llm_client.py       # Groq API wrapper
```

---

## Design Decisions

**Why Groq?**  
Groq's inference is extremely fast (often <1 s for 8b models), which keeps the chat feel snappy without a streaming implementation — ideal for an MVP.

**Why sentence-transformers (no Pinecone/Chroma)?**  
A numpy-backed in-memory store is sufficient for documents up to ~200 pages. It removes the need for external API keys or persistent storage, keeping deployment friction near zero.

**Chunking strategy**  
800-character windows with 150-character overlap ensures that sentences near chunk boundaries appear in at least one complete chunk, reducing retrieval misses.

**Not-in-document detection**  
Rather than a cosine-threshold heuristic, the model itself is instructed to emit a `NOT_IN_DOCUMENT` sentinel. This is more reliable because the LLM can reason about semantic absence, not just embedding distance.

---

## Known Limitations

- **Scanned / image-only PDFs** are not supported (no OCR). The app detects empty extraction and shows an error.
- Very large PDFs (>300 pages) will slow down the embedding step on first load.
- Chat history is session-only; refreshing the page resets it (MVP scope).
- The `all-MiniLM-L6-v2` model cold-starts on first run (~20 s on Streamlit Cloud free tier).

