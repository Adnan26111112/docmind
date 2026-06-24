"""
pdf_processor.py
Extract text from an uploaded PDF and split it into overlapping chunks
suitable for semantic retrieval.
"""

import io
from typing import List
import fitz  # PyMuPDF


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Accept a Streamlit UploadedFile (or any file-like object) and return
    the full text content of the PDF.
    """
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages_text = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages_text.append(text)

    doc.close()
    return "\n\n".join(pages_text)


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[str]:
    """
    Split *text* into overlapping chunks of roughly *chunk_size* characters.

    Overlap ensures that sentences spanning a boundary are captured in at
    least one chunk's context window.
    """
    words = text.split()
    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        # Slide forward, keeping *overlap* words from the previous chunk
        start += chunk_size - overlap

    return [c for c in chunks if len(c) > 50]  # drop near-empty tail chunks
