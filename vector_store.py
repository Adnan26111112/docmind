"""
vector_store.py
Lightweight in-memory vector store using sentence-transformers embeddings
and cosine similarity.  No external vector DB required.
"""

from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

# Cache the model across Streamlit reruns via st.cache_resource if desired,
# but a module-level singleton is sufficient for an MVP.
_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        # Small, fast, good quality for retrieval
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def build_vector_store(chunks: List[str]) -> Dict[str, Any]:
    """
    Embed every chunk and return a simple dict acting as a vector store.

    Structure:
        {
            "chunks": List[str],
            "embeddings": np.ndarray  shape (N, D)
        }
    """
    model = _get_model()
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    return {"chunks": chunks, "embeddings": embeddings}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return cosine similarity between vector *a* and every row of matrix *b*."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_norm @ a_norm


def retrieve_relevant_chunks(
    query: str,
    vector_store: Dict[str, Any],
    chunks: List[str],
    top_k: int = 4,
) -> List[str]:
    """
    Embed *query* and return the *top_k* most similar chunks.
    """
    model = _get_model()
    query_embedding = model.encode([query], convert_to_numpy=True)[0]

    similarities = _cosine_similarity(query_embedding, vector_store["embeddings"])
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [chunks[i] for i in top_indices]
