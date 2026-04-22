"""
rag.py
======
Core retrieval-augmented generation layer:
  - Sentence-transformer embeddings (local, offline)
  - FAISS IndexFlatIP vector store (inner-product / cosine after L2-norm)
  - Ollama HTTP client with model auto-detection and fallback
"""

import json
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from config import (
    DEFAULT_MODEL,
    DOCSTORE_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FALLBACK_MODELS,
    MIN_SCORE,
    OLLAMA_TAGS_URL,
    OLLAMA_URL,
    REQUEST_TIMEOUT,
    SYSTEM_PROMPT,
)

# ── Module-level singletons (lazy-loaded) ──────────────────────────────────
_MODEL: Optional[SentenceTransformer] = None
_INDEX: Optional[faiss.Index] = None
_DOCS: List[Dict[str, str]] = []


# ── Helpers ────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _new_index() -> faiss.Index:
    """Create a fresh inner-product FAISS index."""
    return faiss.IndexFlatIP(EMBEDDING_DIM)


# ── Embedding ──────────────────────────────────────────────────────────────

def load_embedding_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Embed a batch of texts, returns float32 array shaped (N, DIM)."""
    model = load_embedding_model()
    vecs = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,   # enables cosine via inner-product
        show_progress_bar=len(texts) > 20,
    )
    return np.asarray(vecs, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string, returns float32 array shaped (1, DIM)."""
    model = load_embedding_model()
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vec, dtype="float32")


# ── Index persistence ──────────────────────────────────────────────────────

def _load_docstore() -> List[Dict[str, str]]:
    if not DOCSTORE_PATH.exists():
        return []
    with DOCSTORE_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_docstore(docs: List[Dict[str, str]]) -> None:
    _ensure_dirs()
    with DOCSTORE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(docs, fh, ensure_ascii=False, indent=2)


def load_index() -> faiss.Index:
    """Load (or create) the FAISS index and populate _DOCS."""
    global _INDEX, _DOCS
    if _INDEX is not None:
        return _INDEX

    _ensure_dirs()
    if FAISS_INDEX_PATH.exists():
        _INDEX = faiss.read_index(str(FAISS_INDEX_PATH))
    else:
        _INDEX = _new_index()

    _DOCS = _load_docstore()
    return _INDEX


def save_index(index: faiss.Index, docs: List[Dict[str, str]]) -> None:
    """Persist the FAISS index and docstore to disk."""
    global _DOCS
    _ensure_dirs()
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    _save_docstore(docs)
    _DOCS = docs         # keep in-memory mirror in sync


# ── Search ─────────────────────────────────────────────────────────────────

def search(query: str, top_k: int = 4) -> List[Dict[str, str]]:
    """
    Return top-k most relevant chunks for *query*.
    Chunks with score < MIN_SCORE are discarded.
    """
    index = load_index()
    if index.ntotal == 0:
        return []

    qvec = embed_query(query)
    scores, ids = index.search(qvec, min(top_k, index.ntotal))

    results: List[Dict[str, str]] = []
    for doc_id, score in zip(ids[0], scores[0]):
        if doc_id < 0 or doc_id >= len(_DOCS):
            continue
        if score < MIN_SCORE:
            continue
        doc = dict(_DOCS[doc_id])
        doc["score"] = round(float(score), 4)
        results.append(doc)
    return results


def build_context(docs: Sequence[Dict[str, str]]) -> str:
    """
    Format retrieved docs into a numbered context block for the LLM prompt.
    Deduplicates identical chunk texts.
    """
    if not docs:
        return ""

    blocks: List[str] = []
    seen: set = set()
    for i, doc in enumerate(docs, start=1):
        text = (doc.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        source = doc.get("source", "unknown")
        chunk_id = doc.get("chunk_id", str(i))
        score = doc.get("score", "?")
        blocks.append(
            f"--- Context [{i}] | Source: {source} | Chunk: {chunk_id} | Score: {score} ---\n{text}"
        )
    return "\n\n".join(blocks)


def index_stats() -> Dict[str, int]:
    """Return total vectors stored and number of documents."""
    index = load_index()
    return {"vectors": int(index.ntotal), "chunks": len(_DOCS)}


# ── Ollama client ──────────────────────────────────────────────────────────

def _fetch_ollama_models() -> List[str]:
    """Query Ollama for locally installed model names."""
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", []) if m.get("name")]
    except requests.RequestException:
        return []


def _available_models() -> List[str]:
    installed = _fetch_ollama_models()
    return installed if installed else FALLBACK_MODELS


def _build_messages(
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})
    return messages


def call_llm(
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
) -> str:
    """
    Call the local Ollama LLM.
    Tries the specified model first, then falls back through available models.
    Raises RuntimeError if nothing responds.
    """
    candidates = []
    if model:
        candidates.append(model)
    available = _available_models()
    for m in available:
        if m not in candidates:
            candidates.append(m)
    if DEFAULT_MODEL not in candidates:
        candidates.insert(0, DEFAULT_MODEL)

    messages = _build_messages(prompt, history=history)
    last_error: Optional[str] = None

    for candidate in candidates:
        body = {"model": candidate, "messages": messages, "stream": False}
        try:
            resp = requests.post(OLLAMA_URL, json=body, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "").strip()
            if content:
                return content
            last_error = f"Empty response from '{candidate}'."
        except requests.RequestException as exc:
            last_error = str(exc)

    raise RuntimeError(
        "Ollama is unreachable or no installed model responded.\n"
        f"Last error: {last_error}\n"
        "Make sure Ollama is running:  ollama serve"
    )