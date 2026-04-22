import json
from pathlib import Path
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
    OLLAMA_TAGS_URL,
    OLLAMA_URL,
    REQUEST_TIMEOUT,
    SYSTEM_PROMPT,
)

_MODEL: Optional[SentenceTransformer] = None
_INDEX: Optional[faiss.Index] = None
_DOCS: List[Dict[str, str]] = []


def _ensure_dirs() -> None:
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_embedding_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def _new_index() -> faiss.Index:
    return faiss.IndexFlatIP(EMBEDDING_DIM)


def _load_docstore() -> List[Dict[str, str]]:
    if not DOCSTORE_PATH.exists():
        return []
    with DOCSTORE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_docstore(docs: List[Dict[str, str]]) -> None:
    _ensure_dirs()
    with DOCSTORE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(docs, handle, ensure_ascii=False, indent=2)


def load_index() -> faiss.Index:
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
    _ensure_dirs()
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    _save_docstore(docs)


def embed_query(query: str) -> np.ndarray:
    model = load_embedding_model()
    vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vector, dtype="float32")


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    model = load_embedding_model()
    vectors = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vectors, dtype="float32")


def search(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    index = load_index()
    if index.ntotal == 0:
        return []

    query_vector = embed_query(query)
    scores, ids = index.search(query_vector, top_k)
    results: List[Dict[str, str]] = []
    for doc_id, score in zip(ids[0], scores[0]):
        if doc_id < 0 or doc_id >= len(_DOCS):
            continue
        doc = dict(_DOCS[doc_id])
        doc["score"] = float(score)
        results.append(doc)
    return results


def build_context(docs: Sequence[Dict[str, str]]) -> str:
    if not docs:
        return ""

    blocks: List[str] = []
    seen = set()
    for i, doc in enumerate(docs, start=1):
        text = (doc.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        source = doc.get("source", "unknown")
        chunk_id = doc.get("chunk_id", str(i))
        blocks.append(f"[Source {i}] {source} | Chunk {chunk_id}\n{text}")
    return "\n\n".join(blocks)


def _fetch_ollama_models() -> List[str]:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=10)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        names = []
        for item in models:
            name = item.get("name")
            if name:
                names.append(name)
        return names
    except requests.RequestException:
        return []


def _available_models() -> List[str]:
    installed = _fetch_ollama_models()
    if installed:
        return installed
    return FALLBACK_MODELS


def _build_messages(prompt: str, history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for item in history[-5:]:
            if item.get("role") in {"user", "assistant"} and item.get("content"):
                messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": prompt})
    return messages


def call_llm(prompt: str, history: Optional[List[Dict[str, str]]] = None, model: Optional[str] = None) -> str:
    models_to_try = [model] if model else []
    models_to_try.extend([name for name in _available_models() if name not in models_to_try])
    if DEFAULT_MODEL not in models_to_try:
        models_to_try.insert(0, DEFAULT_MODEL)

    payload_messages = _build_messages(prompt, history=history)
    last_error: Optional[str] = None

    for candidate in models_to_try:
        if not candidate:
            continue
        body = {"model": candidate, "messages": payload_messages, "stream": False}
        try:
            response = requests.post(OLLAMA_URL, json=body, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            message = data.get("message", {})
            content = message.get("content", "").strip()
            if content:
                return content
            last_error = f"Ollama returned an empty response for model '{candidate}'."
        except requests.RequestException as exc:
            last_error = f"{candidate}: {exc}"

    raise RuntimeError(
        "Could not reach Ollama or no model responded successfully. "
        f"Last error: {last_error}"
    )

