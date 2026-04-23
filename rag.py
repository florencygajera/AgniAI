"""
rag.py
======
Core retrieval-augmented generation layer:
  - Sentence-transformer embeddings (local, offline)
  - FAISS IndexFlatIP vector store (inner-product / cosine after L2-norm)
  - Ollama HTTP client with model auto-detection and fallback

Speed patches applied:
  1. load_embedding_model() returns cached _MODEL immediately on subsequent calls
     (no re-init, no HF warning after first load)
  2. load_index() guards both _INDEX and _DOCS together — no partial reloads
  3. embed_query() uses batch_size=1 to avoid unnecessary overhead
  4. build_context() strips verbose header metadata to reduce context tokens
"""

import json
import warnings
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

# ── Module-level singletons (lazy-loaded, never evicted) ───────────────────
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


def _escape_control_chars_in_json_strings(raw: str) -> str:
    """
    Repair JSON text that contains literal control characters inside strings.
    Aimed at salvaging docstore files where chunk text was written with
    raw newlines instead of escaped \\n sequences.
    """
    repaired: List[str] = []
    in_string = False
    escaped = False

    for ch in raw:
        if in_string:
            if escaped:
                repaired.append(ch)
                escaped = False
                continue
            if ch == "\\":
                repaired.append(ch)
                escaped = True
                continue
            if ch == '"':
                repaired.append(ch)
                in_string = False
                continue
            if ch == "\n":
                repaired.append("\\n")
                continue
            if ch == "\r":
                repaired.append("\\r")
                continue
            if ch == "\t":
                repaired.append("\\t")
                continue
            if ord(ch) < 32:
                repaired.append(f"\\u{ord(ch):04x}")
                continue
            repaired.append(ch)
            continue

        repaired.append(ch)
        if ch == '"':
            in_string = True

    return "".join(repaired)


def _extract_json_scalar(line: str) -> str:
    """Extract a simple JSON string value from a single indented key/value line."""
    value = line.split(":", 1)[1].strip()
    if value.endswith(","):
        value = value[:-1].rstrip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value


def _repair_docstore_from_lines(raw: str) -> List[Dict[str, str]]:
    """
    Best-effort salvage parser for docstore.json.
    Assumes an array of objects with keys: source, doc_type, chunk_id, text.
    """
    docs: List[Dict[str, str]] = []
    obj: Dict[str, str] = {}
    text_lines: List[str] = []
    in_object = False
    in_text = False

    for line in raw.splitlines():
        stripped = line.strip()

        if stripped == "[" or not stripped:
            continue
        if stripped == "]":
            break

        if stripped.startswith("{"):
            obj = {}
            text_lines = []
            in_object = True
            in_text = False
            continue

        if not in_object:
            continue

        if in_text:
            if stripped in {"}", "},"}:
                obj["text"] = "\n".join(text_lines)
                docs.append(obj)
                obj = {}
                text_lines = []
                in_object = False
                in_text = False
                continue
            text_lines.append(line)
            continue

        if stripped.startswith('"source":'):
            obj["source"] = _extract_json_scalar(line)
        elif stripped.startswith('"doc_type":'):
            obj["doc_type"] = _extract_json_scalar(line)
        elif stripped.startswith('"chunk_id":'):
            obj["chunk_id"] = _extract_json_scalar(line)
        elif stripped.startswith('"text":'):
            fragment = line.split(":", 1)[1].lstrip()
            if fragment.startswith('"'):
                fragment = fragment[1:]
            if fragment.endswith('",'):
                fragment = fragment[:-2]
                obj["text"] = fragment
                text_lines = []
            elif fragment.endswith('"'):
                fragment = fragment[:-1]
                obj["text"] = fragment
                text_lines = []
            else:
                text_lines = [fragment]
                in_text = True
            continue
        elif stripped in {"}", "},"}:
            if "text" not in obj and text_lines:
                obj["text"] = "\n".join(text_lines)
            if obj:
                docs.append(obj)
            obj = {}
            text_lines = []
            in_object = False
            in_text = False

    if in_object and obj:
        if "text" not in obj and text_lines:
            obj["text"] = "\n".join(text_lines)
        docs.append(obj)

    return docs


# ── Embedding ──────────────────────────────────────────────────────────────

def load_embedding_model() -> SentenceTransformer:
    """
    Load the sentence-transformer model ONCE and keep it in module memory.

    Speed patch: _MODEL is checked first — if already loaded, returns
    immediately with zero cost. The HF unauthenticated-request warning only
    fires on the very first call, never again within the same session.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL   # fast path — already in RAM, no work to do

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress HF hub noise on first load
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Embed a batch of texts; returns float32 array shaped (N, DIM)."""
    model = load_embedding_model()
    vecs = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity via inner-product
        show_progress_bar=len(texts) > 20,
        batch_size=32,               # explicit batch size — avoids default overhead
    )
    return np.asarray(vecs, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string; returns float32 array shaped (1, DIM)."""
    model = load_embedding_model()
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=1,   # single query — no batching overhead
    )
    return np.asarray(vec, dtype="float32")


# ── Index persistence ──────────────────────────────────────────────────────

def load_docstore() -> List[Dict[str, str]]:
    if not DOCSTORE_PATH.exists():
        return []

    raw = DOCSTORE_PATH.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = _escape_control_chars_in_json_strings(raw)
        try:
            docs = json.loads(repaired)
        except json.JSONDecodeError:
            docs = _repair_docstore_from_lines(raw)
            if not docs:
                raise
        _save_docstore(docs)
        print("[WARNING] Repaired malformed docstore.json and saved cleaned copy.")
        return docs


def _save_docstore(docs: List[Dict[str, str]]) -> None:
    _ensure_dirs()
    with DOCSTORE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(docs, fh, ensure_ascii=False, indent=2)


def load_index() -> faiss.Index:
    """
    Load (or create) the FAISS index and populate _DOCS.

    Speed patch: guards _INDEX and _DOCS together. Previously _INDEX could be
    non-None while _DOCS was empty, causing a redundant docstore disk read on
    every search call. Now both are loaded atomically on the first call only.
    """
    global _INDEX, _DOCS

    # Fast path — both already in memory
    if _INDEX is not None and _DOCS:
        return _INDEX

    # First call (or _DOCS was wiped by /reset): load from disk
    _ensure_dirs()
    if FAISS_INDEX_PATH.exists():
        _INDEX = faiss.read_index(str(FAISS_INDEX_PATH))
    else:
        _INDEX = _new_index()

    _DOCS = load_docstore()

    if _INDEX.ntotal > 0 and len(_DOCS) == 0:
        print("[WARNING] FAISS index has vectors but docstore is empty. "
              "Run /reset and re-ingest your documents.")

    return _INDEX


def save_index(index: faiss.Index, docs: List[Dict[str, str]]) -> None:
    """Persist the FAISS index and docstore to disk."""
    global _DOCS
    _ensure_dirs()
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    _save_docstore(docs)
    _DOCS = docs   # keep in-memory mirror in sync


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
    Format retrieved docs into a context block for the LLM prompt.

    Speed patch: stripped the verbose header line per chunk
    (was ~60 chars of metadata per chunk the LLM had to tokenise but didn't use).
    Now uses a compact "Source: X" prefix only, saving ~30-40 tokens per query.
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
        # Compact header — fewer tokens for the model to process
        blocks.append(f"[{i}] Source: {source}\n{text}")

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
    Call the local Ollama LLM (non-streaming).
    Tries the specified model first, then falls back through available models.
    Raises RuntimeError if nothing responds.
    """
    candidates: List[str] = []

    if model:
        candidates.append(model)
    if DEFAULT_MODEL not in candidates:
        candidates.append(DEFAULT_MODEL)
    for m in _available_models():
        if m not in candidates:
            candidates.append(m)

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