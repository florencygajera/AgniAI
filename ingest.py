"""
ingest.py
=========
Ingestion pipeline for AgniAI.
Supports: PDF files, web URLs, raw text strings, and .txt files.

All content is:
  1. Extracted → cleaned → chunked (word-level sliding window)
  2. Embedded via sentence-transformers
  3. Added to the FAISS vector index + JSON docstore
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import fitz          # PyMuPDF
import requests
from bs4 import BeautifulSoup

from config import (
    CHUNK_OVERLAP,
    CHUNK_WORDS,
    DATA_DIR,
    DOCSTORE_PATH,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
)
from rag import embed_texts, load_index, save_index


# ── Directory helpers ──────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Text utilities ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Strip null bytes and collapse whitespace."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)          # collapse horizontal space
    text = re.sub(r"\n{3,}", "\n\n", text)        # at most 2 blank lines
    return text.strip()


def chunk_text(
    text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split *text* into overlapping word-count chunks."""
    words = clean_text(text).split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = max(0, end - overlap)
    return chunks


# ── Docstore helpers ───────────────────────────────────────────────────────

def _load_docstore() -> List[Dict[str, str]]:
    if not DOCSTORE_PATH.exists():
        return []
    with DOCSTORE_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _source_already_ingested(source: str) -> bool:
    """Return True if *source* is already in the docstore (avoid duplicates)."""
    docs = _load_docstore()
    return any(d.get("source") == source for d in docs)


# ── Core append ───────────────────────────────────────────────────────────

def _append_documents(
    chunks: Sequence[str],
    source: str,
    doc_type: str,
) -> int:
    """
    Embed *chunks* and append them to the FAISS index + docstore.
    Returns number of chunks actually added.
    """
    if not chunks:
        return 0

    _ensure_dirs()
    index = load_index()
    docs = _load_docstore()

    vectors = embed_texts(chunks)
    if vectors.size == 0:
        return 0

    if index.ntotal == 0 and vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {vectors.shape[1]}"
        )

    index.add(vectors)
    for i, chunk in enumerate(chunks, start=1):
        docs.append(
            {
                "source": source,
                "doc_type": doc_type,
                "chunk_id": str(i),
                "text": chunk,
            }
        )

    save_index(index, docs)
    return len(chunks)


# ── Public ingest functions ────────────────────────────────────────────────

def ingest_pdf(file_path: str, force: bool = False) -> int:
    """
    Extract text from a PDF and add it to the knowledge base.

    Args:
        file_path: Absolute or relative path to the PDF.
        force:     Re-ingest even if this source was previously ingested.

    Returns:
        Number of chunks added.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    source = str(path)
    if not force and _source_already_ingested(source):
        return 0   # already ingested — skip silently

    pages: List[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            page_text = page.get_text("text")
            if page_text and page_text.strip():
                pages.append(page_text)

    if not pages:
        raise ValueError("The PDF contains no extractable text. "
                         "It may be a scanned/image-only PDF.")

    text = clean_text("\n".join(pages))
    chunks = chunk_text(text)
    return _append_documents(chunks, source=source, doc_type="pdf")


def ingest_txt(file_path: str, force: bool = False) -> int:
    """Ingest a plain .txt file."""
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    source = str(path)
    if not force and _source_already_ingested(source):
        return 0

    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_text(text)
    return _append_documents(chunks, source=source, doc_type="txt")


def ingest_url(url: str, force: bool = False) -> int:
    """
    Fetch a webpage, extract visible text, and add it to the knowledge base.
    """
    if not force and _source_already_ingested(url):
        return 0

    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "AgniAI/1.0 (offline-chatbot)"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    parts: List[str] = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "p", "li", "td", "th"]):
        text = element.get_text(" ", strip=True)
        if text:
            parts.append(text)

    text = clean_text(" ".join(parts))
    if not text:
        raise ValueError("No readable text found at the URL.")

    chunks = chunk_text(text)
    return _append_documents(chunks, source=url, doc_type="url")


def ingest_text(text: str, label: str = "manual_text") -> int:
    """Ingest raw text directly (e.g. pasted content)."""
    chunks = chunk_text(text)
    return _append_documents(chunks, source=label, doc_type="text")


def list_sources() -> List[Dict[str, str]]:
    """
    Return a summary of all ingested sources.
    Each entry has: source, doc_type, chunk_count.
    """
    docs = _load_docstore()
    counts: Dict[str, Dict[str, str]] = {}
    for d in docs:
        src = d.get("source", "unknown")
        if src not in counts:
            counts[src] = {"source": src, "doc_type": d.get("doc_type", "?"), "chunk_count": 0}
        counts[src]["chunk_count"] += 1  # type: ignore[assignment]
    return list(counts.values())


def clear_index() -> None:
    """
    Delete all indexed data (FAISS + docstore).
    Use with caution — requires re-ingestion of all sources.
    """
    from rag import _INDEX, _DOCS
    import faiss as _faiss

    if FAISS_INDEX_PATH.exists():
        FAISS_INDEX_PATH.unlink()
    if DOCSTORE_PATH.exists():
        DOCSTORE_PATH.unlink()

    # Reset module-level singletons in rag.py
    import rag
    rag._INDEX = None
    rag._DOCS = []