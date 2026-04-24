"""
ingest.py
=========
Ingestion pipeline for AgniAI.
Supports: PDF files, web URLs, raw text strings, .txt files, and .docx files.

All content is:
  1. Extracted → cleaned → chunked (word-level sliding window)
  2. Embedded via sentence-transformers
  3. Added to the FAISS vector index + JSON docstore
"""

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Sequence

import requests

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    BeautifulSoup = None

from config import (
    CHUNK_OVERLAP,
    CHUNK_WORDS,
    CHUNK_MIN_WORDS,
    DATA_DIR,
    DOCSTORE_PATH,  # BUG-3 FIX
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
)
from rag import embed_texts, load_docstore, load_index, save_index


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
            word_count = len(chunk.split())
            if word_count < CHUNK_MIN_WORDS and chunks:
                # Merge tiny trailing fragments into the previous chunk instead
                # of indexing them as low-signal standalone vectors.
                chunks[-1] = f"{chunks[-1]} {chunk}".strip()
            else:
                chunks.append(chunk)
        if end >= len(words):
            break
        start = max(0, end - overlap)
    return chunks


def _source_already_ingested(source: str) -> bool:
    """Return True if *source* is already in the docstore (avoid duplicates)."""
    docs = load_docstore()
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

    Bug #4 fix: chunk_id now continues from the last stored ID instead
    of always restarting at 1, so IDs are globally unique across ingestions.
    """
    if not chunks:
        return 0

    _ensure_dirs()
    index = load_index()
    docs = load_docstore()

    vectors = embed_texts(chunks)
    if vectors.size == 0:
        return 0

    if index.ntotal == 0 and vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {vectors.shape[1]}"
        )

    index.add(vectors)

    # Continue chunk IDs from the last stored entry (globally unique IDs)
    start_id = len(docs) + 1
    for i, chunk in enumerate(chunks, start=start_id):
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


class _VisibleTextExtractor(HTMLParser):
    """Best-effort fallback HTML text extractor when BeautifulSoup is unavailable."""

    _BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "p", "li", "td", "th", "br", "div"}
    _SKIP_TAGS = {"script", "style", "noscript", "header", "footer", "nav"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS and self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._BLOCK_TAGS and self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self.parts.append(text + " ")


def _extract_visible_text(html: str) -> str:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        parts: List[str] = []
        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "p", "li", "td", "th"]):
            text = element.get_text(" ", strip=True)
            if text:
                parts.append(text)
        return " ".join(parts)

    parser = _VisibleTextExtractor()
    parser.feed(html)
    parser.close()
    return clean_text("".join(parser.parts))


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

    try:
        import fitz  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyMuPDF is not installed. Install the project dependencies or add "
            "'PyMuPDF' to your environment before ingesting PDFs."
        ) from exc

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


def ingest_docx(file_path: str, force: bool = False) -> int:
    """Ingest a Microsoft Word (.docx) file."""
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Word file not found: {path}")
    if path.suffix.lower() != ".docx":
        raise ValueError(f"Expected a .docx file, got: {path.suffix}")

    try:
        from docx import Document as DocxDocument  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "python-docx is not installed. Install the project dependencies or "
            "add 'python-docx' to your environment before ingesting DOCX files."
        ) from exc

    source = str(path)
    if not force and _source_already_ingested(source):
        return 0

    doc = DocxDocument(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = clean_text("\n".join(paragraphs))
    if not text:
        raise ValueError("No extractable text found in the Word document.")

    chunks = chunk_text(text)
    return _append_documents(chunks, source=source, doc_type="docx")


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

    text = clean_text(_extract_visible_text(response.text))
    if not text:
        raise ValueError("No readable text found at the URL.")

    chunks = chunk_text(text)
    return _append_documents(chunks, source=url, doc_type="url")


def ingest_text(text: str, label: str = "manual_text") -> int:
    """
    Ingest raw text directly (e.g. pasted content).

    Bug fix: each call now gets a unique label using a counter suffix,
    so multiple raw-text ingestions don't block each other via deduplication.
    """
    # Make label unique if "manual_text" default is used multiple times
    if label == "manual_text":
        docs = load_docstore()
        existing = len(set(  # BUG-9 FIX
            d["source"] for d in docs
            if d.get("source", "").startswith("manual_text")
        ))
        if existing > 0:
            label = f"manual_text_{existing + 1}"

    chunks = chunk_text(text)
    return _append_documents(chunks, source=label, doc_type="text")


def list_sources() -> List[Dict[str, str]]:
    """
    Return a summary of all ingested sources.
    Each entry has: source, doc_type, chunk_count.
    """
    docs = load_docstore()
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

    Bug #1 fix: removed unused imports (_INDEX, _DOCS, faiss as _faiss).
    Module-level singletons in rag.py are reset directly via the rag module.
    """
    if FAISS_INDEX_PATH.exists():
        FAISS_INDEX_PATH.unlink()
    if DOCSTORE_PATH.exists():
        DOCSTORE_PATH.unlink()

    # Reset module-level singletons in rag.py
    import rag
    rag._INDEX = None
    rag._DOCS = []
