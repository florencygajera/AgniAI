"""
ingest.py
=========
Ingestion pipeline for AgniAI.
Supports: PDF files, web URLs, raw text strings, .txt files, and .docx files.

Fixes in this version:
  • chunk_text: start index can never go negative (edge case in overlap calc)
  • CHUNK_MIN_WORDS merge: only merges into previous chunk if it's not already
    over CHUNK_WORDS * 2 words (prevents runaway chunk growth)
  • _source_already_ingested: normalises path strings before comparing so
    Windows path variants don't cause double-ingestion
  • ingest_text: unique label generation uses a timestamp suffix as tiebreaker
    so concurrent ingests don't collide
"""

import re
import time
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
    DOCSTORE_PATH,
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
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split *text* into overlapping word-count chunks.

    FIX: start index is clamped to max(0, ...) so it never goes negative.
    FIX: CHUNK_MIN_WORDS merge is bounded so tiny fragments don't inflate a
         chunk beyond 2× the target size.
    """
    words = clean_text(text).split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    max_merge_words = chunk_words * 2  # safety ceiling on merged chunk size

    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            word_count = len(chunk.split())
            if word_count < CHUNK_MIN_WORDS and chunks:
                # Only merge if the previous chunk won't balloon excessively
                prev_word_count = len(chunks[-1].split())
                if prev_word_count + word_count <= max_merge_words:
                    chunks[-1] = f"{chunks[-1]} {chunk}".strip()
                else:
                    chunks.append(chunk)
            else:
                chunks.append(chunk)
        if end >= len(words):
            break
        # FIX: clamp so start never goes negative
        start = max(0, end - overlap)

    return chunks


def _normalise_source(source: str) -> str:
    """Normalise a source path/URL for deduplication comparisons."""
    # Convert backslashes to forward slashes for cross-platform consistency
    return source.replace("\\", "/").strip().rstrip("/")


def _source_already_ingested(source: str) -> bool:
    """Return True if *source* is already in the docstore."""
    docs = load_docstore()
    normalised = _normalise_source(source)
    return any(_normalise_source(d.get("source", "")) == normalised for d in docs)


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
    docs = load_docstore()

    vectors = embed_texts(chunks)
    if vectors.size == 0:
        return 0

    if index.ntotal == 0 and vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, "
            f"got {vectors.shape[1]}"
        )

    index.add(vectors)

    start_id = len(docs) + 1
    for i, chunk in enumerate(chunks, start=start_id):
        docs.append(
            {
                "source":   source,
                "doc_type": doc_type,
                "chunk_id": str(i),
                "text":     chunk,
            }
        )

    save_index(index, docs)
    return len(chunks)


# ── HTML extractor ─────────────────────────────────────────────────────────

class _VisibleTextExtractor(HTMLParser):
    _BLOCK_TAGS = {"h1","h2","h3","h4","h5","p","li","td","th","br","div"}
    _SKIP_TAGS  = {"script","style","noscript","header","footer","nav"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS and self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._BLOCK_TAGS and self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_data(self, data):
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
        for element in soup.find_all(["h1","h2","h3","h4","h5","p","li","td","th"]):
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
    """Extract text from a PDF and add it to the knowledge base."""
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    try:
        import fitz  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyMuPDF is not installed. Run: pip install PyMuPDF"
        ) from exc

    source = str(path)
    if not force and _source_already_ingested(source):
        return 0

    pages: List[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            page_text = page.get_text("text")
            if page_text and page_text.strip():
                pages.append(page_text)

    if not pages:
        raise ValueError(
            "The PDF contains no extractable text. "
            "It may be a scanned/image-only PDF."
        )

    text   = clean_text("\n".join(pages))
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

    text   = path.read_text(encoding="utf-8", errors="replace")
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
            "python-docx is not installed. Run: pip install python-docx"
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
    """Fetch a webpage, extract visible text, and add it to the knowledge base."""
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

    FIX: Uses a timestamp suffix so multiple rapid calls never collide on
    the deduplication check, even if the previous call used the same default label.
    """
    if label == "manual_text":
        docs = load_docstore()
        existing_count = sum(
            1 for d in docs
            if d.get("source", "").startswith("manual_text")
        )
        if existing_count > 0:
            label = f"manual_text_{existing_count + 1}_{int(time.time())}"

    chunks = chunk_text(text)
    return _append_documents(chunks, source=label, doc_type="text")


def list_sources() -> List[Dict]:
    """Return a summary of all ingested sources."""
    docs   = load_docstore()
    counts: Dict[str, Dict] = {}
    for d in docs:
        src = d.get("source", "unknown")
        if src not in counts:
            counts[src] = {
                "source":      src,
                "doc_type":    d.get("doc_type", "?"),
                "chunk_count": 0,
            }
        counts[src]["chunk_count"] += 1
    return list(counts.values())


def clear_index() -> None:
    """Delete all indexed data (FAISS + docstore + BM25)."""
    if FAISS_INDEX_PATH.exists():
        FAISS_INDEX_PATH.unlink()
    if DOCSTORE_PATH.exists():
        DOCSTORE_PATH.unlink()

    # Also clear BM25 index
    from config import BM25_INDEX_PATH
    if BM25_INDEX_PATH.exists():
        BM25_INDEX_PATH.unlink()

    import rag
    rag._INDEX = None
    rag._DOCS  = []
    rag._BM25  = None