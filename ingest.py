import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import fitz
import requests
from bs4 import BeautifulSoup

from config import CHUNK_OVERLAP, CHUNK_WORDS, DATA_DIR, DOCSTORE_PATH, EMBEDDING_DIM, FAISS_INDEX_PATH
from rag import embed_texts, load_index, save_index


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> List[str]:
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


def _load_docstore() -> List[Dict[str, str]]:
    if not DOCSTORE_PATH.exists():
        return []
    with DOCSTORE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _append_documents(chunks: Sequence[str], source: str, doc_type: str) -> int:
    _ensure_dirs()
    index = load_index()
    docs = _load_docstore()
    vectors = embed_texts(chunks)
    if vectors.size == 0:
        return 0

    if index.ntotal == 0 and vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {vectors.shape[1]}")

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


def ingest_pdf(file_path: str) -> int:
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("ingest_pdf expects a .pdf file")

    extracted_pages: List[str] = []
    with fitz.open(path) as document:
        for page in document:
            page_text = page.get_text("text")
            if page_text:
                extracted_pages.append(page_text)

    text = clean_text("\n".join(extracted_pages))
    chunks = chunk_text(text)
    return _append_documents(chunks, source=str(path), doc_type="pdf")


def ingest_text(text: str) -> int:
    chunks = chunk_text(text)
    return _append_documents(chunks, source="manual_text", doc_type="text")


def ingest_url(url: str) -> int:
    response = requests.get(url, timeout=30, headers={"User-Agent": "AgniAI/1.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    parts: List[str] = []
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = element.get_text(" ", strip=True)
        if text:
            parts.append(text)

    text = clean_text(" ".join(parts))
    chunks = chunk_text(text)
    return _append_documents(chunks, source=url, doc_type="url")

