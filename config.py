"""Central configuration for AgniAI."""

from __future__ import annotations

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

DOCSTORE_PATH = INDEX_DIR / "docstore.json"
FAISS_INDEX_PATH = INDEX_DIR / "agni.index"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_RERANKER = os.getenv("USE_RERANKER", "1") not in {"0", "false", "False"}

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_URL = os.getenv("OLLAMA_CHAT_URL", f"{OLLAMA_BASE_URL}/api/chat")
OLLAMA_TAGS_URL = os.getenv("OLLAMA_TAGS_URL", f"{OLLAMA_BASE_URL}/api/tags")

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")
FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "OLLAMA_FALLBACK_MODELS",
        "mistral:7b-instruct,mistral:7b-instruct-q4_0,llama3:8b,llama3:8b-instruct,"
        "llama3.1:8b,llama3.2:3b,gemma2:2b",
    ).split(",")
    if m.strip()
]

# Chunking
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "420"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
CHUNK_MIN_WORDS = int(os.getenv("CHUNK_MIN_WORDS", "12"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "8"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.20"))
STRICT_MIN_SCORE = float(os.getenv("STRICT_MIN_SCORE", "0.70"))
STRICT_TOP_K = int(os.getenv("STRICT_TOP_K", "5"))
LOW_RETRIEVAL_CONFIDENCE = float(os.getenv("LOW_RETRIEVAL_CONFIDENCE", "0.45"))
HIGH_RETRIEVAL_CONFIDENCE = float(os.getenv("HIGH_RETRIEVAL_CONFIDENCE", "0.70"))
MIN_RETRIEVAL_CONFIDENCE = LOW_RETRIEVAL_CONFIDENCE

# Hybrid retrieval
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.55"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.45"))
USE_HYBRID = os.getenv("USE_HYBRID", "1") not in {"0", "false", "False"}

# Memory
MEMORY_MAX_MESSAGES = int(os.getenv("MEMORY_MAX_MESSAGES", "10"))

# Network and cache
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))
FIRST_TOKEN_TIMEOUT = int(os.getenv("FIRST_TOKEN_TIMEOUT", "30"))
STREAM_TIMEOUT = int(os.getenv("STREAM_TIMEOUT", "300"))
RETRIEVAL_CACHE_TTL = int(os.getenv("RETRIEVAL_CACHE_TTL", "300"))
RESPONSE_CACHE_TTL = int(os.getenv("RESPONSE_CACHE_TTL", "300"))
EMBED_CACHE_TTL = int(os.getenv("EMBED_CACHE_TTL", "3600"))
MAX_CACHE_ENTRIES = int(os.getenv("MAX_CACHE_ENTRIES", "2048"))
SESSION_HEADER = os.getenv("SESSION_HEADER", "X-Session-Id")

# Context budgets
MAX_CONTEXT_CHARS = {
    "short": int(os.getenv("MAX_CONTEXT_CHARS_SHORT", "1800")),
    "elaborate": int(os.getenv("MAX_CONTEXT_CHARS_ELABORATE", "2400")),
    "detail": int(os.getenv("MAX_CONTEXT_CHARS_DETAIL", "3000")),
}
MAX_CONTEXT_CHARS_DEFAULT = int(os.getenv("MAX_CONTEXT_CHARS_DEFAULT", "3000"))

# Token budgets
MAX_TOKENS_STYLE = {
    "short": int(os.getenv("MAX_TOKENS_SHORT", "300")),
    "elaborate": int(os.getenv("MAX_TOKENS_ELABORATE", "500")),
    "detail": int(os.getenv("MAX_TOKENS_DETAIL", "800")),
}
MAX_TOKENS_DEFAULT = int(os.getenv("MAX_TOKENS_DEFAULT", "500"))

# CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# Answer-style keywords
STYLE_SHORT_KEYWORDS = [
    "in short", "briefly", "brief", "quick answer", "short answer",
    "summarise", "summarize", "tldr", "tl;dr", "in brief",
    "give me short", "one line", "one-line",
    "give a short", "keep it short", "summary", "summarise it",
    "quick summary",
]

STYLE_DETAIL_KEYWORDS = [
    "in detail", "detailed", "explain in detail", "full detail",
    "comprehensive", "thoroughly", "exhaustive", "step by step",
    "step-by-step", "explain fully", "tell me everything",
    "give me detail", "elaborate in detail", "full explanation",
    "complete explanation", "everything about", "all about",
    "full breakdown", "break it down",
]

STYLE_ELABORATE_KEYWORDS = [
    "elaborate", "explain", "elaborate on", "tell me more",
    "expand on", "describe", "give more", "more info", "more detail",
    "walk me through", "how does", "how do",
]

STYLE_OUTPUT_GUIDANCE = {
    "short": "Answer in 2 to 4 bullet points. Keep it concise and factual.",
    "elaborate": "Answer in 4 to 8 bullet points. Be factual and moderately detailed.",
    "detail": "Answer in structured sections with bullet points and concrete facts.",
}

# Strict prompt
REFERENCE_FALLBACK = "Not available in the document"

STRICT_RAG_PROMPT = """You are a strict question-answering system.

Rules:
- Answer ONLY using the provided context
- Do NOT add any external knowledge
- If partial information is available, answer using that
- If completely missing, say: 'Not available in the document'
- Do NOT hallucinate
- Keep answers precise and structured
- Ignore irrelevant context

Format:
- Use bullet points
- Be concise and factual"""

STRICT_RAG_PROMPT_COMPUTE = """You are a strict question-answering system.

Rules:
- Answer ONLY using the provided context
- Do NOT add any external knowledge
- You may compute or aggregate values ONLY from the provided context
- If partial information is available, answer using that
- If completely missing, say: 'Not available in the document'
- Do NOT hallucinate
- Keep answers precise and structured
- Ignore irrelevant context

Format:
- Use bullet points
- Be concise and factual"""

SYSTEM_PROMPT_SHORT = STRICT_RAG_PROMPT
SYSTEM_PROMPT_ELABORATE = STRICT_RAG_PROMPT
SYSTEM_PROMPT_DETAIL = STRICT_RAG_PROMPT
SYSTEM_PROMPT = STRICT_RAG_PROMPT
