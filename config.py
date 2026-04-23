"""
config.py
=========
Central configuration for AgniAI.
All values can be overridden by environment variables where noted.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

DOCSTORE_PATH    = INDEX_DIR / "docstore.json"
FAISS_INDEX_PATH = INDEX_DIR / "agni.index"

# ── Embeddings ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# Prefer smallest capable models for CPU-only hardware.
# llama3:latest (8B) is deliberately last — it is too slow on most CPUs.
DEFAULT_MODEL   = "phi3:mini"
FALLBACK_MODELS = [
    "phi3:mini",
    "phi3:3.8b",
    "gemma2:2b",
    "llama3.2:3b",
    "llama3.2:1b",
    "mistral:7b-instruct-q4_0",
    "llama3:latest",
]

# ── Chunking ───────────────────────────────────────────────────────────────
# Smaller chunks → smaller embedding batches → faster ingestion.
CHUNK_WORDS   = 400   # was 650 — reduced to keep each chunk under ~2 KB
CHUNK_OVERLAP = 40

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K     = 3
MIN_SCORE = 0.05   # low threshold — avoids silently discarding weak-but-relevant chunks

# ── Memory ─────────────────────────────────────────────────────────────────
MEMORY_MAX_MESSAGES = 6   # 3 user + 3 assistant turns

# ── Network ────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 90   # seconds — for the non-streaming rag.py call_llm path

# ── Context budget sent to the LLM ─────────────────────────────────────────
# Keeping this small is the #1 way to reduce time-to-first-token on CPU.
MAX_CONTEXT_CHARS = 1800   # ~450 tokens — fits comfortably in num_ctx=1024

# ── Prompts ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are AgniAI — a concise expert on India's Agniveer / Agnipath recruitment.

Rules:
1. Answer ONLY from the retrieved context provided.
2. If the answer is not in the context, say exactly:
   "I don't have that information. Please ingest the relevant document first."
3. Never invent details.
4. Be concise and structured. Use plain-text bullets (•) or numbered steps.
5. End with "📌 Source: <filename/url>" when a source is available.
"""