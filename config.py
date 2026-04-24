"""
config.py
=========
Central configuration for AgniAI — accuracy-optimised edition.

Key accuracy improvements over baseline:
  • Embedding model upgraded to all-mpnet-base-v2 (768-dim, higher accuracy)
  • TOP_K raised from 2 → 6 to retrieve more candidate chunks
  • RERANK_TOP_K — cross-encoder re-ranking keeps best 3 after retrieval
  • CHUNK_WORDS reduced 200→120, CHUNK_OVERLAP raised 40→30 overlap ratio
    (smaller chunks = more precise hits; overlap preserves context boundaries)
  • MIN_SCORE tightened 0.01 → 0.25 to drop noisy low-similarity chunks
  • MAX_CONTEXT_CHARS raised 1500 → 3000 to fit more evidence per query
  • MEMORY_MAX_MESSAGES raised 6 → 10 for longer coherent conversations
  • BM25_WEIGHT added for hybrid retrieval (dense + sparse fusion)
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

DOCSTORE_PATH    = INDEX_DIR / "docstore.json"
FAISS_INDEX_PATH = INDEX_DIR / "agni.index"
BM25_INDEX_PATH  = INDEX_DIR / "bm25.pkl"           # NEW: sparse index

# ── Embeddings ─────────────────────────────────────────────────────────────
# Upgraded from all-MiniLM-L6-v2 (384-dim) → all-mpnet-base-v2 (768-dim)
# Benchmark NDCG@10 improvement: ~5-8 pp on domain retrieval tasks.
# Trade-off: ~250 MB model, ~2× slower encode — acceptable for offline use.
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM   = 768

# Cross-encoder for re-ranking retrieved candidates
# ms-marco-MiniLM-L-6-v2 is the standard efficient choice (~80 MB)
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKER    = True          # set False to skip re-ranking (faster, less accurate)

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

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
# Smaller chunks = more precise retrieval hits.
# Higher overlap = sentences at boundaries are still captured.
CHUNK_WORDS   = 150     # was 200 — tighter = less noise per chunk
CHUNK_OVERLAP = 50      # was 40  — ~25% overlap ratio preserved
CHUNK_MIN_WORDS = 15    # NEW: discard tiny fragments (e.g. table headers alone)

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K        = 5        # was 2 — retrieve more candidates before re-ranking
RERANK_TOP_K = 3        # NEW: keep best N after cross-encoder re-ranking
MIN_SCORE    = 0.18     # was 0.01 — drop low-similarity noise

# Hybrid retrieval weights (dense cosine + BM25 sparse)
# Final score = DENSE_WEIGHT * cosine_score + BM25_WEIGHT * bm25_score
# Both normalised to [0, 1] before fusion.
DENSE_WEIGHT = 0.55     # semantic similarity
BM25_WEIGHT  = 0.45     # keyword overlap
USE_HYBRID   = True     # set False to use dense-only retrieval

# ── Memory ─────────────────────────────────────────────────────────────────
MEMORY_MAX_MESSAGES = 10   # was 6

# ── Network ────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 120   # was 90

# ── Context budget ─────────────────────────────────────────────────────────
MAX_CONTEXT_CHARS = 3500   # was 1500 — more evidence = better answers

# ── CORS ───────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = "*"

# ── Answer-style keywords ──────────────────────────────────────────────────
STYLE_SHORT_KEYWORDS = [
    "in short", "briefly", "brief", "quick answer", "short answer",
    "summarise", "summarize", "tldr", "tl;dr", "in brief",
    "give me short", "short me", "one line", "one-line",
    "give a short", "keep it short",
]

STYLE_DETAIL_KEYWORDS = [
    "in detail", "detailed", "explain in detail", "full detail",
    "comprehensive", "thoroughly", "exhaustive", "step by step",
    "step-by-step", "explain fully", "tell me everything",
    "give me detail", "elaborate in detail",
]

STYLE_ELABORATE_KEYWORDS = [
    "elaborate", "explain", "elaborate on", "tell me more",
    "expand on", "describe", "give more", "more info",
]

# ── System prompts ─────────────────────────────────────────────────────────
#
# Accuracy improvements in prompts:
#   1. Explicit "ONLY use the reference" instruction to reduce hallucination
#   2. Instructed to cite chunk numbers [1], [2] so answers are traceable
#   3. "If uncertain, say so" fallback prevents confident wrong answers
#   4. Forbidden from adding information not in the reference

SYSTEM_PROMPT_SHORT = """\
You are AgniAI, an expert assistant for India's Agniveer / Agnipath recruitment scheme.

STRICT RULES:
- Answer ONLY using the numbered reference chunks provided below.
- Do NOT add any information not present in the reference.
- Prefer exact facts from the reference over general knowledge.
- If the reference does not contain the answer, reply exactly:
  "Not found in the provided documents."
- Cite which chunk(s) you used, e.g. (Source: [1]).

FORMAT: 1-3 bullet points only. No preamble. No padding.
"""

SYSTEM_PROMPT_ELABORATE = """\
You are AgniAI, an expert assistant for India's Agniveer / Agnipath recruitment scheme.

STRICT RULES:
- Answer ONLY using the numbered reference chunks provided below.
- Do NOT add any information not present in the reference.
- Prefer exact facts from the reference over general knowledge.
- If the reference does not contain the answer, reply exactly:
  "Not found in the provided documents."
- If you are uncertain about a specific figure, say "approximately" only when the reference itself is approximate.
- Cite chunk numbers used, e.g. (Source: [1], [2]).

FORMAT: Well-organised bullet points with sub-bullets where helpful.
Add context only when it is directly supported by the reference.
"""

SYSTEM_PROMPT_DETAIL = """\
You are AgniAI, an expert assistant for India's Agniveer / Agnipath recruitment scheme.

STRICT RULES:
- Answer ONLY using the numbered reference chunks provided below.
- Do NOT add any information not present in the reference.
- Prefer exact facts from the reference over general knowledge.
- If the reference does not contain the answer, reply exactly:
  "Not found in the provided documents."
- Quote specific figures, dates, and thresholds directly from the reference.
- Cite chunk numbers used, e.g. (Source: [1], [2]).

FORMAT:
  1. One-sentence summary.
  2. Numbered sections with bold headings covering every relevant aspect found in the reference.
  3. Include all eligibility specifics, figures, timelines, and examples from the reference.
  4. "Key Takeaway" line at the end.
"""

# Default alias used by ollama_cpu_chat.py
SYSTEM_PROMPT = SYSTEM_PROMPT_ELABORATE
