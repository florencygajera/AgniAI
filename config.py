"""
config.py
=========
Central configuration for AgniAI.
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
CHUNK_WORDS   = 200
CHUNK_OVERLAP = 40

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K     = 2
MIN_SCORE = 0.01

# ── Memory ─────────────────────────────────────────────────────────────────
MEMORY_MAX_MESSAGES = 6

# ── Network ────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 90

# ── Context budget ─────────────────────────────────────────────────────────
MAX_CONTEXT_CHARS = 1500

# ── CORS ───────────────────────────────────────────────────────────────────
# Origins allowed to call this API.
# During development: "*" allows everything (React dev server, .NET, Postman).
# Before production: replace with exact URLs, e.g.:
#   ALLOWED_ORIGINS = [
#       "http://localhost:3000",        # React dev
#       "http://localhost:5173",        # Vite dev
#       "https://yourapp.azurewebsites.net",  # deployed .NET
#   ]
ALLOWED_ORIGINS = "*"

# ── Answer-style keywords ──────────────────────────────────────────────────
# These are checked against the user's question (lowercase) in main.py.
# First match wins; if none match, ELABORATE is used as the default.

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

# ── System prompts — one per answer style ──────────────────────────────────
#
# SHORT     : 1-3 bullet points only, no padding.
# ELABORATE : Structured bullets with sub-points and brief context.
# DETAIL    : Full explanation — summary, numbered sections, figures, takeaway.
#
# The plain SYSTEM_PROMPT alias is kept for backward-compat with
# ollama_cpu_chat.py which imports it directly; it maps to ELABORATE.

SYSTEM_PROMPT_SHORT = """\
You are AgniAI, an assistant for India's Agniveer recruitment.
Read the reference information below carefully.
Answer in 1-3 SHORT bullet points ONLY — no extra commentary, no filler.
End your answer with:  Source: <source name from the reference>.
If the answer is not in the reference, reply only: "Not found in provided documents."
"""

SYSTEM_PROMPT_ELABORATE = """\
You are AgniAI, an assistant for India's Agniveer recruitment.
Read the reference information below carefully.
Answer using well-organised bullet points with sub-points where helpful.
Add one or two sentences of context so the reader understands each point.
End your answer with:  Source: <source name from the reference>.
If the answer is not in the reference, reply only: "Not found in provided documents."
"""

SYSTEM_PROMPT_DETAIL = """\
You are AgniAI, an assistant for India's Agniveer recruitment.
Read the reference information below carefully and give a DETAILED, comprehensive answer.

Structure your response like this:
  1. One-sentence summary of the topic.
  2. Numbered sections with clear headings covering every relevant aspect.
  3. Include eligibility specifics, figures, timelines, and examples from the reference.
  4. A "Key Takeaway" line at the end.
  5. Source: <source name from the reference>.

If the answer is not in the reference, reply only: "Not found in provided documents."
"""

# Default — used by ollama_cpu_chat.py and as the fallback style
SYSTEM_PROMPT = SYSTEM_PROMPT_ELABORATE