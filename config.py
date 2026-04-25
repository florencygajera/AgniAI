"""
config.py
=========
Central configuration for AgniAI — accuracy-maximised edition.

Key improvements over previous version:
  • Per-style MAX_CONTEXT_CHARS so SHORT gets tight context and DETAIL gets full context
  • Per-style MAX_TOKENS so LLM is never truncated mid-sentence in detail mode
  • Prompts are now iron-clad: zero tolerance for hallucination,
    mandatory "Not found in the provided documents." fallback
  • MIN_SCORE raised to 0.20 — noisy chunks below this threshold are worthless
  • RERANK_TOP_K raised to 4 so more evidence reaches the LLM after re-ranking
  • Style keyword lists use whole-word boundaries (enforced in main.py)
  • BM25 STOPWORDS list extended to avoid dropping "age", "pay", "rank"
  • CHUNK_WORDS lowered to 120 for tighter, more precise chunks
  • CHUNK_OVERLAP raised to 40 to preserve boundary context
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

DOCSTORE_PATH    = INDEX_DIR / "docstore.json"
FAISS_INDEX_PATH = INDEX_DIR / "agni.index"
BM25_INDEX_PATH  = INDEX_DIR / "bm25.pkl"

# ── Embeddings ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM   = 768

RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKER    = True   # set False to skip re-ranking (faster, less accurate)

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
# Smaller chunks = more precise retrieval hits per vector.
# Higher overlap = sentences at boundaries are still captured in both chunks.
CHUNK_WORDS     = 120   # tighter chunks → higher precision per vector
CHUNK_OVERLAP   = 40    # ~33% overlap ratio — preserves boundary sentences
CHUNK_MIN_WORDS = 12    # discard micro-fragments (e.g. lone table headers)

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K        = 6        # retrieve more candidates before re-ranking
RERANK_TOP_K = 4        # keep best N after cross-encoder re-ranking
MIN_SCORE    = 0.20     # drop low-similarity noise (raised from 0.18)

# Hybrid retrieval weights  (dense cosine + BM25 sparse)
DENSE_WEIGHT = 0.55
BM25_WEIGHT  = 0.45
USE_HYBRID   = True

# ── Memory ─────────────────────────────────────────────────────────────────
MEMORY_MAX_MESSAGES = 10

# ── Network ────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 120

# ── Context budget (per answer style) ─────────────────────────────────────
# Different styles get different context budgets so SHORT is punchy and
# DETAIL has enough room to show every fact the retriever found.
MAX_CONTEXT_CHARS = {
    "short":     2500,
    "elaborate": 4000,
    "detail":    6000,
}
# Fallback for code paths that haven't been updated to pass style yet
MAX_CONTEXT_CHARS_DEFAULT = 4000

# ── LLM token budget (per answer style) ────────────────────────────────────
# Prevents LLM being cut off mid-sentence in DETAIL mode.
MAX_TOKENS_STYLE = {
    "short":     180,
    "elaborate": 400,
    "detail":    700,
}
MAX_TOKENS_DEFAULT = 400

# ── CORS ───────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = "*"

# ── Answer-style keywords ──────────────────────────────────────────────────
# All matching is done with whole-word boundaries in detect_answer_style()
# so "shorting" does NOT trigger SHORT mode.
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

# ── System prompts ─────────────────────────────────────────────────────────
#
# IRON-CLAD rules applied to all three prompts:
#   1. "Answer ONLY from the numbered reference chunks" — no hallucination
#   2. Mandatory fallback phrase if answer is not in docs
#   3. Cite chunk numbers so answers are traceable
#   4. Forbidden from inferring, guessing, or adding external knowledge
#   5. If a figure is approximate in the source, say "approximately"

_COMMON_RULES = """\
STRICT RULES — FOLLOW EXACTLY:
1. Answer ONLY using the numbered reference chunks provided below.
2. Do NOT add, infer, or assume any information not explicitly stated in the reference.
3. Do NOT use your general training knowledge. If it is not in the reference, it does not exist.
4. If the reference does not contain enough information to answer, reply with EXACTLY:
   "Not found in the provided documents. Please ingest the relevant official notification."
5. Quote exact figures, dates, percentages, and rupee amounts directly from the reference.
6. Cite the chunk numbers you used at the end, e.g. (Source: [1], [3]).
7. If you are uncertain about a specific figure, say "approximately" — ONLY when the source itself is approximate.
"""

SYSTEM_PROMPT_SHORT = f"""\
You are AgniAI, an expert assistant for India's Agniveer / Agnipath recruitment scheme.

{_COMMON_RULES}

OUTPUT FORMAT — SHORT:
• Answer in 2 to 4 bullet points ONLY.
• Each bullet must be one sentence maximum.
• No preamble. No padding. No introduction. Start directly with the first bullet.
• Example format:
  • Age: 17.5 – 21 years (Source: [2])
  • Salary Year 1: ₹30,000/month in-hand ₹21,000 (Source: [3])
"""

SYSTEM_PROMPT_ELABORATE = f"""\
You are AgniAI, an expert assistant for India's Agniveer / Agnipath recruitment scheme.

{_COMMON_RULES}

OUTPUT FORMAT — ELABORATE:
• Start with a 1-sentence summary of what the answer covers.
• Then use organised bullet points with sub-bullets where the reference supports it.
• Add context and relationships between facts ONLY when directly stated in the reference.
• Aim for 6–12 bullet points. No unnecessary padding.
• Finish with the chunk citations on a separate line.
"""

SYSTEM_PROMPT_DETAIL = f"""\
You are AgniAI, an expert assistant for India's Agniveer / Agnipath recruitment scheme.

{_COMMON_RULES}

OUTPUT FORMAT — DETAIL:
1. One-sentence executive summary.
2. Numbered sections with BOLD headings covering every relevant aspect found in the reference.
   Include ALL eligibility figures, salary breakdowns, timelines, test marks, and procedures.
3. Use sub-bullets under each heading for granular facts.
4. If the reference contains tables (salary, PFT marks, bonus marks), reproduce them as
   structured bullet lists — do NOT omit any row.
5. End with a "Key Takeaway" line that is one sentence.
6. Finish with chunk citations.

Do NOT summarise away specific numbers. Include every figure from the reference.
"""

# Default alias used by ollama_cpu_chat.py standalone mode
SYSTEM_PROMPT = SYSTEM_PROMPT_ELABORATE