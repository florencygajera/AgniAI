from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

DOCSTORE_PATH = INDEX_DIR / "docstore.json"
FAISS_INDEX_PATH = INDEX_DIR / "agni.index"

# ── Embeddings ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
DEFAULT_MODEL = "llama3"
FALLBACK_MODELS = ["llama3", "mistral", "phi3", "gemma", "llama2"]

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_WORDS = 650
CHUNK_OVERLAP = 50

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K = 4
MIN_SCORE = 0.15          # discard chunks below this cosine similarity

# ── Memory ─────────────────────────────────────────────────────────────────
MEMORY_MAX_MESSAGES = 10  # keeps last 5 exchanges (user + assistant each)

# ── Network ────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 120

# ── Prompts ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are AgniAI — an expert assistant for everything related to India's Agniveer / Agnipath military recruitment scheme.

Rules:
1. Answer ONLY from the retrieved context provided below the question.
2. If the answer is not present in the context, respond with exactly:
   "I don't have that information in my knowledge base. Please ingest the relevant document first."
3. Never hallucinate or make up details.
4. Keep your tone helpful, professional, and concise.
5. Format lists and numbers clearly using plain-text bullets (•) or numbered steps.
6. Always cite the source file/URL at the end in a "📌 Source:" line when available.
"""