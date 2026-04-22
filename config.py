from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

DOCSTORE_PATH = INDEX_DIR / "docstore.json"
FAISS_INDEX_PATH = INDEX_DIR / "agni.index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
DEFAULT_MODEL = "llama3"
FALLBACK_MODELS = ["llama3", "mistral", "phi3"]

CHUNK_WORDS = 650
CHUNK_OVERLAP = 50
TOP_K = 3
MEMORY_MAX_MESSAGES = 5

REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are AgniAI, an expert in Agniveer recruitment.\n"
    "Answer only from the given context.\n"
    "If not found, say 'I don't know'.\n"
    "Give structured answers."
)

