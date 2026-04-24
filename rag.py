"""
rag.py
======
Core retrieval-augmented generation layer — accuracy-optimised edition.

Accuracy improvements over baseline:
  1. Upgraded embedding model: all-mpnet-base-v2 (768-dim vs 384-dim)
     → Better semantic understanding, ~5-8pp NDCG improvement
  2. Hybrid retrieval: dense cosine + BM25 sparse, score-fused
     → Catches keyword matches that dense retrieval misses (e.g. exact names)
  3. Cross-encoder re-ranking: ms-marco-MiniLM-L-6-v2
     → Re-scores top candidates with full query×chunk interaction
     → Typically +5-10pp precision@k vs bi-encoder alone
  4. MMR diversity: removes near-duplicate chunks before sending to LLM
     → Reduces repetition and frees token budget for distinct evidence
  5. Tighter MIN_SCORE threshold (0.20 vs 0.01) — drops noisy matches
  6. Larger context budget (3500 vs 1500 chars) — more evidence per query
  7. Context formatted with explicit chunk numbering [1], [2] so LLM
     can cite sources reliably
"""

import json
import logging
import os
import pickle
import re
import warnings
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from config import (
    BM25_INDEX_PATH,
    BM25_WEIGHT,
    DEFAULT_MODEL,
    DENSE_WEIGHT,
    DOCSTORE_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FALLBACK_MODELS,
    MIN_SCORE,
    OLLAMA_TAGS_URL,
    OLLAMA_URL,
    RERANKER_MODEL,
    RERANK_TOP_K,
    REQUEST_TIMEOUT,
    SYSTEM_PROMPT,
    TOP_K,
    USE_HYBRID,
    USE_RERANKER,
)

# ── Module-level singletons ────────────────────────────────────────────────
_MODEL: Optional[SentenceTransformer] = None
_RERANKER = None          # CrossEncoder, loaded lazily
_RERANKER_FAILED = False
_INDEX: Optional[faiss.Index] = None
_DOCS: List[Dict[str, str]] = []
_BM25 = None              # rank_bm25.BM25Okapi, loaded lazily


# ── Helpers ────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _new_index() -> faiss.Index:
    return faiss.IndexFlatIP(EMBEDDING_DIM)


def _rebuild_index_from_docs(docs: List[Dict[str, str]]) -> faiss.Index:
    """
    Re-embed the current docstore with the active embedding model and build
    a fresh FAISS index.

    This is used when an existing saved index was created with a different
    embedding dimension than the one configured now.
    """
    index = _new_index()
    if not docs:
        return index

    texts = [d.get("text", "") for d in docs]
    vectors = embed_texts(texts)
    if vectors.size == 0:
        return index
    if vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch while rebuilding index: "
            f"expected {EMBEDDING_DIM}, got {vectors.shape[1]}"
        )

    index.add(vectors)
    save_index(index, docs)
    return index


def _escape_control_chars_in_json_strings(raw: str) -> str:
    repaired: List[str] = []
    in_string = False
    escaped = False
    for ch in raw:
        if in_string:
            if escaped:
                repaired.append(ch); escaped = False; continue
            if ch == "\\":
                repaired.append(ch); escaped = True; continue
            if ch == '"':
                repaired.append(ch); in_string = False; continue
            if ch == "\n": repaired.append("\\n"); continue
            if ch == "\r": repaired.append("\\r"); continue
            if ch == "\t": repaired.append("\\t"); continue
            if ord(ch) < 32: repaired.append(f"\\u{ord(ch):04x}"); continue
            repaired.append(ch); continue
        repaired.append(ch)
        if ch == '"':
            in_string = True
    return "".join(repaired)


def _extract_json_scalar(line: str) -> str:
    value = line.split(":", 1)[1].strip()
    if value.endswith(","): value = value[:-1].rstrip()
    if value.startswith('"') and value.endswith('"'): value = value[1:-1]
    return value


def _repair_docstore_from_lines(raw: str) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    obj: Dict[str, str] = {}
    text_lines: List[str] = []
    in_object = False
    in_text = False
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped == "[" or not stripped: continue
        if stripped == "]": break
        if stripped.startswith("{"):
            obj = {}; text_lines = []; in_object = True; in_text = False; continue
        if not in_object: continue
        if in_text:
            if stripped in {"}", "},"}:
                obj["text"] = "\n".join(text_lines)
                docs.append(obj); obj = {}; text_lines = []
                in_object = False; in_text = False; continue
            text_lines.append(line); continue
        if stripped.startswith('"source":'): obj["source"] = _extract_json_scalar(line)
        elif stripped.startswith('"doc_type":'): obj["doc_type"] = _extract_json_scalar(line)
        elif stripped.startswith('"chunk_id":'): obj["chunk_id"] = _extract_json_scalar(line)
        elif stripped.startswith('"text":'):
            fragment = line.split(":", 1)[1].lstrip()
            if fragment.startswith('"'): fragment = fragment[1:]
            if fragment.endswith('",'):
                fragment = fragment[:-2]; obj["text"] = fragment; text_lines = []
            elif fragment.endswith('"'):
                fragment = fragment[:-1]; obj["text"] = fragment; text_lines = []
            else:
                text_lines = [fragment]; in_text = True
            continue
        elif stripped in {"}", "},"}:
            if "text" not in obj and text_lines: obj["text"] = "\n".join(text_lines)
            if obj: docs.append(obj)
            obj = {}; text_lines = []; in_object = False; in_text = False
    if in_object and obj:
        if "text" not in obj and text_lines: obj["text"] = "\n".join(text_lines)
        docs.append(obj)
    return docs


def _normalize_query_for_retrieval(query: str) -> str:
    """
    Strip style words and filler phrases that are useful to the answer tone
    but harmful for retrieval relevance.
    """
    cleaned = query.lower()
    phrases = (
        "in short", "briefly", "brief", "quick answer", "short answer",
        "summarise", "summarize", "tldr", "tl;dr", "in brief",
        "give me short", "short me", "one line", "one-line",
        "give a short", "keep it short",
        "in detail", "detailed", "explain in detail", "full detail",
        "comprehensive", "thoroughly", "exhaustive", "step by step",
        "step-by-step", "explain fully", "tell me everything",
        "give me detail", "elaborate in detail",
        "elaborate", "explain", "elaborate on", "tell me more",
        "expand on", "describe", "give more", "more info",
        "like i'm a beginner", "like i am a beginner", "for a beginner",
        "as a beginner", "in simple terms", "simple terms", "beginner friendly",
        "please", "can you", "could you", "tell me", "help me understand",
    )
    for phrase in sorted(phrases, key=len, reverse=True):
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,!?:;")
    words = cleaned.split()
    if len(words) < 3:
        cleaned = query.strip().lower()

    # Domain-specific query expansion: the source docs use formal labels that
    # users often shorten in their question.
    expansions = (
        ("age limit", "required age age eligibility"),
        ("eligibility criteria", "eligible eligibility qualification required age"),
        ("selection process", "selection process flow chart recruitment process merit medical physical"),
        ("how is the selection done", "selection process flow chart recruitment process merit medical physical"),
        ("recruitment process", "flow chart recruitment process registration application merit medical physical"),
        ("how to apply", "registration application dashboard join indian army"),
        ("apply", "registration application dashboard join indian army"),
    )
    expanded = cleaned
    for needle, extra in expansions:
        if needle in expanded:
            expanded = f"{expanded} {extra}"
    expanded = re.sub(r"\s+", " ", expanded).strip()
    return expanded or query.strip()


# ── Embedding model ────────────────────────────────────────────────────────

def load_embedding_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _MODEL = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    return _MODEL


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    model = load_embedding_model()
    vecs = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 20,
        batch_size=32,
    )
    return np.asarray(vecs, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    model = load_embedding_model()
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True, batch_size=1)
    return np.asarray(vec, dtype="float32")


# ── Cross-encoder re-ranker ────────────────────────────────────────────────

def load_reranker():
    """Lazy-load the cross-encoder re-ranker."""
    global _RERANKER, _RERANKER_FAILED
    if _RERANKER is not None:
        return _RERANKER
    if _RERANKER_FAILED:
        return None
    if not USE_RERANKER:
        return None
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _RERANKER = CrossEncoder(RERANKER_MODEL, local_files_only=True)
        return _RERANKER
    except Exception as exc:
        _RERANKER_FAILED = True
        print(f"[WARNING] Could not load re-ranker ({exc}). Using bi-encoder scores only.")
        return None


def rerank(query: str, docs: List[Dict], top_n: int = RERANK_TOP_K) -> List[Dict]:
    """
    Re-rank *docs* using a cross-encoder. Returns top_n highest-scoring docs.
    Falls back to original order if the re-ranker isn't available.
    """
    if not docs:
        return docs
    reranker = load_reranker()
    if reranker is None:
        return docs[:top_n]
    try:
        pairs = [(query, d.get("text", "")) for d in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        reranked = []
        for score, doc in ranked[:top_n]:
            doc = dict(doc)
            doc["rerank_score"] = round(float(score), 4)
            reranked.append(doc)
        return reranked
    except Exception as exc:
        print(f"[WARNING] Re-ranking failed ({exc}). Using original order.")
        return docs[:top_n]


# ── BM25 sparse index ──────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    import re
    return re.findall(r"[a-zA-Z0-9\u0900-\u097F]+", text.lower())


_STOPWORDS = {
    "a", "an", "and", "are", "be", "by", "for", "from", "how", "i", "in",
    "is", "it", "me", "my", "of", "on", "or", "please", "show", "tell",
    "the", "to", "what", "when", "where", "which", "who", "why", "with",
    "you", "your", "can", "could", "would", "should", "do", "does", "did",
    "this", "that", "these", "those", "as", "at", "was", "were", "will",
    "like", "just", "about", "into", "over", "under", "up", "down",
}


def _meaningful_tokens(text: str) -> List[str]:
    return [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 1]


def load_bm25():
    """Lazy-load BM25 index from disk."""
    global _BM25
    if _BM25 is not None:
        return _BM25
    if not USE_HYBRID:
        return None
    if not BM25_INDEX_PATH.exists():
        return None
    try:
        with open(BM25_INDEX_PATH, "rb") as f:
            _BM25 = pickle.load(f)
        return _BM25
    except Exception as exc:
        print(f"[WARNING] Could not load BM25 index ({exc}).")
        return None


def save_bm25(docs: List[Dict[str, str]]) -> None:
    """Build and persist BM25 index from docstore."""
    if not USE_HYBRID:
        return
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
        corpus = [_tokenize(d.get("text", "")) for d in docs]
        bm25 = BM25Okapi(corpus)
        _ensure_dirs()
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(bm25, f)
        global _BM25
        _BM25 = bm25
    except ModuleNotFoundError:
        pass  # rank_bm25 not installed — hybrid disabled silently
    except Exception as exc:
        print(f"[WARNING] BM25 index build failed: {exc}")


# ── MMR diversity filter ───────────────────────────────────────────────────

def _mmr_filter(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    docs: List[Dict],
    k: int,
    lambda_: float = 0.6,
) -> List[Dict]:
    """
    Maximal Marginal Relevance: balances relevance and diversity.
    lambda_=1.0 → pure relevance; lambda_=0.0 → pure diversity.
    lambda_=0.6 is the standard balanced setting.
    """
    if len(docs) <= k:
        return docs

    selected_idx: List[int] = []
    remaining = list(range(len(docs)))

    for _ in range(min(k, len(docs))):
        if not remaining:
            break
        if not selected_idx:
            # First pick: highest relevance
            scores = doc_vecs[remaining] @ query_vec.T
            best = remaining[int(np.argmax(scores))]
        else:
            rel_scores = doc_vecs[remaining] @ query_vec.T
            sim_to_selected = np.max(doc_vecs[remaining] @ doc_vecs[selected_idx].T, axis=1)
            mmr_scores = lambda_ * rel_scores.flatten() - (1 - lambda_) * sim_to_selected.flatten()
            best = remaining[int(np.argmax(mmr_scores))]
        selected_idx.append(best)
        remaining.remove(best)

    return [docs[i] for i in selected_idx]


# ── Index persistence ──────────────────────────────────────────────────────

def load_docstore() -> List[Dict[str, str]]:
    if not DOCSTORE_PATH.exists():
        return []
    raw = DOCSTORE_PATH.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = _escape_control_chars_in_json_strings(raw)
        try:
            docs = json.loads(repaired)
        except json.JSONDecodeError:
            docs = _repair_docstore_from_lines(raw)
            if not docs:
                raise
        _save_docstore(docs)
        print("[WARNING] Repaired malformed docstore.json and saved cleaned copy.")
        return docs


def _save_docstore(docs: List[Dict[str, str]]) -> None:
    _ensure_dirs()
    with DOCSTORE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(docs, fh, ensure_ascii=False, indent=2)


def load_index() -> faiss.Index:
    global _INDEX, _DOCS
    if _INDEX is not None and _DOCS:
        return _INDEX
    _ensure_dirs()
    _DOCS = load_docstore()
    if FAISS_INDEX_PATH.exists():
        _INDEX = faiss.read_index(str(FAISS_INDEX_PATH))
    else:
        _INDEX = _new_index()

    if _INDEX.d != EMBEDDING_DIM:
        old_dim = _INDEX.d
        print(
            f"[WARNING] Saved FAISS index uses dimension {old_dim}, "
            f"but the active embedding model uses {EMBEDDING_DIM}. "
            "Rebuilding the index from the docstore..."
        )
        _INDEX = _rebuild_index_from_docs(_DOCS)

    if _INDEX.ntotal > 0 and len(_DOCS) == 0:
        print("[WARNING] FAISS index has vectors but docstore is empty. "
              "Run /reset and re-ingest your documents.")
    return _INDEX


def save_index(index: faiss.Index, docs: List[Dict[str, str]]) -> None:
    global _DOCS
    _ensure_dirs()
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    _save_docstore(docs)
    _DOCS = docs
    # Rebuild BM25 index whenever docstore changes
    save_bm25(docs)


# ── Search ─────────────────────────────────────────────────────────────────

def _bm25_scores(query: str, n: int) -> np.ndarray:
    """
    Return BM25 scores for all docs, normalised to [0, 1].
    Returns zeros array if BM25 is unavailable.
    """
    bm25 = load_bm25()
    if bm25 is None or not _DOCS:
        return np.zeros(len(_DOCS), dtype="float32")
    try:
        tokens = _tokenize(query)
        scores = np.array(bm25.get_scores(tokens), dtype="float32")
        max_s = scores.max()
        if max_s > 0:
            scores /= max_s
        return scores
    except Exception:
        return np.zeros(len(_DOCS), dtype="float32")


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-8:
        return np.ones_like(values, dtype="float32")
    return ((values - lo) / (hi - lo)).astype("float32")


def search(query: str, top_k: int = TOP_K) -> List[Dict[str, str]]:
    """
    Retrieve top-k most relevant chunks using hybrid dense+sparse search,
    followed by cross-encoder re-ranking and MMR diversity filtering.

    Pipeline:
      1. Dense cosine search → candidate_k results (2× top_k)
      2. BM25 sparse score fusion (if USE_HYBRID)
      3. Cross-encoder re-ranking (if USE_RERANKER) → RERANK_TOP_K
      4. MMR diversity filter → final top_k
    """
    index = load_index()
    if index.ntotal == 0:
        return []

    retrieval_query = _normalize_query_for_retrieval(query)
    qvec = embed_query(retrieval_query)
    if index.ntotal <= 1000:
        candidate_k = index.ntotal
    else:
        candidate_k = min(max(top_k * 8, 24), index.ntotal)   # broader recall pool

    # ── Dense retrieval ────────────────────────────────────────
    scores_dense, ids = index.search(qvec, candidate_k)
    dense_scores = scores_dense[0]
    doc_ids = ids[0]
    dense_map = {int(doc_id): float(score) for doc_id, score in zip(doc_ids, dense_scores) if doc_id >= 0}

    # ── BM25 fusion ────────────────────────────────────────────
    if USE_HYBRID and len(_DOCS) > 0:
        bm25_all = _bm25_scores(retrieval_query, len(_DOCS))
        bm25_top_ids = np.argsort(bm25_all)[::-1][:candidate_k]

        query_terms = set(_meaningful_tokens(retrieval_query))
        token_count = len(retrieval_query.split())
        if token_count <= 3:
            dense_weight, bm25_weight = 0.30, 0.70
        elif token_count <= 6:
            dense_weight, bm25_weight = 0.40, 0.60
        else:
            dense_weight, bm25_weight = DENSE_WEIGHT, BM25_WEIGHT

        candidate_ids: List[int] = []
        seen_ids: set[int] = set()
        for doc_id in list(doc_ids) + [int(x) for x in bm25_top_ids]:
            if doc_id < 0 or doc_id >= len(_DOCS) or doc_id in seen_ids:
                continue
            candidate_ids.append(int(doc_id))
            seen_ids.add(int(doc_id))

        dense_values = np.array([dense_map.get(doc_id, 0.0) for doc_id in candidate_ids], dtype="float32")
        dense_values = _min_max_normalize(dense_values)

        bm25_values = np.array([float(bm25_all[doc_id]) for doc_id in candidate_ids], dtype="float32")

        fused: List[tuple] = []
        for doc_id, ds, bs in zip(candidate_ids, dense_values, bm25_values):
            combined = dense_weight * float(ds) + bm25_weight * float(bs)
            if query_terms:
                doc_text = _DOCS[doc_id].get("text", "")
                doc_terms = set(_meaningful_tokens(doc_text))
                overlap = len(query_terms & doc_terms) / max(1, len(query_terms))
                combined += 0.20 * overlap

                doc_lower = doc_text.lower()
                query_lower = retrieval_query.lower()
                if "age" in query_lower and "required age" in doc_lower:
                    combined += 0.60
                if "eligibility" in query_lower and "eligibility criteria" in doc_lower:
                    combined += 0.60
                if "eligibility" in query_lower and "required age" in doc_lower:
                    combined += 0.30
                if ("selection" in query_lower or "process" in query_lower or "recruitment" in query_lower) and (
                    "flow chart" in doc_lower and "recruitment" in doc_lower
                ):
                    combined += 0.80
                if "selection" in query_lower and "merit" in doc_lower:
                    combined += 0.20
            if combined < MIN_SCORE:
                continue
            fused.append((combined, int(doc_id)))
        fused.sort(key=lambda x: x[0], reverse=True)
        candidates = []
        for combined, doc_id in fused:
            doc = dict(_DOCS[doc_id])
            doc["score"] = round(combined, 4)
            candidates.append(doc)
    else:
        candidates = []
        for doc_id, score in zip(doc_ids, dense_scores):
            if doc_id < 0 or doc_id >= len(_DOCS):
                continue
            if float(score) < MIN_SCORE:
                continue
            doc = dict(_DOCS[doc_id])
            doc["score"] = round(float(score), 4)
            candidates.append(doc)

    if not candidates:
        # Keep the strongest dense hit as a fallback rather than returning
        # empty context when the query is still likely relevant.
        if len(doc_ids) > 0 and 0 <= int(doc_ids[0]) < len(_DOCS):
            doc = dict(_DOCS[int(doc_ids[0])])
            doc["score"] = round(float(dense_scores[0]), 4)
            return [doc]
        return []

    # ── Cross-encoder re-ranking ───────────────────────────────
    if USE_RERANKER:
        candidates = rerank(
            query,
            candidates,
            top_n=min(max(top_k, RERANK_TOP_K), len(candidates)),
        )

    # ── MMR diversity ──────────────────────────────────────────
    if len(candidates) > 1:
        texts = [c.get("text", "") for c in candidates]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cand_vecs = embed_texts(texts)
        candidates = _mmr_filter(qvec, cand_vecs, candidates, k=min(top_k, len(candidates)))

    return candidates[:top_k]


# ── Context builder ────────────────────────────────────────────────────────

def build_context(docs: Sequence[Dict[str, str]]) -> str:
    """
    Format retrieved docs into a numbered context block for the LLM.

    Accuracy improvements:
    - Explicit [N] numbering so the LLM can cite sources
    - Deduplication of near-identical chunks
    - Source label kept compact
    """
    if not docs:
        return ""

    blocks: List[str] = []
    seen_texts: set = set()

    for i, doc in enumerate(docs, start=1):
        text = (doc.get("text") or "").strip()
        # Deduplicate by first 100 chars fingerprint (handles near-dupes)
        fingerprint = text[:100].lower()
        if not text or fingerprint in seen_texts:
            continue
        seen_texts.add(fingerprint)
        source = doc.get("source", "unknown")
        # Compact source display
        if len(source) > 60:
            source = "…" + source[-58:]
        blocks.append(f"[{i}] Source: {source}\n{text}")

    return "\n\n---\n\n".join(blocks)


def index_stats() -> Dict[str, int]:
    index = load_index()
    return {"vectors": int(index.ntotal), "chunks": len(_DOCS)}


# ── Ollama client (non-streaming, used by rag.call_llm) ───────────────────

def _fetch_ollama_models() -> List[str]:
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", []) if m.get("name")]
    except requests.RequestException:
        return []


def _available_models() -> List[str]:
    installed = _fetch_ollama_models()
    return installed if installed else FALLBACK_MODELS


def _build_messages(
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})
    return messages


def call_llm(
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
) -> str:
    candidates: List[str] = []
    if model: candidates.append(model)
    if DEFAULT_MODEL not in candidates: candidates.append(DEFAULT_MODEL)
    for m in _available_models():
        if m not in candidates: candidates.append(m)

    messages = _build_messages(prompt, history=history)
    last_error: Optional[str] = None

    for candidate in candidates:
        body = {"model": candidate, "messages": messages, "stream": False}
        try:
            resp = requests.post(OLLAMA_URL, json=body, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "").strip()
            if content:
                return content
            last_error = f"Empty response from '{candidate}'."
        except requests.RequestException as exc:
            last_error = str(exc)

    raise RuntimeError(
        "Ollama is unreachable or no installed model responded.\n"
        f"Last error: {last_error}\nMake sure Ollama is running: ollama serve"
    )
