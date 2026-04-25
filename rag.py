"""Retrieval helpers for AgniAI."""

from __future__ import annotations

import json
import logging
import os
import pickle
import hashlib
import re
import time
import warnings
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from runtime_cache import TTLCache

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
    REFERENCE_FALLBACK,
    RERANKER_MODEL,
    RERANK_TOP_K,
    REQUEST_TIMEOUT,
    STRICT_MIN_SCORE,
    STRICT_RAG_PROMPT,
    STRICT_TOP_K,
    MIN_RETRIEVAL_CONFIDENCE,
    SYSTEM_PROMPT,
    TOP_K,
    USE_HYBRID,
    USE_RERANKER,
    RETRIEVAL_CACHE_TTL,
    RESPONSE_CACHE_TTL,
    EMBED_CACHE_TTL,
    MAX_CACHE_ENTRIES,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHARS_DEFAULT,
    HIGH_RETRIEVAL_CONFIDENCE,
    LOW_RETRIEVAL_CONFIDENCE,
    style_structure_instruction,
)

logger = logging.getLogger(__name__)

_MODEL: Optional[SentenceTransformer] = None
_RERANKER = None
_RERANKER_FAILED = False
_INDEX: Optional[faiss.Index] = None
_DOCS: List[Dict[str, str]] = []
_BM25 = None
_QUERY_EMBED_CACHE = TTLCache(maxsize=MAX_CACHE_ENTRIES, ttl=EMBED_CACHE_TTL)
_RETRIEVAL_CACHE = TTLCache(maxsize=MAX_CACHE_ENTRIES, ttl=RETRIEVAL_CACHE_TTL)
_RESPONSE_CACHE = TTLCache(maxsize=MAX_CACHE_ENTRIES, ttl=RESPONSE_CACHE_TTL)


def _ensure_dirs() -> None:
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _new_index() -> faiss.Index:
    return faiss.IndexFlatIP(EMBEDDING_DIM)


def _escape_control_chars_in_json_strings(raw: str) -> str:
    repaired: List[str] = []
    in_string = False
    escaped = False
    for ch in raw:
        if in_string:
            if escaped:
                repaired.append(ch)
                escaped = False
                continue
            if ch == "\\":
                repaired.append(ch)
                escaped = True
                continue
            if ch == '"':
                repaired.append(ch)
                in_string = False
                continue
            if ch == "\n":
                repaired.append("\\n")
                continue
            if ch == "\r":
                repaired.append("\\r")
                continue
            if ch == "\t":
                repaired.append("\\t")
                continue
            if ord(ch) < 32:
                repaired.append(f"\\u{ord(ch):04x}")
                continue
            repaired.append(ch)
            continue
        repaired.append(ch)
        if ch == '"':
            in_string = True
    return "".join(repaired)


def _extract_json_scalar(line: str) -> str:
    value = line.split(":", 1)[1].strip()
    if value.endswith(","):
        value = value[:-1].rstrip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value


def _repair_docstore_from_lines(raw: str) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    obj: Dict[str, str] = {}
    text_lines: List[str] = []
    in_object = False
    in_text = False
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped == "[" or not stripped:
            continue
        if stripped == "]":
            break
        if stripped.startswith("{"):
            obj = {}
            text_lines = []
            in_object = True
            in_text = False
            continue
        if not in_object:
            continue
        if in_text:
            if stripped in {"}", "},"}:
                obj["text"] = "\n".join(text_lines)
                docs.append(obj)
                obj = {}
                text_lines = []
                in_object = False
                in_text = False
                continue
            text_lines.append(line)
            continue
        if stripped.startswith('"source":'):
            obj["source"] = _extract_json_scalar(line)
        elif stripped.startswith('"doc_type":'):
            obj["doc_type"] = _extract_json_scalar(line)
        elif stripped.startswith('"chunk_id":'):
            obj["chunk_id"] = _extract_json_scalar(line)
        elif stripped.startswith('"text":'):
            fragment = line.split(":", 1)[1].lstrip()
            if fragment.startswith('"'):
                fragment = fragment[1:]
            if fragment.endswith('",'):
                fragment = fragment[:-2]
                obj["text"] = fragment
                text_lines = []
            elif fragment.endswith('"'):
                fragment = fragment[:-1]
                obj["text"] = fragment
                text_lines = []
            else:
                text_lines = [fragment]
                in_text = True
            continue
        elif stripped in {"}", "},"}:
            if "text" not in obj and text_lines:
                obj["text"] = "\n".join(text_lines)
            if obj:
                docs.append(obj)
            obj = {}
            text_lines = []
            in_object = False
            in_text = False
    if in_object and obj:
        if "text" not in obj and text_lines:
            obj["text"] = "\n".join(text_lines)
        docs.append(obj)
    return docs


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9\u0900-\u097F]+", text.lower())


def _meaningful_tokens(text: str) -> List[str]:
    stopwords = {
        "a", "an", "and", "are", "be", "by", "for", "from", "how", "i", "in",
        "is", "it", "me", "my", "of", "on", "or", "please", "show", "tell",
        "the", "to", "what", "when", "where", "which", "who", "why", "with",
        "you", "your", "can", "could", "would", "should", "do", "does", "did",
        "this", "that", "these", "those", "as", "at", "was", "were", "will",
        "just", "about", "into", "over", "under", "up", "down",
    }
    return [t for t in _tokenize(text) if t not in stopwords and len(t) > 1]


def _chunk_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a_norm = _normalise_text(a)
    b_norm = _normalise_text(b)
    ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    if ratio >= 0.95:
        return ratio
    a_tokens = set(_meaningful_tokens(a_norm))
    b_tokens = set(_meaningful_tokens(b_norm))
    if not a_tokens or not b_tokens:
        return ratio
    jaccard = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    return max(ratio, jaccard)


def _dedupe_docs(docs: List[Dict[str, str]], similarity_threshold: float = 0.88) -> List[Dict[str, str]]:
    deduped: List[Dict[str, str]] = []
    seen_fingerprints: set[str] = set()
    for doc in docs:
        text = _normalise_text(doc.get("text", ""))
        if not text:
            continue
        fingerprint = text[:180]
        if fingerprint in seen_fingerprints:
            continue
        if any(_chunk_similarity(text, existing.get("text", "")) >= similarity_threshold for existing in deduped):
            continue
        seen_fingerprints.add(fingerprint)
        deduped.append(doc)
    return deduped


def _normalize_query_for_retrieval(query: str) -> str:
    cleaned = query.lower()
    filler_phrases = (
        "in short", "briefly", "brief", "quick answer", "short answer",
        "summarise", "summarize", "tldr", "tl;dr", "in brief",
        "give me short", "one line", "one-line",
        "give a short", "keep it short", "in detail", "detailed",
        "explain in detail", "full detail", "comprehensive", "exhaustive",
        "step by step", "step-by-step", "explain fully", "tell me everything",
        "give me detail", "elaborate in detail", "full explanation",
        "full breakdown", "break it down", "please", "can you", "could you",
    )
    for phrase in sorted(filler_phrases, key=len, reverse=True):
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,!?:;")
    if len(cleaned.split()) < 3:
        cleaned = query.strip().lower()

    expansions = (
        (r"\bage limit\b", "required age eligibility"),
        (r"\bage\b", "required age eligibility"),
        (r"\beligibilit", "eligibility criteria required age qualification"),
        (r"\bselection process\b", "recruitment process merit medical physical fitness"),
        (r"\bhow.*select", "recruitment process merit medical physical fitness"),
        (r"\brecruitment process\b", "registration rally medical"),
        (r"\bhow.*appl", "registration application"),
        (r"\bsalary\b", "customised package in hand seva nidhi monthly"),
        (r"\bpay\b", "customised package in hand monthly"),
        (r"\bphysical test\b", "physical fitness test pft 1.6 km run"),
        (r"\bpft\b", "physical fitness test 1.6 km run"),
        (r"\bbonus mark", "bonus marks ncc sports"),
        (r"\binsurance\b", "life insurance cover 48 lakhs"),
        (r"\bseva nidhi\b", "seva nidhi corpus fund exit after 4 year lakh"),
        (r"\btraining\b", "military training regimental centre"),
        (r"\bdocument", "documents required matric aadhaar domicile"),
        (r"\bmedical\b", "medical examination army medical standards"),
    )
    for pattern, extra in expansions:
        if re.search(pattern, cleaned):
            cleaned = f"{cleaned} {extra}"
            break
    return re.sub(r"\s+", " ", cleaned).strip() or query.strip()


def _rewrite_query_candidates(query: str) -> List[str]:
    q = query.strip().lower()
    candidates = [q]
    if any(word in q for word in ("calculate", "total", "sum", "overall", "aggregate", "combined")):
        candidates.append(
            re.sub(r"\b(calculate|total|sum|overall|aggregate|combined)\b", " ", q)
        )
    return [re.sub(r"\s+", " ", cand).strip() for cand in candidates if cand.strip()]


def _query_similarity(a: str, b: str) -> float:
    try:
        av = _cache_query_embedding(a)
        bv = _cache_query_embedding(b)
        return float(np.dot(av[0], bv[0]))
    except Exception:
        return 0.0


def safe_rewrite_query(query: str) -> str:
    candidates = _rewrite_query_candidates(query)
    if len(candidates) == 1:
        return candidates[0]
    original = candidates[0]
    best = original
    best_score = -1.0
    for candidate in candidates:
        score = _query_similarity(original, candidate)
        if score > best_score:
            best_score = score
            best = candidate
    if best != original and best_score < 0.88:
        logger.debug("Rewrite rejected for query=%r: best_score=%.3f", query, best_score)
        return original
    if best != original:
        logger.debug("Rewrite accepted for query=%r -> %r (score=%.3f)", query, best, best_score)
    return best


def _query_cache_key(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip().lower()


def _cache_query_embedding(query: str) -> np.ndarray:
    key = _query_cache_key(query)
    cached = _QUERY_EMBED_CACHE.get(key)
    if cached is not None:
        return cached
    vec = embed_query(query)
    _QUERY_EMBED_CACHE.set(key, vec)
    return vec


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def make_retrieval_cache_key(query: str, top_k: int) -> str:
    return f"{_query_cache_key(query)}|k={top_k}"


def make_response_cache_key(
    query: str,
    *,
    style: str,
    model: str,
    context: str,
    session_id: str,
) -> str:
    payload = "|".join(
        [style, model, session_id, _query_cache_key(query), _hash_text(context)]
    )
    return _hash_text(payload)


def get_cached_retrieval(query: str, top_k: int) -> Optional[List[Dict[str, str]]]:
    return _RETRIEVAL_CACHE.get(make_retrieval_cache_key(query, top_k))


def set_cached_retrieval(query: str, top_k: int, docs: List[Dict[str, str]]) -> None:
    _RETRIEVAL_CACHE.set(make_retrieval_cache_key(query, top_k), docs)


def get_cached_response(key: str) -> Optional[str]:
    return _RESPONSE_CACHE.get(key)


def set_cached_response(key: str, value: str) -> None:
    _RESPONSE_CACHE.set(key, value)


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
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=1,
    )
    return np.asarray(vec, dtype="float32")


def _reranker_local_files_available(model_name: str) -> bool:
    model_path = Path(model_name)
    if model_path.exists():
        return True
    hf_hub_cache = os.getenv("HF_HUB_CACHE")
    hf_home = os.getenv("HF_HOME")
    if hf_hub_cache:
        cache_root = Path(hf_hub_cache)
    elif hf_home:
        cache_root = Path(hf_home) / "hub"
    else:
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    if not repo_dir.exists():
        return False
    snapshots = repo_dir / "snapshots"
    if not snapshots.exists():
        return False
    return any(snapshot.is_dir() and any(snapshot.iterdir()) for snapshot in snapshots.iterdir())


def load_reranker():
    global _RERANKER, _RERANKER_FAILED
    if _RERANKER is not None:
        return _RERANKER
    if _RERANKER_FAILED or not USE_RERANKER:
        return None
    if not _reranker_local_files_available(RERANKER_MODEL):
        _RERANKER_FAILED = True
        logger.info("Reranker not available locally, skipping rerank step.")
        return None
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _RERANKER = CrossEncoder(RERANKER_MODEL, local_files_only=True)
        return _RERANKER
    except Exception as exc:
        _RERANKER_FAILED = True
        logger.warning("Could not load reranker: %s", exc)
        return None


def rerank(query: str, docs: List[Dict], top_n: int = RERANK_TOP_K) -> List[Dict]:
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
            new_doc = dict(doc)
            new_doc["rerank_score"] = round(float(score), 4)
            reranked.append(new_doc)
        return reranked
    except Exception as exc:
        logger.warning("Re-ranking failed, using original order: %s", exc)
        return docs[:top_n]


def load_bm25():
    global _BM25
    if _BM25 is not None:
        return _BM25
    if not USE_HYBRID or not BM25_INDEX_PATH.exists():
        return None
    try:
        with open(BM25_INDEX_PATH, "rb") as f:
            _BM25 = pickle.load(f)
        return _BM25
    except Exception as exc:
        logger.warning("Could not load BM25 index: %s", exc)
        return None


def save_bm25(docs: List[Dict[str, str]]) -> None:
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
        return
    except Exception as exc:
        logger.warning("BM25 index build failed: %s", exc)


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
        logger.warning("Repaired malformed docstore.json and saved cleaned copy.")
        return docs


def _save_docstore(docs: List[Dict[str, str]]) -> None:
    _ensure_dirs()
    with DOCSTORE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(docs, fh, ensure_ascii=False, indent=2)


def _rebuild_index_from_docs(docs: List[Dict[str, str]]) -> faiss.Index:
    index = _new_index()
    if not docs:
        return index
    texts = [d.get("text", "") for d in docs]
    vectors = embed_texts(texts)
    if vectors.size == 0:
        return index
    if vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch while rebuilding: expected {EMBEDDING_DIM}, got {vectors.shape[1]}"
        )
    index.add(vectors)
    save_index(index, docs)
    return index


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
        _INDEX = _rebuild_index_from_docs(_DOCS)
    if _INDEX.ntotal > 0 and len(_DOCS) == 0:
        logger.warning("FAISS index has vectors but docstore is empty.")
    return _INDEX


def save_index(index: faiss.Index, docs: List[Dict[str, str]]) -> None:
    global _DOCS, _INDEX
    _ensure_dirs()
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    _save_docstore(docs)
    _DOCS = docs
    _INDEX = index
    save_bm25(docs)


def _bm25_scores(query: str) -> np.ndarray:
    bm25 = load_bm25()
    if bm25 is None or not _DOCS:
        return np.zeros(len(_DOCS), dtype="float32")
    try:
        scores = np.array(bm25.get_scores(_tokenize(query)), dtype="float32")
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


_DOMAIN_BOOSTS = [
    (r"\bage\b", r"required age|eligibility", 0.70),
    (r"eligibilit", r"eligibility criteria|required age", 0.70),
    (r"selection|how.*select", r"recruitment process|flow chart", 0.90),
    (r"how.*appl|apply", r"registration|application", 0.70),
    (r"salary|pay|package", r"seva nidhi|in hand|monthly", 0.80),
    (r"physical|pft", r"physical fitness test|1\.6 km run", 0.80),
    (r"bonus mark", r"bonus marks|ncc|sports", 0.80),
    (r"insurance", r"48 lakh|life insurance", 0.80),
    (r"training", r"military training|regimental", 0.70),
    (r"document", r"documents required|matric|aadhaar|domicile", 0.70),
    (r"medical", r"medical examination|army medical", 0.70),
    (r"ncc", r"ncc.*certificate|bonus.*ncc", 0.70),
]


def _apply_domain_boosts(query_lower: str, doc_text_lower: str) -> float:
    best = 0.0
    for q_pat, d_pat, boost in _DOMAIN_BOOSTS:
        if re.search(q_pat, query_lower) and re.search(d_pat, doc_text_lower):
            best = max(best, boost)
    return best


def search(query: str, top_k: int = TOP_K) -> List[Dict[str, str]]:
    cached = get_cached_retrieval(query, top_k)
    if cached is not None:
        logger.debug("Retrieval cache hit for query=%r", query)
        return [dict(doc) for doc in cached]

    index = load_index()
    if index.ntotal == 0:
        return []

    rewritten_query = _normalize_query_for_retrieval(query)
    retrieval_query = safe_rewrite_query(rewritten_query)
    if retrieval_query != rewritten_query:
        logger.debug(
            "Query rewrite downgraded to preserve intent. original=%r rewritten=%r final=%r",
            query,
            rewritten_query,
            retrieval_query,
        )
    logger.debug("Retrieval query: original=%r rewritten=%r final=%r", query, rewritten_query, retrieval_query)
    qvec = _cache_query_embedding(retrieval_query)

    candidate_k = min(max(top_k * 8, 20), 60, index.ntotal)
    scores_dense, ids = index.search(qvec, candidate_k)
    dense_scores = scores_dense[0]
    doc_ids = ids[0]
    dense_map = {
        int(doc_id): float(score)
        for doc_id, score in zip(doc_ids, dense_scores)
        if doc_id >= 0
    }

    if USE_HYBRID and _DOCS:
        bm25_all = _bm25_scores(retrieval_query)
        bm25_top_ids = np.argsort(bm25_all)[::-1][:candidate_k]

        token_count = len(retrieval_query.split())
        if token_count <= 3:
            dense_weight, bm25_weight = 0.25, 0.75
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

        dense_values = np.array(
            [dense_map.get(doc_id, 0.0) for doc_id in candidate_ids], dtype="float32"
        )
        dense_values = _min_max_normalize(dense_values)
        bm25_values = np.array(
            [float(bm25_all[doc_id]) for doc_id in candidate_ids], dtype="float32"
        )
        query_terms = set(_meaningful_tokens(retrieval_query))
        query_lower = retrieval_query.lower()

        fused: List[tuple] = []
        for doc_id, ds, bs in zip(candidate_ids, dense_values, bm25_values):
            combined = dense_weight * float(ds) + bm25_weight * float(bs)
            doc_text = _DOCS[doc_id].get("text", "")
            if query_terms:
                doc_terms = set(_meaningful_tokens(doc_text))
                overlap = len(query_terms & doc_terms) / max(1, len(query_terms))
                combined += 0.15 * overlap
            combined += _apply_domain_boosts(query_lower, doc_text.lower())
            if combined >= MIN_SCORE:
                fused.append((combined, int(doc_id)))

        fused.sort(key=lambda item: item[0], reverse=True)
        candidates = []
        for combined, doc_id in fused:
            doc = dict(_DOCS[doc_id])
            doc["score"] = round(float(combined), 4)
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
        return []

    candidates = _dedupe_docs(candidates)

    if USE_RERANKER:
        candidates = rerank(query, candidates, top_n=min(max(top_k, RERANK_TOP_K), len(candidates)))

    candidates = sorted(candidates, key=lambda doc: float(doc.get("score", 0.0)), reverse=True)

    logger.debug(
        "Retrieved chunks for query=%r: %s",
        query,
        [
            {"score": doc.get("score"), "source": doc.get("source"), "chunk_id": doc.get("chunk_id")}
            for doc in candidates[: max(top_k, STRICT_TOP_K)]
        ],
    )
    final = candidates[: max(top_k, STRICT_TOP_K)]
    set_cached_retrieval(query, top_k, final)
    return [dict(doc) for doc in final]


def build_context(
    docs: Sequence[Dict[str, str]],
    *,
    max_chunks: int = STRICT_TOP_K,
    min_score: float = STRICT_MIN_SCORE,
    max_chars: int = 3000,
) -> str:
    if not docs:
        return ""

    ordered = sorted(docs, key=lambda doc: float(doc.get("score", 0.0)), reverse=True)
    ordered = [doc for doc in ordered if float(doc.get("score", 0.0)) >= min_score]
    ordered = _dedupe_docs(ordered)[:max_chunks]
    if not ordered:
        ordered = _dedupe_docs(sorted(docs, key=lambda doc: float(doc.get("score", 0.0)), reverse=True))[:max_chunks]

    if not ordered:
        return ""

    def _truncate_to_limit(text: str, limit: int) -> str:
        text = (text or "").strip()
        if limit <= 0 or len(text) <= limit:
            return text
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        if len(sentences) <= 1:
            return text[:limit].rstrip()
        pieces: List[str] = []
        total = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            extra = len(sentence) + (1 if pieces else 0)
            if total + extra > limit:
                break
            pieces.append(sentence)
            total += extra
        if pieces:
            return " ".join(pieces).strip()
        return text[:limit].rstrip()

    blocks: List[str] = []
    total_chars = 0
    for i, doc in enumerate(ordered, start=1):
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        source = doc.get("source", "unknown")
        if len(source) > 60:
            source = "..." + source[-57:]
        score = float(doc.get("score", 0.0))
        header = f"[{i}] score={score:.3f} source={source}\n"
        remaining = max_chars - total_chars - len(header)
        if remaining <= 0:
            break
        truncated_text = _truncate_to_limit(text, remaining)
        block = f"{header}{truncated_text}".strip()
        if not truncated_text:
            continue
        blocks.append(block)
        total_chars += len(block)
        if total_chars >= max_chars:
            break

    logger.debug(
        "Final context blocks: %s",
        [{"score": doc.get("score"), "source": doc.get("source")} for doc in ordered[: len(blocks)]],
    )
    return "\n\n---\n\n".join(blocks)


def retrieval_confidence(docs: Sequence[Dict[str, str]], query: str) -> float:
    if not docs:
        return 0.0
    ordered = sorted(docs, key=lambda doc: float(doc.get("score", 0.0)), reverse=True)
    top_score = float(ordered[0].get("score", 0.0))
    query_terms = set(_meaningful_tokens(query))
    if not query_terms:
        return min(1.0, top_score)
    top_text = ordered[0].get("text", "")
    overlap = len(query_terms & set(_meaningful_tokens(top_text))) / max(1, len(query_terms))
    confidence = (0.65 * top_score) + (0.35 * overlap)
    if len(ordered) > 1:
        confidence = min(1.0, confidence + 0.05 * min(2, len(ordered) - 1))
    return round(float(confidence), 4)


def is_reasoning_query(query: str) -> bool:
    q = query.lower()
    reasoning_terms = (
        "calculate", "total", "sum", "overall", "aggregate", "combined",
        "how much", "how many", "after 4 years", "over 4 years", "for 4 years",
        "what happens after", "in total",
    )
    return any(term in q for term in reasoning_terms)


def decide_answer_mode(
    *,
    query: str,
    docs: Sequence[Dict[str, str]],
    confidence: float,
) -> str:
    if not docs:
        return "reject"
    if confidence < LOW_RETRIEVAL_CONFIDENCE:
        return "strict_answer"
    if confidence < HIGH_RETRIEVAL_CONFIDENCE:
        return "strict_answer"
    return "normal_answer"


def prepare_rag_bundle(
    query: str,
    *,
    top_k: int = TOP_K,
    style: str = "elaborate",
    max_context_chars: Optional[int] = None,
) -> Dict[str, object]:
    retrieval_query = _normalize_query_for_retrieval(query)
    docs = search(retrieval_query, top_k=top_k)
    context_limit = (
        MAX_CONTEXT_CHARS.get(style, MAX_CONTEXT_CHARS_DEFAULT)
        if isinstance(MAX_CONTEXT_CHARS, dict)
        else MAX_CONTEXT_CHARS_DEFAULT
    )
    if max_context_chars is not None:
        context_limit = max(0, min(int(context_limit), int(max_context_chars)))
    confidence = retrieval_confidence(docs, query)
    mode = decide_answer_mode(query=query, docs=docs, confidence=confidence)
    context_min_score = STRICT_MIN_SCORE if mode == "normal_answer" else LOW_RETRIEVAL_CONFIDENCE
    context = build_context(
        docs,
        max_chunks=max(STRICT_TOP_K, min(5, top_k)),
        min_score=context_min_score,
        max_chars=context_limit,
    )
    logger.debug(
        "RAG bundle: confidence=%.3f low=%.2f high=%.2f mode=%s context_min=%.2f docs=%s",
        confidence,
        LOW_RETRIEVAL_CONFIDENCE,
        HIGH_RETRIEVAL_CONFIDENCE,
        mode,
        context_min_score,
        [
            {"score": d.get("score"), "source": d.get("source")}
            for d in docs[: min(5, len(docs))]
        ],
    )
    return {
        "query": query,
        "retrieval_query": retrieval_query,
        "docs": docs,
        "context": context,
        "confidence": confidence,
        "mode": mode,
        "reasoning": is_reasoning_query(query),
        "style": style,
    }


def answer_is_grounded(answer: str, context: str) -> bool:
    answer = (answer or "").strip()
    if not answer:
        return False
    if not context.strip():
        return answer.lower() == REFERENCE_FALLBACK.lower()

    answer_norm = _normalise_text(answer)
    context_norm = _normalise_text(context)
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", answer_norm)
    for num in numbers:
        if num not in context_norm:
            return False
    tokens = [tok for tok in _meaningful_tokens(answer_norm) if len(tok) >= 5]
    if not tokens:
        return True
    supported = sum(1 for tok in tokens if tok in context_norm)
    return supported / max(1, len(tokens)) >= 0.75


def build_strict_messages(
    query: str,
    *,
    context: str,
    style: str = "elaborate",
    reasoning: bool = False,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    system_content = STRICT_RAG_PROMPT
    if reasoning:
        system_content = STRICT_RAG_PROMPT_COMPUTE
    system_content = f"{system_content}\n\n{style_structure_instruction(style)}"
    messages = [{"role": "system", "content": system_content}]
    if history:
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
    user_content = f"Reference information:\n{context}\n\nQuestion: {query}"
    messages.append({"role": "user", "content": user_content})
    return messages


def index_stats() -> Dict[str, int]:
    index = load_index()
    return {"vectors": int(index.ntotal), "chunks": len(_DOCS)}


def _installed_models(session: requests.Session) -> List[str]:
    try:
        resp = session.get(OLLAMA_TAGS_URL, timeout=(8, 10))
        resp.raise_for_status()
        models = resp.json().get("models", [])
        models.sort(key=lambda m: m.get("size", 99_000_000_000))
        return [m["name"] for m in models if m.get("name")]
    except Exception:
        return []


def _candidate_models(requested: str, installed: List[str]) -> List[str]:
    installed_set = set(installed)
    ordered: List[str] = []

    def _add(name: str) -> None:
        if name and name not in ordered:
            ordered.append(name)

    _add(requested)
    for fb in FALLBACK_MODELS:
        if fb in installed_set:
            _add(fb)
    for model in installed:
        _add(model)
    return ordered


def _build_messages(prompt: str, history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    return build_strict_messages(prompt, context="", history=history)


def call_llm(
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
) -> str:
    session = requests.Session()
    requested_models: List[str] = []
    if model:
        requested_models.append(model)
    requested_models.append(DEFAULT_MODEL)
    requested_models.extend(FALLBACK_MODELS)

    installed = _installed_models(session)
    candidate_models: List[str] = []
    for requested in requested_models:
        for candidate in _candidate_models(requested, installed):
            if candidate not in candidate_models:
                candidate_models.append(candidate)
    if not candidate_models:
        raise RuntimeError("No Ollama models found.")

    messages = _build_messages(prompt, history)
    last_error: Optional[str] = None
    for candidate in candidate_models:
        try:
            resp = session.post(
                OLLAMA_URL,
                json={
                    "model": candidate,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 400, "num_ctx": 2048},
                },
                timeout=(8, REQUEST_TIMEOUT),
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except Exception as exc:
            last_error = str(exc)
    raise RuntimeError(last_error or "Ollama request failed")
