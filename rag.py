"""
rag.py
======
Core retrieval-augmented generation layer — accuracy-maximised edition.

Fixes in this version vs previous:
  1. BM25 stopwords: "age", "pay", "rank", "year" removed from stopword list
     — these are crucial Agniveer domain terms that were being dropped
  2. MMR lambda tuned to 0.7 (slightly more relevance-biased) for small
     single-document corpora where diversity hurts more than it helps
  3. build_context deduplication fingerprint extended to 150 chars so
     genuinely different chunks that share a short opening don't get dropped
  4. Dense-only fallback now returns top-3 chunks (not just 1) when nothing
     passes MIN_SCORE — prevents empty context on borderline queries
  5. Query expansion covers more Agniveer-specific term patterns
  6. _normalize_query_for_retrieval is now less aggressive — only strips
     obvious filler, not domain terms like "explain" which add useful signal
  7. BM25 per-domain boosting extended (salary, insurance, training, etc.)
  8. Candidate pool now capped at min(top_k * 12, ntotal) for better recall
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
_RERANKER = None
_RERANKER_FAILED = False
_INDEX: Optional[faiss.Index] = None
_DOCS: List[Dict[str, str]] = []
_BM25 = None


# ── Helpers ────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _new_index() -> faiss.Index:
    return faiss.IndexFlatIP(EMBEDDING_DIM)


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
            f"Embedding dimension mismatch while rebuilding: "
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
    Lightly strip filler phrases from the query to improve retrieval precision.

    IMPORTANT: This version is deliberately less aggressive than the previous
    one. Domain terms like "explain", "how", "process" carry useful signal for
    BM25 matching and are NOT stripped. Only obvious meta-phrases (style words,
    politeness fillers) are removed.
    """
    cleaned = query.lower()

    # Only strip pure style/politeness phrases — not domain terms
    filler_phrases = (
        "in short", "briefly", "brief", "quick answer", "short answer",
        "summarise", "summarize", "tldr", "tl;dr", "in brief",
        "give me short", "one line", "give a short", "keep it short",
        "in detail", "detailed", "explain in detail", "full detail",
        "comprehensive", "exhaustive", "step by step", "step-by-step",
        "explain fully", "tell me everything", "give me detail",
        "elaborate in detail", "full explanation", "full breakdown",
        "break it down", "like i'm a beginner", "like i am a beginner",
        "for a beginner", "as a beginner", "in simple terms",
        "beginner friendly", "please", "can you", "could you",
    )
    for phrase in sorted(filler_phrases, key=len, reverse=True):
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,!?:;")

    # Safety: if stripping left fewer than 3 words, fall back to original
    if len(cleaned.split()) < 3:
        cleaned = query.strip().lower()

    # ── Domain-specific query expansion ───────────────────────────────────
    # Map common user phrasings to terms actually present in the Agniveer docs
    expansions = (
        (r"\bage\b",             "required age age eligibility 17"),
        (r"\bage limit\b",       "required age age eligibility 17"),
        (r"\beligibilit",        "eligibility criteria required age qualification"),
        (r"\bselection process\b","flow chart recruitment process merit medical physical fitness"),
        (r"\bhow.*select",       "flow chart recruitment process merit medical physical fitness"),
        (r"\brecruitment process\b", "flow chart recruitment process registration rally medical"),
        (r"\bhow.*appl",         "registration application dashboard joinindianarmy"),
        (r"\bsalary\b",          "customised package in hand seva nidhi corpus fund monthly"),
        (r"\bpay\b",             "customised package in hand monthly year rupees"),
        (r"\bphysical test\b",   "physical fitness test PFT 1.6 km run pull ups beam"),
        (r"\bpft\b",             "physical fitness test 1.6 km run pull ups beam marks group"),
        (r"\bbonus mark",        "bonus marks NCC sports ward servicemen"),
        (r"\binsurance\b",       "life insurance cover 48 lakhs non-contributory"),
        (r"\bseva nidhi\b",      "seva nidhi corpus fund exit after 4 year lakh"),
        (r"\btraining\b",        "military training regimental centre enrolment"),
        (r"\bdocument",          "documents required matric aadhaar domicile caste certificate"),
        (r"\bmedical\b",         "medical examination army medical standards fitness rally"),
    )
    expanded = cleaned
    for pattern, extra in expansions:
        if re.search(pattern, expanded):
            expanded = f"{expanded} {extra}"
            break  # only apply first matching expansion to avoid bloating query

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
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=1,
    )
    return np.asarray(vec, dtype="float32")


# ── Cross-encoder re-ranker ────────────────────────────────────────────────

def load_reranker():
    global _RERANKER, _RERANKER_FAILED
    if _RERANKER is not None:
        return _RERANKER
    if _RERANKER_FAILED or not USE_RERANKER:
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
    return re.findall(r"[a-zA-Z0-9\u0900-\u097F]+", text.lower())


# FIX: removed "age", "pay", "rank", "year", "km", "run" from stopwords.
# These are critical Agniveer domain terms — dropping them destroys BM25 recall.
_STOPWORDS = {
    "a", "an", "and", "are", "be", "by", "for", "from", "how", "i", "in",
    "is", "it", "me", "my", "of", "on", "or", "please", "show", "tell",
    "the", "to", "what", "when", "where", "which", "who", "why", "with",
    "you", "your", "can", "could", "would", "should", "do", "does", "did",
    "this", "that", "these", "those", "as", "at", "was", "were", "will",
    "just", "about", "into", "over", "under", "up", "down",
    # NOTE: "like", "give", "make", "take" retained — sometimes useful
}


def _meaningful_tokens(text: str) -> List[str]:
    return [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 1]


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
        print(f"[WARNING] Could not load BM25 index ({exc}).")
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
        pass  # rank_bm25 not installed — hybrid disabled silently
    except Exception as exc:
        print(f"[WARNING] BM25 index build failed: {exc}")


# ── MMR diversity filter ───────────────────────────────────────────────────

def _mmr_filter(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    docs: List[Dict],
    k: int,
    lambda_: float = 0.7,   # FIX: raised from 0.6 → 0.7 (more relevance-biased)
) -> List[Dict]:
    """
    Maximal Marginal Relevance.
    lambda_=0.7 balances relevance and diversity.
    For single-document corpora, higher lambda_ is better: relevant chunks
    may be similar to each other by nature, and we want them all.
    """
    if len(docs) <= k:
        return docs

    selected_idx: List[int] = []
    remaining = list(range(len(docs)))

    for _ in range(min(k, len(docs))):
        if not remaining:
            break
        if not selected_idx:
            scores = doc_vecs[remaining] @ query_vec.T
            best = remaining[int(np.argmax(scores))]
        else:
            rel_scores = doc_vecs[remaining] @ query_vec.T
            sim_to_selected = np.max(
                doc_vecs[remaining] @ doc_vecs[selected_idx].T, axis=1
            )
            mmr_scores = (
                lambda_ * rel_scores.flatten()
                - (1 - lambda_) * sim_to_selected.flatten()
            )
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
            f"but the active model uses {EMBEDDING_DIM}. "
            "Rebuilding the index from the docstore..."
        )
        _INDEX = _rebuild_index_from_docs(_DOCS)

    if _INDEX.ntotal > 0 and len(_DOCS) == 0:
        print(
            "[WARNING] FAISS index has vectors but docstore is empty. "
            "Run /reset and re-ingest your documents."
        )
    return _INDEX


def save_index(index: faiss.Index, docs: List[Dict[str, str]]) -> None:
    global _DOCS
    _ensure_dirs()
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    _save_docstore(docs)
    _DOCS = docs
    save_bm25(docs)


# ── Search ─────────────────────────────────────────────────────────────────

def _bm25_scores(query: str, n: int) -> np.ndarray:
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


# ── Domain boosting rules ──────────────────────────────────────────────────
# Each entry: (query_pattern, doc_pattern, boost_score)
# Applied during hybrid fusion to surface highly relevant chunks.
_DOMAIN_BOOSTS = [
    (r"\bage\b",              r"required age",                  0.70),
    (r"\bage\b",              r"17.*22|17½",                    0.60),
    (r"eligibilit",           r"eligibility criteria",          0.70),
    (r"eligibilit",           r"required age",                  0.40),
    (r"selection|how.*select",r"flow chart.*recruitment|recruitment process", 0.90),
    (r"recruitment process",  r"flow chart.*recruitment",       0.90),
    (r"how.*appl|apply",      r"registration|application",      0.70),
    (r"salary|pay|package",   r"customis.*package|in hand|seva nidhi", 0.80),
    (r"salary|pay|package",   r"30,000|33,000|36,500|40,000",  0.70),
    (r"physical|pft",         r"physical fitness test|1\.6 km run", 0.80),
    (r"physical|pft",         r"pull.up|beam|group.i|group.ii", 0.60),
    (r"bonus mark",           r"bonus marks|ncc|sportsmen",     0.80),
    (r"insurance",            r"48 lakh|life insurance",        0.80),
    (r"seva nidhi",           r"seva nidhi|10\.04 lakh|corpus", 0.80),
    (r"training",             r"military training|regimental",  0.70),
    (r"document",             r"documents required|matric|aadhaar|domicile", 0.70),
    (r"medical",              r"medical examination|army medical", 0.70),
    (r"women|female|girl",    r"women military police|women mp", 0.80),
    (r"tradesman",            r"tradesman|tradesmen",           0.70),
    (r"clerk|skt",            r"clerk.*store keeper|skt",       0.70),
    (r"technical",            r"agniveer.*tech|tech.*agniveer", 0.70),
    (r"cee|entrance exam",    r"common entrance examination|cee", 0.80),
    (r"ncc",                  r"ncc.*certificate|bonus.*ncc",   0.70),
]


def _apply_domain_boosts(query_lower: str, doc_text_lower: str) -> float:
    """Return the highest applicable boost for this query+doc pair."""
    best = 0.0
    for q_pat, d_pat, boost in _DOMAIN_BOOSTS:
        if re.search(q_pat, query_lower) and re.search(d_pat, doc_text_lower):
            best = max(best, boost)
    return best


def search(query: str, top_k: int = TOP_K) -> List[Dict[str, str]]:
    """
    Retrieve top-k chunks using hybrid dense+sparse search, cross-encoder
    re-ranking, and MMR diversity filtering.

    Pipeline:
      1. Dense cosine search → candidate_k results
      2. BM25 sparse score fusion (if USE_HYBRID)
      3. Domain boosting (Agniveer-specific heuristics)
      4. Cross-encoder re-ranking (if USE_RERANKER)
      5. MMR diversity filter → final top_k
    """
    index = load_index()
    if index.ntotal == 0:
        return []

    retrieval_query = _normalize_query_for_retrieval(query)
    qvec = embed_query(retrieval_query)

    # Wider candidate pool → better recall before re-ranking
    candidate_k = min(max(top_k * 12, 30), index.ntotal)

    # ── Dense retrieval ────────────────────────────────────────
    scores_dense, ids = index.search(qvec, candidate_k)
    dense_scores = scores_dense[0]
    doc_ids = ids[0]
    dense_map = {
        int(doc_id): float(score)
        for doc_id, score in zip(doc_ids, dense_scores)
        if doc_id >= 0
    }

    # ── BM25 fusion ────────────────────────────────────────────
    if USE_HYBRID and len(_DOCS) > 0:
        bm25_all = _bm25_scores(retrieval_query, len(_DOCS))
        bm25_top_ids = np.argsort(bm25_all)[::-1][:candidate_k]

        # Adaptive weighting: short queries benefit more from BM25 (keyword)
        token_count = len(retrieval_query.split())
        if token_count <= 3:
            dense_weight, bm25_weight = 0.25, 0.75
        elif token_count <= 6:
            dense_weight, bm25_weight = 0.40, 0.60
        else:
            dense_weight, bm25_weight = DENSE_WEIGHT, BM25_WEIGHT

        candidate_ids: List[int] = []
        seen_ids: set = set()
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

        query_terms  = set(_meaningful_tokens(retrieval_query))
        query_lower  = retrieval_query.lower()

        fused: List[tuple] = []
        for doc_id, ds, bs in zip(candidate_ids, dense_values, bm25_values):
            combined = dense_weight * float(ds) + bm25_weight * float(bs)

            # Term overlap bonus
            if query_terms:
                doc_text = _DOCS[doc_id].get("text", "")
                doc_terms = set(_meaningful_tokens(doc_text))
                overlap = len(query_terms & doc_terms) / max(1, len(query_terms))
                combined += 0.15 * overlap

                # Domain-specific boost
                doc_lower = doc_text.lower()
                combined += _apply_domain_boosts(query_lower, doc_lower)

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
        # Dense-only path
        candidates = []
        for doc_id, score in zip(doc_ids, dense_scores):
            if doc_id < 0 or doc_id >= len(_DOCS):
                continue
            if float(score) < MIN_SCORE:
                continue
            doc = dict(_DOCS[doc_id])
            doc["score"] = round(float(score), 4)
            candidates.append(doc)

    # FIX: Return top-3 fallback chunks instead of just 1 when nothing passes
    # MIN_SCORE. Prevents empty context on borderline queries (e.g. "age limit").
    if not candidates:
        fallback = []
        for i in range(min(3, len(doc_ids))):
            fid = int(doc_ids[i])
            if 0 <= fid < len(_DOCS):
                doc = dict(_DOCS[fid])
                doc["score"] = round(float(dense_scores[i]), 4)
                fallback.append(doc)
        return fallback

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
        candidates = _mmr_filter(
            qvec, cand_vecs, candidates, k=min(top_k, len(candidates))
        )

    return candidates[:top_k]


# ── Context builder ────────────────────────────────────────────────────────

def build_context(docs: Sequence[Dict[str, str]]) -> str:
    """
    Format retrieved docs into a numbered context block for the LLM.

    FIX: Deduplication fingerprint extended to 150 chars (was 100) so
    genuinely different chunks that share a short common opening are both
    kept. This matters for salary tables and eligibility sections that
    all start with the same sentence fragment.
    """
    if not docs:
        return ""

    blocks: List[str] = []
    seen_texts: set = set()

    for i, doc in enumerate(docs, start=1):
        text = (doc.get("text") or "").strip()
        fingerprint = text[:150].lower()
        if not text or fingerprint in seen_texts:
            continue
        seen_texts.add(fingerprint)
        source = doc.get("source", "unknown")
        if len(source) > 60:
            source = "…" + source[-58:]
        blocks.append(f"[{i}] Source: {source}\n{text}")

    return "\n\n---\n\n".join(blocks)


def index_stats() -> Dict[str, int]:
    index = load_index()
    return {"vectors": int(index.ntotal), "chunks": len(_DOCS)}


# ── Ollama client (non-streaming) ──────────────────────────────────────────

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
            role    = msg.get("role")
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