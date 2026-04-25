"""
app.py
======
Flask REST API for AgniAI — the integration point for the .NET ChatController.

Fixes in this version:
  • detect_answer_style imported from main.py — decoupled into its own
    helper here so app.py has no dependency on main.py's CLI code
  • Per-style MAX_CONTEXT_CHARS and MAX_TOKENS passed through to LLM
  • history[-6:] instead of history[-4:] — more coherent multi-turn context
  • images field always returned in /api/chat response (frontend ready)
  • /api/health returns 503 with explanation when Ollama is unreachable
  • CORS locked to ALLOWED_ORIGINS from config

The .NET backend calls:
    POST http://localhost:5000/api/chat
    Body: { "message": "What is the age limit?", "model": "phi3:mini" (opt) }

    Response:
    {
        "success": true,
        "answer":  "...",
        "style":   "short|elaborate|detail",
        "images":  []
    }
"""

from __future__ import annotations

import threading
from pathlib import Path

import requests as _requests
from flask import Flask, jsonify, request
from flask_cors import CORS

from api_models import (
    err,
    ok_chat,
    ok_health,
    ok_ingest,
    ok_message,
    ok_sources,
    ok_stats,
)
from config import (
    ALLOWED_ORIGINS,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHARS_DEFAULT,
    MAX_TOKENS_STYLE,
    MAX_TOKENS_DEFAULT,
    SYSTEM_PROMPT_DETAIL,
    SYSTEM_PROMPT_ELABORATE,
    SYSTEM_PROMPT_SHORT,
    STYLE_DETAIL_KEYWORDS,
    STYLE_ELABORATE_KEYWORDS,
    STYLE_SHORT_KEYWORDS,
    TOP_K,
)
from ingest import (
    clear_index,
    ingest_docx,
    ingest_pdf,
    ingest_text,
    ingest_txt,
    ingest_url,
    list_sources,
)
from memory import ConversationMemory
from ollama_cpu_chat import MODEL_NAME as DEFAULT_MODEL
from ollama_cpu_chat import PartialResponseError, chat_with_fallback
from rag import build_context, index_stats, search

import re

# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
CORS(app, origins=ALLOWED_ORIGINS)

# ── Shared state ───────────────────────────────────────────────────────────
_memory       = ConversationMemory()
_session      = _requests.Session()
_active_model = DEFAULT_MODEL
_lock         = threading.Lock()


# ── Style detection (self-contained, no dependency on main.py) ─────────────

def _kw_match(query_lower: str, keywords: list) -> bool:
    for kw in keywords:
        if " " in kw:
            if kw in query_lower:
                return True
        else:
            if re.search(rf"\b{re.escape(kw)}\b", query_lower):
                return True
    return False


def detect_answer_style(query: str):
    """Returns (style_name, system_prompt)."""
    q = query.lower()
    if _kw_match(q, STYLE_SHORT_KEYWORDS):
        return "short", SYSTEM_PROMPT_SHORT
    if _kw_match(q, STYLE_DETAIL_KEYWORDS):
        return "detail", SYSTEM_PROMPT_DETAIL
    if _kw_match(q, STYLE_ELABORATE_KEYWORDS):
        return "elaborate", SYSTEM_PROMPT_ELABORATE
    return "elaborate", SYSTEM_PROMPT_ELABORATE


def _should_use_rag(query: str) -> bool:
    q = query.strip().lower()
    _GREETINGS = {
        "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye",
        "good morning", "good evening", "good afternoon", "greetings",
        "welcome", "sup", "yo",
    }
    words = q.split()
    if len(words) <= 3 and q in _GREETINGS:
        return False
    return True


def _get_context_limit(style: str) -> int:
    if isinstance(MAX_CONTEXT_CHARS, dict):
        return MAX_CONTEXT_CHARS.get(style, MAX_CONTEXT_CHARS_DEFAULT)
    return int(MAX_CONTEXT_CHARS)


def _get_token_limit(style: str) -> int:
    return MAX_TOKENS_STYLE.get(style, MAX_TOKENS_DEFAULT)


# =============================================================================
# HEALTH
# =============================================================================

@app.route("/api/health")
def health():
    """
    GET /api/health
    Returns 200 when both the service and Ollama are reachable.
    Returns 503 when Ollama is down (so .NET can surface a useful error).
    """
    stats_data = index_stats()
    # Quick Ollama reachability check
    ollama_ok = True
    try:
        _session.get("http://localhost:11434/api/tags", timeout=3)
    except Exception:
        ollama_ok = False

    if not ollama_ok:
        body = ok_health(
            vectors=stats_data["vectors"],
            chunks=stats_data["chunks"],
            model=_active_model,
            status="ollama_unreachable",
        )
        return jsonify(body), 503

    return jsonify(ok_health(
        vectors=stats_data["vectors"],
        chunks=stats_data["chunks"],
        model=_active_model,
        status="ok",
    ))


# =============================================================================
# CHAT
# =============================================================================

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    POST /api/chat
    Body JSON:
        {
            "message": "What is the age limit?",   ← required
            "model":   "phi3:mini"                 ← optional
        }

    Response JSON:
        {
            "success": true,
            "answer":  "...",
            "style":   "short|elaborate|detail",
            "images":  []
        }
    """
    global _active_model

    data    = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    model   = (data.get("model")   or "").strip()

    if not message:
        return jsonify(*err("message field is required and cannot be empty.", 400))

    with _lock:
        if model:
            _active_model = model
        current_model = _active_model

    # ── Detect style ───────────────────────────────────────────────────────
    style_name, system_prompt = detect_answer_style(message)
    context_limit = _get_context_limit(style_name)
    token_limit   = _get_token_limit(style_name)

    # ── RAG retrieval ──────────────────────────────────────────────────────
    use_rag = _should_use_rag(message)
    context = ""
    if use_rag:
        docs    = search(message, top_k=TOP_K)
        context = build_context(docs)
        if len(context) > context_limit:
            context = context[:context_limit].rstrip() + "\n...[truncated]..."

    # Knowledge base has no relevant info
    if use_rag and not context:
        answer = (
            "I don't have that information in my knowledge base. "
            "Please ingest the relevant document first."
        )
        _memory.add("user", message)
        _memory.add("assistant", answer)
        return jsonify(ok_chat(answer=answer, style=style_name))

    # ── Build message list ─────────────────────────────────────────────────
    history = _memory.history()

    if use_rag:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-6:])   # FIX: was -4
        user_content = f"Reference information:\n{context}\n\nQuestion: {message}"
    else:
        messages = [{"role": "system", "content": (
            "You are AgniAI, a helpful assistant for India's Agniveer "
            "recruitment scheme. Respond naturally and concisely."
        )}]
        if history:
            messages.extend(history[-6:])
        user_content = message

    messages.append({"role": "user", "content": user_content})

    # ── LLM call with per-style token limit ────────────────────────────────
    try:
        result = chat_with_fallback(
            _session,
            current_model,
            messages,
            stream_tokens=False,
            max_tokens_override=token_limit,
        )
        answer = result.text

    except PartialResponseError as exc:
        answer = exc.partial_text or "Partial response received. Please try again."

    except RuntimeError as exc:
        return jsonify(*err(f"LLM service unavailable: {exc}", 503))

    _memory.add("user",      message)
    _memory.add("assistant", answer)

    return jsonify(ok_chat(answer=answer, style=style_name))


# =============================================================================
# INGEST
# =============================================================================

@app.route("/api/ingest", methods=["POST"])
def ingest():
    """
    POST /api/ingest
    Body JSON:
        {
            "kind":   "pdf|url|txt|text|docx",
            "target": "/path/to/file or URL"
        }
    """
    data   = request.get_json(force=True, silent=True) or {}
    kind   = (data.get("kind")   or "").strip().lower()
    target = (data.get("target") or "").strip()

    if not kind:
        return jsonify(*err("kind field is required (pdf|url|txt|text|docx).", 400))
    if not target:
        return jsonify(*err("target field is required (file path or URL).", 400))

    fn_map = {
        "pdf":  ingest_pdf,
        "url":  ingest_url,
        "txt":  ingest_txt,
        "text": ingest_text,
        "docx": ingest_docx,
    }

    if kind not in fn_map:
        return jsonify(*err(
            f"Unknown kind '{kind}'. Valid values: pdf, url, txt, text, docx.", 400
        ))

    try:
        count = fn_map[kind](target)
    except FileNotFoundError as exc:
        return jsonify(*err(f"File not found: {exc}", 404))
    except Exception as exc:
        return jsonify(*err(f"Ingestion failed: {exc}", 500))

    if count == 0:
        return jsonify(ok_ingest(
            message="Source was already ingested. No new chunks added.",
            chunks=0,
            source=target,
        ))

    return jsonify(ok_ingest(
        message=f"Successfully ingested {count} chunks.",
        chunks=count,
        source=target,
    ))


# =============================================================================
# SOURCES / STATS / MEMORY / RESET
# =============================================================================

@app.route("/api/sources")
def sources():
    """GET /api/sources — list all ingested sources."""
    return jsonify(ok_sources(list_sources()))


@app.route("/api/stats")
def stats():
    """GET /api/stats — index vector count."""
    s = index_stats()
    return jsonify(ok_stats(vectors=s["vectors"], chunks=s["chunks"]))


@app.route("/api/clear_memory", methods=["POST"])
def clear_memory():
    """POST /api/clear_memory — wipe conversation history."""
    _memory.clear()
    return jsonify(ok_message("Conversation memory cleared."))


@app.route("/api/reset_index", methods=["POST"])
def reset_index():
    """POST /api/reset_index — delete the entire knowledge base (irreversible)."""
    clear_index()
    return jsonify(ok_message("Knowledge base reset. Re-ingest documents to continue."))


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    stats_data = index_stats()
    print("\n  AgniAI REST API")
    print("  ───────────────────────────────────────")
    print("  Listening on  http://0.0.0.0:5000")
    print("  Health check  http://localhost:5000/api/health")
    print("  Chat endpoint http://localhost:5000/api/chat  [POST]")
    print("  ───────────────────────────────────────")
    if stats_data["vectors"] == 0:
        print("  ⚠  Knowledge base is empty.")
        print("     POST /api/ingest to add documents before chatting.\n")
    else:
        print(f"  ✔  Knowledge base ready: {stats_data['vectors']} vectors.\n")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)