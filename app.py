"""
app.py
======
Flask REST API for AgniAI — the integration point for the .NET ChatController.

The .NET backend calls these endpoints:
    POST  /api/chat          — main chat (send message, get answer)
    POST  /api/ingest        — add a document to the knowledge base
    GET   /api/sources       — list all ingested sources
    GET   /api/stats         — index vector count
    GET   /api/health        — health check (used by .NET before routing requests)
    POST  /api/clear_memory  — wipe conversation history
    POST  /api/reset_index   — ⚠ delete entire knowledge base

Start the service:
    python app.py

.NET ChatController.cs calls:
    POST http://localhost:5000/api/chat
    Body: { "message": "What is the age limit?", "model": "phi3:mini" (optional) }

    Response:
    {
        "success": true,
        "answer": "...",
        "style": "short|elaborate|detail"
    }

CORS is enabled for all origins during development.
Lock it down via ALLOWED_ORIGINS in config.py before production.
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
    SYSTEM_PROMPT_DETAIL,
    SYSTEM_PROMPT_ELABORATE,
    SYSTEM_PROMPT_SHORT,
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
from main import detect_answer_style, _should_use_rag
from memory import ConversationMemory
from ollama_cpu_chat import MODEL_NAME as DEFAULT_MODEL
from ollama_cpu_chat import PartialResponseError, chat_with_fallback
from rag import build_context, index_stats, search

# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# CORS — allows the React frontend (any origin during dev) and the
# .NET backend to call this service without browser/server header errors.
# Lock ALLOWED_ORIGINS down to specific URLs before going to production.
CORS(app, origins=ALLOWED_ORIGINS)

# ── Shared state (one session per server process) ─────────────────────────
_memory       = ConversationMemory()
_session      = _requests.Session()
_active_model = DEFAULT_MODEL
_lock         = threading.Lock()


# =============================================================================
# HEALTH  — .NET calls this to verify the Python service is up
# =============================================================================

@app.route("/api/health")
def health():
    """
    GET /api/health

    .NET ChatController should call this before routing chat requests.
    Returns 200 when the service and knowledge base are ready.

    C# usage:
        var resp = await _http.GetAsync("http://localhost:5000/api/health");
        resp.EnsureSuccessStatusCode();
    """
    stats_data = index_stats()
    return jsonify(ok_health(
        vectors=stats_data["vectors"],
        chunks=stats_data["chunks"],
        model=_active_model,
        status="ok",
    ))


# =============================================================================
# CHAT  — main endpoint the .NET ChatController posts to
# =============================================================================

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    POST /api/chat
    Body JSON:
        {
            "message": "What is the age limit?",   ← required
            "model":   "phi3:mini"                 ← optional, overrides default
        }

    Response JSON:
        {
            "success": true,
            "answer":  "...",
            "style":   "short|elaborate|detail"
        }

    On error:
        {
            "success": false,
            "error": "reason"
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

    # ── Detect answer style from question wording ──────────────────────────
    style_name, system_prompt = detect_answer_style(message)

    # ── RAG retrieval ──────────────────────────────────────────────────────
    use_rag = _should_use_rag(message)
    context = ""
    if use_rag:
        docs    = search(message, top_k=TOP_K)
        context = build_context(docs)
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS].rstrip() + "\n...[truncated]..."

    # Knowledge base has no relevant info
    if use_rag and not context:
        answer = (
            "I don't have that information in my knowledge base. "
            "Please ingest the relevant document first."
        )
        _memory.add("user", message)
        _memory.add("assistant", answer)
        return jsonify(ok_chat(answer=answer, style=style_name))

    # ── Build message list for the LLM ────────────────────────────────────
    history = _memory.history()

    if use_rag:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-4:])
        user_content = f"Reference information:\n{context}\n\nQuestion: {message}"
    else:
        messages = [{"role": "system", "content": (
            "You are AgniAI, a helpful assistant for India's Agniveer "
            "recruitment scheme. Respond naturally and concisely."
        )}]
        if history:
            messages.extend(history[-4:])
        user_content = message

    messages.append({"role": "user", "content": user_content})

    # ── LLM call ───────────────────────────────────────────────────────────
    try:
        result = chat_with_fallback(
            _session,
            current_model,
            messages,
            stream_tokens=False,   # REST API — no streaming, return full answer
        )
        answer = result.text

    except PartialResponseError as exc:
        # Ollama returned something before cutting out — use what we got
        answer = exc.partial_text or "Partial response received. Please try again."

    except RuntimeError as exc:
        # Ollama is down or no model loaded
        return jsonify(*err(f"LLM service unavailable: {exc}", 503))

    _memory.add("user",      message)
    _memory.add("assistant", answer)

    return jsonify(ok_chat(answer=answer, style=style_name))


# =============================================================================
# INGEST  — add documents to the knowledge base
# =============================================================================

@app.route("/api/ingest", methods=["POST"])
def ingest():
    """
    POST /api/ingest
    Body JSON:
        {
            "kind":   "pdf|url|txt|text|docx",   ← required
            "target": "/path/to/file or URL"     ← required
        }

    Response JSON:
        {
            "success": true,
            "message": "Ingested 12 chunks.",
            "chunks":  12,
            "source":  "/path/to/file"
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
# SOURCES  — list everything in the knowledge base
# =============================================================================

@app.route("/api/sources")
def sources():
    """
    GET /api/sources

    Response JSON:
        {
            "success": true,
            "count": 3,
            "sources": [
                { "source": "...", "doc_type": "pdf", "chunk_count": 12 },
                ...
            ]
        }
    """
    return jsonify(ok_sources(list_sources()))


# =============================================================================
# STATS  — index size
# =============================================================================

@app.route("/api/stats")
def stats():
    """
    GET /api/stats

    Response JSON:
        { "success": true, "vectors": 115, "chunks": 115 }
    """
    s = index_stats()
    return jsonify(ok_stats(vectors=s["vectors"], chunks=s["chunks"]))


# =============================================================================
# MEMORY & RESET
# =============================================================================

@app.route("/api/clear_memory", methods=["POST"])
def clear_memory():
    """POST /api/clear_memory — wipe conversation history for this session."""
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