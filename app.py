"""
app.py
======
Flask web server for AgniAI.

Runs the existing RAG + Ollama pipeline and serves a browser chat UI.
Images (web-fetched + generated charts) are returned alongside answers.

Start:
    python app.py
Then open:  http://localhost:5000
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional

import requests as _requests
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

# ── Ensure project root is importable ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MAX_CONTEXT_CHARS,
    SYSTEM_PROMPT_SHORT,
    SYSTEM_PROMPT_ELABORATE,
    SYSTEM_PROMPT_DETAIL,
    TOP_K,
)
from image_engine import get_images
from ingest import (
    clear_index,
    ingest_pdf,
    ingest_text,
    ingest_txt,
    ingest_url,
    ingest_docx,
    list_sources,
)
from main import detect_answer_style, _should_use_rag
from memory import ConversationMemory
from ollama_cpu_chat import MODEL_NAME as DEFAULT_MODEL
from ollama_cpu_chat import PartialResponseError, chat_with_fallback
from rag import build_context, index_stats, search

# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_AS_ASCII"] = False

# ── Shared state (one session per server process) ─────────────────────────
_memory       = ConversationMemory()
_session      = _requests.Session()
_active_model = DEFAULT_MODEL
_lock         = threading.Lock()   # guard _active_model


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    POST  { "message": "...", "model": "phi3:mini" (optional) }
    Returns JSON:
    {
      "answer":  "...",
      "style":   "short|elaborate|detail",
      "images":  [ { "kind": "chart|web", "url": "...", "caption": "...", "alt": "..." } ]
    }
    """
    global _active_model

    data    = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    model   = (data.get("model")   or "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    with _lock:
        if model:
            _active_model = model
        current_model = _active_model

    # ── Detect style ───────────────────────────────────────────────────────
    style_name, system_prompt = detect_answer_style(message)

    # ── RAG retrieval ──────────────────────────────────────────────────────
    use_rag = _should_use_rag(message)
    context = ""
    if use_rag:
        docs    = search(message, top_k=TOP_K)
        context = build_context(docs)
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS].rstrip() + "\n...[truncated]..."

    if use_rag and not context:
        answer = (
            "I don't have that information in my knowledge base. "
            "Please ingest the relevant document first."
        )
        _memory.add("user", message)
        _memory.add("assistant", answer)
        return jsonify({"answer": answer, "style": style_name, "images": []})

    # ── Build messages ─────────────────────────────────────────────────────
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
            _session, current_model, messages, stream_tokens=False
        )
        answer = result.text
    except PartialResponseError as exc:
        answer = exc.partial_text or "Partial response received."
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    _memory.add("user",      message)
    _memory.add("assistant", answer)

    # ── Fetch images ───────────────────────────────────────────────────────
    images_raw = get_images(message, answer)
    images = [
        {"kind": img.kind, "url": img.url, "caption": img.caption, "alt": img.alt}
        for img in images_raw
    ]

    return jsonify({"answer": answer, "style": style_name, "images": images})


@app.route("/api/ingest", methods=["POST"])
def ingest():
    """POST { "kind": "pdf|url|txt|text|docx", "target": "..." }"""
    data   = request.get_json(force=True, silent=True) or {}
    kind   = (data.get("kind")   or "").strip().lower()
    target = (data.get("target") or "").strip()

    if not kind or not target:
        return jsonify({"error": "Need 'kind' and 'target'"}), 400

    try:
        fn_map = {
            "pdf":  ingest_pdf,
            "url":  ingest_url,
            "txt":  ingest_txt,
            "text": ingest_text,
            "docx": ingest_docx,
        }
        if kind not in fn_map:
            return jsonify({"error": f"Unknown kind: {kind}"}), 400

        count = fn_map[kind](target)
        if count == 0:
            return jsonify({"message": "Already ingested.", "chunks": 0})
        return jsonify({"message": f"Ingested {count} chunks.", "chunks": count})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/sources")
def sources():
    return jsonify(list_sources())


@app.route("/api/stats")
def stats():
    return jsonify(index_stats())


@app.route("/api/clear_memory", methods=["POST"])
def clear_memory():
    _memory.clear()
    return jsonify({"message": "Memory cleared."})


@app.route("/api/reset_index", methods=["POST"])
def reset_index():
    clear_index()
    return jsonify({"message": "Knowledge base reset."})


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("\n  AgniAI Web UI  →  http://localhost:5000\n")
    stats_data = index_stats()
    if stats_data["vectors"] == 0:
        print("  ⚠  Knowledge base empty — use the Ingest panel to add documents.\n")
    else:
        print(f"  ✔  {stats_data['vectors']} vectors ready.\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
