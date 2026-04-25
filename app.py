"""Flask REST API for AgniAI."""

from __future__ import annotations

import json
import logging
import threading
import time
from hashlib import sha1
from queue import Empty, Queue

import requests as _requests
from flask import Flask, Response, g, jsonify, request, stream_with_context
from flask_cors import CORS

from api_models import err, ok_chat, ok_health, ok_ingest, ok_message, ok_sources, ok_stats
from config import (
    ALLOWED_ORIGINS,
    FIRST_TOKEN_TIMEOUT,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHARS_DEFAULT,
    MAX_TOKENS_STYLE,
    MAX_TOKENS_DEFAULT,
    MODEL_MAX_CONTEXT_TOKENS,
    MIN_RETRIEVAL_CONFIDENCE,
    OLLAMA_TAGS_URL,
    REFERENCE_FALLBACK,
    SESSION_HEADER,
    TOKEN_SAFETY_BUFFER,
    STYLE_DETAIL_KEYWORDS,
    STYLE_ELABORATE_KEYWORDS,
    STYLE_SHORT_KEYWORDS,
    TOP_K,
    estimate_message_tokens,
    style_structure_instruction,
    trim_to_complete_sentence,
)
from ingest import clear_index, ingest_docx, ingest_pdf, ingest_text, ingest_txt, ingest_url, list_sources
from memory import ConversationMemory
from ollama_cpu_chat import MODEL_NAME as DEFAULT_MODEL
from ollama_cpu_chat import PartialResponseError, chat_with_fallback
from rag import (
    answer_is_grounded,
    build_strict_messages,
    get_cached_response,
    index_stats,
    make_response_cache_key,
    decide_answer_mode,
    is_reasoning_query,
    prepare_rag_bundle,
    set_cached_response,
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
CORS(app, origins=ALLOWED_ORIGINS)

_memory = ConversationMemory()
_session = _requests.Session()
_active_model = DEFAULT_MODEL
_lock = threading.Lock()


@app.before_request
def _start_timer() -> None:
    g.request_start = time.time()


@app.after_request
def _log_request(response):
    elapsed_ms = (time.time() - getattr(g, "request_start", time.time())) * 1000.0
    response.headers["X-Request-Duration-Ms"] = f"{elapsed_ms:.1f}"
    logger.info("%s %s -> %s in %.1fms", request.method, request.path, response.status_code, elapsed_ms)
    return response


def _kw_match(query_lower: str, keywords: list) -> bool:
    for kw in keywords:
        if " " in kw:
            if kw in query_lower:
                return True
        elif f" {kw} " in f" {query_lower} ":
            return True
    return False


def detect_answer_style(query: str) -> tuple[str, str]:
    q = query.lower()
    if _kw_match(q, STYLE_SHORT_KEYWORDS):
        return "short", "short"
    if _kw_match(q, STYLE_DETAIL_KEYWORDS):
        return "detail", "detail"
    if _kw_match(q, STYLE_ELABORATE_KEYWORDS):
        return "elaborate", "elaborate"
    return "elaborate", "elaborate"


def _classify_intent(query: str) -> str:
    q = query.strip().lower()
    tokens = [t for t in q.split() if t]
    if not tokens:
        return "reject"

    greeting_like = {
        "hi", "hello", "hey", "thanks", "thank you", "good morning",
        "good afternoon", "good evening", "bye", "greetings", "welcome",
        "ok", "okay",
    }
    if q in greeting_like and len(tokens) <= 2:
        return "chat"

    small_talk = (
        "how are you", "what's up", "whats up", "good morning", "good afternoon",
        "good evening", "thank you", "thanks",
    )
    if any(phrase in q for phrase in small_talk):
        return "chat"

    domain_terms = (
        "age", "eligibility", "salary", "pay", "selection", "medical", "pft",
        "physical", "training", "insurance", "ncc", "document", "apply",
        "application", "seva", "nidhi", "recruitment", "joining", "service",
        "agni", "agniveer", "benefit", "package", "rally", "medical", "fitness",
    )
    if any(term in q for term in domain_terms):
        return "rag"

    reasoning_terms = ("calculate", "total", "sum", "overall", "aggregate", "combined", "after 4 years", "over 4 years")
    if any(term in q for term in reasoning_terms) and any(term in q for term in ("salary", "pay", "service", "seva", "benefit", "nidhi", "year", "years")):
        return "rag"

    return "reject"


def _should_use_rag(query: str) -> bool:
    return _classify_intent(query) == "rag"


def _get_session_id(data: dict) -> str:
    session_id = (data.get("session_id") or request.headers.get(SESSION_HEADER) or "").strip()
    return session_id or "default"


def _history_fingerprint(history: list[dict]) -> str:
    payload = json.dumps(history[-6:], ensure_ascii=False, sort_keys=True)
    return sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def _get_context_limit(style: str) -> int:
    if isinstance(MAX_CONTEXT_CHARS, dict):
        return MAX_CONTEXT_CHARS.get(style, MAX_CONTEXT_CHARS_DEFAULT)
    return int(MAX_CONTEXT_CHARS)


def _get_token_limit(style: str) -> int:
    return MAX_TOKENS_STYLE.get(style, MAX_TOKENS_DEFAULT)


def _build_budget_probe_messages(
    *,
    query: str,
    style: str,
    history: list[dict] | None,
    reasoning: bool,
    use_rag: bool,
) -> list[dict]:
    if use_rag:
        return build_strict_messages(
            query,
            context="",
            style=style,
            reasoning=reasoning,
            history=history,
        )

    messages = [{
        "role": "system",
        "content": (
            "You are AgniAI, a helpful assistant for India's Agniveer Training scheme. "
            "Respond naturally and concisely."
        ),
    }]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})
    return messages


def _compute_context_char_budget(*, query: str, style: str, history: list[dict] | None, reasoning: bool, use_rag: bool) -> tuple[int, int]:
    style_budget = _get_token_limit(style)
    probe_messages = _build_budget_probe_messages(
        query=query,
        style=style,
        history=history,
        reasoning=reasoning,
        use_rag=use_rag,
    )
    prompt_tokens = estimate_message_tokens(probe_messages)
    available_after_prompt = MODEL_MAX_CONTEXT_TOKENS - prompt_tokens - TOKEN_SAFETY_BUFFER
    completion_budget = max(1, min(style_budget, available_after_prompt)) if available_after_prompt > 0 else 1
    context_tokens = max(0, MODEL_MAX_CONTEXT_TOKENS - prompt_tokens - completion_budget - TOKEN_SAFETY_BUFFER)
    return completion_budget, context_tokens * 4


def _finalize_answer(answer: str) -> str:
    final = trim_to_complete_sentence(answer)
    return final or REFERENCE_FALLBACK


def _build_messages(
    *,
    query: str,
    style: str,
    context: str,
    reasoning: bool,
    history: list[dict] | None,
    use_rag: bool,
) -> list[dict]:
    if use_rag:
        return build_strict_messages(
            query,
            context=context,
            style=style,
            reasoning=reasoning,
            history=history,
        )

    messages = [{"role": "system", "content": (
        "You are AgniAI, a helpful assistant for India's Agniveer recruitment scheme. "
        "Respond naturally and concisely."
        f"\n\n{style_structure_instruction(style)}"
    )}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})
    return messages


def _answer_via_llm(*, messages: list[dict], model: str, token_limit: int, stream: bool, on_token=None) -> str:
    result = chat_with_fallback(
        _session,
        model,
        messages,
        stream_tokens=stream,
        on_token=on_token,
        max_tokens_override=token_limit,
    )
    return result.text


def _stream_answer_response(answer_generator, status_payload: dict) -> Response:
    def generate():
        yield f"event: meta\ndata: {json.dumps(status_payload, ensure_ascii=False)}\n\n"
        try:
            for token in answer_generator():
                if isinstance(token, str) and token.startswith("event:"):
                    yield token
                else:
                    yield f"event: token\ndata: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/health")
def health():
    stats_data = index_stats()
    ollama_ok = True
    try:
        _session.get(OLLAMA_TAGS_URL, timeout=3)
    except Exception:
        ollama_ok = False

    if not ollama_ok:
        return jsonify(ok_health(
            vectors=stats_data["vectors"],
            chunks=stats_data["chunks"],
            model=_active_model,
            status="ollama_unreachable",
        )), 503

    return jsonify(ok_health(
        vectors=stats_data["vectors"],
        chunks=stats_data["chunks"],
        model=_active_model,
        status="ok",
    ))


@app.route("/api/chat", methods=["POST"])
def chat():
    global _active_model

    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    model = (data.get("model") or "").strip()
    stream_value = data.get("stream")
    stream = str(stream_value).lower() in {"1", "true", "yes", "on"} if stream_value is not None else False
    session_id = _get_session_id(data)

    if not message:
        return jsonify(*err("message field is required and cannot be empty.", 400))

    with _lock:
        if model:
            _active_model = model
        current_model = _active_model

    style_name, _ = detect_answer_style(message)
    intent = _classify_intent(message)
    use_rag = intent == "rag"
    history = _memory.history(session_id)
    reasoning = is_reasoning_query(message) if use_rag else False

    token_limit, context_char_budget = _compute_context_char_budget(
        query=message,
        style=style_name,
        history=history,
        reasoning=reasoning,
        use_rag=use_rag,
    )

    bundle = {"docs": [], "context": "", "confidence": 0.0}
    if use_rag:
        bundle = prepare_rag_bundle(message, top_k=TOP_K, style=style_name, max_context_chars=context_char_budget)

    context = bundle.get("context", "") if isinstance(bundle, dict) else ""
    confidence = float(bundle.get("confidence", 0.0)) if isinstance(bundle, dict) else 0.0
    mode = bundle.get("mode", "reject") if isinstance(bundle, dict) else "reject"
    reasoning = bool(bundle.get("reasoning", False)) if isinstance(bundle, dict) else False

    history_hash = _history_fingerprint(history)
    response_key = make_response_cache_key(
        message,
        style=style_name,
        model=current_model,
        context=context,
        session_id=f"{session_id}:{history_hash}",
    )

    cached_answer = get_cached_response(response_key)
    if cached_answer is not None:
        _memory.add("user", message, session_id=session_id)
        _memory.add("assistant", cached_answer, session_id=session_id)
        return jsonify(ok_chat(answer=cached_answer, style=style_name, session_id=session_id))

    if intent == "reject":
        answer = "Not available in the document"
        _memory.add("user", message, session_id=session_id)
        _memory.add("assistant", answer, session_id=session_id)
        set_cached_response(response_key, answer)
        if stream:
            return _stream_answer_response(
                answer_generator=lambda: iter([answer]),
                status_payload={
                    "success": True,
                    "style": style_name,
                    "session_id": session_id,
                    "cached": False,
                    "grounded": False,
                    "confidence": confidence,
                    "mode": mode,
                },
            )
        return jsonify(ok_chat(answer=answer, style=style_name, session_id=session_id))

    messages = _build_messages(
        query=message,
        style=style_name,
        context=context,
        reasoning=reasoning,
        history=history[-6:] if history else None,
        use_rag=use_rag,
    )

    if stream:
        token_queue: Queue[str | None] = Queue()
        outcome: dict[str, object] = {}

        def _worker() -> None:
            try:
                outcome["answer"] = _answer_via_llm(
                    messages=messages,
                    model=current_model,
                    token_limit=token_limit,
                    stream=True,
                    on_token=token_queue.put,
                )
            except PartialResponseError as exc:
                outcome["answer"] = exc.partial_text or "Not available in the document"
            except Exception as exc:
                outcome["error"] = str(exc)
            finally:
                token_queue.put(None)

        threading.Thread(target=_worker, daemon=True).start()

        def _generator():
            pieces: list[str] = []
            while True:
                try:
                    token = token_queue.get(timeout=FIRST_TOKEN_TIMEOUT)
                except Empty:
                    yield f"event: error\ndata: {json.dumps({'error': 'First token timeout'}, ensure_ascii=False)}\n\n"
                    return
                if token is None:
                    break
                pieces.append(token)
                yield token

            if "error" in outcome:
                yield f"event: error\ndata: {json.dumps({'error': str(outcome['error'])}, ensure_ascii=False)}\n\n"
                return

            answer = "".join(pieces).strip() or str(outcome.get("answer", "")).strip()
            if use_rag and mode == "normal_answer" and not answer_is_grounded(answer, context):
                answer = REFERENCE_FALLBACK
            elif use_rag and mode == "strict_answer" and not context.strip() and not bundle.get("docs"):
                answer = REFERENCE_FALLBACK
            answer = _finalize_answer(answer)
            _memory.add("user", message, session_id=session_id)
            _memory.add("assistant", answer, session_id=session_id)
            set_cached_response(response_key, answer)

        return _stream_answer_response(
            answer_generator=_generator,
            status_payload={
                "success": True,
                "style": style_name,
                "session_id": session_id,
                "cached": False,
                "grounded": bool(use_rag),
                "confidence": confidence,
                "mode": mode,
            },
        )

    try:
        answer = _answer_via_llm(
            messages=messages,
            model=current_model,
            token_limit=token_limit,
            stream=False,
        )
    except PartialResponseError as exc:
        answer = exc.partial_text or "Partial response received. Please try again."
    except RuntimeError as exc:
        return jsonify(*err(f"LLM service unavailable: {exc}", 503))

    if use_rag and mode == "normal_answer" and not answer_is_grounded(answer, context):
        answer = REFERENCE_FALLBACK
    elif use_rag and mode == "strict_answer" and not context.strip() and not bundle.get("docs"):
        answer = REFERENCE_FALLBACK
    answer = _finalize_answer(answer)

    _memory.add("user", message, session_id=session_id)
    _memory.add("assistant", answer, session_id=session_id)
    set_cached_response(response_key, answer)
    return jsonify(ok_chat(answer=answer, style=style_name, session_id=session_id))


@app.route("/api/ingest", methods=["POST"])
def ingest():
    data = request.get_json(force=True, silent=True) or {}
    kind = (data.get("kind") or "").strip().lower()
    target = (data.get("target") or "").strip()

    if not kind:
        return jsonify(*err("kind field is required (pdf|url|txt|text|docx).", 400))
    if not target:
        return jsonify(*err("target field is required (file path or URL).", 400))

    fn_map = {
        "pdf": ingest_pdf,
        "url": ingest_url,
        "txt": ingest_txt,
        "text": ingest_text,
        "docx": ingest_docx,
    }

    if kind not in fn_map:
        return jsonify(*err(f"Unknown kind '{kind}'. Valid values: pdf, url, txt, text, docx.", 400))

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


@app.route("/api/sources")
def sources():
    return jsonify(ok_sources(list_sources()))


@app.route("/api/stats")
def stats():
    s = index_stats()
    return jsonify(ok_stats(vectors=s["vectors"], chunks=s["chunks"]))


@app.route("/api/clear_memory", methods=["POST"])
def clear_memory():
    session_id = _get_session_id(request.get_json(force=True, silent=True) or {})
    _memory.clear(session_id if session_id != "default" else None)
    return jsonify(ok_message("Conversation memory cleared."))


@app.route("/api/reset_index", methods=["POST"])
def reset_index():
    clear_index()
    return jsonify(ok_message("Knowledge base reset. Re-ingest documents to continue."))


if __name__ == "__main__":
    stats_data = index_stats()
    print("\n  AgniAI REST API")
    print("  Listening on  http://0.0.0.0:5000")
    print("  Health check  http://localhost:5000/api/health")
    print("  Chat endpoint http://localhost:5000/api/chat  [POST]")
    if stats_data["vectors"] == 0:
        print("  Warning: Knowledge base is empty.")
        print("  POST /api/ingest to add documents before chatting.\n")
    else:
        print(f"  Knowledge base ready: {stats_data['vectors']} vectors.\n")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
