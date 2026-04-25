"""CLI chatbot for AgniAI."""

from __future__ import annotations

import json
import re
import sys
from hashlib import sha1
from typing import Optional, Tuple

import requests

from config import (
    DATA_DIR,
    INDEX_DIR,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHARS_DEFAULT,
    MAX_TOKENS_DEFAULT,
    MAX_TOKENS_STYLE,
    MODEL_MAX_CONTEXT_TOKENS,
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
from ollama_cpu_chat import MODEL_NAME as DEFAULT_MODEL_NAME
from ollama_cpu_chat import PartialResponseError, chat_with_fallback
from rag import (
    build_strict_messages,
    get_cached_response,
    index_stats,
    make_response_cache_key,
    generate_structured_answer,
    is_reasoning_query,
    prepare_rag_bundle,
    set_cached_response,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def dim(t): return _c("2", t)
def bold(t): return _c("1", t)
def cyan(t): return _c("96", t)
def green(t): return _c("92", t)
def yellow(t): return _c("93", t)
def red(t): return _c("91", t)
def blue(t): return _c("94", t)


BANNER = cyan(r"""
   ___                  _ ___    ___
  / _ |___  ___ _  ___ (_) _ \  / _ \
 / __ / _ \/ _ \ |/ _ \| | | | | (_) |
/_/ |_\___/_//_/___|___/|_|___/  \___/
""") + bold("  Agniveer AI Assistant  - Offline · Local · Private\n")

HELP_TEXT = """
Available commands:

  /ingest pdf  <path>     Add a PDF to the knowledge base
  /ingest url  <url>      Add a web page to the knowledge base
  /ingest txt  <path>     Add a plain .txt file
  /ingest text <content>  Add raw text
  /ingest docx <path>     Add a Word (.docx) file
  /sources                List all ingested sources
  /stats                  Show index vector count
  /clear                  Clear conversation memory
  /reset                  Delete the entire knowledge base
  /model <name>           Switch the Ollama model
  /help                   Show this help
  /exit  or  /quit        Exit AgniAI

Answer style is detected automatically from your question.
"""

_STYLE_LABEL = {"short": "SHORT", "elaborate": "ELABORATE", "detail": "DETAIL"}
_STYLE_COLOR = {"short": yellow, "elaborate": cyan, "detail": blue}


def _kw_match(query_lower: str, keywords: list) -> bool:
    for kw in keywords:
        if " " in kw:
            if kw in query_lower:
                return True
        elif re.search(rf"\b{re.escape(kw)}\b", query_lower):
            return True
    return False


def detect_answer_style(query: str) -> Tuple[str, str]:
    q = query.lower()
    if _kw_match(q, STYLE_SHORT_KEYWORDS):
        return "short", "short"
    if _kw_match(q, STYLE_DETAIL_KEYWORDS):
        return "detail", "detail"
    if _kw_match(q, STYLE_ELABORATE_KEYWORDS):
        return "elaborate", "elaborate"
    return "elaborate", "elaborate"


def get_context_limit(style: str) -> int:
    if isinstance(MAX_CONTEXT_CHARS, dict):
        return MAX_CONTEXT_CHARS.get(style, MAX_CONTEXT_CHARS_DEFAULT)
    return int(MAX_CONTEXT_CHARS)


def get_token_limit(style: str) -> int:
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
            "You are AgniAI, a helpful assistant for India's Agniveer recruitment scheme. "
            "Respond naturally and concisely."
            f"\n\n{style_structure_instruction(style)}"
        ),
    }]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})
    return messages


def _compute_context_char_budget(*, query: str, style: str, history: list[dict] | None, reasoning: bool, use_rag: bool) -> tuple[int, int]:
    style_budget = get_token_limit(style)
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


def _generate_structured_rag_answer(
    *,
    query: str,
    style: str,
    docs: list[dict],
    model: str,
    session,
    reasoning: bool,
    history: list[dict] | None,
) -> dict:
    return generate_structured_answer(
        query,
        docs=docs,
        style=style,
        model=model,
        session=session,
        reasoning=reasoning,
        history=history,
    )


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
        "agni", "agniveer", "benefit", "package", "rally", "fitness",
    )
    if any(term in q for term in domain_terms):
        return "rag"

    reasoning_terms = ("calculate", "total", "sum", "overall", "aggregate", "combined", "after 4 years", "over 4 years")
    if any(term in q for term in reasoning_terms) and any(term in q for term in ("salary", "pay", "service", "seva", "benefit", "nidhi", "year", "years")):
        return "rag"

    return "reject"


def _should_use_rag(query: str) -> bool:
    return _classify_intent(query) == "rag"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _handle_ingest(command: str) -> None:
    parts = command.split(maxsplit=2)
    if len(parts) < 3:
        print(yellow("Usage: /ingest pdf <path> | /ingest url <url> | /ingest txt <path> | /ingest text <content> | /ingest docx <path>"))
        return

    kind = parts[1].lower()
    target = parts[2].strip()
    fn_map = {"pdf": ingest_pdf, "url": ingest_url, "txt": ingest_txt, "text": ingest_text, "docx": ingest_docx}
    if kind not in fn_map:
        print(yellow(f"Unknown type '{kind}'. Use: pdf, url, txt, text, docx."))
        return
    try:
        print(dim(f"  Ingesting {kind}..."))
        count = fn_map[kind](target)
        if count == 0:
            print(yellow("  Source already ingested (use /reset to re-ingest)."))
        else:
            print(green(f"  Ingested {count} chunk(s) successfully."))
    except FileNotFoundError as exc:
        print(red(f"  File not found: {exc}"))
    except Exception as exc:
        print(red(f"  Ingestion failed: {exc}"))


def _handle_sources() -> None:
    sources = list_sources()
    if not sources:
        print(yellow("  No sources ingested yet."))
        return
    print(bold(f"\n  Ingested Sources ({len(sources)} total):"))
    for s in sources:
        chunk_info = dim(f"({s['chunk_count']} chunks)")
        print(f"    - [{s['doc_type'].upper()}] {s['source']}  {chunk_info}")


def _handle_stats() -> None:
    stats = index_stats()
    print(f"\n  Index stats: {stats['vectors']} vectors / {stats['chunks']} chunks")


def _handle_reset() -> None:
    confirm = input(yellow("  This will DELETE the entire knowledge base. Type YES to confirm: ")).strip()
    if confirm == "YES":
        clear_index()
        print(green("  Knowledge base cleared."))
    else:
        print(dim("  Reset cancelled."))


def _history_fingerprint(history: list[dict]) -> str:
    payload = json.dumps(history[-6:], ensure_ascii=False, sort_keys=True)
    return sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def run_chat() -> None:
    _ensure_dirs()
    memory = ConversationMemory()
    active_model: Optional[str] = DEFAULT_MODEL_NAME
    session = requests.Session()

    print(BANNER)
    stats = index_stats()
    if stats["vectors"] == 0:
        print(yellow("  Knowledge base is empty. Use /ingest to add PDFs, URLs, or text.\n"))
    else:
        print(dim(f"  Knowledge base ready: {stats['vectors']} vectors loaded.\n"))
    print(dim(f"  Active model: {active_model}  (type /model <name> to switch)\n"))

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw:
            continue

        low = raw.lower()
        if low in {"/exit", "/quit"}:
            print("Goodbye.")
            break
        if low == "/help":
            print(HELP_TEXT)
            continue
        if low == "/sources":
            _handle_sources()
            continue
        if low == "/stats":
            _handle_stats()
            continue
        if low == "/clear":
            memory.clear()
            print(green("  Conversation memory cleared."))
            continue
        if low == "/reset":
            _handle_reset()
            continue
        if low.startswith("/model"):
            parts = raw.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                active_model = parts[1].strip()
                print(green(f"  Model switched to '{active_model}'."))
            else:
                print(yellow("  Usage: /model <model-name>"))
            continue
        if low.startswith("/ingest "):
            _handle_ingest(raw)
            continue
        if low.startswith("/"):
            print(yellow(f"  Unknown command: {raw} (type /help for a list)"))
            continue

        style_name, _ = detect_answer_style(raw)
        style_label = _STYLE_LABEL[style_name]
        style_color = _STYLE_COLOR[style_name]
        context_limit = get_context_limit(style_name)
        history = memory.history()
        intent = _classify_intent(raw)
        use_rag = intent == "rag"
        reasoning = is_reasoning_query(raw) if use_rag else False
        token_limit, context_char_budget = _compute_context_char_budget(
            query=raw,
            style=style_name,
            history=history,
            reasoning=reasoning,
            use_rag=use_rag,
        )
        print(dim("  Answer style: ") + style_color(style_label) + dim(f"  [ctx={context_limit} chars, tokens≤{token_limit}]"))

        bundle = {"docs": [], "context": "", "confidence": 0.0, "mode": "reject", "reasoning": False}
        if use_rag:
            print(dim("  Preparing retrieval..."))
            bundle = prepare_rag_bundle(raw, top_k=TOP_K, style=style_name, max_context_chars=context_char_budget)
        context = bundle.get("context", "") if isinstance(bundle, dict) else ""
        confidence = float(bundle.get("confidence", 0.0)) if isinstance(bundle, dict) else 0.0
        mode = bundle.get("mode", "reject") if isinstance(bundle, dict) else "reject"
        reasoning = bool(bundle.get("reasoning", False)) if isinstance(bundle, dict) else False
        if use_rag:
            print(dim(f"  Retrieval confidence: {confidence:.3f} | mode={mode} | reasoning={reasoning}"))
        history_hash = _history_fingerprint(history)
        response_key = make_response_cache_key(
            raw,
            style=style_name,
            model=active_model or DEFAULT_MODEL_NAME,
            context=context,
            session_id=f"cli:{history_hash}",
        )
        cached_answer = get_cached_response(response_key)
        if cached_answer is not None:
            print(dim("  Cache hit."))
            print(f"\nAgniAI: {cached_answer}\n")
            memory.add("user", raw)
            memory.add("assistant", cached_answer)
            continue

        if intent == "reject":
            answer = REFERENCE_FALLBACK
            print(f"\nAgniAI: {answer}\n")
            memory.add("user", raw)
            memory.add("assistant", answer)
            set_cached_response(response_key, answer)
            continue

        try:
            print("\nAgniAI: ", end="", flush=True)
            if use_rag:
                structured = _generate_structured_rag_answer(
                    query=raw,
                    style=style_name,
                    docs=bundle.get("docs", []) if isinstance(bundle, dict) else [],
                    model=active_model or DEFAULT_MODEL_NAME,
                    session=session,
                    reasoning=reasoning,
                    history=history[-6:] if history else None,
                )
                answer = str(structured.get("answer", "")).strip()
                if not answer and structured.get("points"):
                    answer = "\n".join(
                        f"{idx}. {point.get('title', '').strip()}"
                        for idx, point in enumerate(structured.get("points", []), start=1)
                        if point.get("title")
                    )
                if not answer:
                    answer = REFERENCE_FALLBACK
                print(answer)
            else:
                messages = [{"role": "system", "content": (
                    "You are AgniAI, a helpful assistant for India's Agniveer recruitment scheme. Respond naturally and concisely."
                    f"\n\n{style_structure_instruction(style_name)}"
                )}]
                if history:
                    messages.extend(history[-6:])
                messages.append({"role": "user", "content": raw})
                result = chat_with_fallback(
                    session,
                    active_model or DEFAULT_MODEL_NAME,
                    messages,
                    stream_tokens=True,
                    max_tokens_override=token_limit,
                )
                answer = _finalize_answer(result.text)
                print()
        except PartialResponseError as exc:
            print(f"\n  Partial response: {exc}\n")
            answer = _finalize_answer(exc.partial_text or REFERENCE_FALLBACK)
        except RuntimeError as exc:
            print(f"\n  LLM Error: {exc}\n")
            continue
        except KeyboardInterrupt:
            print("\n\n  [Generation stopped]\n")
            continue

        memory.add("user", raw)
        memory.add("assistant", answer)
        set_cached_response(response_key, answer)


if __name__ == "__main__":
    run_chat()
