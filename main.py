"""
main.py
=======
AgniAI — Offline CLI chatbot for Agniveer recruitment queries.
Used ONLY for local testing. The .NET team integrates via app.py (REST API).

Run:
    python main.py

Answer styles (auto-detected from your question):
    Short    — "What is the age limit in short?"
    Elaborate— "Elaborate on the salary structure."   ← default
    Detail   — "Explain the selection process in detail."

Commands (during chat):
    /ingest pdf  <path>    — Add a PDF file
    /ingest url  <url>     — Add a web page
    /ingest txt  <path>    — Add a .txt file
    /ingest text <content> — Add raw text
    /ingest docx <path>    — Add a Word (.docx) file
    /sources               — List all ingested sources
    /stats                 — Show index statistics
    /clear                 — Clear conversation memory
    /reset                 — ⚠ Delete the entire knowledge base index
    /model <name>          — Switch Ollama model mid-session
    /help                  — Show this help message
    /exit  or  /quit       — Exit
"""

import re
import sys
import textwrap
from typing import Optional, Tuple

import requests

from config import (
    DATA_DIR,
    INDEX_DIR,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHARS_DEFAULT,
    MAX_TOKENS_STYLE,
    MAX_TOKENS_DEFAULT,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHORT,
    SYSTEM_PROMPT_ELABORATE,
    SYSTEM_PROMPT_DETAIL,
    STYLE_SHORT_KEYWORDS,
    STYLE_ELABORATE_KEYWORDS,
    STYLE_DETAIL_KEYWORDS,
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
from ollama_cpu_chat import MODEL_NAME as DEFAULT_MODEL_NAME
from ollama_cpu_chat import PartialResponseError, chat_with_fallback
from rag import build_context, index_stats, search

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── ANSI colour helpers ────────────────────────────────────────────────────

def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"

def dim(t):    return _c("2",  t)
def bold(t):   return _c("1",  t)
def cyan(t):   return _c("96", t)
def green(t):  return _c("92", t)
def yellow(t): return _c("93", t)
def red(t):    return _c("91", t)
def blue(t):   return _c("94", t)


BANNER = cyan(r"""
   ___                  _ ___    ___
  / _ |___  ___ _  ___ (_) _ \  / _ \
 / __ / _ \/ _ \ |/ _ \| | | | | (_) |
/_/ |_\___/_//_/___|___/|_|___/  \___/
""") + bold("  Agniveer AI Assistant  — Offline · Local · Private\n")

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
  /model <name>           Switch the Ollama model (e.g. phi3:mini)
  /help                   Show this help
  /exit  or  /quit        Exit AgniAI

Answer style is detected automatically from your question:
  • "... in short"     → SHORT  (2-4 tight bullets)
  • "elaborate ..."    → ELABORATE (6-12 bullets + context)   ← default
  • "... in detail"    → DETAIL (full numbered sections, every figure)

Recommended small models for CPU:
  ollama pull phi3:mini      (~2.3 GB)
  ollama pull llama3.2:3b   (~2.0 GB)
  ollama pull gemma2:2b     (~1.6 GB)
"""

_STYLE_LABEL = {
    "short":     "SHORT",
    "elaborate": "ELABORATE",
    "detail":    "DETAIL",
}

_STYLE_COLOR = {
    "short":     yellow,
    "elaborate": cyan,
    "detail":    blue,
}


# ── Answer-style detection ─────────────────────────────────────────────────

def _kw_match(query_lower: str, keywords: list) -> bool:
    """
    Whole-word / whole-phrase keyword match.
    Prevents "shorting" from triggering SHORT mode, etc.
    Multi-word phrases are matched as substrings (already specific enough).
    Single words are matched with word boundaries.
    """
    for kw in keywords:
        if " " in kw:
            # Multi-word phrase: substring match is fine (specific enough)
            if kw in query_lower:
                return True
        else:
            # Single word: require word boundary to avoid false positives
            if re.search(rf"\b{re.escape(kw)}\b", query_lower):
                return True
    return False


def detect_answer_style(query: str) -> Tuple[str, str]:
    """
    Inspect *query* for style keywords and return (style_name, system_prompt).

    Priority:  short  >  detail  >  elaborate  >  elaborate (default)

    Returns
    -------
    style_name   : "short" | "elaborate" | "detail"
    system_prompt: matching SYSTEM_PROMPT_* string
    """
    q = query.lower()

    if _kw_match(q, STYLE_SHORT_KEYWORDS):
        return "short", SYSTEM_PROMPT_SHORT

    if _kw_match(q, STYLE_DETAIL_KEYWORDS):
        return "detail", SYSTEM_PROMPT_DETAIL

    if _kw_match(q, STYLE_ELABORATE_KEYWORDS):
        return "elaborate", SYSTEM_PROMPT_ELABORATE

    return "elaborate", SYSTEM_PROMPT_ELABORATE


def get_context_limit(style: str) -> int:
    """Return the MAX_CONTEXT_CHARS for the given style."""
    if isinstance(MAX_CONTEXT_CHARS, dict):
        return MAX_CONTEXT_CHARS.get(style, MAX_CONTEXT_CHARS_DEFAULT)
    # backwards compat if someone passes old int config
    return int(MAX_CONTEXT_CHARS)


def get_token_limit(style: str) -> int:
    return MAX_TOKENS_STYLE.get(style, MAX_TOKENS_DEFAULT)


# ── Directory bootstrap ────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _should_use_rag(query: str) -> bool:
    """
    Skip RAG only for pure small-talk / greetings.
    Any query with 2+ words that looks substantive gets RAG.
    """
    q = query.strip().lower()
    _GREETINGS = {
        "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye",
        "good morning", "good evening", "good afternoon", "greetings",
        "welcome", "sup", "yo",
    }
    words = q.split()
    # Single greeting word → no RAG
    if len(words) == 1 and q in _GREETINGS:
        return False
    # Two-word greeting phrases → no RAG
    if len(words) <= 3 and q in _GREETINGS:
        return False
    return True


# ── Command handlers ───────────────────────────────────────────────────────

def _handle_ingest(command: str) -> None:
    parts = command.split(maxsplit=2)
    if len(parts) < 3:
        print(yellow(
            "Usage:  /ingest pdf <path>  |  /ingest url <url>  "
            "|  /ingest txt <path>  |  /ingest text <content>  "
            "|  /ingest docx <path>"
        ))
        return

    kind   = parts[1].lower()
    target = parts[2].strip()

    fn_map = {
        "pdf":  ingest_pdf,
        "url":  ingest_url,
        "txt":  ingest_txt,
        "text": ingest_text,
        "docx": ingest_docx,
    }

    if kind not in fn_map:
        print(yellow(f"  Unknown type '{kind}'. Use: pdf, url, txt, text, docx."))
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
    print(
        f"\n  Index stats: "
        f"{stats['vectors']} vectors  /  "
        f"{stats['chunks']} chunks"
    )


def _handle_reset() -> None:
    confirm = input(
        yellow("  This will DELETE the entire knowledge base. Type YES to confirm: ")
    ).strip()
    if confirm == "YES":
        clear_index()
        print(green("  Knowledge base cleared."))
    else:
        print(dim("  Reset cancelled."))


# ── Main chat loop ─────────────────────────────────────────────────────────

def run_chat() -> None:
    _ensure_dirs()
    memory       = ConversationMemory()
    active_model: Optional[str] = DEFAULT_MODEL_NAME
    session      = requests.Session()

    print(BANNER)

    stats = index_stats()
    if stats["vectors"] == 0:
        print(yellow(
            "  Knowledge base is empty.  "
            "Use /ingest to add PDFs, URLs, or text.\n"
        ))
    else:
        print(dim(f"  Knowledge base ready: {stats['vectors']} vectors loaded.\n"))

    print(dim(f"  Active model: {active_model}  (type /model <name> to switch)\n"))
    print(dim("  Type /help for commands or just ask a question.\n"))
    print(dim(
        "  Tip: add 'in short', 'elaborate', or 'in detail' to your question "
        "to control answer length.\n"
    ))

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
                print(yellow("  Usage: /model <model-name>  e.g.  /model phi3:mini"))
            continue

        if low.startswith("/ingest "):
            _handle_ingest(raw)
            continue

        if low.startswith("/"):
            print(yellow(f"  Unknown command: {raw}  (type /help for a list)"))
            continue

        # ── Detect answer style ────────────────────────────────────────────
        style_name, active_system_prompt = detect_answer_style(raw)
        style_label = _STYLE_LABEL[style_name]
        style_color = _STYLE_COLOR[style_name]
        context_limit = get_context_limit(style_name)
        token_limit   = get_token_limit(style_name)

        print(dim(f"  Answer style: ") + style_color(style_label)
              + dim(f"  [ctx={context_limit} chars, tokens≤{token_limit}]"))

        # ── RAG pipeline ───────────────────────────────────────────────────
        use_rag = _should_use_rag(raw)
        context = ""
        if use_rag:
            print(dim("  Searching knowledge base..."))
            docs    = search(raw, top_k=TOP_K)
            context = build_context(docs)
            if len(context) > context_limit:
                context = context[:context_limit].rstrip() + "\n...[truncated]..."

        if use_rag and not context:
            no_info = (
                "I don't have that information in my knowledge base. "
                "Please ingest the relevant document first using "
                "/ingest pdf, /ingest url, /ingest docx, or /ingest text."
            )
            print(f"\nAgniAI: {no_info}\n")
            memory.add("user",      raw)
            memory.add("assistant", no_info)
            continue

        print(dim("  Generating answer..."))

        history = memory.history()

        if use_rag:
            messages = [{"role": "system", "content": active_system_prompt}]
            if history:
                messages.extend(history[-6:])   # fixed: was -4
            user_content = (
                f"Reference information:\n{context}\n\n"
                f"Question: {raw}"
            )
        else:
            messages = [{"role": "system", "content": (
                "You are AgniAI, a helpful assistant for India's Agniveer "
                "recruitment scheme. Respond naturally and concisely."
            )}]
            if history:
                messages.extend(history[-6:])
            user_content = raw

        messages.append({"role": "user", "content": user_content})

        # ── LLM call with per-style token limit ────────────────────────────
        try:
            print(f"\nAgniAI: ", end="", flush=True)
            result = chat_with_fallback(
                session,
                active_model,
                messages,
                stream_tokens=True,
                max_tokens_override=token_limit,
            )
            answer = result.text
            print("\n")
        except PartialResponseError as exc:
            print(f"\n  Partial response: {exc}\n")
            answer = exc.partial_text
            if answer:
                print(f"{answer}\n")
        except RuntimeError as exc:
            print(f"\n  LLM Error: {exc}\n")
            continue
        except KeyboardInterrupt:
            print("\n\n  [Generation stopped]\n")
            continue

        memory.add("user",      raw)
        memory.add("assistant", answer)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_chat()