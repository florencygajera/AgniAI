"""
main.py
=======
AgniAI — Offline CLI chatbot for Agniveer recruitment queries.

Run:
    python main.py

Commands (during chat):
    /ingest pdf  <path>    — Add a PDF file to the knowledge base
    /ingest url  <url>     — Add a web page to the knowledge base
    /ingest txt  <path>    — Add a plain .txt file to the knowledge base
    /ingest text <content> — Add raw text to the knowledge base
    /sources               — List all ingested sources
    /stats                 — Show index statistics
    /clear                 — Clear the conversation memory
    /reset                 — ⚠ Delete the entire knowledge base index
    /model <name>          — Switch the Ollama model mid-session
    /help                  — Show this help message
    /exit  or  /quit       — Exit AgniAI
"""

import sys
import textwrap
from typing import Optional

from config import DATA_DIR, INDEX_DIR, TOP_K
from ingest import clear_index, ingest_pdf, ingest_text, ingest_txt, ingest_url, list_sources
from memory import ConversationMemory
from rag import build_context, call_llm, index_stats, search


# ── ANSI colour helpers ────────────────────────────────────────────────────

def _c(code: str, text: str) -> str:
    """Wrap *text* in an ANSI colour code (disabled on non-tty)."""
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"

def dim(t: str) -> str:   return _c("2", t)
def bold(t: str) -> str:  return _c("1", t)
def cyan(t: str) -> str:  return _c("96", t)
def green(t: str) -> str: return _c("92", t)
def yellow(t: str) -> str:return _c("93", t)
def red(t: str) -> str:   return _c("91", t)
def blue(t: str) -> str:  return _c("94", t)


BANNER = cyan(r"""
   ___                  _ ___    ___ 
  / _ |___  ___ _  ___ (_) _ \  / _ \
 / __ / _ \/ _ \ |/ _ \| | | | | (_) |
/_/ |_\___/_//_/___|___/|_|___/  \___/
""") + bold("  Agniveer AI Assistant  — Offline · Local · Private\n")

HELP_TEXT = f"""
{bold("Available commands:")}

  {cyan("/ingest pdf  <path>")}    Add a PDF to the knowledge base
  {cyan("/ingest url  <url>")}     Add a web page to the knowledge base
  {cyan("/ingest txt  <path>")}    Add a plain .txt file to the knowledge base
  {cyan("/ingest text <content>")} Add raw text to the knowledge base
  {cyan("/sources")}               List all ingested sources
  {cyan("/stats")}                 Show index vector count
  {cyan("/clear")}                 Clear conversation memory
  {cyan("/reset")}                 ⚠  Delete the entire knowledge base
  {cyan("/model <name>")}          Switch the Ollama model (e.g. mistral)
  {cyan("/help")}                  Show this help message
  {cyan("/exit")}  or  {cyan("/quit")}       Exit AgniAI
"""


# ── Directory bootstrap ────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ── Command handlers ───────────────────────────────────────────────────────

def _handle_ingest(command: str) -> None:
    parts = command.split(maxsplit=2)
    if len(parts) < 3:
        print(yellow("Usage:  /ingest pdf <path>  |  /ingest url <url>  "
                     "|  /ingest txt <path>  |  /ingest text <content>"))
        return

    kind = parts[1].lower()
    target = parts[2].strip()

    try:
        print(dim(f"  ⏳ Ingesting {kind}…"))
        if kind == "pdf":
            count = ingest_pdf(target)
        elif kind == "url":
            count = ingest_url(target)
        elif kind == "txt":
            count = ingest_txt(target)
        elif kind == "text":
            count = ingest_text(target)
        else:
            print(yellow(f"  Unknown type '{kind}'. Use: pdf, url, txt, or text."))
            return

        if count == 0:
            print(yellow("  ⚠  Source already ingested (use /reset to re-ingest)."))
        else:
            print(green(f"  ✔  Ingested {count} chunk(s) successfully."))
    except FileNotFoundError as exc:
        print(red(f"  ✘  File not found: {exc}"))
    except Exception as exc:
        print(red(f"  ✘  Ingestion failed: {exc}"))


def _handle_sources() -> None:
    sources = list_sources()
    if not sources:
        print(yellow("  No sources ingested yet."))
        return
    print(bold(f"\n  📚 Ingested Sources ({len(sources)} total):"))
    for s in sources:
        chunk_info = dim(f"({s['chunk_count']} chunks)")
        print(f"    {cyan('•')} [{s['doc_type'].upper()}] {s['source']}  {chunk_info}")


def _handle_stats() -> None:
    stats = index_stats()
    print(f"\n  {bold('Index stats:')}  "
          f"{cyan(str(stats['vectors']))} vectors  ·  "
          f"{cyan(str(stats['chunks']))} chunks")


def _handle_reset() -> None:
    confirm = input(yellow("  ⚠  This will DELETE the entire knowledge base. Type YES to confirm: ")).strip()
    if confirm == "YES":
        clear_index()
        print(green("  ✔  Knowledge base cleared."))
    else:
        print(dim("  Reset cancelled."))


# ── Formatting ─────────────────────────────────────────────────────────────

def _wrap_answer(text: str, width: int = 80) -> str:
    """Soft-wrap answer lines for readability in the terminal."""
    lines = text.splitlines()
    wrapped = []
    for line in lines:
        # Don't wrap bullet/numbered lines at start — just indent them
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        prefix = " " * indent
        wrapped_line = textwrap.fill(line, width=width, subsequent_indent=prefix + "  ")
        wrapped.append(wrapped_line)
    return "\n".join(wrapped)


# ── Main chat loop ─────────────────────────────────────────────────────────

def run_chat() -> None:
    _ensure_dirs()
    memory = ConversationMemory()
    active_model: Optional[str] = None   # None → auto-detect from Ollama

    print(BANNER)
    stats = index_stats()
    if stats["vectors"] == 0:
        print(yellow("  ℹ  Knowledge base is empty.  "
                     "Use /ingest to add PDFs, URLs, or text.\n"))
    else:
        print(dim(f"  Knowledge base ready: {stats['vectors']} vectors loaded.\n"))

    print(dim("  Type /help for commands or just ask a question.\n"))

    while True:
        # ── Input ──────────────────────────────────────────────────────────
        try:
            raw = input(f"{bold(cyan('You'))}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{dim('Goodbye.')}")
            break

        if not raw:
            continue

        low = raw.lower()

        # ── Built-in commands ──────────────────────────────────────────────
        if low in {"/exit", "/quit"}:
            print(dim("Goodbye."))
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
            print(green("  ✔  Conversation memory cleared."))
            continue

        if low == "/reset":
            _handle_reset()
            continue

        if low.startswith("/model "):
            parts = raw.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                active_model = parts[1].strip()
                print(green(f"  ✔  Model switched to '{active_model}'."))
            else:
                print(yellow("  Usage: /model <model-name>  e.g.  /model mistral"))
            continue

        if low.startswith("/ingest "):
            _handle_ingest(raw)
            continue

        if low.startswith("/"):
            print(yellow(f"  Unknown command: {raw}  (type /help for a list)"))
            continue

        # ── RAG pipeline ───────────────────────────────────────────────────
        print(dim("  🔍 Searching knowledge base…"))
        docs = search(raw, top_k=TOP_K)
        context = build_context(docs)

        if not context:
            no_info = ("I don't have that information in my knowledge base. "
                       "Please ingest the relevant document first using "
                       "/ingest pdf, /ingest url, or /ingest text.")
            print(f"\n{bold(blue('AgniAI'))}: {yellow(no_info)}\n")
            memory.add("user", raw)
            memory.add("assistant", no_info)
            continue

        print(dim("  🤖 Generating answer…"))

        history = memory.history()
        prompt = (
            f"Question: {raw}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Answer in a structured, concise format using ONLY the retrieved context above."
        )

        try:
            answer = call_llm(prompt, history=history, model=active_model)
        except RuntimeError as exc:
            print(f"\n{red('  ✘  LLM Error:')} {exc}\n")
            continue

        print(f"\n{bold(blue('AgniAI'))}:\n{_wrap_answer(answer)}\n")

        memory.add("user", raw)
        memory.add("assistant", answer)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_chat()