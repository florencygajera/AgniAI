"""
ollama_cpu_chat.py
==================
CPU-optimised Ollama streaming client for AgniAI.

Key design decisions for CPU-only hardware:
  - Prefers tiny/small models (phi3:mini, llama3.2:3b, gemma2:2b, …)
  - Shrinks num_ctx to 1024 and max tokens to 160 to cut time-to-first-token
  - Uses a two-phase timeout: short connect + first-token timeout, then a
    generous streaming timeout so we don't cut off mid-answer
  - Retries with exponential back-off on transient errors
  - Gives up fast on heavy models and moves to the next candidate
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# =============================================================================
# CONFIG  (all overridable via environment variables)
# =============================================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
CHAT_ENDPOINT   = f"{OLLAMA_BASE_URL}/api/chat"
TAGS_ENDPOINT   = f"{OLLAMA_BASE_URL}/api/tags"

# ── Model priority ──────────────────────────────────────────────────────────
# Smallest capable models first — these run well on CPU-only hardware.
# llama3:latest (8B) is intentionally placed last; it's too slow for most CPUs.
MODEL_NAME = os.getenv("OLLAMA_MODEL", "phi3:mini")

FALLBACK_MODELS: List[str] = [
    m.strip()
    for m in os.getenv(
        "OLLAMA_FALLBACK_MODELS",
        "phi3:mini,phi3:3.8b,gemma2:2b,llama3.2:3b,llama3.2:1b,mistral:7b-instruct-q4_0,llama3:latest",
    ).split(",")
    if m.strip()
]

# ── Timeouts ────────────────────────────────────────────────────────────────
# FIRST_TOKEN_TIMEOUT: how long to wait before *any* token arrives.
#   Set low so we abandon a model that hasn't started generating quickly.
# STREAM_TIMEOUT: per-chunk read timeout once tokens are flowing.
TIMEOUT_CONNECT     = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "8"))
FIRST_TOKEN_TIMEOUT = float(os.getenv("OLLAMA_FIRST_TOKEN_TIMEOUT", "45"))  # seconds
STREAM_TIMEOUT      = float(os.getenv("OLLAMA_STREAM_TIMEOUT", "120"))      # seconds per chunk

MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))

# ── Generation knobs ────────────────────────────────────────────────────────
# Keep these small — they directly control how much the CPU has to crunch.
MAX_TOKENS     = int(os.getenv("OLLAMA_MAX_TOKENS",     "160"))
NUM_CTX        = int(os.getenv("OLLAMA_NUM_CTX",        "1024"))   # ← critical for CPU speed
TEMPERATURE    = float(os.getenv("OLLAMA_TEMPERATURE",  "0.2"))
TOP_K          = int(os.getenv("OLLAMA_TOP_K",          "20"))
TOP_P          = float(os.getenv("OLLAMA_TOP_P",        "0.9"))
REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY","1.1"))
KEEP_ALIVE     = os.getenv("OLLAMA_KEEP_ALIVE",          "15m")    # keep model hot

# ── RAG limits ───────────────────────────────────────────────────────────────
MAX_HISTORY_MESSAGES = int(os.getenv("OLLAMA_MAX_HISTORY_MESSAGES", "4"))
MAX_RAG_CHARS        = int(os.getenv("OLLAMA_MAX_RAG_CHARS",        "1800"))

SYSTEM_PROMPT = (
    "You are AgniAI, a concise assistant for India's Agniveer/Agnipath recruitment. "
    "Answer using only the retrieved context. If the context is insufficient, say so. "
    "Keep answers short and structured."
)


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class ChatResult:
    model: str
    text: str
    duration_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


# =============================================================================
# ERRORS
# =============================================================================

class OllamaError(RuntimeError):
    pass


class PartialResponseError(OllamaError):
    def __init__(self, message: str, partial_text: str):
        super().__init__(message)
        self.partial_text = partial_text


# =============================================================================
# RAG HOOK
# =============================================================================

def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.70)
    tail = max(0, limit - head - 20)
    return f"{text[:head].rstrip()}\n\n...[truncated]...\n\n{text[-tail:].lstrip()}"


def build_rag_context(query: str) -> str:
    """Pull top-2 chunks from FAISS and return a truncated context string."""
    try:
        from rag import build_context, search  # type: ignore
        docs = search(query, top_k=2)
        context = build_context(docs)
        return _truncate(context, MAX_RAG_CHARS)
    except Exception:
        return ""


def build_messages(query: str, history: List[dict]) -> List[dict]:
    context = build_rag_context(query)
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history[-MAX_HISTORY_MESSAGES:])

    if context:
        user_content = (
            f"Question:\n{query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer concisely using only the context above."
        )
    else:
        user_content = query

    messages.append({"role": "user", "content": user_content})
    return messages


# =============================================================================
# OLLAMA CLIENT
# =============================================================================

def _installed_models(session: requests.Session) -> List[str]:
    """Return names of locally installed Ollama models, smallest-first heuristic."""
    try:
        resp = session.get(TAGS_ENDPOINT, timeout=(TIMEOUT_CONNECT, 10))
        resp.raise_for_status()
        models = resp.json().get("models", [])
        # Sort by size ascending so smaller models are tried first
        models.sort(key=lambda m: m.get("size", 99_000_000_000))
        return [m["name"] for m in models if m.get("name")]
    except Exception:
        return []


def _candidate_models(requested: str, installed: List[str]) -> List[str]:
    """
    Build an ordered list of models to try.
    Priority: requested → FALLBACK_MODELS list (filtered to installed) → remaining installed.
    """
    installed_set = set(installed)
    ordered: List[str] = []

    def _add(name: str) -> None:
        if name and name not in ordered:
            ordered.append(name)

    # 1. User-requested model first (even if not installed — Ollama may pull it)
    _add(requested)

    # 2. Walk FALLBACK_MODELS in order, but only if installed
    for fb in FALLBACK_MODELS:
        if fb in installed_set:
            _add(fb)

    # 3. Any remaining installed models (already sorted small→large above)
    for m in installed:
        _add(m)

    # 4. Full fallback list in case nothing installed matched
    for fb in FALLBACK_MODELS:
        _add(fb)

    return ordered


def _iter_ndjson(resp: requests.Response) -> Iterable[dict]:
    """Parse newline-delimited JSON from a streaming response."""
    buffer = b""
    for chunk in resp.iter_content(chunk_size=2048):
        if not chunk:
            continue
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            yield json.loads(line.decode("utf-8", errors="replace"))
    tail = buffer.strip()
    if tail:
        yield json.loads(tail.decode("utf-8", errors="replace"))


def _ollama_chat_once(
    session: requests.Session,
    model: str,
    messages: List[dict],
    *,
    stream_tokens: bool = True,
    on_token: Optional[Callable[[str], None]] = None,
) -> ChatResult:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature":    TEMPERATURE,
            "num_ctx":        NUM_CTX,
            "num_predict":    MAX_TOKENS,
            "top_k":          TOP_K,
            "top_p":          TOP_P,
            "repeat_penalty": REPEAT_PENALTY,
            # CPU threading — use all available cores
            "num_thread":     int(os.getenv("OLLAMA_NUM_THREAD", "0")),  # 0 = auto
        },
    }

    pieces: List[str] = []
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    first_token_received = False
    start = time.time()
    deadline_first_token = start + FIRST_TOKEN_TIMEOUT

    try:
        with session.post(
            CHAT_ENDPOINT,
            json=payload,
            stream=True,
            timeout=(TIMEOUT_CONNECT, STREAM_TIMEOUT),
        ) as resp:
            if resp.status_code == 404:
                raise OllamaError(
                    f"Model '{model}' not found in Ollama. "
                    "Run:  ollama pull phi3:mini"
                )
            if resp.status_code >= 400:
                body = resp.text[:500].strip()
                raise OllamaError(
                    f"Ollama HTTP {resp.status_code} for '{model}': {body}"
                )

            for event in _iter_ndjson(resp):
                if "error" in event:
                    raise OllamaError(str(event["error"]))

                if event.get("done"):
                    prompt_tokens     = event.get("prompt_eval_count", prompt_tokens)
                    completion_tokens = event.get("eval_count", completion_tokens)
                    break

                token = event.get("message", {}).get("content", "")
                if token:
                    if not first_token_received:
                        first_token_received = True
                        # First token arrived — good to go regardless of wall time
                    pieces.append(token)
                    if stream_tokens:
                        if on_token is not None:
                            on_token(token)
                        else:
                            sys.stdout.write(token)
                            sys.stdout.flush()

                # Enforce first-token deadline (check only before we get any token)
                if not first_token_received and time.time() > deadline_first_token:
                    raise OllamaError(
                        f"Model '{model}' did not produce a first token within "
                        f"{FIRST_TOKEN_TIMEOUT:.0f}s. "
                        "It is likely too large for this CPU. "
                        "Install a smaller model:  ollama pull phi3:mini"
                    )

    except requests.Timeout as exc:
        if pieces:
            raise PartialResponseError(
                f"Stream timeout from '{model}' after partial response.",
                "".join(pieces),
            ) from exc
        raise OllamaError(
            f"Timed out waiting for '{model}'. "
            "Try a smaller model or increase OLLAMA_FIRST_TOKEN_TIMEOUT."
        ) from exc

    except requests.ConnectionError as exc:
        if pieces:
            raise PartialResponseError(
                f"Connection dropped mid-stream from '{model}'.",
                "".join(pieces),
            ) from exc
        raise OllamaError(
            "Cannot connect to Ollama. Is it running?  Run:  ollama serve"
        ) from exc

    except json.JSONDecodeError as exc:
        if pieces:
            raise PartialResponseError(
                f"Malformed JSON from '{model}'.", "".join(pieces)
            ) from exc
        raise OllamaError(f"Malformed JSON from Ollama: {exc}") from exc

    duration = time.time() - start
    return ChatResult(
        model=model,
        text="".join(pieces),
        duration_s=duration,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def chat_with_fallback(
    session: requests.Session,
    requested_model: str,
    messages: List[dict],
    *,
    stream_tokens: bool = True,
) -> ChatResult:
    """
    Try models in priority order until one responds successfully.
    Moves to the next model quickly on OllamaError (includes first-token timeout).
    """
    installed = _installed_models(session)
    candidates = _candidate_models(requested_model, installed)

    if not candidates:
        raise OllamaError(
            "No Ollama models available. Run:  ollama pull phi3:mini"
        )

    last_error: Optional[str] = None
    for model in candidates:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = _ollama_chat_once(
                    session,
                    model,
                    messages,
                    stream_tokens=stream_tokens,
                )
                return result

            except PartialResponseError:
                # Partial response already streamed — surface it immediately.
                raise

            except OllamaError as exc:
                last_error = str(exc)
                if attempt < MAX_RETRIES:
                    wait = 0.5 * attempt
                    time.sleep(wait)
                    continue
                # Exhausted retries for this model → try next candidate
                break

    raise OllamaError(
        "All model candidates failed.\n"
        f"Last error: {last_error}\n"
        f"Tried: {', '.join(candidates)}\n\n"
        "Fix: run  ollama pull phi3:mini  then restart AgniAI."
    )


# =============================================================================
# CLI (standalone usage)
# =============================================================================

def print_banner() -> None:
    print("=" * 68)
    print("AgniAI — Ollama CPU-safe streaming client")
    print(f"  Base URL      : {OLLAMA_BASE_URL}")
    print(f"  Default model : {MODEL_NAME}")
    print(f"  num_ctx       : {NUM_CTX}   max_tokens: {MAX_TOKENS}")
    print(f"  1st-token t/o : {FIRST_TOKEN_TIMEOUT}s   stream t/o: {STREAM_TIMEOUT}s")
    print("=" * 68)


def main() -> int:
    session = requests.Session()
    model   = MODEL_NAME
    history: List[dict] = []

    print_banner()

    installed = _installed_models(session)
    if installed:
        print(f"Installed models (smallest first): {', '.join(installed)}\n")
    else:
        print("[warning] Could not list installed models — is Ollama running?\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not user:
            continue

        low = user.lower()

        if low in {"/exit", "/quit"}:
            print("Goodbye.")
            return 0
        if low == "/help":
            print(
                "\nCommands: /exit  /model <name>  /clear  /tags  /health  /help\n"
            )
            continue
        if low == "/clear":
            history.clear()
            print("History cleared.")
            continue
        if low == "/tags":
            models = _installed_models(session)
            print(", ".join(models) if models else "(none detected)")
            continue
        if low == "/health":
            try:
                session.get(TAGS_ENDPOINT, timeout=(TIMEOUT_CONNECT, 10)).raise_for_status()
                print("Ollama API reachable.")
            except Exception as exc:
                print(f"[error] {exc}")
            continue
        if low.startswith("/model "):
            model = user.split(maxsplit=1)[1].strip()
            print(f"Model switched to {model}")
            continue
        if low.startswith("/"):
            print("Unknown command. Type /help.")
            continue

        messages = build_messages(user, history)
        print("Assistant: ", end="", flush=True)

        try:
            result = chat_with_fallback(session, model, messages, stream_tokens=True)
            print()
            history.append({"role": "user",      "content": user})
            history.append({"role": "assistant",  "content": result.text})
            if len(history) > MAX_HISTORY_MESSAGES * 2:
                history = history[-(MAX_HISTORY_MESSAGES * 2):]
            if result.prompt_tokens is not None:
                print(
                    f"[model={result.model} time={result.duration_s:.1f}s "
                    f"prompt_tok={result.prompt_tokens} "
                    f"completion_tok={result.completion_tokens}]"
                )
        except PartialResponseError as exc:
            print("\n[warning] Partial response received (connection dropped).")
            if exc.partial_text:
                print(exc.partial_text)
        except OllamaError as exc:
            print(f"\n[error] {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())