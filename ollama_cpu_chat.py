"""
ollama_cpu_chat.py
==================
CPU-optimised Ollama streaming client for AgniAI.

Speed strategy (fastest to slowest impact):
  1. keep_alive= 5m  → model stays in RAM permanently, zero reload cost
  2. num_ctx=512    → half the KV-cache to fill, biggest time-to-first-token win
  3. num_predict=120→ stop generating sooner; RAG answers are short anyway
  4. top_k=10       → less sampling work per token
  5. num_thread=0   → Ollama uses all CPU cores automatically
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import requests

from config import MAX_CONTEXT_CHARS as MAX_RAG_CHARS, SYSTEM_PROMPT 

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
TIMEOUT_CONNECT     = float(os.getenv("OLLAMA_CONNECT_TIMEOUT",   "8"))
FIRST_TOKEN_TIMEOUT = float(os.getenv("OLLAMA_FIRST_TOKEN_TIMEOUT","45"))
STREAM_TIMEOUT      = float(os.getenv("OLLAMA_STREAM_TIMEOUT",    "120"))
MAX_RETRIES         = int(os.getenv("OLLAMA_MAX_RETRIES",          "2"))

# ── Generation knobs — tuned for maximum CPU speed ──────────────────────────
#
#  NUM_CTX    : KV-cache size. The single biggest lever for speed on CPU.
#               512 tokens is enough for system prompt + 2 RAG chunks + answer.
#               Cutting from 1024 → 512 roughly halves prefill time.
#
#  MAX_TOKENS : Cap output length. RAG answers rarely need >100 tokens.
#               Shorter cap = faster wall-clock finish time.
#
#  TOP_K=10   : Fewer candidates sampled per token → faster sampling loop.
#
#  KEEP_ALIVE= 5m : Never unload the model from RAM. The single biggest
#               latency win if you ask multiple questions: zero reload cost.
#               Uses ~2-4 GB RAM permanently. Change to "10m" if RAM is tight.

MAX_TOKENS     = int(os.getenv("OLLAMA_MAX_TOKENS",      "256")) 
NUM_CTX        = int(os.getenv("OLLAMA_NUM_CTX",         "512"))  
TEMPERATURE    = float(os.getenv("OLLAMA_TEMPERATURE",   "0.1"))   
TOP_K          = int(os.getenv("OLLAMA_TOP_K",           "5"))    
TOP_P          = float(os.getenv("OLLAMA_TOP_P",         "0.9"))
REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY","1.1"))
KEEP_ALIVE     = os.getenv("OLLAMA_KEEP_ALIVE",          "5m")     

# ── RAG limits ───────────────────────────────────────────────────────────────
MAX_HISTORY_MESSAGES = int(os.getenv("OLLAMA_MAX_HISTORY_MESSAGES", "5"))  


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
            f"=== REFERENCE TEXT ===\n{context}\n=== END ===\n\n"
            f"Question: {query}\n\nAnswer from the reference text above."
        )
    else:
        user_content = query
    messages.append({"role": "user", "content": user_content})
    return messages


# =============================================================================
# OLLAMA CLIENT
# =============================================================================

def _installed_models(session: requests.Session) -> List[str]:
    try:
        resp = session.get(TAGS_ENDPOINT, timeout=(TIMEOUT_CONNECT, 10))
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
    for m in installed:
        _add(m)
    # BUG-6 FIX: do not append fallback models that are not installed.
    return ordered


def _iter_ndjson(resp: requests.Response) -> Iterable[dict]:
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
            "num_thread":     int(os.getenv("OLLAMA_NUM_THREAD", "0")),  
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
                    f"Model '{model}' not found. Run:  ollama pull phi3:mini"
                )
            if resp.status_code >= 400:
                raise OllamaError(
                    f"Ollama HTTP {resp.status_code} for '{model}': {resp.text[:300]}"
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
                    first_token_received = True
                    pieces.append(token)
                    if stream_tokens:
                        if on_token is not None:
                            on_token(token)
                        else:
                            sys.stdout.write(token)
                            sys.stdout.flush()

                if not first_token_received and time.time() > deadline_first_token:
                    raise OllamaError(
                        f"Model '{model}' no first token in {FIRST_TOKEN_TIMEOUT:.0f}s. "
                        "Too large for this CPU. Try:  ollama pull phi3:mini"
                    )

    except requests.Timeout as exc:
        if pieces:
            raise PartialResponseError("Stream timeout after partial response.", "".join(pieces)) from exc
        raise OllamaError(f"Timeout waiting for '{model}'.") from exc

    except requests.ConnectionError as exc:
        if pieces:
            raise PartialResponseError("Connection dropped mid-stream.", "".join(pieces)) from exc
        raise OllamaError("Cannot connect to Ollama. Run:  ollama serve") from exc

    except json.JSONDecodeError as exc:
        if pieces:
            raise PartialResponseError("Malformed JSON mid-stream.", "".join(pieces)) from exc
        raise OllamaError(f"Malformed JSON: {exc}") from exc
    except KeyboardInterrupt:  # CLEANUP-FIX
        if pieces:  # CLEANUP-FIX
            raise PartialResponseError("Interrupted by user.", "".join(pieces))  
        raise  # CLEANUP-FIX

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
    installed  = _installed_models(session)
    candidates = _candidate_models(requested_model, installed)

    if not candidates:
        raise OllamaError("No Ollama models found. Run:  ollama pull phi3:mini")

    last_error: Optional[str] = None
    for model in candidates:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return _ollama_chat_once(session, model, messages, stream_tokens=stream_tokens)
            except PartialResponseError:
                raise
            except OllamaError as exc:
                last_error = str(exc)
                if attempt < MAX_RETRIES:
                    time.sleep(0.5 * attempt)
                    continue
                break

    raise OllamaError(
        f"All models failed. Last error: {last_error}\n"
        f"Tried: {', '.join(candidates)}\n"
        "Fix: run  ollama pull phi3:mini"
    )


# =============================================================================
# CLI (standalone usage)
# =============================================================================

def main() -> int:
    session = requests.Session()
    model   = MODEL_NAME
    history: List[dict] = []

    print(f"AgniAI CPU chat | model={model} | num_ctx={NUM_CTX} | max_tokens={MAX_TOKENS} | keep_alive={KEEP_ALIVE}")
    installed = _installed_models(session)
    if installed:
        print(f"Installed (smallest first): {', '.join(installed)}\n")

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
        if low == "/clear":
            history.clear()
            print("History cleared.")
            continue
        if low.startswith("/model "):
            model = user.split(maxsplit=1)[1].strip()
            print(f"Model: {model}")
            continue
        if low.startswith("/"):
            print("Commands: /exit  /model <n>  /clear")
            continue

        messages = build_messages(user, history)
        print("AgniAI: ", end="", flush=True)

        try:
            result = chat_with_fallback(session, model, messages, stream_tokens=True)
            print()
            history.append({"role": "user",      "content": user})
            history.append({"role": "assistant",  "content": result.text})
            if len(history) > MAX_HISTORY_MESSAGES * 2:
                history = history[-(MAX_HISTORY_MESSAGES * 2):]
            print(f"[{result.model} | {result.duration_s:.1f}s | prompt={result.prompt_tokens} completion={result.completion_tokens}]")
        except PartialResponseError as exc:
            print(f"\n[partial] {exc.partial_text}")
        except OllamaError as exc:
            print(f"\n[error] {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
