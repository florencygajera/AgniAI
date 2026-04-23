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
# CONFIG
# =============================================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat"
TAGS_ENDPOINT = f"{OLLAMA_BASE_URL}/api/tags"

# Best CPU-friendly default. Override with: set OLLAMA_MODEL=...
MODEL_NAME = os.getenv("OLLAMA_MODEL", "phi3:mini")
FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "OLLAMA_FALLBACK_MODELS",
        "llama3.2:3b, llama3:8b, mistral:7b, phi3:mini",
    ).split(",")
    if m.strip()
]

TIMEOUT_CONNECT = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "10"))
TIMEOUT_READ = float(os.getenv("OLLAMA_READ_TIMEOUT", "300"))
MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))

# Keep generation tight for CPU-only systems.
MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "192"))
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "1536"))
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
TOP_K = int(os.getenv("OLLAMA_TOP_K", "40"))
TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.9"))
REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.1"))
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")

# RAG-friendly limits
MAX_HISTORY_MESSAGES = int(os.getenv("OLLAMA_MAX_HISTORY_MESSAGES", "8"))
MAX_RAG_CHARS = int(os.getenv("OLLAMA_MAX_RAG_CHARS", "6000"))

SYSTEM_PROMPT = (
    "You are a fast, reliable local assistant. "
    "Answer concisely. If retrieved context is provided, use it first and do not invent facts."
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
# RAG HOOKS
# =============================================================================


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = max(0, limit - head - 20)
    return f"{text[:head].rstrip()}\n\n...[truncated]...\n\n{text[-tail:].lstrip()}"


def build_rag_context(query: str) -> str:
    """
    Optional hook into your FAISS/RAG layer.
    If rag.py is available, this pulls top hits and formats them.
    """
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
            f"Retrieved context:\n{context}\n\n"
            "Answer using only the retrieved context. If the context is insufficient, say so plainly."
        )
    else:
        user_content = query

    messages.append({"role": "user", "content": user_content})
    return messages


# =============================================================================
# OLLAMA CLIENT
# =============================================================================


class OllamaError(RuntimeError):
    pass


class PartialResponseError(OllamaError):
    def __init__(self, message: str, partial_text: str):
        super().__init__(message)
        self.partial_text = partial_text


def _installed_models(session: requests.Session) -> List[str]:
    try:
        resp = session.get(TAGS_ENDPOINT, timeout=(TIMEOUT_CONNECT, 15))
        resp.raise_for_status()
        payload = resp.json()
        return [m["name"] for m in payload.get("models", []) if m.get("name")]
    except Exception:
        return []


def _candidate_models(requested: str, installed: List[str]) -> List[str]:
    ordered: List[str] = []
    for name in [requested, *FALLBACK_MODELS, *installed]:
        if name and name not in ordered:
            ordered.append(name)
    return ordered


def _iter_ndjson(resp: requests.Response) -> Iterable[dict]:
    buffer = b""
    for chunk in resp.iter_content(chunk_size=4096):
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
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
            "num_predict": MAX_TOKENS,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "repeat_penalty": REPEAT_PENALTY,
        },
    }

    started = False
    pieces: List[str] = []
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    start = time.time()

    try:
        with session.post(
            CHAT_ENDPOINT,
            json=payload,
            stream=True,
            timeout=(TIMEOUT_CONNECT, TIMEOUT_READ),
        ) as resp:
            if resp.status_code >= 400:
                body = resp.text[:1000].strip()
                raise OllamaError(
                    f"Ollama returned HTTP {resp.status_code} for model '{model}'. "
                    f"{body or 'No response body.'}"
                )

            for event in _iter_ndjson(resp):
                if "error" in event:
                    raise OllamaError(str(event["error"]))

                if event.get("done"):
                    prompt_tokens = event.get("prompt_eval_count", prompt_tokens)
                    completion_tokens = event.get("eval_count", completion_tokens)
                    break

                token = event.get("message", {}).get("content", "")
                if token:
                    started = True
                    pieces.append(token)
                    if stream_tokens:
                        if on_token is not None:
                            on_token(token)
                        else:
                            sys.stdout.write(token)
                            sys.stdout.flush()

    except requests.Timeout as exc:
        if started:
            raise PartialResponseError(
                f"Timeout while streaming from '{model}'. The model started responding, but the connection stalled.",
                "".join(pieces),
            ) from exc
        raise OllamaError(
            f"Timed out contacting Ollama while waiting for '{model}'. "
            f"Increase OLLAMA_READ_TIMEOUT or use a smaller model."
        ) from exc
    except requests.ConnectionError as exc:
        if started:
            raise PartialResponseError(
                f"Connection dropped while streaming from '{model}'.",
                "".join(pieces),
            ) from exc
        raise OllamaError(
            "Could not connect to Ollama. Make sure the server is running on "
            f"{OLLAMA_BASE_URL}."
        ) from exc
    except json.JSONDecodeError as exc:
        if started:
            raise PartialResponseError(
                f"Received malformed streaming JSON from '{model}'.",
                "".join(pieces),
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
    installed = _installed_models(session)
    candidates = _candidate_models(requested_model, installed)

    last_error: Optional[str] = None
    for model in candidates:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return _ollama_chat_once(
                    session,
                    model,
                    messages,
                    stream_tokens=stream_tokens,
                )
            except PartialResponseError as exc:
                # Don't auto-retry after tokens have already been emitted.
                if exc.partial_text:
                    if not stream_tokens:
                        print(exc.partial_text, end="", flush=True)
                    print()
                raise
            except OllamaError as exc:
                last_error = str(exc)
                if attempt < MAX_RETRIES:
                    time.sleep(0.75 * attempt)
                    continue
                break

    raise OllamaError(
        "All Ollama model attempts failed.\n"
        f"Last error: {last_error}\n"
        f"Tried: {', '.join(candidates) if candidates else requested_model}"
    )


# =============================================================================
# CLI
# =============================================================================


def print_banner() -> None:
    print("=" * 72)
    print("Ollama CPU-safe streaming chat")
    print(f"Base URL : {OLLAMA_BASE_URL}")
    print(f"Model    : {MODEL_NAME}")
    print(f"Timeout  : connect={TIMEOUT_CONNECT}s read={TIMEOUT_READ}s")
    print(f"Context  : num_ctx={NUM_CTX} max_tokens={MAX_TOKENS}")
    print("=" * 72)


def help_text() -> None:
    print(
        "\nCommands:\n"
        "  /exit, /quit   Exit\n"
        "  /model NAME    Switch model\n"
        "  /clear         Clear history\n"
        "  /tags          List installed models\n"
        "  /health        Check Ollama API\n"
        "  /help          Show this help\n"
    )


def main() -> int:
    session = requests.Session()
    model = MODEL_NAME
    history: List[dict] = []

    print_banner()
    help_text()

    try:
        resp = session.get(TAGS_ENDPOINT, timeout=(TIMEOUT_CONNECT, 15))
        resp.raise_for_status()
        installed = [m["name"] for m in resp.json().get("models", []) if m.get("name")]
        print(f"\nInstalled models: {', '.join(installed) if installed else '(none detected)'}\n")
    except Exception as exc:
        print(f"\n[warning] Could not list installed models: {exc}\n")

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
            help_text()
            continue
        if low == "/clear":
            history.clear()
            print("History cleared.")
            continue
        if low == "/tags":
            try:
                resp = session.get(TAGS_ENDPOINT, timeout=(TIMEOUT_CONNECT, 15))
                resp.raise_for_status()
                installed = [m["name"] for m in resp.json().get("models", []) if m.get("name")]
                print(", ".join(installed) if installed else "(none detected)")
            except Exception as exc:
                print(f"[error] {exc}")
            continue
        if low == "/health":
            try:
                resp = session.get(TAGS_ENDPOINT, timeout=(TIMEOUT_CONNECT, 15))
                resp.raise_for_status()
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
            result = chat_with_fallback(
                session,
                model,
                messages,
                stream_tokens=True,
            )
            print()
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": result.text})

            if len(history) > MAX_HISTORY_MESSAGES * 2:
                history = history[-MAX_HISTORY_MESSAGES * 2 :]

            if result.prompt_tokens is not None or result.completion_tokens is not None:
                print(
                    f"[model={result.model} time={result.duration_s:.1f}s "
                    f"prompt={result.prompt_tokens} completion={result.completion_tokens}]"
                )
        except PartialResponseError as exc:
            print("\n[warning] The connection dropped after a partial answer.")
            if exc.partial_text:
                print(exc.partial_text)
        except OllamaError as exc:
            print(f"\n[error] {exc}")
            print("Try a smaller model, shorter context, or a longer OLLAMA_READ_TIMEOUT.")


if __name__ == "__main__":
    raise SystemExit(main())
