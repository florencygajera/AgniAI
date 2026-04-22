from config import DATA_DIR, INDEX_DIR, TOP_K
from ingest import ingest_pdf, ingest_text, ingest_url
from memory import ConversationMemory
from rag import build_context, call_llm, search


BANNER = r"""
   ___                  _ ___    ___ 
  / _ |___  ___ _  ___ (_) _ \  / _ \
 / __ / _ \/ _ \ |/ _ \| | | | | (_) |
/_/ |_\___/_//_/___|___/|_|___/  \___/
"""


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _handle_ingest(command: str) -> None:
    parts = command.split(maxsplit=2)
    if len(parts) < 3:
        print("Usage: /ingest pdf <path> | /ingest url <url> | /ingest text <text>")
        return

    kind = parts[1].lower()
    target = parts[2].strip()

    try:
        if kind == "pdf":
            count = ingest_pdf(target)
        elif kind == "url":
            count = ingest_url(target)
        elif kind == "text":
            count = ingest_text(target)
        else:
            print("Unsupported ingest type. Use pdf, url, or text.")
            return
        print(f"Ingested {count} chunks.")
    except Exception as exc:
        print(f"Ingestion failed: {exc}")


def run_chat() -> None:
    _ensure_dirs()
    memory = ConversationMemory()
    print(BANNER)
    print("AgniAI is ready. Type /exit to quit.")
    print("Commands: /ingest pdf <path>, /ingest url <url>, /ingest text <text>")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("Goodbye.")
            break

        if user_input.lower().startswith("/ingest "):
            _handle_ingest(user_input)
            continue

        docs = search(user_input, top_k=TOP_K)
        context = build_context(docs)
        if not context:
            print("\nAgniAI:\nI don't know.")
            memory.add("user", user_input)
            memory.add("assistant", "I don't know.")
            continue

        history = memory.history()
        prompt = (
            f"Question: {user_input}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Answer in a clean, structured format using only the retrieved context."
        )

        try:
            answer = call_llm(prompt, history=history)
        except Exception as exc:
            answer = f"I don't know.\n\nReason: {exc}"

        print(f"\nAgniAI:\n{answer}")
        memory.add("user", user_input)
        memory.add("assistant", answer)


if __name__ == "__main__":
    run_chat()
