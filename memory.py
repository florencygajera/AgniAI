from collections import deque
from typing import Deque, Dict, List

from config import MEMORY_MAX_MESSAGES


class ConversationMemory:
    """
    Sliding-window conversation history.
    Stores at most `max_messages` total messages (user + assistant combined).
    Oldest messages are evicted automatically as the deque fills.
    """

    def __init__(self, max_messages: int = MEMORY_MAX_MESSAGES) -> None:
        self._messages: Deque[Dict[str, str]] = deque(maxlen=max_messages)

    # ── Mutation ───────────────────────────────────────────────────────────

    def add(self, role: str, content: str) -> None:
        """Append a message.  role must be 'user' or 'assistant'."""
        if role not in {"user", "assistant"}:
            raise ValueError(f"Invalid role: {role!r}. Must be 'user' or 'assistant'.")
        self._messages.append({"role": role, "content": content})

    def clear(self) -> None:
        """Wipe all history."""
        self._messages.clear()

    # ── Query ──────────────────────────────────────────────────────────────

    def history(self) -> List[Dict[str, str]]:
        """Return the current message list (oldest → newest)."""
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)