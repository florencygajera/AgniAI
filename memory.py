from collections import deque
from typing import Deque, Dict, List

from config import MEMORY_MAX_MESSAGES


class ConversationMemory:
    def __init__(self, max_messages: int = MEMORY_MAX_MESSAGES) -> None:
        self._messages: Deque[Dict[str, str]] = deque(maxlen=max_messages)

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def history(self) -> List[Dict[str, str]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

