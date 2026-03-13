"""Minimal in-memory task queue skeleton."""
from collections import deque
from typing import Any


class TaskQueue:
    """Simple FIFO queue for action requests (MVP skeleton)."""

    def __init__(self) -> None:
        self._q: deque[dict[str, Any]] = deque()

    def enqueue(self, item: dict[str, Any]) -> None:
        self._q.append(item)

    def dequeue(self) -> dict[str, Any] | None:
        if not self._q:
            return None
        return self._q.popleft()

    def __len__(self) -> int:
        return len(self._q)
