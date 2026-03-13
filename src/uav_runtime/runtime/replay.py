"""Replay helper skeleton for loading latest audit events."""
import json
from pathlib import Path


class ReplayStore:
    def __init__(self, path: str = "logs/audit/events.jsonl") -> None:
        self.path = Path(path)

    def load_last(self) -> dict | None:
        if not self.path.exists():
            return None
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return None
        return json.loads(lines[-1])
