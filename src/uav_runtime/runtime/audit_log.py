"""Audit logger skeleton writing newline JSON records to local file."""
import json
from pathlib import Path
from typing import Any


class AuditLog:
    def __init__(self, path: str = "logs/audit/events.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
