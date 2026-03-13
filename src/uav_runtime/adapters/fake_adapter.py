"""Fake adapter that simulates deterministic execution responses."""
from datetime import datetime


class FakeAdapter:
    def execute(self, command: dict) -> dict:
        return {
            "status": "success",
            "code": "FAKE_OK",
            "message": "executed by fake adapter",
            "command_echo": command,
            "timestamp": datetime.utcnow().isoformat(),
        }
