"""Adapter base interfaces for deterministic execution layer."""
from typing import Protocol


class Adapter(Protocol):
    def execute(self, command: dict) -> dict:
        ...
