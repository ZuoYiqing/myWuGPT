"""Base skill abstractions for semantic capability wrappers."""
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class SkillMetadata:
    name: str
    version: str
    skill_group: str
    safety_level: str
    permission_level: str
    timeout_ms: int
    audit_tags: list[str] = field(default_factory=list)


class Skill(Protocol):
    metadata: SkillMetadata

    def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return semantic execution intent (not low-level protocol frames)."""
        ...
