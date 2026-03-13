"""Skill registry skeleton with explicit registration API."""
from uav_runtime.skills.base import Skill


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, key: str, skill: Skill) -> None:
        self._skills[key] = skill

    def get(self, key: str) -> Skill:
        return self._skills[key]

    def keys(self) -> list[str]:
        return list(self._skills.keys())
