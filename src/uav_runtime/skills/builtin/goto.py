"""Goto skill stub (semantic capability, no direct protocol frame control)."""
from uav_runtime.skills.base import SkillMetadata


class GotoSkill:
    metadata = SkillMetadata(
        name="goto",
        version="0.1.0",
        skill_group="navigation",
        safety_level="R2",
        permission_level="operator",
        timeout_ms=3000,
        audit_tags=["builtin", "goto"],
    )

    def execute(self, params: dict) -> dict:
        """TODO: return semantic intent for adapter gateway mapping."""
        return {"skill": "goto", "status": "stub", "params": params}
