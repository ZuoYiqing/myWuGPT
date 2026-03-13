"""Policy profile model (MVP required fields + runtime constraints)."""
from dataclasses import dataclass, field


@dataclass(slots=True)
class PolicyProfile:
    profile_id: str
    max_risk_level: str
    allowed_skill_groups: list[str] = field(default_factory=list)
    forbidden_skill_groups: list[str] = field(default_factory=list)
    confirm_rules: list[dict[str, object]] = field(default_factory=list)
    requires_higher_confirmation_for: list[str] = field(default_factory=list)
    target_validation_rules: list[dict[str, object]] = field(default_factory=list)
    preemption_behavior: dict[str, str] = field(default_factory=dict)
    degradation_behavior: dict[str, str] = field(default_factory=dict)
    fallback_behavior: dict[str, str] = field(default_factory=dict)
    recovery_behavior: dict[str, str] = field(default_factory=dict)
    max_parallel_actions: int = 1
    runtime_constraints: dict[str, object] = field(default_factory=dict)
