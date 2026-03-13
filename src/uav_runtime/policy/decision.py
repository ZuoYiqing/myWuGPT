"""Policy decision envelope used as unified gate output contract."""
from dataclasses import dataclass, field
from typing import Any

from uav_runtime.protocol.enums import DecisionCode


@dataclass(slots=True)
class HandoverPlanOut:
    mode: str = "none"
    takeover_target_request_id: str | None = None
    resume_policy: str | None = None


@dataclass(slots=True)
class PolicyDecisionEnvelope:
    decision_code: DecisionCode
    effective_profile_id: str
    effective_scope: str
    enforced_constraints: dict[str, Any] = field(default_factory=dict)
    handover_plan: HandoverPlanOut = field(default_factory=HandoverPlanOut)
    primary_reason_code: str | None = None
    secondary_reason_codes: list[str] = field(default_factory=list)
    audit_tags: list[str] = field(default_factory=list)
    error_code: str | None = None
