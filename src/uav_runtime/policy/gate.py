"""Unified policy gate: the only policy decision entrypoint in MVP."""
from datetime import datetime

from uav_runtime.policy.context import PolicyContext
from uav_runtime.policy.decision import HandoverPlanOut, PolicyDecisionEnvelope
from uav_runtime.policy.delegation import DelegationGrant
from uav_runtime.policy.profile import PolicyProfile
from uav_runtime.protocol.enums import DecisionCode
from uav_runtime.protocol.schema import ActionRequestPayload


def unified_policy_gate(
    *,
    context: PolicyContext,
    request: ActionRequestPayload,
    delegations: list[DelegationGrant],
    profile: PolicyProfile,
    now: datetime,
) -> PolicyDecisionEnvelope:
    """Run fixed policy decision sequence (skeleton only).

    Fixed decision steps (TODO implement):
    1) identity/source checks
    2) request/ttl shape checks
    3) delegation validity checks
    4) source priority compute
    5) preemption decision
    6) scope shrinking
    7) profile checks
    8) target validation
    9) risk/confirmation decision
    10) runtime constraints
    11) final decision envelope
    12) audit tag preparation
    """
    # TODO: implement complete contract-aligned policy decision logic.
    if context.link_state.value == "lost" and request.skill_group != "fallback":
        return PolicyDecisionEnvelope(
            decision_code=DecisionCode.DENY,
            effective_profile_id=context.active_profile_id,
            effective_scope="self_only",
            primary_reason_code="LINK_LOST_FALLBACK_ONLY",
            secondary_reason_codes=["LINK_STATE_INCOMPATIBLE_WITH_ACTION"],
            audit_tags=["skeleton", "link_lost_guard"],
        )

    # Skeleton default path: allow with nullable primary_reason_code.
    return PolicyDecisionEnvelope(
        decision_code=DecisionCode.ALLOW,
        effective_profile_id=context.active_profile_id,
        effective_scope=request.requested_scope,
        enforced_constraints={"rate_limit": "TODO", "param_clamp": "TODO"},
        handover_plan=HandoverPlanOut(mode="none"),
        primary_reason_code=None,
        audit_tags=["skeleton"],
    )
