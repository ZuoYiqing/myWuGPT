"""Unit tests for policy contract shapes and gate skeleton behavior."""
from datetime import datetime, timedelta

from uav_runtime.policy.context import PolicyContext
from uav_runtime.policy.gate import unified_policy_gate
from uav_runtime.policy.profile import PolicyProfile
from uav_runtime.protocol.enums import AutonomyState, CommandSource, DecisionCode, LinkState
from uav_runtime.protocol.schema import ActionRequestPayload


def _base_context(link_state: LinkState = LinkState.OK) -> PolicyContext:
    return PolicyContext(
        mission_id="m1",
        current_phase="execute",
        link_state=link_state,
        autonomy_state=AutonomyState.NOMINAL,
        active_controller_source=CommandSource.GROUND_STATION,
        active_profile_id="profile-standard",
        active_scope="self_only",
    )


def _base_request() -> ActionRequestPayload:
    return ActionRequestPayload(
        request_id="r1",
        action_type="hover",
        skill_group="navigation",
        target_set=["self_uav"],
        requested_scope="self_only",
        risk_hint="R1",
        idempotency_key="idem-r1",
    )


def test_allow_path_shape_skeleton() -> None:
    d = unified_policy_gate(
        context=_base_context(),
        request=_base_request(),
        delegations=[],
        profile=PolicyProfile(profile_id="profile-standard", max_risk_level="R3"),
        now=datetime.utcnow(),
    )
    assert d.decision_code == DecisionCode.ALLOW


def test_link_lost_deny_path_shape_skeleton() -> None:
    d = unified_policy_gate(
        context=_base_context(LinkState.LOST),
        request=_base_request(),
        delegations=[],
        profile=PolicyProfile(profile_id="profile-standard", max_risk_level="R3"),
        now=datetime.utcnow() + timedelta(seconds=1),
    )
    assert d.decision_code == DecisionCode.DENY
    assert d.primary_reason_code == "LINK_LOST_FALLBACK_ONLY"


def test_require_confirm_path_todo() -> None:
    # TODO: implement once confirmation rules are wired.
    assert True


def test_preempt_path_todo() -> None:
    # TODO: implement once preemption rules are wired.
    assert True
