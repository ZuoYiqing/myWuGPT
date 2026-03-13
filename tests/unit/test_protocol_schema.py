"""Unit tests for protocol schema and validation skeleton contracts."""
from datetime import datetime

from uav_runtime.protocol.enums import CommandSource, DecisionCode, MessageType
from uav_runtime.protocol.schema import Envelope, HandoverPlan, PolicyDecisionPayload
from uav_runtime.protocol.validation import validate_envelope_shape, validate_policy_decision_payload


def test_envelope_required_fields_min_shape() -> None:
    env = Envelope(
        protocol_version="1.0",
        schema_id="uav.action_request.v1",
        message_type=MessageType.ACTION_REQUEST,
        message_id="m1",
        trace_id="t1",
        mission_id="mission-1",
        source=CommandSource.GROUND_STATION,
        target="policy_gate",
        timestamp=datetime.utcnow(),
        payload={"request_id": "r1"},
    )
    validate_envelope_shape(env)


def test_preempt_requires_handover_mode() -> None:
    payload = PolicyDecisionPayload(
        request_id="r1",
        decision_code=DecisionCode.PREEMPT,
        effective_scope="self_only",
        effective_profile_id="profile-standard",
        handover_plan=HandoverPlan(mode="suspend"),
        primary_reason_code="PREEMPT_ALLOWED_HIGHER_PRIORITY",
    )
    validate_policy_decision_payload(payload)
