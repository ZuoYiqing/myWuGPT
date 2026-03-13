"""Validation entry points for protocol envelope and policy invariants."""
from .enums import DecisionCode, MessageType
from .schema import Envelope, PolicyDecisionPayload


REQUIRED_ENVELOPE_FIELDS = (
    "protocol_version",
    "schema_id",
    "message_type",
    "message_id",
    "trace_id",
    "mission_id",
    "source",
    "target",
    "timestamp",
    "payload",
)


def validate_envelope_shape(envelope: Envelope) -> None:
    """Basic shape validation for required envelope fields.

    TODO: extend with schema-id specific payload validators.
    """
    for name in REQUIRED_ENVELOPE_FIELDS:
        if getattr(envelope, name) in (None, ""):
            raise ValueError(f"missing required envelope field: {name}")


def validate_policy_decision_payload(payload: PolicyDecisionPayload) -> None:
    """Contract checks for decision semantics.

    - PREEMPT requires handover_plan.mode != none
    - ALLOW may keep primary_reason_code as None
    """
    if payload.decision_code == DecisionCode.PREEMPT and payload.handover_plan.mode == "none":
        raise ValueError("PREEMPT requires handover_plan.mode != none")
    if payload.decision_code in {
        DecisionCode.DENY,
        DecisionCode.DEFER,
        DecisionCode.REQUIRE_CONFIRM,
        DecisionCode.PREEMPT,
    } and not payload.primary_reason_code:
        raise ValueError("non-ALLOW decisions require primary_reason_code")


def is_internal_policy_event(message_type: MessageType) -> bool:
    """Policy decision event is internal cross-module message in MVP."""
    return message_type == MessageType.POLICY_DECISION_EVENT
