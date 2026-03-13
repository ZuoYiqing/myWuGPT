"""Datamodels for canonical protocol_json envelope and payload shapes."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .enums import CommandSource, DecisionCode, MessageType


@dataclass(slots=True)
class Envelope:
    """Generic control-plane envelope used by all message types."""

    protocol_version: str
    schema_id: str
    message_type: MessageType
    message_id: str
    trace_id: str
    mission_id: str
    source: CommandSource
    target: str
    timestamp: datetime
    payload: dict[str, Any]
    correlation_id: str | None = None
    causation_id: str | None = None
    ttl: int | None = None
    audit_ref: str | None = None
    replay_ref: str | None = None


@dataclass(slots=True)
class ActionRequestPayload:
    """Action request with authoritative vs hint fields per contract."""

    request_id: str
    action_type: str
    skill_group: str
    target_set: list[str]
    requested_scope: str  # hint
    risk_hint: str
    idempotency_key: str
    requires_confirmation_hint: bool | None = None
    priority_hint: int | None = None
    delegation_id: str | None = None


@dataclass(slots=True)
class HandoverPlan:
    """Handover plan is required when decision_code == PREEMPT."""

    mode: str = "none"
    takeover_target_request_id: str | None = None
    resume_policy: str | None = None


@dataclass(slots=True)
class PolicyDecisionPayload:
    """Internal policy decision event payload."""

    request_id: str
    decision_code: DecisionCode
    effective_scope: str
    effective_profile_id: str
    handover_plan: HandoverPlan = field(default_factory=HandoverPlan)
    primary_reason_code: str | None = None
    secondary_reason_codes: list[str] = field(default_factory=list)
    error_code: str | None = None
    effective_risk_level: str | None = None
    enforced_constraints: dict[str, Any] = field(default_factory=dict)
    audit_tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ActionResultPayload:
    """Result payload produced by adapter execution path."""

    request_id: str
    status: str
    code: str
    message: str
    decision_code: DecisionCode | None = None
    primary_reason_code: str | None = None
    secondary_reason_codes: list[str] = field(default_factory=list)
    error_code: str | None = None
    adapter_trace: dict[str, Any] = field(default_factory=dict)
